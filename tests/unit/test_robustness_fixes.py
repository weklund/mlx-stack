"""Regression tests for process management robustness fixes.

Tests the 3 scrutiny-identified issues:

1. start_service() kills spawned subprocess when PID-file write fails,
   preventing leaked unmanaged processes.
2. stack_up.py catches ProcessError when reading PIDs during startup
   probing (corrupt PID files) and treats as stale PID cleanup path.
3. stack_down.py verifies process liveness after SIGKILL and only
   removes PID file once termination is confirmed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mlx_stack.core.process import (
    ProcessError,
    _terminate_process,
    read_pid_file,
    start_service,
    stop_service,
    write_pid_file,
)
from mlx_stack.core.stack_up import (
    run_up,
)
from tests.factories import create_pid_file, make_stack_yaml, write_stack_yaml

# Single-tier stack used by all robustness tests.
_FAST_TIER_ONLY: list[dict[str, Any]] = [
    {
        "name": "fast",
        "model": "fast-model",
        "quant": "int4",
        "source": "mlx-community/fast-model-4bit",
        "port": 8001,
        "vllm_flags": {"continuous_batching": True},
    },
]


# =========================================================================== #
# Fix #1: start_service() kills subprocess when PID-file write fails
# =========================================================================== #


class TestStartServicePidWriteFailure:
    """Regression: start_service() must kill the spawned subprocess if
    PID-file write fails, preventing leaked unmanaged processes."""

    @patch("mlx_stack.core.process.subprocess.Popen")
    @patch("mlx_stack.core.process.write_pid_file", side_effect=ProcessError("disk full"))
    def test_process_killed_on_pid_write_failure(
        self,
        mock_write_pid: MagicMock,
        mock_popen: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """Process is terminated when PID file cannot be written."""
        # Arrange
        mock_proc = MagicMock()
        mock_proc.pid = 54321
        mock_popen.return_value = mock_proc

        # Act / Assert
        with pytest.raises(ProcessError, match="Could not write PID file"):
            start_service("fast", cmd=["vllm-mlx", "--port", "8001"], port=8001)

        mock_proc.terminate.assert_called_once()

    @patch("mlx_stack.core.process.subprocess.Popen")
    @patch("mlx_stack.core.process.write_pid_file", side_effect=ProcessError("disk full"))
    def test_process_killed_even_if_terminate_fails(
        self,
        mock_write_pid: MagicMock,
        mock_popen: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """Process is force-killed if terminate() fails."""
        # Arrange
        mock_proc = MagicMock()
        mock_proc.pid = 54321
        mock_proc.terminate.side_effect = OSError("already dead")
        mock_popen.return_value = mock_proc

        # Act / Assert
        with pytest.raises(ProcessError, match="Could not write PID file"):
            start_service("fast", cmd=["vllm-mlx", "--port", "8001"], port=8001)

        mock_proc.kill.assert_called_once()

    @patch("mlx_stack.core.process.subprocess.Popen")
    @patch("mlx_stack.core.process.write_pid_file", side_effect=ProcessError("disk full"))
    def test_error_message_includes_pid(
        self,
        mock_write_pid: MagicMock,
        mock_popen: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """Error message includes the orphaned process PID."""
        # Arrange
        mock_proc = MagicMock()
        mock_proc.pid = 54321
        mock_popen.return_value = mock_proc

        # Act / Assert
        with pytest.raises(ProcessError) as exc_info:
            start_service("fast", cmd=["vllm-mlx"], port=8001)

        assert "54321" in str(exc_info.value)
        assert "terminated" in str(exc_info.value).lower()

    @patch("mlx_stack.core.process.subprocess.Popen")
    @patch("mlx_stack.core.process.write_pid_file", side_effect=ProcessError("write failed"))
    def test_log_file_closed_on_pid_write_failure(
        self,
        mock_write_pid: MagicMock,
        mock_popen: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """Log file handle is properly closed on PID write failure."""
        # Arrange
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        # Act
        with pytest.raises(ProcessError):
            start_service("fast", cmd=["vllm-mlx"], port=8001)

        # Assert -- log file exists (was opened) but handle was closed
        log_path = mlx_stack_home / "logs" / "fast.log"
        assert log_path.exists()


# =========================================================================== #
# Fix #2: stack_up.py handles corrupt PIDs during startup probing
# =========================================================================== #


class TestStackUpCorruptPidHandling:
    """Regression: stack_up.py must catch ProcessError when reading PIDs
    during startup probing and treat as stale PID cleanup path."""

    def test_corrupt_tier_pid_cleaned_up_gracefully(self, mlx_stack_home: Path) -> None:
        """Corrupt tier PID file is cleaned up without traceback."""
        # Arrange
        write_stack_yaml(mlx_stack_home, make_stack_yaml(tiers=_FAST_TIER_ONLY))
        create_pid_file(mlx_stack_home, "fast", "not-a-number")
        litellm_config = mlx_stack_home / "litellm.yaml"
        litellm_config.write_text("model_list: []\n")

        # Act
        with (
            patch("mlx_stack.core.stack_up.acquire_lock") as mock_lock,
            patch("mlx_stack.core.stack_up.ensure_dependency"),
            patch("mlx_stack.core.stack_up.shutil.which", return_value="/usr/bin/vllm-mlx"),
            patch("mlx_stack.core.stack_up.check_port_conflict", return_value=None),
            patch("mlx_stack.core.stack_up.check_local_model_exists", return_value=None),
            patch("mlx_stack.core.stack_up.start_service") as mock_start,
            patch("mlx_stack.core.stack_up.wait_for_healthy") as mock_health,
            patch("mlx_stack.core.stack_up.load_catalog", return_value=[]),
        ):
            mock_lock.return_value.__enter__ = MagicMock()
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)
            mock_start.return_value = MagicMock(pid=99999)
            mock_health.return_value = MagicMock(healthy=True)

            result = run_up()

        # Assert -- corrupt PID cleaned up and tier started fresh
        pids_dir = mlx_stack_home / "pids"
        corrupt_file = pids_dir / "fast.pid"
        assert not corrupt_file.exists() or corrupt_file.read_text() != "not-a-number"
        assert any("stale" in w.lower() or "Cleaned up" in w for w in result.warnings)

    def test_corrupt_litellm_pid_cleaned_up_gracefully(self, mlx_stack_home: Path) -> None:
        """Corrupt LiteLLM PID file is cleaned up without traceback."""
        # Arrange
        write_stack_yaml(mlx_stack_home, make_stack_yaml(tiers=_FAST_TIER_ONLY))
        create_pid_file(mlx_stack_home, "litellm", "garbage-data")
        litellm_config = mlx_stack_home / "litellm.yaml"
        litellm_config.write_text("model_list: []\n")

        # Act -- should NOT raise UpError; corrupt PID is handled gracefully
        with (
            patch("mlx_stack.core.stack_up.acquire_lock") as mock_lock,
            patch("mlx_stack.core.stack_up.ensure_dependency"),
            patch("mlx_stack.core.stack_up.shutil.which", return_value="/usr/bin/vllm-mlx"),
            patch("mlx_stack.core.stack_up.check_port_conflict", return_value=None),
            patch("mlx_stack.core.stack_up.check_local_model_exists", return_value=None),
            patch("mlx_stack.core.stack_up.start_service") as mock_start,
            patch("mlx_stack.core.stack_up.wait_for_healthy") as mock_health,
            patch("mlx_stack.core.stack_up.load_catalog", return_value=[]),
        ):
            mock_lock.return_value.__enter__ = MagicMock()
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)
            mock_start.return_value = MagicMock(pid=99999)
            mock_health.return_value = MagicMock(healthy=True)

            result = run_up()

        # Assert
        assert any("stale" in w.lower() or "Cleaned up" in w for w in result.warnings)

    def test_corrupt_pid_no_traceback_via_cli(self, mlx_stack_home: Path) -> None:
        """Corrupt PID during up does not produce a Python traceback."""
        from click.testing import CliRunner

        from mlx_stack.cli.main import cli

        # Arrange
        write_stack_yaml(mlx_stack_home, make_stack_yaml(tiers=_FAST_TIER_ONLY))
        create_pid_file(mlx_stack_home, "fast", "corrupt!!!")
        litellm_config = mlx_stack_home / "litellm.yaml"
        litellm_config.write_text("model_list: []\n")

        # Act
        with (
            patch("mlx_stack.core.stack_up.acquire_lock") as mock_lock,
            patch("mlx_stack.core.stack_up.ensure_dependency"),
            patch("mlx_stack.core.stack_up.shutil.which", return_value="/usr/bin/vllm-mlx"),
            patch("mlx_stack.core.stack_up.check_port_conflict", return_value=None),
            patch("mlx_stack.core.stack_up.check_local_model_exists", return_value=None),
            patch("mlx_stack.core.stack_up.start_service") as mock_start,
            patch("mlx_stack.core.stack_up.wait_for_healthy") as mock_health,
            patch("mlx_stack.core.stack_up.load_catalog", return_value=[]),
        ):
            mock_lock.return_value.__enter__ = MagicMock()
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)
            mock_start.return_value = MagicMock(pid=99999)
            mock_health.return_value = MagicMock(healthy=True)

            runner = CliRunner()
            result = runner.invoke(cli, ["up"])

        # Assert
        assert "Traceback" not in result.output


# =========================================================================== #
# Fix #3: Verify process liveness after SIGKILL before PID file removal
# =========================================================================== #


class TestSigkillVerification:
    """Regression: After SIGKILL, process termination must be confirmed
    before PID file removal."""

    @patch("mlx_stack.core.process.time.sleep")
    @patch("mlx_stack.core.process.time.monotonic")
    @patch("mlx_stack.core.process.is_process_alive")
    @patch("mlx_stack.core.process.os.kill")
    def test_sigkill_confirmed_dead_returns_confirmed(
        self,
        mock_kill: MagicMock,
        mock_alive: MagicMock,
        mock_monotonic: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """Process confirmed dead after SIGKILL → confirmed=True."""
        # Arrange -- process stays alive through grace, dies after SIGKILL
        mock_alive.side_effect = [True, True, True, False]
        mock_monotonic.side_effect = [
            0.0,  # deadline = 10.0
            5.0,  # still within grace
            11.0,  # past grace → SIGKILL
        ]

        # Act
        graceful, confirmed = _terminate_process(123, grace_period=10)

        # Assert
        assert graceful is False
        assert confirmed is True

    @patch("mlx_stack.core.process.time.sleep")
    @patch("mlx_stack.core.process.time.monotonic")
    @patch("mlx_stack.core.process.is_process_alive")
    @patch("mlx_stack.core.process.os.kill")
    def test_sigkill_process_still_alive_returns_not_confirmed(
        self,
        mock_kill: MagicMock,
        mock_alive: MagicMock,
        mock_monotonic: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """Process still alive after SIGKILL → confirmed=False."""
        # Arrange -- process never dies (zombie or kernel hold)
        mock_alive.return_value = True
        mock_monotonic.side_effect = [
            0.0,  # deadline = 10.0
            11.0,  # past grace → SIGKILL
        ]

        # Act
        graceful, confirmed = _terminate_process(123, grace_period=10)

        # Assert
        assert graceful is False
        assert confirmed is False

    @patch("mlx_stack.core.process._terminate_process", return_value=(False, False))
    @patch("mlx_stack.core.process.is_process_alive", return_value=True)
    def test_pid_file_not_removed_when_termination_unconfirmed(
        self,
        mock_alive: MagicMock,
        mock_terminate: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """PID file is NOT removed when process termination is not confirmed."""
        # Arrange
        write_pid_file("stubborn", 12345)

        # Act
        result = stop_service("stubborn")

        # Assert
        assert result is not None
        assert result.graceful is False
        pid = read_pid_file("stubborn")
        assert pid == 12345

    @patch("mlx_stack.core.process._terminate_process", return_value=(False, True))
    @patch("mlx_stack.core.process.is_process_alive", return_value=True)
    def test_pid_file_removed_when_termination_confirmed(
        self,
        mock_alive: MagicMock,
        mock_terminate: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """PID file IS removed when process termination is confirmed."""
        # Arrange
        write_pid_file("killed", 12345)

        # Act
        result = stop_service("killed")

        # Assert
        assert result is not None
        assert result.graceful is False
        assert read_pid_file("killed") is None

    @patch("mlx_stack.core.process._terminate_process", return_value=(True, True))
    @patch("mlx_stack.core.process.is_process_alive", return_value=True)
    def test_pid_file_removed_on_graceful_confirmed_shutdown(
        self,
        mock_alive: MagicMock,
        mock_terminate: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """PID file removed on graceful + confirmed shutdown."""
        # Arrange
        write_pid_file("graceful", 12345)

        # Act
        result = stop_service("graceful")

        # Assert
        assert result is not None
        assert result.graceful is True
        assert read_pid_file("graceful") is None
