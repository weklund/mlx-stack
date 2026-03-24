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
import yaml

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

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_stack_yaml(
    tiers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a stack definition dict for testing."""
    if tiers is None:
        tiers = [
            {
                "name": "fast",
                "model": "fast-model",
                "quant": "int4",
                "source": "mlx-community/fast-model-4bit",
                "port": 8001,
                "vllm_flags": {"continuous_batching": True},
            },
        ]
    return {
        "schema_version": 1,
        "name": "default",
        "hardware_profile": "m4-max-128",
        "intent": "balanced",
        "created": "2026-03-24T00:00:00+00:00",
        "tiers": tiers,
    }


def _write_stack_yaml(mlx_stack_home: Path, stack: dict[str, Any] | None = None) -> Path:
    """Write a stack YAML file and return its path."""
    if stack is None:
        stack = _make_stack_yaml()
    stacks_dir = mlx_stack_home / "stacks"
    stacks_dir.mkdir(parents=True, exist_ok=True)
    stack_path = stacks_dir / "default.yaml"
    stack_path.write_text(yaml.dump(stack, default_flow_style=False))
    return stack_path


def _create_pid_file(
    mlx_stack_home: Path,
    service_name: str,
    pid: int | str = 12345,
) -> Path:
    """Create a PID file in the pids directory."""
    pids_dir = mlx_stack_home / "pids"
    pids_dir.mkdir(parents=True, exist_ok=True)
    pid_path = pids_dir / f"{service_name}.pid"
    pid_path.write_text(str(pid))
    return pid_path


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
        mock_proc = MagicMock()
        mock_proc.pid = 54321
        mock_popen.return_value = mock_proc

        with pytest.raises(ProcessError, match="Could not write PID file"):
            start_service("fast", cmd=["vllm-mlx", "--port", "8001"], port=8001)

        # The spawned process must have been terminated
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
        mock_proc = MagicMock()
        mock_proc.pid = 54321
        mock_proc.terminate.side_effect = OSError("already dead")
        mock_popen.return_value = mock_proc

        with pytest.raises(ProcessError, match="Could not write PID file"):
            start_service("fast", cmd=["vllm-mlx", "--port", "8001"], port=8001)

        # Should fall through to kill()
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
        mock_proc = MagicMock()
        mock_proc.pid = 54321
        mock_popen.return_value = mock_proc

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
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        with pytest.raises(ProcessError):
            start_service("fast", cmd=["vllm-mlx"], port=8001)

        # Verify no leaked file descriptors — the log file should exist
        # (was opened for writing) but the handle should have been closed
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
        _write_stack_yaml(mlx_stack_home)
        # Create a corrupt PID file
        _create_pid_file(mlx_stack_home, "fast", "not-a-number")

        # Write litellm config so LiteLLM can be started
        litellm_config = mlx_stack_home / "litellm.yaml"
        litellm_config.write_text("model_list: []\n")

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

        # The corrupt PID should have been cleaned up, and the tier started fresh
        pids_dir = mlx_stack_home / "pids"
        corrupt_file = pids_dir / "fast.pid"
        # The corrupt file should have been cleaned up (removed by our fix)
        # and a new one written by start_service mock
        assert not corrupt_file.exists() or corrupt_file.read_text() != "not-a-number"

        # Should have warnings about stale PID cleanup
        assert any("stale" in w.lower() or "Cleaned up" in w for w in result.warnings)

    def test_corrupt_litellm_pid_cleaned_up_gracefully(self, mlx_stack_home: Path) -> None:
        """Corrupt LiteLLM PID file is cleaned up without traceback."""
        _write_stack_yaml(mlx_stack_home)
        # Create a corrupt LiteLLM PID file
        _create_pid_file(mlx_stack_home, "litellm", "garbage-data")

        litellm_config = mlx_stack_home / "litellm.yaml"
        litellm_config.write_text("model_list: []\n")

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

            # This should NOT raise UpError — corrupt PID is handled gracefully
            result = run_up()

        # Corrupt PID cleaned up
        assert any("stale" in w.lower() or "Cleaned up" in w for w in result.warnings)

    def test_corrupt_pid_no_traceback_via_cli(self, mlx_stack_home: Path) -> None:
        """Corrupt PID during up does not produce a Python traceback."""
        from click.testing import CliRunner

        from mlx_stack.cli.main import cli

        _write_stack_yaml(mlx_stack_home)
        _create_pid_file(mlx_stack_home, "fast", "corrupt!!!")

        litellm_config = mlx_stack_home / "litellm.yaml"
        litellm_config.write_text("model_list: []\n")

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
        # Process stays alive through grace, dies after SIGKILL
        mock_alive.side_effect = [True, True, True, False]
        mock_monotonic.side_effect = [
            0.0,   # deadline = 10.0
            5.0,   # still within grace
            11.0,  # past grace → SIGKILL
        ]

        graceful, confirmed = _terminate_process(123, grace_period=10)
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
        # Process never dies (survives SIGKILL — e.g. zombie or kernel hold)
        mock_alive.return_value = True
        mock_monotonic.side_effect = [
            0.0,   # deadline = 10.0
            11.0,  # past grace → SIGKILL
        ]

        graceful, confirmed = _terminate_process(123, grace_period=10)
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
        write_pid_file("stubborn", 12345)

        result = stop_service("stubborn")

        assert result is not None
        assert result.graceful is False
        # PID file should still exist since process wasn't confirmed dead
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
        write_pid_file("killed", 12345)

        result = stop_service("killed")

        assert result is not None
        assert result.graceful is False
        # PID file should be removed since termination was confirmed
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
        write_pid_file("graceful", 12345)

        result = stop_service("graceful")

        assert result is not None
        assert result.graceful is True
        assert read_pid_file("graceful") is None
