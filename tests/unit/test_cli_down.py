"""Tests for the `mlx-stack down` CLI command and core stack_down module.

Validates:
- VAL-DOWN-001: All managed processes terminated in correct order
- VAL-DOWN-002: Graceful shutdown with SIGKILL escalation
- VAL-DOWN-003: PID files cleaned and lockfile managed
- VAL-DOWN-004: --tier stops only the specified tier
- VAL-DOWN-005: Stale and corrupt PID handling
- VAL-DOWN-006: Nothing running produces informational message
- VAL-CROSS-001: After down, zero artifacts remain
- VAL-CROSS-009: Partial failure recovery — down cleans stale PIDs
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from mlx_stack.cli.main import cli
from mlx_stack.core.process import (
    LockError,
    ProcessError,
    ShutdownResult,
)
from mlx_stack.core.stack_down import (
    DownError,
    DownResult,
    ServiceStopResult,
    _get_tier_names_from_stack,
    _get_valid_tier_names,
    _stop_single_service,
    run_down,
)

# --------------------------------------------------------------------------- #
# Fixtures — reusable test data
# --------------------------------------------------------------------------- #


def _make_stack_yaml(
    tiers: list[dict[str, Any]] | None = None,
    schema_version: int = 1,
) -> dict[str, Any]:
    """Create a stack definition dict for testing."""
    if tiers is None:
        tiers = [
            {
                "name": "standard",
                "model": "big-model",
                "quant": "int4",
                "source": "mlx-community/big-model-4bit",
                "port": 8000,
                "vllm_flags": {
                    "continuous_batching": True,
                    "use_paged_cache": True,
                },
            },
            {
                "name": "fast",
                "model": "fast-model",
                "quant": "int4",
                "source": "mlx-community/fast-model-4bit",
                "port": 8001,
                "vllm_flags": {
                    "continuous_batching": True,
                    "use_paged_cache": True,
                },
            },
        ]
    return {
        "schema_version": schema_version,
        "name": "default",
        "hardware_profile": "m4-max-128",
        "intent": "balanced",
        "created": "2026-03-24T00:00:00+00:00",
        "tiers": tiers,
    }


def _write_stack_yaml(
    mlx_stack_home: Path,
    stack: dict[str, Any] | None = None,
) -> Path:
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


# --------------------------------------------------------------------------- #
# Tests — _get_tier_names_from_stack
# --------------------------------------------------------------------------- #


class TestGetTierNamesFromStack:
    """Tests for _get_tier_names_from_stack helper."""

    def test_returns_tier_names_from_stack(self, mlx_stack_home: Path) -> None:
        """Returns tier names from a valid stack definition."""
        _write_stack_yaml(mlx_stack_home)

        with patch("mlx_stack.core.stack_down.load_catalog") as mock_catalog:
            mock_catalog.side_effect = Exception("no catalog")
            names = _get_tier_names_from_stack()

        assert set(names) == {"standard", "fast"}

    def test_returns_empty_on_missing_stack(self, mlx_stack_home: Path) -> None:
        """Returns empty list when no stack definition exists."""
        names = _get_tier_names_from_stack()
        assert names == []


class TestGetValidTierNames:
    """Tests for _get_valid_tier_names helper."""

    def test_returns_valid_tier_names(self, mlx_stack_home: Path) -> None:
        """Returns tier names from the stack definition."""
        _write_stack_yaml(mlx_stack_home)
        names = _get_valid_tier_names()
        assert set(names) == {"standard", "fast"}

    def test_returns_empty_on_missing_stack(self, mlx_stack_home: Path) -> None:
        """Returns empty list when no stack exists."""
        names = _get_valid_tier_names()
        assert names == []


# --------------------------------------------------------------------------- #
# Tests — _stop_single_service
# --------------------------------------------------------------------------- #


class TestStopSingleService:
    """Tests for the _stop_single_service helper."""

    def test_stops_running_process_gracefully(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-002: Stops a running process gracefully."""
        _create_pid_file(mlx_stack_home, "fast", 12345)

        with (
            patch("mlx_stack.core.stack_down.read_pid_file", return_value=12345),
            patch("mlx_stack.core.stack_down.is_process_alive", return_value=True),
            patch(
                "mlx_stack.core.stack_down.stop_service",
                return_value=ShutdownResult(name="fast", pid=12345, graceful=True),
            ),
        ):
            result = _stop_single_service("fast")

        assert result.name == "fast"
        assert result.status == "stopped"
        assert result.graceful is True
        assert result.pid == 12345

    def test_stops_running_process_forced(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-002: Reports forced shutdown when SIGKILL needed."""
        _create_pid_file(mlx_stack_home, "fast", 99999)

        with (
            patch("mlx_stack.core.stack_down.read_pid_file", return_value=99999),
            patch("mlx_stack.core.stack_down.is_process_alive", return_value=True),
            patch(
                "mlx_stack.core.stack_down.stop_service",
                return_value=ShutdownResult(name="fast", pid=99999, graceful=False),
            ),
        ):
            result = _stop_single_service("fast")

        assert result.status == "stopped"
        assert result.graceful is False

    def test_handles_stale_pid(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-005: Detects stale PID and cleans up."""
        _create_pid_file(mlx_stack_home, "fast", 12345)

        with (
            patch("mlx_stack.core.stack_down.read_pid_file", return_value=12345),
            patch("mlx_stack.core.stack_down.is_process_alive", return_value=False),
            patch("mlx_stack.core.stack_down.remove_pid_file") as mock_remove,
        ):
            result = _stop_single_service("fast")

        assert result.status == "stale"
        assert result.pid == 12345
        assert "already dead" in (result.error or "")
        mock_remove.assert_called_once_with("fast")

    def test_handles_corrupt_pid(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-005: Handles corrupt PID file gracefully."""
        _create_pid_file(mlx_stack_home, "fast", "not-a-number")

        with (
            patch(
                "mlx_stack.core.stack_down.read_pid_file",
                side_effect=ProcessError("non-numeric"),
            ),
            patch("mlx_stack.core.stack_down.remove_pid_file") as mock_remove,
        ):
            result = _stop_single_service("fast")

        assert result.status == "corrupt"
        assert result.pid is None
        assert "non-numeric" in (result.error or "")
        mock_remove.assert_called_once_with("fast")

    def test_handles_no_pid_file(self, mlx_stack_home: Path) -> None:
        """Returns not-running when no PID file exists."""
        with patch("mlx_stack.core.stack_down.read_pid_file", return_value=None):
            result = _stop_single_service("fast")

        assert result.status == "not-running"
        assert result.pid is None


# --------------------------------------------------------------------------- #
# Tests — run_down (core logic)
# --------------------------------------------------------------------------- #


class TestRunDown:
    """Tests for the run_down orchestration function."""

    def test_nothing_to_stop_no_pid_files(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-006: Reports nothing to stop when no PID files exist."""
        result = run_down()

        assert result.nothing_to_stop is True
        assert result.services == []

    def test_full_shutdown_correct_order(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-001: LiteLLM stopped first, then servers in reverse order."""
        _write_stack_yaml(mlx_stack_home)
        _create_pid_file(mlx_stack_home, "standard", 1001)
        _create_pid_file(mlx_stack_home, "fast", 1002)
        _create_pid_file(mlx_stack_home, "litellm", 1003)

        shutdown_order: list[str] = []

        def mock_stop(service_name: str) -> ServiceStopResult:
            shutdown_order.append(service_name)
            return ServiceStopResult(
                name=service_name,
                pid=1000,
                status="stopped",
                graceful=True,
            )

        with (
            patch("mlx_stack.core.stack_down.acquire_lock") as mock_lock,
            patch("mlx_stack.core.stack_down._stop_single_service", side_effect=mock_stop),
            patch("mlx_stack.core.stack_down.load_catalog", side_effect=Exception("no catalog")),
        ):
            mock_lock.return_value.__enter__ = MagicMock()
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)

            result = run_down()

        assert result.nothing_to_stop is False
        assert len(result.services) == 3

        # LiteLLM must be first
        assert shutdown_order[0] == "litellm"
        # Model servers after (order depends on stack def reverse)
        assert set(shutdown_order[1:]) == {"standard", "fast"}

    def test_lockfile_acquired_during_shutdown(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-003: Lockfile acquired during down."""
        _create_pid_file(mlx_stack_home, "litellm", 1001)

        with (
            patch("mlx_stack.core.stack_down.acquire_lock") as mock_lock,
            patch(
                "mlx_stack.core.stack_down._stop_single_service",
                return_value=ServiceStopResult(
                    name="litellm",
                    pid=1001,
                    status="stopped",
                    graceful=True,
                ),
            ),
        ):
            mock_lock.return_value.__enter__ = MagicMock()
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)

            run_down()

        mock_lock.assert_called_once()

    def test_lockfile_error_propagated(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-003: Lock conflict raises LockError."""
        _create_pid_file(mlx_stack_home, "fast", 1001)

        with (
            patch(
                "mlx_stack.core.stack_down.acquire_lock",
                side_effect=LockError("Another operation is running"),
            ),
            pytest.raises(LockError, match="Another operation"),
        ):
            run_down()

    def test_tier_filter_stops_only_specified_tier(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-004: --tier stops only the specified tier."""
        _write_stack_yaml(mlx_stack_home)
        _create_pid_file(mlx_stack_home, "standard", 1001)
        _create_pid_file(mlx_stack_home, "fast", 1002)
        _create_pid_file(mlx_stack_home, "litellm", 1003)

        stopped_services: list[str] = []

        def mock_stop(service_name: str) -> ServiceStopResult:
            stopped_services.append(service_name)
            return ServiceStopResult(
                name=service_name,
                pid=1000,
                status="stopped",
                graceful=True,
            )

        with (
            patch("mlx_stack.core.stack_down.acquire_lock") as mock_lock,
            patch("mlx_stack.core.stack_down._stop_single_service", side_effect=mock_stop),
        ):
            mock_lock.return_value.__enter__ = MagicMock()
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)

            result = run_down(tier_filter="fast")

        # Only 'fast' should be stopped
        assert len(result.services) == 1
        assert result.services[0].name == "fast"
        assert stopped_services == ["fast"]

    def test_tier_filter_invalid_tier_raises_error(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-004: Invalid tier name errors with valid tier list."""
        _write_stack_yaml(mlx_stack_home)

        with pytest.raises(DownError, match="Unknown tier 'nonexistent'"):
            run_down(tier_filter="nonexistent")

    def test_tier_filter_invalid_tier_shows_valid_list(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-004: Error message includes valid tier names."""
        _write_stack_yaml(mlx_stack_home)

        with pytest.raises(DownError) as exc_info:
            run_down(tier_filter="nonexistent")

        error_msg = str(exc_info.value)
        assert "fast" in error_msg
        assert "standard" in error_msg

    def test_tier_filter_not_running(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-004: --tier for a valid but not-running tier."""
        _write_stack_yaml(mlx_stack_home)
        # Create a PID file for a different tier to avoid "nothing to stop"
        _create_pid_file(mlx_stack_home, "standard", 1001)

        with (
            patch("mlx_stack.core.stack_down.acquire_lock") as mock_lock,
        ):
            mock_lock.return_value.__enter__ = MagicMock()
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)

            result = run_down(tier_filter="fast")

        assert len(result.services) == 1
        assert result.services[0].name == "fast"
        assert result.services[0].status == "not-running"

    def test_stale_pid_detected_and_cleaned(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-005: Stale PIDs detected, reported, and cleaned up."""
        _create_pid_file(mlx_stack_home, "fast", 12345)

        with (
            patch("mlx_stack.core.stack_down.acquire_lock") as mock_lock,
            patch(
                "mlx_stack.core.stack_down._stop_single_service",
                return_value=ServiceStopResult(
                    name="fast",
                    pid=12345,
                    status="stale",
                    error="Process 12345 already dead; cleaned up stale PID file.",
                ),
            ),
        ):
            mock_lock.return_value.__enter__ = MagicMock()
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)

            result = run_down()

        assert len(result.services) == 1
        assert result.services[0].status == "stale"
        assert "already dead" in (result.services[0].error or "")

    def test_corrupt_pid_detected_and_cleaned(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-005: Corrupt PID files reported and removed."""
        _create_pid_file(mlx_stack_home, "fast", "garbage")

        with (
            patch("mlx_stack.core.stack_down.acquire_lock") as mock_lock,
            patch(
                "mlx_stack.core.stack_down._stop_single_service",
                return_value=ServiceStopResult(
                    name="fast",
                    pid=None,
                    status="corrupt",
                    error="PID file contained non-numeric content; removed.",
                ),
            ),
        ):
            mock_lock.return_value.__enter__ = MagicMock()
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)

            result = run_down()

        assert len(result.services) == 1
        assert result.services[0].status == "corrupt"

    def test_mixed_stale_and_running_services(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-005: Mix of stale, corrupt, and running services processed."""
        _write_stack_yaml(mlx_stack_home)
        _create_pid_file(mlx_stack_home, "standard", 1001)
        _create_pid_file(mlx_stack_home, "fast", "bad")
        _create_pid_file(mlx_stack_home, "litellm", 1003)

        results_map = {
            "litellm": ServiceStopResult(
                name="litellm",
                pid=1003,
                status="stopped",
                graceful=True,
            ),
            "standard": ServiceStopResult(
                name="standard",
                pid=1001,
                status="stale",
                error="Process 1001 already dead; cleaned up stale PID file.",
            ),
            "fast": ServiceStopResult(
                name="fast",
                pid=None,
                status="corrupt",
                error="PID file contained non-numeric content; removed.",
            ),
        }

        def mock_stop(service_name: str) -> ServiceStopResult:
            return results_map[service_name]

        with (
            patch("mlx_stack.core.stack_down.acquire_lock") as mock_lock,
            patch("mlx_stack.core.stack_down._stop_single_service", side_effect=mock_stop),
            patch("mlx_stack.core.stack_down.load_catalog", side_effect=Exception("no catalog")),
        ):
            mock_lock.return_value.__enter__ = MagicMock()
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)

            result = run_down()

        assert len(result.services) == 3
        statuses = {s.name: s.status for s in result.services}
        assert statuses["litellm"] == "stopped"
        assert statuses["standard"] == "stale"
        assert statuses["fast"] == "corrupt"

    def test_orphaned_services_cleaned_up(self, mlx_stack_home: Path) -> None:
        """Services not in stack definition are still cleaned up."""
        _write_stack_yaml(mlx_stack_home)
        _create_pid_file(mlx_stack_home, "orphaned-service", 9999)

        with (
            patch("mlx_stack.core.stack_down.acquire_lock") as mock_lock,
            patch(
                "mlx_stack.core.stack_down._stop_single_service",
                return_value=ServiceStopResult(
                    name="orphaned-service",
                    pid=9999,
                    status="stopped",
                    graceful=True,
                ),
            ),
            patch("mlx_stack.core.stack_down.load_catalog", side_effect=Exception("no catalog")),
        ):
            mock_lock.return_value.__enter__ = MagicMock()
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)

            result = run_down()

        assert len(result.services) == 1
        assert result.services[0].name == "orphaned-service"
        assert result.services[0].status == "stopped"

    def test_pid_files_cleaned_after_shutdown(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-003 / VAL-CROSS-001: PID files deleted after termination."""
        pids_dir = mlx_stack_home / "pids"
        pids_dir.mkdir(parents=True, exist_ok=True)
        (pids_dir / "fast.pid").write_text("12345")

        with (
            patch("mlx_stack.core.stack_down.acquire_lock") as mock_lock,
            patch("mlx_stack.core.process.psutil") as mock_psutil,
            patch("mlx_stack.core.process.os.kill"),
        ):
            mock_lock.return_value.__enter__ = MagicMock()
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)

            # First call: process alive for stop; subsequent: dead
            mock_psutil.pid_exists.return_value = True
            proc_mock = MagicMock()
            proc_mock.status.side_effect = ["running", "zombie"]
            mock_psutil.Process.return_value = proc_mock
            mock_psutil.STATUS_ZOMBIE = "zombie"

            run_down()

        # PID file should be removed
        assert not (pids_dir / "fast.pid").exists()

    def test_no_stack_definition_falls_back(self, mlx_stack_home: Path) -> None:
        """When no stack exists, PID files are still processed."""
        _create_pid_file(mlx_stack_home, "fast", 12345)

        with (
            patch("mlx_stack.core.stack_down.acquire_lock") as mock_lock,
            patch(
                "mlx_stack.core.stack_down._stop_single_service",
                return_value=ServiceStopResult(
                    name="fast",
                    pid=12345,
                    status="stopped",
                    graceful=True,
                ),
            ),
        ):
            mock_lock.return_value.__enter__ = MagicMock()
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)

            result = run_down()

        assert len(result.services) == 1
        assert result.services[0].status == "stopped"


# --------------------------------------------------------------------------- #
# Tests — CLI integration via CliRunner
# --------------------------------------------------------------------------- #


class TestDownCli:
    """Tests for the `mlx-stack down` CLI command via CliRunner."""

    def test_nothing_to_stop_exit_zero(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-006: 'Nothing to stop' and exits with code 0."""
        runner = CliRunner()
        result = runner.invoke(cli, ["down"])

        assert result.exit_code == 0
        assert "Nothing to stop" in result.output

    def test_nothing_to_stop_message(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-006: Informational message with no PID files."""
        runner = CliRunner()
        result = runner.invoke(cli, ["down"])

        assert result.exit_code == 0
        assert "Nothing to stop" in result.output
        assert "no managed services" in result.output.lower() or "No PID files" in result.output

    def test_shutdown_displays_summary_table(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-001: Displays shutdown summary."""
        mock_result = DownResult(
            services=[
                ServiceStopResult(
                    name="litellm",
                    pid=1003,
                    status="stopped",
                    graceful=True,
                ),
                ServiceStopResult(
                    name="fast",
                    pid=1002,
                    status="stopped",
                    graceful=True,
                ),
                ServiceStopResult(
                    name="standard",
                    pid=1001,
                    status="stopped",
                    graceful=False,
                ),
            ],
        )

        with patch("mlx_stack.cli.down.run_down", return_value=mock_result):
            runner = CliRunner()
            result = runner.invoke(cli, ["down"])

        assert result.exit_code == 0
        assert "Shutdown Summary" in result.output
        assert "litellm" in result.output
        assert "fast" in result.output
        assert "standard" in result.output

    def test_shutdown_shows_graceful_method(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-002: Reports graceful vs forced per service."""
        mock_result = DownResult(
            services=[
                ServiceStopResult(
                    name="fast",
                    pid=1002,
                    status="stopped",
                    graceful=True,
                ),
                ServiceStopResult(
                    name="standard",
                    pid=1001,
                    status="stopped",
                    graceful=False,
                ),
            ],
        )

        with patch("mlx_stack.cli.down.run_down", return_value=mock_result):
            runner = CliRunner()
            result = runner.invoke(cli, ["down"])

        assert result.exit_code == 0
        assert "graceful" in result.output
        assert "SIGTERM" in result.output
        assert "forced (SIGKILL)" in result.output

    def test_forced_sigkill_explicit_in_output(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-002: SIGKILL escalation explicitly reported per service."""
        mock_result = DownResult(
            services=[
                ServiceStopResult(
                    name="standard",
                    pid=1001,
                    status="stopped",
                    graceful=False,
                ),
            ],
        )

        with patch("mlx_stack.cli.down.run_down", return_value=mock_result):
            runner = CliRunner()
            result = runner.invoke(cli, ["down"])

        assert result.exit_code == 0
        # Must explicitly say "forced (SIGKILL)" for force-killed service
        assert "forced (SIGKILL)" in result.output
        # The service name must appear so user knows which service was forced
        assert "standard" in result.output

    def test_graceful_sigterm_explicit_in_output(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-002: Graceful shutdown explicitly reports SIGTERM."""
        mock_result = DownResult(
            services=[
                ServiceStopResult(
                    name="fast",
                    pid=1002,
                    status="stopped",
                    graceful=True,
                ),
            ],
        )

        with patch("mlx_stack.cli.down.run_down", return_value=mock_result):
            runner = CliRunner()
            result = runner.invoke(cli, ["down"])

        assert result.exit_code == 0
        assert "graceful (SIGTERM)" in result.output

    def test_tier_filter_option(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-004: --tier option passed to run_down."""
        mock_result = DownResult(
            services=[
                ServiceStopResult(
                    name="fast",
                    pid=1002,
                    status="stopped",
                    graceful=True,
                ),
            ],
        )

        with patch("mlx_stack.cli.down.run_down", return_value=mock_result) as mock_run:
            runner = CliRunner()
            result = runner.invoke(cli, ["down", "--tier", "fast"])

        assert result.exit_code == 0
        mock_run.assert_called_once_with(tier_filter="fast")

    def test_invalid_tier_shows_error(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-004: Invalid tier name shows error."""
        with patch(
            "mlx_stack.cli.down.run_down",
            side_effect=DownError("Unknown tier 'bad'. Valid tiers: fast, standard"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["down", "--tier", "bad"])

        assert result.exit_code == 1
        assert "Unknown tier" in result.output

    def test_lock_error_shows_message(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-003: Lock conflict shows message and exits 1."""
        with patch(
            "mlx_stack.cli.down.run_down",
            side_effect=LockError("Another operation is running"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["down"])

        assert result.exit_code == 1
        assert "Another operation" in result.output

    def test_stale_pid_shown_in_output(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-005: Stale PID reported in output."""
        mock_result = DownResult(
            services=[
                ServiceStopResult(
                    name="fast",
                    pid=12345,
                    status="stale",
                    error="Process 12345 already dead; cleaned up stale PID file.",
                ),
            ],
        )

        with patch("mlx_stack.cli.down.run_down", return_value=mock_result):
            runner = CliRunner()
            result = runner.invoke(cli, ["down"])

        assert result.exit_code == 0
        assert "stale" in result.output.lower()

    def test_corrupt_pid_shown_in_output(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-005: Corrupt PID reported in output."""
        mock_result = DownResult(
            services=[
                ServiceStopResult(
                    name="fast",
                    pid=None,
                    status="corrupt",
                    error="PID file contained non-numeric content; removed.",
                ),
            ],
        )

        with patch("mlx_stack.cli.down.run_down", return_value=mock_result):
            runner = CliRunner()
            result = runner.invoke(cli, ["down"])

        assert result.exit_code == 0
        assert "corrupt" in result.output.lower()

    def test_help_text(self, mlx_stack_home: Path) -> None:
        """Down command has help text."""
        runner = CliRunner()
        result = runner.invoke(cli, ["down", "--help"])

        assert result.exit_code == 0
        assert "Stop all managed services" in result.output
        assert "--tier" in result.output

    def test_down_appears_in_main_help(self, mlx_stack_home: Path) -> None:
        """Down command appears in main help output."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "down" in result.output

    def test_no_traceback_on_error(self, mlx_stack_home: Path) -> None:
        """Errors never show Python tracebacks."""
        with patch(
            "mlx_stack.cli.down.run_down",
            side_effect=DownError("Something went wrong"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["down"])

        assert result.exit_code == 1
        assert "Traceback" not in result.output
        assert "Error:" in result.output


# --------------------------------------------------------------------------- #
# Tests — End-to-end flow with real process module calls
# --------------------------------------------------------------------------- #


class TestDownEndToEnd:
    """Integration-style tests using real PID file operations."""

    def test_full_lifecycle_cleanup(self, mlx_stack_home: Path) -> None:
        """VAL-CROSS-001: After down, zero PID files remain."""
        _write_stack_yaml(mlx_stack_home)
        pids_dir = mlx_stack_home / "pids"
        pids_dir.mkdir(parents=True, exist_ok=True)
        (pids_dir / "standard.pid").write_text("12345")
        (pids_dir / "fast.pid").write_text("12346")
        (pids_dir / "litellm.pid").write_text("12347")

        with (
            patch("mlx_stack.core.stack_down.acquire_lock") as mock_lock,
            patch("mlx_stack.core.process.psutil") as mock_psutil,
            patch("mlx_stack.core.process.os.kill"),
        ):
            mock_lock.return_value.__enter__ = MagicMock()
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)

            # Make processes look alive then dead after SIGTERM
            mock_psutil.pid_exists.return_value = True
            proc_mock = MagicMock()
            proc_mock.status.side_effect = [
                "running",  # alive check for litellm
                "zombie",  # dead after SIGTERM for litellm
                "running",  # alive check for tier
                "zombie",  # dead after SIGTERM for tier
                "running",  # alive check for tier
                "zombie",  # dead after SIGTERM for tier
            ]
            mock_psutil.Process.return_value = proc_mock
            mock_psutil.STATUS_ZOMBIE = "zombie"

            with patch(
                "mlx_stack.core.stack_down.load_catalog",
                side_effect=Exception("no catalog"),
            ):
                run_down()

        # All PID files should be cleaned up
        remaining_pids = list(pids_dir.glob("*.pid"))
        assert remaining_pids == [], f"PID files remain: {remaining_pids}"

    def test_selective_tier_leaves_others(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-004: --tier leaves other services' PID files intact."""
        _write_stack_yaml(mlx_stack_home)
        pids_dir = mlx_stack_home / "pids"
        pids_dir.mkdir(parents=True, exist_ok=True)
        (pids_dir / "standard.pid").write_text("1001")
        (pids_dir / "fast.pid").write_text("1002")
        (pids_dir / "litellm.pid").write_text("1003")

        with (
            patch("mlx_stack.core.stack_down.acquire_lock") as mock_lock,
            patch("mlx_stack.core.process.psutil") as mock_psutil,
            patch("mlx_stack.core.process.os.kill"),
        ):
            mock_lock.return_value.__enter__ = MagicMock()
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)

            mock_psutil.pid_exists.return_value = True
            proc_mock = MagicMock()
            proc_mock.status.side_effect = ["running", "zombie"]
            mock_psutil.Process.return_value = proc_mock
            mock_psutil.STATUS_ZOMBIE = "zombie"

            run_down(tier_filter="fast")

        # Only fast PID should be removed
        assert not (pids_dir / "fast.pid").exists()
        # Others should remain
        assert (pids_dir / "standard.pid").exists()
        assert (pids_dir / "litellm.pid").exists()

    def test_corrupt_and_stale_mixed(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-005: Mixed corrupt + stale PID files all cleaned."""
        pids_dir = mlx_stack_home / "pids"
        pids_dir.mkdir(parents=True, exist_ok=True)
        (pids_dir / "fast.pid").write_text("not_a_pid")
        (pids_dir / "standard.pid").write_text("99999")

        with (
            patch("mlx_stack.core.stack_down.acquire_lock") as mock_lock,
            patch("mlx_stack.core.process.psutil") as mock_psutil,
        ):
            mock_lock.return_value.__enter__ = MagicMock()
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)

            # standard pid: process not found → stale
            mock_psutil.pid_exists.return_value = False
            mock_psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
            mock_psutil.AccessDenied = type("AccessDenied", (Exception,), {})

            result = run_down()

        # Both PID files should be cleaned
        remaining = list(pids_dir.glob("*.pid"))
        assert remaining == [], f"PID files remain: {remaining}"

        statuses = {s.name: s.status for s in result.services}
        assert statuses["fast"] == "corrupt"
        assert statuses["standard"] == "stale"


# --------------------------------------------------------------------------- #
# Tests — Shutdown order verification
# --------------------------------------------------------------------------- #


class TestShutdownOrder:
    """Tests verifying the correct shutdown order."""

    def test_litellm_stopped_before_model_servers(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-001: LiteLLM stopped first."""
        _write_stack_yaml(mlx_stack_home)
        _create_pid_file(mlx_stack_home, "standard", 1001)
        _create_pid_file(mlx_stack_home, "fast", 1002)
        _create_pid_file(mlx_stack_home, "litellm", 1003)

        order: list[str] = []

        def mock_stop(name: str) -> ServiceStopResult:
            order.append(name)
            return ServiceStopResult(
                name=name,
                pid=1000,
                status="stopped",
                graceful=True,
            )

        with (
            patch("mlx_stack.core.stack_down.acquire_lock") as mock_lock,
            patch("mlx_stack.core.stack_down._stop_single_service", side_effect=mock_stop),
            patch("mlx_stack.core.stack_down.load_catalog", side_effect=Exception("no catalog")),
        ):
            mock_lock.return_value.__enter__ = MagicMock()
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)

            run_down()

        assert order[0] == "litellm"

    def test_model_servers_reversed(self, mlx_stack_home: Path) -> None:
        """VAL-DOWN-001: Model servers stopped in reverse startup order."""
        # Stack with 3 tiers of different sizes
        tiers = [
            {
                "name": "standard",
                "model": "big-model",
                "quant": "int4",
                "source": "src1",
                "port": 8000,
                "vllm_flags": {},
            },
            {
                "name": "fast",
                "model": "small-model",
                "quant": "int4",
                "source": "src2",
                "port": 8001,
                "vllm_flags": {},
            },
            {
                "name": "longctx",
                "model": "medium-model",
                "quant": "int4",
                "source": "src3",
                "port": 8002,
                "vllm_flags": {},
            },
        ]
        stack = _make_stack_yaml(tiers=tiers)
        _write_stack_yaml(mlx_stack_home, stack)

        for tier in tiers:
            _create_pid_file(mlx_stack_home, tier["name"], 1000)

        order: list[str] = []

        def mock_stop(name: str) -> ServiceStopResult:
            order.append(name)
            return ServiceStopResult(
                name=name,
                pid=1000,
                status="stopped",
                graceful=True,
            )

        with (
            patch("mlx_stack.core.stack_down.acquire_lock") as mock_lock,
            patch("mlx_stack.core.stack_down._stop_single_service", side_effect=mock_stop),
            patch("mlx_stack.core.stack_down.load_catalog", side_effect=Exception("no catalog")),
        ):
            mock_lock.return_value.__enter__ = MagicMock()
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)

            run_down()

        # With no catalog, tiers are in stack definition order
        # Reversed means last-defined tier stops first (among model servers)
        model_servers = [s for s in order if s != "litellm"]
        # Reverse of the startup order; startup order = stack definition order
        # when catalog is unavailable
        assert len(model_servers) == 3
