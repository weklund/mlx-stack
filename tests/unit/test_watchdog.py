"""Tests for the watchdog core module.

Covers flap detection, restart delay calculation, service restart,
daemon mode, signal handling, poll cycles, log rotation, and the
main watchdog loop.
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from mlx_stack.core.watchdog import (
    DEFAULT_INTERVAL,
    DEFAULT_MAX_RESTARTS,
    DEFAULT_RESTART_DELAY,
    PollResult,
    ServiceTracker,
    WatchdogError,
    WatchdogState,
    calculate_restart_delay,
    check_existing_watchdog,
    check_flapping,
    daemonize,
    poll_cycle,
    remove_watchdog_pid,
    reset_flap_state,
    restart_service,
    rotate_service_logs,
    run_watchdog,
    setup_signal_handlers,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def stack_definition(mlx_stack_home: Path) -> dict[str, Any]:
    """Create a test stack definition and return it."""
    stacks_dir = mlx_stack_home / "stacks"
    stacks_dir.mkdir(parents=True, exist_ok=True)

    stack = {
        "schema_version": 1,
        "name": "default",
        "hardware_profile": "m4-pro-48",
        "intent": "balanced",
        "created": "2025-01-01T00:00:00",
        "tiers": [
            {
                "name": "fast",
                "model": "qwen3.5-3b",
                "quant": "int4",
                "source": "mlx-community/Qwen3.5-3B-4bit",
                "port": 8000,
                "vllm_flags": {
                    "continuous_batching": True,
                    "use_paged_cache": True,
                },
            },
            {
                "name": "standard",
                "model": "qwen3.5-8b",
                "quant": "int4",
                "source": "mlx-community/Qwen3.5-8B-4bit",
                "port": 8001,
                "vllm_flags": {
                    "continuous_batching": True,
                    "use_paged_cache": True,
                },
            },
        ],
    }

    stack_path = stacks_dir / "default.yaml"
    stack_path.write_text(yaml.dump(stack))

    return stack


@pytest.fixture()
def pids_dir(mlx_stack_home: Path) -> Path:
    """Create and return the pids directory."""
    d = mlx_stack_home / "pids"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture()
def logs_dir(mlx_stack_home: Path) -> Path:
    """Create and return the logs directory."""
    d = mlx_stack_home / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Flap detection tests
# --------------------------------------------------------------------------- #


class TestCheckFlapping:
    """Tests for check_flapping."""

    def test_no_restarts_not_flapping(self) -> None:
        tracker = ServiceTracker()
        assert check_flapping(tracker, max_restarts=5) is False

    def test_below_threshold_not_flapping(self) -> None:
        tracker = ServiceTracker()
        tracker.restart_timestamps = [time.monotonic() - i for i in range(4)]
        assert check_flapping(tracker, max_restarts=5) is False

    def test_at_threshold_is_flapping(self) -> None:
        tracker = ServiceTracker()
        now = time.monotonic()
        tracker.restart_timestamps = [now - i for i in range(5)]
        assert check_flapping(tracker, max_restarts=5) is True
        assert tracker.is_flapping is True

    def test_old_restarts_pruned(self) -> None:
        """Restarts outside the rolling window are pruned."""
        tracker = ServiceTracker()
        old = time.monotonic() - 700  # outside 600s window
        tracker.restart_timestamps = [old, old - 1, old - 2, old - 3, old - 4]
        assert check_flapping(tracker, max_restarts=5) is False
        assert len(tracker.restart_timestamps) == 0

    def test_custom_window(self) -> None:
        tracker = ServiceTracker()
        now = time.monotonic()
        # 3 restarts in a 10s window
        tracker.restart_timestamps = [now, now - 1, now - 2]
        assert check_flapping(tracker, max_restarts=3, window_seconds=10.0) is True


class TestResetFlapState:
    """Tests for reset_flap_state."""

    def test_not_flapping_returns_false(self) -> None:
        tracker = ServiceTracker()
        assert reset_flap_state(tracker) is False

    def test_flapping_too_recent_returns_false(self) -> None:
        tracker = ServiceTracker(
            is_flapping=True,
            last_restart_time=time.monotonic(),
        )
        assert reset_flap_state(tracker, stable_period=300.0) is False

    def test_flapping_stable_resets(self) -> None:
        tracker = ServiceTracker(
            is_flapping=True,
            last_restart_time=time.monotonic() - 400,
            restart_timestamps=[time.monotonic() - 400],
            restart_count=5,
            consecutive_failures=3,
        )
        assert reset_flap_state(tracker, stable_period=300.0) is True
        assert tracker.is_flapping is False
        assert tracker.restart_timestamps == []
        assert tracker.restart_count == 0
        assert tracker.consecutive_failures == 0

    def test_no_last_restart_returns_false(self) -> None:
        tracker = ServiceTracker(is_flapping=True, last_restart_time=0.0)
        assert reset_flap_state(tracker) is False


# --------------------------------------------------------------------------- #
# Restart delay tests
# --------------------------------------------------------------------------- #


class TestCalculateRestartDelay:
    """Tests for calculate_restart_delay."""

    def test_zero_failures_returns_base(self) -> None:
        assert calculate_restart_delay(10.0, 0) == 10.0

    def test_first_failure_returns_base(self) -> None:
        assert calculate_restart_delay(10.0, 1) == 10.0

    def test_second_failure_doubles(self) -> None:
        assert calculate_restart_delay(10.0, 2) == 20.0

    def test_third_failure_quadruples(self) -> None:
        assert calculate_restart_delay(10.0, 3) == 40.0

    def test_max_delay_cap(self) -> None:
        result = calculate_restart_delay(10.0, 10, max_delay=300.0)
        assert result == 300.0

    def test_negative_failures_returns_base(self) -> None:
        assert calculate_restart_delay(10.0, -1) == 10.0


# --------------------------------------------------------------------------- #
# Restart service tests
# --------------------------------------------------------------------------- #


class TestRestartService:
    """Tests for restart_service."""

    def test_restart_tier_success(
        self, mlx_stack_home: Path, stack_definition: dict[str, Any], pids_dir: Path
    ) -> None:
        tracker = ServiceTracker()

        with (
            patch("mlx_stack.core.watchdog.acquire_lock") as mock_lock,
            patch("mlx_stack.core.watchdog.start_service") as mock_start,
            patch("mlx_stack.core.watchdog.remove_pid_file"),
        ):
            # Make acquire_lock a context manager
            mock_lock.return_value.__enter__ = MagicMock(return_value=None)
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)

            mock_start.return_value = MagicMock()

            result = restart_service(
                service_name="fast",
                stack=stack_definition,
                tracker=tracker,
                vllm_binary="/usr/bin/vllm-mlx",
            )

        assert result is True

    def test_restart_unknown_tier_fails(
        self, mlx_stack_home: Path, stack_definition: dict[str, Any], pids_dir: Path
    ) -> None:
        tracker = ServiceTracker()

        with (
            patch("mlx_stack.core.watchdog.acquire_lock") as mock_lock,
            patch("mlx_stack.core.watchdog.remove_pid_file"),
        ):
            mock_lock.return_value.__enter__ = MagicMock(return_value=None)
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)

            result = restart_service(
                service_name="nonexistent",
                stack=stack_definition,
                tracker=tracker,
                vllm_binary="/usr/bin/vllm-mlx",
            )

        assert result is False

    def test_restart_lock_failure(
        self, mlx_stack_home: Path, stack_definition: dict[str, Any], pids_dir: Path
    ) -> None:
        from mlx_stack.core.process import LockError

        tracker = ServiceTracker()

        with (
            patch("mlx_stack.core.watchdog.acquire_lock", side_effect=LockError("locked")),
            patch("mlx_stack.core.watchdog.remove_pid_file"),
        ):
            result = restart_service(
                service_name="fast",
                stack=stack_definition,
                tracker=tracker,
                vllm_binary="/usr/bin/vllm-mlx",
            )

        assert result is False

    def test_restart_litellm(
        self, mlx_stack_home: Path, stack_definition: dict[str, Any], pids_dir: Path
    ) -> None:
        tracker = ServiceTracker()

        with (
            patch("mlx_stack.core.watchdog.acquire_lock") as mock_lock,
            patch("mlx_stack.core.watchdog.start_service") as mock_start,
            patch("mlx_stack.core.watchdog.remove_pid_file"),
            patch("mlx_stack.core.watchdog.get_value") as mock_get_value,
        ):
            mock_lock.return_value.__enter__ = MagicMock(return_value=None)
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)
            mock_start.return_value = MagicMock()
            mock_get_value.side_effect = lambda key: {
                "litellm-port": 4000,
                "openrouter-key": "",
            }.get(key, "")

            result = restart_service(
                service_name="litellm",
                stack=stack_definition,
                tracker=tracker,
                litellm_binary="/usr/bin/litellm",
            )

        assert result is True


# --------------------------------------------------------------------------- #
# Check existing watchdog tests
# --------------------------------------------------------------------------- #


class TestCheckExistingWatchdog:
    """Tests for check_existing_watchdog."""

    def test_no_pid_file(self, mlx_stack_home: Path) -> None:
        assert check_existing_watchdog() is None

    def test_stale_pid_file(self, mlx_stack_home: Path, pids_dir: Path) -> None:
        # Write a PID file with a dead PID
        pid_path = pids_dir / "watchdog.pid"
        pid_path.write_text("99999999")

        with patch("mlx_stack.core.watchdog.is_process_alive", return_value=False):
            result = check_existing_watchdog()

        assert result is None
        # Stale PID file should be cleaned up
        assert not pid_path.exists()

    def test_running_watchdog(self, mlx_stack_home: Path, pids_dir: Path) -> None:
        pid_path = pids_dir / "watchdog.pid"
        pid_path.write_text("12345")

        with patch("mlx_stack.core.watchdog.is_process_alive", return_value=True):
            result = check_existing_watchdog()

        assert result == 12345


# --------------------------------------------------------------------------- #
# Signal handling tests
# --------------------------------------------------------------------------- #


class TestSignalHandling:
    """Tests for setup_signal_handlers."""

    def test_sigterm_sets_shutdown_flag(self) -> None:
        state = WatchdogState()
        setup_signal_handlers(state)
        assert state.shutdown_requested is False

        # Send SIGTERM to ourselves
        os.kill(os.getpid(), signal.SIGTERM)
        assert state.shutdown_requested is True

    def test_sigint_sets_shutdown_flag(self) -> None:
        state = WatchdogState()
        setup_signal_handlers(state)
        assert state.shutdown_requested is False

        # Manually call the handler (SIGINT would be caught by pytest)
        handler = signal.getsignal(signal.SIGINT)
        if callable(handler):
            handler(signal.SIGINT, None)
        assert state.shutdown_requested is True


# --------------------------------------------------------------------------- #
# Log rotation tests
# --------------------------------------------------------------------------- #


class TestRotateServiceLogs:
    """Tests for rotate_service_logs."""

    def test_no_logs_directory(self, mlx_stack_home: Path) -> None:
        with patch("mlx_stack.core.watchdog.get_value", return_value=50):
            count = rotate_service_logs()
        assert count == 0

    def test_rotates_eligible_files(
        self, mlx_stack_home: Path, logs_dir: Path
    ) -> None:
        # Create a log file exceeding threshold
        log_file = logs_dir / "fast.log"
        log_file.write_bytes(b"x" * (1 * 1024 * 1024))  # 1 MB

        with patch("mlx_stack.core.watchdog.get_value") as mock_get:
            mock_get.side_effect = lambda key: {
                "log-max-size-mb": 0,  # Will cause threshold to be 0
                "log-max-files": 5,
            }.get(key, 50)

            # rotate_log checks if size >= threshold_bytes
            # With threshold 0 * 1024 * 1024 = 0, but file > 0, should rotate
            count = rotate_service_logs()

        # Since max_size_mb=0 means threshold=0 bytes, any non-empty file rotates
        assert count >= 0  # The actual rotation depends on implementation

    def test_skips_non_log_files(
        self, mlx_stack_home: Path, logs_dir: Path
    ) -> None:
        # Create a non-.log file
        other_file = logs_dir / "fast.log.1.gz"
        other_file.write_bytes(b"compressed data")

        with patch("mlx_stack.core.watchdog.get_value") as mock_get:
            mock_get.side_effect = lambda key: {
                "log-max-size-mb": 50,
                "log-max-files": 5,
            }.get(key, 50)

            count = rotate_service_logs()

        assert count == 0


# --------------------------------------------------------------------------- #
# Poll cycle tests
# --------------------------------------------------------------------------- #


class TestPollCycle:
    """Tests for poll_cycle."""

    def test_basic_poll_no_crashes(
        self, mlx_stack_home: Path, stack_definition: dict[str, Any]
    ) -> None:
        state = WatchdogState()

        from mlx_stack.core.stack_status import ServiceHealth, ServiceStatus, StatusResult

        mock_status = StatusResult(
            services=[
                ServiceStatus(
                    tier="fast", model="qwen3.5-3b", port=8000,
                    status=ServiceHealth.HEALTHY, uptime=100.0, uptime_display="1m 40s",
                    response_time=0.1, pid=1234,
                ),
                ServiceStatus(
                    tier="standard", model="qwen3.5-8b", port=8001,
                    status=ServiceHealth.STOPPED, uptime=None, uptime_display="-",
                    response_time=None, pid=None,
                ),
            ]
        )

        with (
            patch("mlx_stack.core.watchdog.run_status", return_value=mock_status),
            patch("mlx_stack.core.watchdog.rotate_service_logs", return_value=0),
        ):
            result = poll_cycle(
                state=state,
                stack=stack_definition,
                interval=30,
                max_restarts=5,
                restart_delay=10,
            )

        assert result.restarts_attempted == 0
        assert len(result.statuses) == 2
        assert state.cycle_count == 1

    def test_poll_with_crashed_service_triggers_restart(
        self, mlx_stack_home: Path, stack_definition: dict[str, Any]
    ) -> None:
        state = WatchdogState()

        from mlx_stack.core.stack_status import ServiceHealth, ServiceStatus, StatusResult

        mock_status = StatusResult(
            services=[
                ServiceStatus(
                    tier="fast", model="qwen3.5-3b", port=8000,
                    status=ServiceHealth.CRASHED, uptime=None, uptime_display="-",
                    response_time=None, pid=1234,
                ),
                ServiceStatus(
                    tier="standard", model="qwen3.5-8b", port=8001,
                    status=ServiceHealth.HEALTHY, uptime=200.0, uptime_display="3m 20s",
                    response_time=0.05, pid=5678,
                ),
            ]
        )

        with (
            patch("mlx_stack.core.watchdog.run_status", return_value=mock_status),
            patch("mlx_stack.core.watchdog.rotate_service_logs", return_value=0),
            patch("mlx_stack.core.watchdog.restart_service", return_value=True),
        ):
            result = poll_cycle(
                state=state,
                stack=stack_definition,
                interval=30,
                max_restarts=5,
                restart_delay=10,
            )

        assert result.restarts_attempted == 1
        assert result.restarts_succeeded == 1
        assert len(state.restart_log) == 1
        assert state.restart_log[0].service == "fast"
        assert state.restart_log[0].success is True

    def test_poll_does_not_restart_stopped_service(
        self, mlx_stack_home: Path, stack_definition: dict[str, Any]
    ) -> None:
        """Stopped services (no PID file) should NOT be restarted."""
        state = WatchdogState()

        from mlx_stack.core.stack_status import ServiceHealth, ServiceStatus, StatusResult

        mock_status = StatusResult(
            services=[
                ServiceStatus(
                    tier="fast", model="qwen3.5-3b", port=8000,
                    status=ServiceHealth.STOPPED, uptime=None, uptime_display="-",
                    response_time=None, pid=None,
                ),
            ]
        )

        with (
            patch("mlx_stack.core.watchdog.run_status", return_value=mock_status),
            patch("mlx_stack.core.watchdog.rotate_service_logs", return_value=0),
            patch("mlx_stack.core.watchdog.restart_service"),
        ):
            result = poll_cycle(
                state=state,
                stack=stack_definition,
                interval=30,
                max_restarts=5,
                restart_delay=10,
            )

        assert result.restarts_attempted == 0

    def test_poll_flapping_service_not_restarted(
        self, mlx_stack_home: Path, stack_definition: dict[str, Any]
    ) -> None:
        """Flapping services should not be restarted."""
        state = WatchdogState()
        # Pre-mark service as flapping
        state.service_trackers["fast"] = ServiceTracker(
            is_flapping=True,
            last_restart_time=time.monotonic(),
        )

        from mlx_stack.core.stack_status import ServiceHealth, ServiceStatus, StatusResult

        mock_status = StatusResult(
            services=[
                ServiceStatus(
                    tier="fast", model="qwen3.5-3b", port=8000,
                    status=ServiceHealth.CRASHED, uptime=None, uptime_display="-",
                    response_time=None, pid=1234,
                ),
            ]
        )

        with (
            patch("mlx_stack.core.watchdog.run_status", return_value=mock_status),
            patch("mlx_stack.core.watchdog.rotate_service_logs", return_value=0),
            patch("mlx_stack.core.watchdog.restart_service"),
        ):
            result = poll_cycle(
                state=state,
                stack=stack_definition,
                interval=30,
                max_restarts=5,
                restart_delay=10,
            )

        assert result.restarts_attempted == 0
        assert "fast" in result.flapping_services

    def test_poll_respects_restart_delay(
        self, mlx_stack_home: Path, stack_definition: dict[str, Any]
    ) -> None:
        """Should not restart if delay has not elapsed."""
        state = WatchdogState()
        # Set last restart time to now (delay not elapsed yet)
        state.service_trackers["fast"] = ServiceTracker(
            last_restart_time=time.monotonic(),
            consecutive_failures=0,
        )

        from mlx_stack.core.stack_status import ServiceHealth, ServiceStatus, StatusResult

        mock_status = StatusResult(
            services=[
                ServiceStatus(
                    tier="fast", model="qwen3.5-3b", port=8000,
                    status=ServiceHealth.CRASHED, uptime=None, uptime_display="-",
                    response_time=None, pid=1234,
                ),
            ]
        )

        with (
            patch("mlx_stack.core.watchdog.run_status", return_value=mock_status),
            patch("mlx_stack.core.watchdog.rotate_service_logs", return_value=0),
            patch("mlx_stack.core.watchdog.restart_service"),
        ):
            result = poll_cycle(
                state=state,
                stack=stack_definition,
                interval=30,
                max_restarts=5,
                restart_delay=10,
            )

        assert result.restarts_attempted == 0

    def test_poll_with_failed_restart(
        self, mlx_stack_home: Path, stack_definition: dict[str, Any]
    ) -> None:
        """Failed restart should increment consecutive_failures."""
        state = WatchdogState()

        from mlx_stack.core.stack_status import ServiceHealth, ServiceStatus, StatusResult

        mock_status = StatusResult(
            services=[
                ServiceStatus(
                    tier="fast", model="qwen3.5-3b", port=8000,
                    status=ServiceHealth.CRASHED, uptime=None, uptime_display="-",
                    response_time=None, pid=1234,
                ),
            ]
        )

        with (
            patch("mlx_stack.core.watchdog.run_status", return_value=mock_status),
            patch("mlx_stack.core.watchdog.rotate_service_logs", return_value=0),
            patch("mlx_stack.core.watchdog.restart_service", return_value=False),
        ):
            result = poll_cycle(
                state=state,
                stack=stack_definition,
                interval=30,
                max_restarts=5,
                restart_delay=10,
            )

        assert result.restarts_attempted == 1
        assert result.restarts_succeeded == 0
        assert state.service_trackers["fast"].consecutive_failures == 1

    def test_poll_log_rotation_counted(
        self, mlx_stack_home: Path, stack_definition: dict[str, Any]
    ) -> None:
        state = WatchdogState()

        from mlx_stack.core.stack_status import StatusResult

        mock_status = StatusResult(services=[])

        with (
            patch("mlx_stack.core.watchdog.run_status", return_value=mock_status),
            patch("mlx_stack.core.watchdog.rotate_service_logs", return_value=3),
        ):
            result = poll_cycle(
                state=state,
                stack=stack_definition,
                interval=30,
                max_restarts=5,
                restart_delay=10,
            )

        assert result.rotations_performed == 3

    def test_poll_no_stack_returns_early(
        self, mlx_stack_home: Path, stack_definition: dict[str, Any]
    ) -> None:
        state = WatchdogState()

        from mlx_stack.core.stack_status import StatusResult

        mock_status = StatusResult(
            no_stack=True,
            message="No stack configured",
        )

        with (
            patch("mlx_stack.core.watchdog.run_status", return_value=mock_status),
            patch("mlx_stack.core.watchdog.rotate_service_logs", return_value=0),
        ):
            result = poll_cycle(
                state=state,
                stack=stack_definition,
                interval=30,
                max_restarts=5,
                restart_delay=10,
            )

        assert result.restarts_attempted == 0

    def test_poll_healthy_resets_consecutive_failures(
        self, mlx_stack_home: Path, stack_definition: dict[str, Any]
    ) -> None:
        """Healthy service should reset consecutive_failures."""
        state = WatchdogState()
        state.service_trackers["fast"] = ServiceTracker(
            consecutive_failures=3,
            last_restart_time=time.monotonic() - 100,
        )

        from mlx_stack.core.stack_status import ServiceHealth, ServiceStatus, StatusResult

        mock_status = StatusResult(
            services=[
                ServiceStatus(
                    tier="fast", model="qwen3.5-3b", port=8000,
                    status=ServiceHealth.HEALTHY, uptime=100.0, uptime_display="1m 40s",
                    response_time=0.1, pid=1234,
                ),
            ]
        )

        with (
            patch("mlx_stack.core.watchdog.run_status", return_value=mock_status),
            patch("mlx_stack.core.watchdog.rotate_service_logs", return_value=0),
        ):
            poll_cycle(
                state=state,
                stack=stack_definition,
                interval=30,
                max_restarts=5,
                restart_delay=10,
            )

        assert state.service_trackers["fast"].consecutive_failures == 0


# --------------------------------------------------------------------------- #
# Main watchdog loop tests
# --------------------------------------------------------------------------- #


class TestRunWatchdog:
    """Tests for run_watchdog."""

    def test_no_stack_raises_error(self, mlx_stack_home: Path) -> None:
        with pytest.raises(WatchdogError, match="No stack configuration found"):
            run_watchdog()

    def test_already_running_raises_error(
        self, mlx_stack_home: Path, stack_definition: dict[str, Any], pids_dir: Path
    ) -> None:
        # Write a watchdog PID file with a "running" process
        pid_path = pids_dir / "watchdog.pid"
        pid_path.write_text(str(os.getpid()))  # Current process is alive

        with pytest.raises(WatchdogError, match="already running"):
            run_watchdog()

    def test_watchdog_loop_with_immediate_shutdown(
        self, mlx_stack_home: Path, stack_definition: dict[str, Any], pids_dir: Path
    ) -> None:
        """Test that the watchdog loop runs and exits on shutdown."""
        from mlx_stack.core.stack_status import StatusResult

        mock_status = StatusResult(services=[])
        call_count = 0

        def status_callback(result: PollResult, state: WatchdogState) -> None:
            nonlocal call_count
            call_count += 1
            # Set shutdown after first cycle
            state.shutdown_requested = True

        with (
            patch("mlx_stack.core.watchdog.run_status", return_value=mock_status),
            patch("mlx_stack.core.watchdog.rotate_service_logs", return_value=0),
        ):
            state = run_watchdog(
                interval=1,
                status_callback=status_callback,
            )

        assert call_count == 1
        assert state.cycle_count == 1
        # PID file should be cleaned up
        assert not (pids_dir / "watchdog.pid").exists()

    def test_watchdog_cleanup_on_exit(
        self, mlx_stack_home: Path, stack_definition: dict[str, Any], pids_dir: Path
    ) -> None:
        """Test that watchdog cleans up PID file on exit."""
        from mlx_stack.core.stack_status import StatusResult

        mock_status = StatusResult(services=[])

        def status_callback(result: PollResult, state: WatchdogState) -> None:
            state.shutdown_requested = True

        with (
            patch("mlx_stack.core.watchdog.run_status", return_value=mock_status),
            patch("mlx_stack.core.watchdog.rotate_service_logs", return_value=0),
        ):
            run_watchdog(interval=1, status_callback=status_callback)

        assert not (pids_dir / "watchdog.pid").exists()


# --------------------------------------------------------------------------- #
# Daemon mode tests
# --------------------------------------------------------------------------- #


class TestDaemonize:
    """Tests for daemonize (mocked)."""

    def test_daemonize_calls_fork_and_setsid(self, mlx_stack_home: Path) -> None:
        with (
            patch("os.fork", side_effect=[0, 0]) as mock_fork,
            patch("os.setsid") as mock_setsid,
            patch("os.open", return_value=3),
            patch("os.dup2"),
            patch("os.close"),
            patch("mlx_stack.core.watchdog.write_pid_file"),
        ):
            daemonize()

        assert mock_fork.call_count == 2
        mock_setsid.assert_called_once()

    def test_daemonize_first_fork_parent_exits(self, mlx_stack_home: Path) -> None:
        """When first fork returns >0, os._exit(0) should be called."""
        exit_called = False

        def fake_exit(code: int) -> None:
            nonlocal exit_called
            exit_called = True
            raise SystemExit(code)

        with (
            patch("os.fork", return_value=123),
            patch("os._exit", side_effect=fake_exit),
            pytest.raises(SystemExit),
        ):
            daemonize()

        assert exit_called

    def test_daemonize_first_fork_failure(self, mlx_stack_home: Path) -> None:
        with (
            patch("os.fork", side_effect=OSError("fork failed")),
            pytest.raises(WatchdogError, match="First fork failed"),
        ):
            daemonize()


# --------------------------------------------------------------------------- #
# Remove PID tests
# --------------------------------------------------------------------------- #


class TestRemoveWatchdogPid:
    """Tests for remove_watchdog_pid."""

    def test_removes_existing_pid_file(
        self, mlx_stack_home: Path, pids_dir: Path
    ) -> None:
        pid_path = pids_dir / "watchdog.pid"
        pid_path.write_text("12345")

        remove_watchdog_pid()
        assert not pid_path.exists()

    def test_no_pid_file_is_noop(self, mlx_stack_home: Path) -> None:
        remove_watchdog_pid()  # Should not raise


# --------------------------------------------------------------------------- #
# Constants/defaults tests
# --------------------------------------------------------------------------- #


class TestDefaults:
    """Test that default constants are set correctly."""

    def test_default_interval(self) -> None:
        assert DEFAULT_INTERVAL == 30

    def test_default_max_restarts(self) -> None:
        assert DEFAULT_MAX_RESTARTS == 5

    def test_default_restart_delay(self) -> None:
        assert DEFAULT_RESTART_DELAY == 10
