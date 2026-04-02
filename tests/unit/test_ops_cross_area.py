"""Cross-ops integration tests for mlx-stack.

Validates cross-feature interactions across the ops milestone:
- VAL-CROSS-OPS-001: Watchdog triggers log rotation automatically when a
  service log exceeds threshold. Concurrent rotation (watchdog + manual
  --rotate) is safe/idempotent.
- VAL-CROSS-OPS-002: logs --follow survives a copytruncate rotation —
  detects file size decrease and continues from beginning.
- VAL-CROSS-OPS-003: launchd-started watchdog performs rotation and restart
  (mock launchctl, verify plist command).
- VAL-CROSS-OPS-004: down sets services to stopped (watchdog does not
  restart), up makes them monitored. No lock conflicts.
- VAL-CROSS-OPS-005: Full lifecycle test: init -> install -> watchdog starts
  -> up -> log generation -> rotation -> follow -> kill service -> watchdog
  restarts -> down -> uninstall.
"""

from __future__ import annotations

import gzip
import os
import plistlib
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from mlx_stack.core.log_rotation import rotate_log
from mlx_stack.core.log_viewer import follow_log
from mlx_stack.core.watchdog import (
    ServiceTracker,
    WatchdogState,
    poll_cycle,
    restart_service,
    rotate_service_logs,
)

# --------------------------------------------------------------------------- #
# Shared fixtures
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


@pytest.fixture()
def config_file(mlx_stack_home: Path) -> Path:
    """Create a config file with rotation settings."""
    config_path = mlx_stack_home / "config.yaml"
    config_data = {
        "log-max-size-mb": 1,
        "log-max-files": 5,
    }
    config_path.write_text(yaml.dump(config_data))
    return config_path


def _create_log(path: Path, size_mb: float = 0, content: str | None = None) -> None:
    """Create a log file with the given size or specific content."""
    if content is not None:
        path.write_text(content)
    else:
        size_bytes = int(size_mb * 1024 * 1024)
        if size_bytes > 0:
            path.write_bytes(b"x" * size_bytes)
        else:
            path.touch()


def _read_gz(path: Path) -> bytes:
    """Read and decompress a gzip file."""
    with gzip.open(str(path), "rb") as f:
        return f.read()


# =========================================================================== #
# VAL-CROSS-OPS-001: Watchdog auto-rotation integration
# =========================================================================== #


class TestWatchdogAutoRotation:
    """Tests for watchdog triggering log rotation during poll cycles.

    VAL-CROSS-OPS-001: With watchdog running and services generating logs,
    rotation occurs automatically when threshold exceeded. On-demand
    `logs --rotate` while watchdog active doesn't conflict (idempotent).
    """

    def test_poll_cycle_rotates_over_threshold_logs(
        self, mlx_stack_home: Path, logs_dir: Path, stack_definition: dict[str, Any]
    ) -> None:
        """Watchdog poll cycle rotates logs exceeding the configured threshold."""
        # Create a log file over the threshold (use 1MB threshold for speed)
        fast_log = logs_dir / "fast.log"
        _create_log(fast_log, size_mb=2)

        # Create a log file under threshold — should NOT be rotated
        standard_log = logs_dir / "standard.log"
        _create_log(standard_log, content="small log\n")

        state = WatchdogState()

        with (
            patch("mlx_stack.core.watchdog.run_status") as mock_status,
            patch("mlx_stack.core.watchdog.get_value") as mock_get_value,
        ):
            # All services healthy — no restarts needed
            mock_result = MagicMock()
            mock_result.no_stack = False
            mock_result.services = []
            mock_status.return_value = mock_result

            # Configure rotation thresholds
            mock_get_value.side_effect = lambda key: {
                "log-max-size-mb": 1,
                "log-max-files": 5,
            }.get(key, "")

            result = poll_cycle(
                state=state,
                stack=stack_definition,
                interval=30,
                max_restarts=5,
                restart_delay=10,
            )

        # fast.log was above 1MB → should have been rotated
        assert result.rotations_performed >= 1
        assert fast_log.stat().st_size == 0  # truncated
        archive = logs_dir / "fast.log.1.gz"
        assert archive.exists()
        data = _read_gz(archive)
        assert len(data) == 2 * 1024 * 1024

        # standard.log should NOT have been rotated (below threshold)
        assert standard_log.stat().st_size > 0
        assert not (logs_dir / "standard.log.1.gz").exists()

    def test_concurrent_watchdog_and_manual_rotation_is_idempotent(
        self, mlx_stack_home: Path, logs_dir: Path
    ) -> None:
        """Concurrent rotation from watchdog and manual --rotate is safe.

        If the watchdog rotates a file, and then manual --rotate runs,
        the file should already be below threshold, so manual rotation
        is a no-op (idempotent).
        """
        log = logs_dir / "fast.log"
        _create_log(log, size_mb=2)

        # First rotation: simulates watchdog
        result_1 = rotate_log(log, max_size_mb=1, max_files=5)
        assert result_1 is True
        assert log.stat().st_size == 0
        archive_1 = logs_dir / "fast.log.1.gz"
        assert archive_1.exists()

        # Second rotation: simulates manual --rotate (concurrent)
        result_2 = rotate_log(log, max_size_mb=1, max_files=5)
        assert result_2 is False  # File is now 0 bytes — skip

        # Archive should still be intact
        assert archive_1.exists()
        data = _read_gz(archive_1)
        assert len(data) == 2 * 1024 * 1024

    def test_concurrent_rotation_with_new_content(
        self, mlx_stack_home: Path, logs_dir: Path
    ) -> None:
        """If new content arrives between rotations, both rotations succeed.

        The second rotation produces a separate archive without corrupting
        the first.
        """
        log = logs_dir / "fast.log"
        _create_log(log, size_mb=2)

        # First rotation (watchdog)
        rotate_log(log, max_size_mb=1, max_files=5)
        assert log.stat().st_size == 0
        assert (logs_dir / "fast.log.1.gz").exists()

        # Simulate new log content arriving above threshold
        _create_log(log, size_mb=1.5)

        # Second rotation (manual --rotate)
        result_2 = rotate_log(log, max_size_mb=1, max_files=5)
        assert result_2 is True
        assert log.stat().st_size == 0

        # Both archives should exist:
        # .1.gz = most recent (second rotation)
        # .2.gz = first rotation (shifted up)
        assert (logs_dir / "fast.log.1.gz").exists()
        assert (logs_dir / "fast.log.2.gz").exists()

    def test_rotate_service_logs_from_watchdog_module(
        self, mlx_stack_home: Path, logs_dir: Path
    ) -> None:
        """rotate_service_logs() rotates all eligible logs in the logs dir."""
        fast_log = logs_dir / "fast.log"
        standard_log = logs_dir / "standard.log"
        _create_log(fast_log, size_mb=2)
        _create_log(standard_log, size_mb=3)

        with patch("mlx_stack.core.watchdog.get_value") as mock_get_value:
            mock_get_value.side_effect = lambda key: {
                "log-max-size-mb": 1,
                "log-max-files": 5,
            }.get(key, "")

            count = rotate_service_logs()

        assert count == 2
        assert fast_log.stat().st_size == 0
        assert standard_log.stat().st_size == 0
        assert (logs_dir / "fast.log.1.gz").exists()
        assert (logs_dir / "standard.log.1.gz").exists()

    def test_concurrent_threaded_rotation_safety(
        self, mlx_stack_home: Path, logs_dir: Path
    ) -> None:
        """Concurrent rotation from multiple threads does not corrupt files.

        This simulates the scenario where watchdog and CLI --rotate
        execute at the exact same time via threading. One rotation
        succeeds; the other either succeeds or gets a benign I/O error
        because the temp file is already in use. Either way, no data
        is lost and the log file remains intact.
        """
        log = logs_dir / "fast.log"
        _create_log(log, size_mb=2)

        results: list[bool] = []
        errors: list[Exception] = []

        def do_rotate() -> None:
            try:
                result = rotate_log(log, max_size_mb=1, max_files=5)
                results.append(result)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=do_rotate)
        t2 = threading.Thread(target=do_rotate)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # At least one rotation should have succeeded or raised a
        # benign LogRotationError. Both outcomes are safe.
        total_outcomes = len(results) + len(errors)
        assert total_outcomes == 2

        # At least one rotation must have completed successfully
        assert len(results) >= 1

        # The log file should still exist (not deleted)
        assert log.exists()

        # The archive should exist (at least one rotation succeeded)
        archive = logs_dir / "fast.log.1.gz"
        assert archive.exists()


# =========================================================================== #
# VAL-CROSS-OPS-002: logs --follow survives rotation
# =========================================================================== #


class TestFollowSurvivesRotation:
    """Tests for logs --follow surviving copytruncate rotation.

    VAL-CROSS-OPS-002: During --follow, if rotation truncates the file,
    follow detects truncation and continues showing new content seamlessly.
    """

    def test_follow_detects_truncation_and_continues(
        self, mlx_stack_home: Path, logs_dir: Path
    ) -> None:
        """Follow detects file size decrease (truncation) and reads from 0."""
        log = logs_dir / "fast.log"
        log.write_text("line1\nline2\nline3\n")

        captured: list[str] = []

        def follow_thread() -> None:
            try:
                follow_log(
                    log,
                    num_lines=0,
                    output_callback=lambda text: captured.append(text),
                )
            except Exception:
                pass

        thread = threading.Thread(target=follow_thread, daemon=True)
        thread.start()

        # Give follow time to start and read initial position
        time.sleep(0.3)

        # Append content before rotation
        with open(log, "a") as f:
            f.write("pre-rotation-line\n")

        time.sleep(0.8)

        # Simulate copytruncate rotation: copy + truncate
        content_before = log.read_bytes()
        archive = logs_dir / "fast.log.1.gz"
        with gzip.open(str(archive), "wb") as gz:
            gz.write(content_before)
        # Truncate in-place (copytruncate)
        open(log, "w").close()

        time.sleep(0.2)

        # Write new content after rotation
        with open(log, "a") as f:
            f.write("post-rotation-line\n")

        # Give follow time to detect truncation and read new content
        time.sleep(1.0)

        # Stop the follow thread by sending KeyboardInterrupt
        # (follow_log catches KeyboardInterrupt for clean exit)
        import ctypes
        assert thread.ident is not None
        try:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(thread.ident),
                ctypes.py_object(KeyboardInterrupt),
            )
        except Exception:
            pass
        thread.join(timeout=3)

        # The follow output should contain the pre-rotation and
        # post-rotation content
        combined = "\n".join(captured)
        assert "pre-rotation-line" in combined
        assert "post-rotation-line" in combined

    def test_follow_handles_empty_file_after_truncation(
        self, mlx_stack_home: Path, logs_dir: Path
    ) -> None:
        """Follow handles the case where the file is empty after truncation."""
        log = logs_dir / "test.log"
        log.write_text("initial content\n")

        captured: list[str] = []

        def follow_thread() -> None:
            try:
                follow_log(
                    log,
                    num_lines=0,
                    output_callback=lambda text: captured.append(text),
                )
            except Exception:
                pass

        thread = threading.Thread(target=follow_thread, daemon=True)
        thread.start()

        time.sleep(0.3)

        # Truncate file (simulating rotation)
        open(log, "w").close()
        time.sleep(0.3)

        # Write new content
        with open(log, "a") as f:
            f.write("after-truncation\n")

        time.sleep(1.0)

        import ctypes
        assert thread.ident is not None
        try:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(thread.ident),
                ctypes.py_object(KeyboardInterrupt),
            )
        except Exception:
            pass
        thread.join(timeout=3)

        combined = "\n".join(captured)
        assert "after-truncation" in combined

    def test_follow_continues_after_multiple_rotations(
        self, mlx_stack_home: Path, logs_dir: Path
    ) -> None:
        """Follow continues seamlessly through multiple rotation events.

        Uses a wait-for-capture approach to avoid timing flakiness.
        After truncation, we wait for the follow thread to observe
        the zero-size file (at least one poll cycle) before writing
        new content. This ensures the truncation detection path fires.
        """
        log = logs_dir / "multi.log"
        log.write_text("")

        captured: list[str] = []

        def follow_thread() -> None:
            try:
                follow_log(
                    log,
                    num_lines=0,
                    output_callback=lambda text: captured.append(text),
                )
            except Exception:
                pass

        def wait_for_content(marker: str, timeout: float = 5.0) -> bool:
            """Wait until marker appears in captured output."""
            end = time.monotonic() + timeout
            while time.monotonic() < end:
                if any(marker in c for c in captured):
                    return True
                time.sleep(0.2)
            return False

        thread = threading.Thread(target=follow_thread, daemon=True)
        thread.start()

        time.sleep(0.3)

        # First round: write content then rotate
        with open(log, "a") as f:
            f.write("round-1-content\n")
        assert wait_for_content("round-1-content")

        # Truncate and wait for the follow thread to observe the
        # truncation (at least one poll cycle of 0.5s)
        open(log, "w").close()
        time.sleep(0.7)

        # Second round: write content then rotate
        with open(log, "a") as f:
            f.write("round-2-content\n")
        assert wait_for_content("round-2-content")

        # Truncate and wait for follow to observe
        open(log, "w").close()
        time.sleep(0.7)

        # Third round: write final content
        with open(log, "a") as f:
            f.write("round-3-content\n")
        assert wait_for_content("round-3-content")

        import ctypes
        assert thread.ident is not None
        try:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(thread.ident),
                ctypes.py_object(KeyboardInterrupt),
            )
        except Exception:
            pass
        thread.join(timeout=3)

        combined = "\n".join(captured)
        assert "round-1-content" in combined
        assert "round-2-content" in combined
        assert "round-3-content" in combined


# =========================================================================== #
# VAL-CROSS-OPS-003: launchd-started watchdog performs rotation and restart
# =========================================================================== #


class TestLaunchdWatchdogIntegration:
    """Tests for launchd-started watchdog performing rotation and restart.

    VAL-CROSS-OPS-003: Watchdog started via launchd (after install) performs
    log rotation and service auto-restart identically to manual execution.
    """

    def test_plist_runs_mlx_stack_watch(self, mlx_stack_home: Path) -> None:
        """The generated plist runs 'mlx-stack watch' as ProgramArguments."""
        from mlx_stack.core.launchd import generate_plist

        plist = generate_plist(mlx_stack_binary="/usr/local/bin/mlx-stack")

        assert plist["ProgramArguments"] == ["/usr/local/bin/mlx-stack", "watch"]
        assert plist["RunAtLoad"] is True
        assert plist["KeepAlive"] is True
        assert plist["Label"] == "com.mlx-stack.watchdog"

    def test_launchd_watchdog_performs_rotation(
        self,
        mlx_stack_home: Path,
        logs_dir: Path,
        stack_definition: dict[str, Any],
    ) -> None:
        """Watchdog (as started by launchd) triggers rotation identically to manual.

        We simulate the launchd-started watchdog by running poll_cycle
        (same code path as when launched from the plist).
        """
        fast_log = logs_dir / "fast.log"
        _create_log(fast_log, size_mb=2)

        state = WatchdogState()

        with (
            patch("mlx_stack.core.watchdog.run_status") as mock_status,
            patch("mlx_stack.core.watchdog.get_value") as mock_get_value,
        ):
            mock_result = MagicMock()
            mock_result.no_stack = False
            mock_result.services = []
            mock_status.return_value = mock_result

            mock_get_value.side_effect = lambda key: {
                "log-max-size-mb": 1,
                "log-max-files": 5,
            }.get(key, "")

            result = poll_cycle(
                state=state,
                stack=stack_definition,
                interval=30,
                max_restarts=5,
                restart_delay=10,
            )

        assert result.rotations_performed >= 1
        assert fast_log.stat().st_size == 0
        assert (logs_dir / "fast.log.1.gz").exists()

    def test_launchd_watchdog_restarts_crashed_service(
        self,
        mlx_stack_home: Path,
        pids_dir: Path,
        stack_definition: dict[str, Any],
    ) -> None:
        """Watchdog (as started by launchd) auto-restarts crashed services.

        A crashed service has a PID file but the process is dead.
        """
        from mlx_stack.core.stack_status import ServiceHealth, ServiceStatus

        # Create a stale PID file (simulates crashed process)
        pid_file = pids_dir / "fast.pid"
        pid_file.write_text("99999")  # non-existent PID

        state = WatchdogState()

        with (
            patch("mlx_stack.core.watchdog.run_status") as mock_status,
            patch("mlx_stack.core.watchdog.restart_service") as mock_restart,
            patch("mlx_stack.core.watchdog.get_value") as mock_get_value,
        ):
            # Report "fast" as crashed
            crashed_svc = ServiceStatus(
                tier="fast",
                model="qwen3.5-3b",
                port=8000,
                status=ServiceHealth.CRASHED,
                uptime=None,
                uptime_display="-",
                response_time=None,
                pid=99999,
            )
            healthy_svc = ServiceStatus(
                tier="standard",
                model="qwen3.5-8b",
                port=8001,
                status=ServiceHealth.HEALTHY,
                uptime=120.0,
                uptime_display="2m",
                response_time=0.1,
                pid=12345,
            )
            mock_result = MagicMock()
            mock_result.no_stack = False
            mock_result.services = [crashed_svc, healthy_svc]
            mock_status.return_value = mock_result

            mock_restart.return_value = True

            mock_get_value.side_effect = lambda key: {
                "log-max-size-mb": 50,
                "log-max-files": 5,
            }.get(key, "")

            result = poll_cycle(
                state=state,
                stack=stack_definition,
                interval=30,
                max_restarts=5,
                restart_delay=10,
            )

        # Should have attempted to restart the crashed service
        assert result.restarts_attempted == 1
        assert result.restarts_succeeded == 1
        mock_restart.assert_called_once()
        call_args = mock_restart.call_args
        # restart_service is called with keyword args from poll_cycle
        assert call_args.kwargs.get("service_name") == "fast"

    def test_launchctl_bootstrap_bootout_subprocess_calls(
        self,
        mlx_stack_home: Path,
        stack_definition: dict[str, Any],
    ) -> None:
        """launchctl bootstrap/bootout calls include gui/<uid> and plist path.

        VAL-CROSS-OPS-003: Verifies that when install_agent() and
        uninstall_agent() invoke launchctl, the command arguments include
        the correct gui/<uid> domain and plist path.
        """
        from mlx_stack.core.launchd import (
            generate_plist,
            load_agent,
            unload_agent,
            write_plist,
        )

        plist_path = mlx_stack_home / "com.mlx-stack.watchdog.plist"
        plist_data = generate_plist(mlx_stack_binary="/usr/local/bin/mlx-stack")
        write_plist(plist_data, plist_path=plist_path)

        uid = os.getuid()

        # Test bootstrap (load_agent) call args
        with patch("mlx_stack.core.launchd.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
            load_agent(plist_path=plist_path)

            mock_run.assert_called_once()
            call_args = mock_run.call_args
            cmd = call_args[0][0]  # positional arg: the command list
            assert cmd[0] == "launchctl"
            assert cmd[1] == "bootstrap"
            assert cmd[2] == f"gui/{uid}"
            assert cmd[3] == str(plist_path)

        # Test bootout (unload_agent) call args
        with patch("mlx_stack.core.launchd.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
            unload_agent(plist_path=plist_path)

            mock_run.assert_called_once()
            call_args = mock_run.call_args
            cmd = call_args[0][0]
            assert cmd[0] == "launchctl"
            assert cmd[1] == "bootout"
            assert cmd[2] == f"gui/{uid}"
            assert cmd[3] == str(plist_path)

    def test_launchd_install_then_watchdog_performs_duties(
        self,
        mlx_stack_home: Path,
        logs_dir: Path,
        pids_dir: Path,
        stack_definition: dict[str, Any],
    ) -> None:
        """Full flow: install generates plist → watchdog poll performs duties.

        Verifies that the plist generated by install points to the correct
        command, and that the watchdog code path (poll_cycle) performs
        rotation and restart when triggered.
        """
        from mlx_stack.core.launchd import generate_plist, write_plist
        from mlx_stack.core.stack_status import ServiceHealth, ServiceStatus

        # Step 1: Generate and write plist
        plist_path = mlx_stack_home / "com.mlx-stack.watchdog.plist"
        plist_data = generate_plist(mlx_stack_binary="/usr/local/bin/mlx-stack")
        write_plist(plist_data, plist_path=plist_path)

        # Verify plist is valid and has correct structure
        with open(plist_path, "rb") as f:
            loaded_plist = plistlib.load(f)
        assert loaded_plist["ProgramArguments"] == ["/usr/local/bin/mlx-stack", "watch"]

        # Step 2: Simulate watchdog started by launchd doing its duties
        fast_log = logs_dir / "fast.log"
        _create_log(fast_log, size_mb=2)

        # Stale PID for standard (crashed)
        pid_file = pids_dir / "standard.pid"
        pid_file.write_text("88888")

        state = WatchdogState()

        with (
            patch("mlx_stack.core.watchdog.run_status") as mock_status,
            patch("mlx_stack.core.watchdog.restart_service") as mock_restart,
            patch("mlx_stack.core.watchdog.get_value") as mock_get_value,
        ):
            crashed_svc = ServiceStatus(
                tier="standard",
                model="qwen3.5-8b",
                port=8001,
                status=ServiceHealth.CRASHED,
                uptime=None,
                uptime_display="-",
                response_time=None,
                pid=88888,
            )
            mock_result = MagicMock()
            mock_result.no_stack = False
            mock_result.services = [crashed_svc]
            mock_status.return_value = mock_result

            mock_restart.return_value = True

            mock_get_value.side_effect = lambda key: {
                "log-max-size-mb": 1,
                "log-max-files": 5,
            }.get(key, "")

            result = poll_cycle(
                state=state,
                stack=stack_definition,
                interval=30,
                max_restarts=5,
                restart_delay=10,
            )

        # Watchdog both rotated and restarted
        assert result.rotations_performed >= 1
        assert result.restarts_attempted == 1
        assert result.restarts_succeeded == 1


# =========================================================================== #
# VAL-CROSS-OPS-004: down/up interaction with watchdog
# =========================================================================== #


class TestDownUpWatchdogInteraction:
    """Tests for down/up interaction with watchdog.

    VAL-CROSS-OPS-004: `mlx-stack down` sets services to stopped (watchdog
    does not restart them). `mlx-stack up` starts services (watchdog monitors
    them). No lock conflicts between poll cycles.
    """

    def test_stopped_services_not_restarted_by_watchdog(
        self,
        mlx_stack_home: Path,
        pids_dir: Path,
        stack_definition: dict[str, Any],
    ) -> None:
        """Services stopped via `down` (no PID file) are NOT restarted.

        The watchdog only restarts crashed services (PID file exists,
        process dead), not stopped services (no PID file).
        """
        from mlx_stack.core.stack_status import ServiceHealth, ServiceStatus

        # No PID files — services are stopped (as after `down`)
        state = WatchdogState()

        with (
            patch("mlx_stack.core.watchdog.run_status") as mock_status,
            patch("mlx_stack.core.watchdog.restart_service") as mock_restart,
            patch("mlx_stack.core.watchdog.get_value") as mock_get_value,
        ):
            stopped_fast = ServiceStatus(
                tier="fast",
                model="qwen3.5-3b",
                port=8000,
                status=ServiceHealth.STOPPED,
                uptime=None,
                uptime_display="-",
                response_time=None,
                pid=None,
            )
            stopped_standard = ServiceStatus(
                tier="standard",
                model="qwen3.5-8b",
                port=8001,
                status=ServiceHealth.STOPPED,
                uptime=None,
                uptime_display="-",
                response_time=None,
                pid=None,
            )
            mock_result = MagicMock()
            mock_result.no_stack = False
            mock_result.services = [stopped_fast, stopped_standard]
            mock_status.return_value = mock_result

            mock_get_value.side_effect = lambda key: {
                "log-max-size-mb": 50,
                "log-max-files": 5,
            }.get(key, "")

            result = poll_cycle(
                state=state,
                stack=stack_definition,
                interval=30,
                max_restarts=5,
                restart_delay=10,
            )

        # Watchdog should NOT have attempted any restarts
        assert result.restarts_attempted == 0
        assert result.restarts_succeeded == 0
        mock_restart.assert_not_called()

    def test_up_services_monitored_by_watchdog(
        self,
        mlx_stack_home: Path,
        pids_dir: Path,
        stack_definition: dict[str, Any],
    ) -> None:
        """Services started via `up` (PID files, healthy) are monitored.

        If one of these services crashes later, the watchdog detects it
        and restarts it.
        """
        from mlx_stack.core.stack_status import ServiceHealth, ServiceStatus

        state = WatchdogState()

        # First poll: all healthy (services just started by `up`)
        with (
            patch("mlx_stack.core.watchdog.run_status") as mock_status,
            patch("mlx_stack.core.watchdog.restart_service") as mock_restart,
            patch("mlx_stack.core.watchdog.get_value") as mock_get_value,
        ):
            healthy_fast = ServiceStatus(
                tier="fast",
                model="qwen3.5-3b",
                port=8000,
                status=ServiceHealth.HEALTHY,
                uptime=60.0,
                uptime_display="1m",
                response_time=0.05,
                pid=10001,
            )
            healthy_standard = ServiceStatus(
                tier="standard",
                model="qwen3.5-8b",
                port=8001,
                status=ServiceHealth.HEALTHY,
                uptime=55.0,
                uptime_display="55s",
                response_time=0.08,
                pid=10002,
            )
            mock_result = MagicMock()
            mock_result.no_stack = False
            mock_result.services = [healthy_fast, healthy_standard]
            mock_status.return_value = mock_result

            mock_get_value.side_effect = lambda key: {
                "log-max-size-mb": 50,
                "log-max-files": 5,
            }.get(key, "")

            result_1 = poll_cycle(
                state=state,
                stack=stack_definition,
                interval=30,
                max_restarts=5,
                restart_delay=10,
            )

        assert result_1.restarts_attempted == 0
        mock_restart.assert_not_called()

        # Second poll: fast crashes → watchdog should restart it
        with (
            patch("mlx_stack.core.watchdog.run_status") as mock_status,
            patch("mlx_stack.core.watchdog.restart_service") as mock_restart,
            patch("mlx_stack.core.watchdog.get_value") as mock_get_value,
        ):
            crashed_fast = ServiceStatus(
                tier="fast",
                model="qwen3.5-3b",
                port=8000,
                status=ServiceHealth.CRASHED,
                uptime=None,
                uptime_display="-",
                response_time=None,
                pid=10001,
            )
            healthy_standard = ServiceStatus(
                tier="standard",
                model="qwen3.5-8b",
                port=8001,
                status=ServiceHealth.HEALTHY,
                uptime=85.0,
                uptime_display="1m 25s",
                response_time=0.08,
                pid=10002,
            )
            mock_result = MagicMock()
            mock_result.no_stack = False
            mock_result.services = [crashed_fast, healthy_standard]
            mock_status.return_value = mock_result

            mock_restart.return_value = True

            mock_get_value.side_effect = lambda key: {
                "log-max-size-mb": 50,
                "log-max-files": 5,
            }.get(key, "")

            result_2 = poll_cycle(
                state=state,
                stack=stack_definition,
                interval=30,
                max_restarts=5,
                restart_delay=10,
            )

        assert result_2.restarts_attempted == 1
        assert result_2.restarts_succeeded == 1

    def test_down_then_watchdog_then_up_then_watchdog(
        self,
        mlx_stack_home: Path,
        pids_dir: Path,
        stack_definition: dict[str, Any],
    ) -> None:
        """Full down → watchdog → up → watchdog sequence.

        After down: watchdog sees stopped services, does nothing.
        After up: watchdog monitors services, restarts crashes.
        """
        from mlx_stack.core.stack_status import ServiceHealth, ServiceStatus

        state = WatchdogState()

        # Phase 1: After `down` — all stopped
        with (
            patch("mlx_stack.core.watchdog.run_status") as mock_status,
            patch("mlx_stack.core.watchdog.restart_service") as _mock_restart,
            patch("mlx_stack.core.watchdog.get_value") as mock_get_value,
        ):
            mock_result = MagicMock()
            mock_result.no_stack = False
            mock_result.services = [
                ServiceStatus(
                    tier="fast", model="qwen3.5-3b", port=8000,
                    status=ServiceHealth.STOPPED, uptime=None, uptime_display="-",
                    response_time=None, pid=None,
                ),
                ServiceStatus(
                    tier="standard", model="qwen3.5-8b", port=8001,
                    status=ServiceHealth.STOPPED, uptime=None, uptime_display="-",
                    response_time=None, pid=None,
                ),
            ]
            mock_status.return_value = mock_result

            mock_get_value.side_effect = lambda key: {
                "log-max-size-mb": 50,
                "log-max-files": 5,
            }.get(key, "")

            result = poll_cycle(
                state=state, stack=stack_definition,
                interval=30, max_restarts=5, restart_delay=10,
            )

        assert result.restarts_attempted == 0

        # Phase 2: After `up` — all healthy
        with (
            patch("mlx_stack.core.watchdog.run_status") as mock_status,
            patch("mlx_stack.core.watchdog.restart_service") as _mock_restart,
            patch("mlx_stack.core.watchdog.get_value") as mock_get_value,
        ):
            mock_result = MagicMock()
            mock_result.no_stack = False
            mock_result.services = [
                ServiceStatus(
                    tier="fast", model="qwen3.5-3b", port=8000,
                    status=ServiceHealth.HEALTHY, uptime=10.0, uptime_display="10s",
                    response_time=0.05, pid=20001,
                ),
                ServiceStatus(
                    tier="standard", model="qwen3.5-8b", port=8001,
                    status=ServiceHealth.HEALTHY, uptime=8.0, uptime_display="8s",
                    response_time=0.08, pid=20002,
                ),
            ]
            mock_status.return_value = mock_result

            mock_get_value.side_effect = lambda key: {
                "log-max-size-mb": 50,
                "log-max-files": 5,
            }.get(key, "")

            result = poll_cycle(
                state=state, stack=stack_definition,
                interval=30, max_restarts=5, restart_delay=10,
            )

        assert result.restarts_attempted == 0

    def test_watchdog_lock_only_during_restart(
        self,
        mlx_stack_home: Path,
        pids_dir: Path,
        stack_definition: dict[str, Any],
    ) -> None:
        """Watchdog acquires lock only during restart, not during polling.

        `status` should be able to run during poll cycles without conflict.
        """
        from mlx_stack.core.stack_status import ServiceHealth, ServiceStatus

        state = WatchdogState()

        # Poll with all healthy — no lock should be acquired
        with (
            patch("mlx_stack.core.watchdog.run_status") as mock_status,
            patch("mlx_stack.core.watchdog.acquire_lock") as mock_lock,
            patch("mlx_stack.core.watchdog.get_value") as mock_get_value,
        ):
            mock_result = MagicMock()
            mock_result.no_stack = False
            mock_result.services = [
                ServiceStatus(
                    tier="fast", model="qwen3.5-3b", port=8000,
                    status=ServiceHealth.HEALTHY, uptime=60.0, uptime_display="1m",
                    response_time=0.05, pid=10001,
                ),
            ]
            mock_status.return_value = mock_result

            mock_get_value.side_effect = lambda key: {
                "log-max-size-mb": 50,
                "log-max-files": 5,
            }.get(key, "")

            poll_cycle(
                state=state, stack=stack_definition,
                interval=30, max_restarts=5, restart_delay=10,
            )

        # acquire_lock should NOT have been called during a healthy poll
        # (lock is only acquired inside restart_service, which wasn't called)
        mock_lock.assert_not_called()


# =========================================================================== #
# VAL-CROSS-OPS-005: Full ops lifecycle
# =========================================================================== #


class TestFullOpsLifecycle:
    """Tests for the full ops lifecycle.

    VAL-CROSS-OPS-005: init -> install -> watchdog starts -> up -> log
    generation -> rotation -> follow -> kill service -> watchdog restarts
    -> down -> uninstall.
    """

    def test_full_lifecycle_sequence(
        self,
        mlx_stack_home: Path,
        logs_dir: Path,
        pids_dir: Path,
        stack_definition: dict[str, Any],
    ) -> None:
        """Full lifecycle: init -> install -> watch -> up -> rotate ->
        follow -> kill -> restart -> down -> uninstall.

        Each step is validated to ensure correct state transitions.
        """
        from mlx_stack.core.launchd import (
            generate_plist,
            write_plist,
        )
        from mlx_stack.core.process import remove_pid_file, write_pid_file
        from mlx_stack.core.stack_status import ServiceHealth, ServiceStatus

        # --- Step 1: init (already done by stack_definition fixture) ---
        stacks_dir = mlx_stack_home / "stacks"
        assert (stacks_dir / "default.yaml").exists()

        # --- Step 2: install (generate plist) ---
        plist_path = mlx_stack_home / "com.mlx-stack.watchdog.plist"
        plist_data = generate_plist(mlx_stack_binary="/usr/local/bin/mlx-stack")
        write_plist(plist_data, plist_path=plist_path)
        assert plist_path.exists()
        with open(plist_path, "rb") as f:
            loaded_plist = plistlib.load(f)
        assert loaded_plist["ProgramArguments"] == ["/usr/local/bin/mlx-stack", "watch"]

        # --- Step 3: watchdog starts (simulate via WatchdogState) ---
        state = WatchdogState()
        watchdog_pid_file = pids_dir / "watchdog.pid"
        write_pid_file("watchdog", os.getpid())
        assert watchdog_pid_file.exists()

        # --- Step 4: up (simulate services started) ---
        write_pid_file("fast", 30001)
        write_pid_file("standard", 30002)
        write_pid_file("litellm", 30003)

        # Create log files (simulating service output)
        fast_log = logs_dir / "fast.log"
        standard_log = logs_dir / "standard.log"
        litellm_log = logs_dir / "litellm.log"
        fast_log.write_text("fast service started\n")
        standard_log.write_text("standard service started\n")
        litellm_log.write_text("litellm proxy started\n")

        # --- Step 5: Log generation exceeding threshold ---
        _create_log(fast_log, size_mb=2)

        # --- Step 6: Rotation (via watchdog poll) ---
        with (
            patch("mlx_stack.core.watchdog.run_status") as mock_status,
            patch("mlx_stack.core.watchdog.get_value") as mock_get_value,
        ):
            mock_result = MagicMock()
            mock_result.no_stack = False
            mock_result.services = [
                ServiceStatus(
                    tier="fast", model="qwen3.5-3b", port=8000,
                    status=ServiceHealth.HEALTHY, uptime=300.0, uptime_display="5m",
                    response_time=0.05, pid=30001,
                ),
                ServiceStatus(
                    tier="standard", model="qwen3.5-8b", port=8001,
                    status=ServiceHealth.HEALTHY, uptime=300.0, uptime_display="5m",
                    response_time=0.08, pid=30002,
                ),
            ]
            mock_status.return_value = mock_result

            mock_get_value.side_effect = lambda key: {
                "log-max-size-mb": 1,
                "log-max-files": 5,
            }.get(key, "")

            poll_result = poll_cycle(
                state=state,
                stack=stack_definition,
                interval=30,
                max_restarts=5,
                restart_delay=10,
            )

        assert poll_result.rotations_performed >= 1
        assert fast_log.stat().st_size == 0
        assert (logs_dir / "fast.log.1.gz").exists()

        # --- Step 7: Follow (verify log following detects new lines) ---
        # Write initial content after rotation
        fast_log.write_text("post-rotation log line\n")

        follow_captured: list[str] = []

        def follow_thread_fn() -> None:
            try:
                follow_log(
                    fast_log,
                    num_lines=0,
                    output_callback=lambda text: follow_captured.append(text),
                )
            except Exception:
                pass

        follow_thread = threading.Thread(target=follow_thread_fn, daemon=True)
        follow_thread.start()

        # Give follow time to start and read initial position
        time.sleep(0.3)

        # Append new content (simulating live service logging)
        with open(fast_log, "a") as f:
            f.write("follow-detected-line\n")

        # Wait for follow to pick up the new content
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if any("follow-detected-line" in c for c in follow_captured):
                break
            time.sleep(0.2)

        # Stop the follow thread
        import ctypes
        assert follow_thread.ident is not None
        try:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(follow_thread.ident),
                ctypes.py_object(KeyboardInterrupt),
            )
        except Exception:
            pass
        follow_thread.join(timeout=3)

        # Verify follow detected the new line
        combined_follow = "\n".join(follow_captured)
        assert "follow-detected-line" in combined_follow

        # --- Step 8: Kill service (simulate crash) ---
        # Remove PID to simulate process death BUT leave PID file
        # (this is how the watchdog detects a crash)
        # In our test, the PID 30001 doesn't correspond to a real process
        # so the watchdog will see it as crashed

        # --- Step 9: Watchdog restarts crashed service ---
        with (
            patch("mlx_stack.core.watchdog.run_status") as mock_status,
            patch("mlx_stack.core.watchdog.restart_service") as mock_restart,
            patch("mlx_stack.core.watchdog.get_value") as mock_get_value,
        ):
            mock_result = MagicMock()
            mock_result.no_stack = False
            mock_result.services = [
                ServiceStatus(
                    tier="fast", model="qwen3.5-3b", port=8000,
                    status=ServiceHealth.CRASHED, uptime=None, uptime_display="-",
                    response_time=None, pid=30001,
                ),
                ServiceStatus(
                    tier="standard", model="qwen3.5-8b", port=8001,
                    status=ServiceHealth.HEALTHY, uptime=600.0, uptime_display="10m",
                    response_time=0.08, pid=30002,
                ),
            ]
            mock_status.return_value = mock_result

            mock_restart.return_value = True

            mock_get_value.side_effect = lambda key: {
                "log-max-size-mb": 50,
                "log-max-files": 5,
            }.get(key, "")

            poll_result_2 = poll_cycle(
                state=state,
                stack=stack_definition,
                interval=30,
                max_restarts=5,
                restart_delay=10,
            )

        assert poll_result_2.restarts_attempted == 1
        assert poll_result_2.restarts_succeeded == 1

        # --- Step 10: down (remove PID files) ---
        remove_pid_file("fast")
        remove_pid_file("standard")
        remove_pid_file("litellm")

        # Verify PID files are gone
        assert not (pids_dir / "fast.pid").exists()
        assert not (pids_dir / "standard.pid").exists()
        assert not (pids_dir / "litellm.pid").exists()

        # --- Step 11: After down, watchdog sees stopped — no restarts ---
        with (
            patch("mlx_stack.core.watchdog.run_status") as mock_status,
            patch("mlx_stack.core.watchdog.restart_service") as mock_restart,
            patch("mlx_stack.core.watchdog.get_value") as mock_get_value,
        ):
            mock_result = MagicMock()
            mock_result.no_stack = False
            mock_result.services = [
                ServiceStatus(
                    tier="fast", model="qwen3.5-3b", port=8000,
                    status=ServiceHealth.STOPPED, uptime=None, uptime_display="-",
                    response_time=None, pid=None,
                ),
                ServiceStatus(
                    tier="standard", model="qwen3.5-8b", port=8001,
                    status=ServiceHealth.STOPPED, uptime=None, uptime_display="-",
                    response_time=None, pid=None,
                ),
            ]
            mock_status.return_value = mock_result

            mock_get_value.side_effect = lambda key: {
                "log-max-size-mb": 50,
                "log-max-files": 5,
            }.get(key, "")

            poll_result_3 = poll_cycle(
                state=state,
                stack=stack_definition,
                interval=30,
                max_restarts=5,
                restart_delay=10,
            )

        assert poll_result_3.restarts_attempted == 0
        mock_restart.assert_not_called()

        # --- Step 12: uninstall (remove plist) ---
        plist_path.unlink()
        assert not plist_path.exists()

        # Clean up watchdog PID
        remove_pid_file("watchdog")
        assert not watchdog_pid_file.exists()

    def test_lifecycle_rotation_preserves_service_logging(
        self,
        mlx_stack_home: Path,
        logs_dir: Path,
    ) -> None:
        """After rotation, new log content is written to the same file path.

        This simulates the key property of copytruncate: the FD remains
        valid because only the file content is truncated, not the inode.
        """
        log = logs_dir / "fast.log"
        # Simulate initial log content
        initial_content = "startup message\nhealth check OK\n"
        log.write_text(initial_content)

        # Make the file big enough to trigger rotation
        _create_log(log, size_mb=2)

        # Rotate
        rotate_log(log, max_size_mb=1, max_files=5)
        assert log.stat().st_size == 0

        # Simulate service continuing to write (using append mode
        # as process.py uses "a" mode)
        with open(log, "a") as f:
            f.write("post-rotation health check OK\n")
            f.write("post-rotation request handled\n")

        content = log.read_text()
        assert "post-rotation health check OK" in content
        assert "post-rotation request handled" in content

        # Archive should contain the old content
        archive = logs_dir / "fast.log.1.gz"
        assert archive.exists()

    def test_lifecycle_down_clears_all_state(
        self,
        mlx_stack_home: Path,
        pids_dir: Path,
    ) -> None:
        """After down, all PID files are removed and no managed processes remain."""
        from mlx_stack.core.process import list_pid_files, remove_pid_file, write_pid_file

        # Simulate running services
        write_pid_file("fast", 40001)
        write_pid_file("standard", 40002)
        write_pid_file("litellm", 40003)

        assert len(list_pid_files()) == 3

        # Simulate down
        for svc in ["litellm", "standard", "fast"]:
            remove_pid_file(svc)

        assert len(list_pid_files()) == 0

    def test_lifecycle_watchdog_restart_uses_stack_definition(
        self,
        mlx_stack_home: Path,
        pids_dir: Path,
        stack_definition: dict[str, Any],
    ) -> None:
        """Watchdog restart rebuilds the command from the stack definition.

        This ensures the restarted service uses the same model, port,
        and flags as the original.
        """

        tracker = ServiceTracker()

        with (
            patch("mlx_stack.core.watchdog.acquire_lock") as mock_lock,
            patch("mlx_stack.core.watchdog.start_service") as mock_start,
            patch("mlx_stack.core.watchdog.remove_pid_file"),
        ):
            mock_lock.return_value.__enter__ = MagicMock(return_value=None)
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)
            mock_start.return_value = MagicMock()

            restart_service(
                service_name="fast",
                stack=stack_definition,
                tracker=tracker,
                vllm_binary="/usr/bin/vllm-mlx",
            )

        # Verify start_service was called with the correct command
        mock_start.assert_called_once()
        call_kwargs = mock_start.call_args
        # The cmd should contain the vllm binary and the model source
        cmd = call_kwargs[1].get("cmd", call_kwargs[0][1] if len(call_kwargs[0]) > 1 else None)
        if cmd is None:
            cmd = call_kwargs.kwargs.get("cmd")
        assert cmd is not None, "start_service was not called with a 'cmd' argument"
        assert "/usr/bin/vllm-mlx" in cmd[0]
        assert "8000" in str(cmd)  # Port from stack definition
