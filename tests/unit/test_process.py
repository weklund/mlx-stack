"""Tests for the process management module (core/process.py).

Covers PID file management, process state checks, lockfile management,
HTTP health checks, port conflict detection, subprocess management,
service shutdown, status reporting, and uptime formatting.
"""

from __future__ import annotations

import os
import signal
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import psutil
import pytest

from mlx_stack.core.process import (
    HEALTH_CHECK_TIMEOUT,
    SHUTDOWN_GRACE_PERIOD,
    STATUS_CHECK_TIMEOUT,
    STATUS_DEGRADED_THRESHOLD,
    HealthCheckError,
    HealthCheckResult,
    LockError,
    PortConflictError,
    ProcessError,
    ServiceInfo,
    ShutdownResult,
    acquire_lock,
    check_port_conflict,
    cleanup_stale_pid,
    detect_port_conflict,
    format_uptime,
    get_service_status,
    http_health_check,
    is_process_alive,
    is_stale_pid,
    list_pid_files,
    read_pid_file,
    remove_pid_file,
    start_service,
    stop_service,
    wait_for_healthy,
    write_pid_file,
)

# =========================================================================== #
# PID file management
# =========================================================================== #


class TestWritePidFile:
    """Tests for write_pid_file."""

    def test_creates_pid_file(self, mlx_stack_home: Path) -> None:
        path = write_pid_file("fast", 12345)
        assert path.exists()
        assert path.read_text() == "12345"

    def test_pid_file_has_no_trailing_whitespace(self, mlx_stack_home: Path) -> None:
        path = write_pid_file("standard", 99999)
        content = path.read_text()
        assert content == "99999"
        assert content == content.strip()

    def test_creates_pids_directory(self, clean_mlx_stack_home: Path) -> None:
        pids_dir = clean_mlx_stack_home / "pids"
        assert not pids_dir.exists()
        write_pid_file("test-svc", 1)
        assert pids_dir.exists()

    def test_overwrites_existing_pid_file(self, mlx_stack_home: Path) -> None:
        write_pid_file("fast", 111)
        write_pid_file("fast", 222)
        path = mlx_stack_home / "pids" / "fast.pid"
        assert path.read_text() == "222"

    def test_write_pid_file_os_error(self, mlx_stack_home: Path) -> None:
        with patch("mlx_stack.core.process._ensure_pids_dir") as mock_dir:
            bad_path = mlx_stack_home / "pids"
            bad_path.mkdir(parents=True, exist_ok=True)
            mock_dir.return_value = bad_path
            # Make the directory read-only to trigger OSError
            pid_file = bad_path / "locked.pid"
            pid_file.write_text("old")
            pid_file.chmod(0o000)
            try:
                with pytest.raises(ProcessError, match="Could not write PID file"):
                    write_pid_file("locked", 999)
            finally:
                pid_file.chmod(0o644)


class TestReadPidFile:
    """Tests for read_pid_file."""

    def test_read_existing_pid(self, mlx_stack_home: Path) -> None:
        (mlx_stack_home / "pids").mkdir(parents=True, exist_ok=True)
        (mlx_stack_home / "pids" / "fast.pid").write_text("54321")
        assert read_pid_file("fast") == 54321

    def test_read_missing_file(self, mlx_stack_home: Path) -> None:
        assert read_pid_file("nonexistent") is None

    def test_read_empty_file(self, mlx_stack_home: Path) -> None:
        (mlx_stack_home / "pids").mkdir(parents=True, exist_ok=True)
        (mlx_stack_home / "pids" / "empty.pid").write_text("")
        assert read_pid_file("empty") is None

    def test_read_corrupt_file_raises(self, mlx_stack_home: Path) -> None:
        (mlx_stack_home / "pids").mkdir(parents=True, exist_ok=True)
        (mlx_stack_home / "pids" / "bad.pid").write_text("not-a-number")
        with pytest.raises(ProcessError, match="non-numeric content"):
            read_pid_file("bad")

    def test_read_pid_with_whitespace(self, mlx_stack_home: Path) -> None:
        (mlx_stack_home / "pids").mkdir(parents=True, exist_ok=True)
        (mlx_stack_home / "pids" / "ws.pid").write_text("  12345  \n")
        assert read_pid_file("ws") == 12345


class TestRemovePidFile:
    """Tests for remove_pid_file."""

    def test_remove_existing(self, mlx_stack_home: Path) -> None:
        (mlx_stack_home / "pids").mkdir(parents=True, exist_ok=True)
        (mlx_stack_home / "pids" / "fast.pid").write_text("123")
        assert remove_pid_file("fast") is True
        assert not (mlx_stack_home / "pids" / "fast.pid").exists()

    def test_remove_nonexistent(self, mlx_stack_home: Path) -> None:
        assert remove_pid_file("nonexistent") is False


class TestListPidFiles:
    """Tests for list_pid_files."""

    def test_empty_directory(self, mlx_stack_home: Path) -> None:
        assert list_pid_files() == {}

    def test_no_pids_dir(self, clean_mlx_stack_home: Path) -> None:
        assert list_pid_files() == {}

    def test_lists_multiple_files(self, mlx_stack_home: Path) -> None:
        pids_dir = mlx_stack_home / "pids"
        pids_dir.mkdir(parents=True, exist_ok=True)
        (pids_dir / "fast.pid").write_text("1")
        (pids_dir / "standard.pid").write_text("2")
        (pids_dir / "litellm.pid").write_text("3")

        result = list_pid_files()
        assert set(result.keys()) == {"fast", "standard", "litellm"}

    def test_ignores_non_pid_files(self, mlx_stack_home: Path) -> None:
        pids_dir = mlx_stack_home / "pids"
        pids_dir.mkdir(parents=True, exist_ok=True)
        (pids_dir / "fast.pid").write_text("1")
        (pids_dir / "readme.txt").write_text("ignore")

        result = list_pid_files()
        assert list(result.keys()) == ["fast"]


# =========================================================================== #
# Process state checks
# =========================================================================== #


class TestIsProcessAlive:
    """Tests for is_process_alive."""

    def test_current_process_is_alive(self) -> None:
        assert is_process_alive(os.getpid()) is True

    def test_nonexistent_pid(self) -> None:
        # Use a very high PID that almost certainly doesn't exist
        assert is_process_alive(999999999) is False

    @patch("mlx_stack.core.process.psutil.pid_exists", return_value=True)
    @patch("mlx_stack.core.process.psutil.Process")
    def test_zombie_process(self, mock_process: MagicMock, mock_exists: MagicMock) -> None:
        mock_proc = MagicMock()
        mock_proc.status.return_value = psutil.STATUS_ZOMBIE
        mock_process.return_value = mock_proc
        assert is_process_alive(123) is False

    @patch("mlx_stack.core.process.psutil.pid_exists", return_value=True)
    @patch("mlx_stack.core.process.psutil.Process", side_effect=psutil.NoSuchProcess(123))
    def test_no_such_process(self, mock_process: MagicMock, mock_exists: MagicMock) -> None:
        assert is_process_alive(123) is False

    @patch("mlx_stack.core.process.psutil.pid_exists", return_value=True)
    @patch("mlx_stack.core.process.psutil.Process", side_effect=psutil.AccessDenied(123))
    def test_access_denied(self, mock_process: MagicMock, mock_exists: MagicMock) -> None:
        assert is_process_alive(123) is False


class TestIsStalePid:
    """Tests for is_stale_pid."""

    def test_no_pid_file(self, mlx_stack_home: Path) -> None:
        assert is_stale_pid("nonexistent") is False

    def test_process_alive(self, mlx_stack_home: Path) -> None:
        write_pid_file("live", os.getpid())
        assert is_stale_pid("live") is False

    def test_process_dead(self, mlx_stack_home: Path) -> None:
        write_pid_file("dead", 999999999)
        assert is_stale_pid("dead") is True

    def test_corrupt_pid_file(self, mlx_stack_home: Path) -> None:
        (mlx_stack_home / "pids").mkdir(parents=True, exist_ok=True)
        (mlx_stack_home / "pids" / "bad.pid").write_text("garbage")
        assert is_stale_pid("bad") is True


class TestCleanupStalePid:
    """Tests for cleanup_stale_pid."""

    def test_cleans_up_dead_process(self, mlx_stack_home: Path) -> None:
        write_pid_file("dead", 999999999)
        assert cleanup_stale_pid("dead") is True
        assert read_pid_file("dead") is None

    def test_does_not_clean_live_process(self, mlx_stack_home: Path) -> None:
        write_pid_file("live", os.getpid())
        assert cleanup_stale_pid("live") is False
        assert read_pid_file("live") == os.getpid()

    def test_no_pid_file(self, mlx_stack_home: Path) -> None:
        assert cleanup_stale_pid("nonexistent") is False


# =========================================================================== #
# Lockfile management
# =========================================================================== #


class TestAcquireLock:
    """Tests for acquire_lock."""

    def test_acquires_and_releases(self, mlx_stack_home: Path) -> None:
        with acquire_lock():
            lock_path = mlx_stack_home / "lock"
            assert lock_path.exists()
        # After release, we should be able to re-acquire
        with acquire_lock():
            pass

    def test_concurrent_lock_raises(self, mlx_stack_home: Path) -> None:
        with acquire_lock():
            with pytest.raises(LockError, match="Another mlx-stack operation"):
                with acquire_lock():
                    pass  # pragma: no cover

    def test_creates_data_home(self, clean_mlx_stack_home: Path) -> None:
        with acquire_lock():
            assert clean_mlx_stack_home.exists()


# =========================================================================== #
# Health checks
# =========================================================================== #


class TestHttpHealthCheck:
    """Tests for http_health_check."""

    @patch("mlx_stack.core.process.httpx.get")
    def test_healthy_response(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = http_health_check(8000)
        assert result.healthy is True
        assert result.status_code == 200
        assert result.response_time is not None

    @patch("mlx_stack.core.process.httpx.get")
    def test_unhealthy_status_code(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_get.return_value = mock_response

        result = http_health_check(8000)
        assert result.healthy is False
        assert result.status_code == 503

    @patch("mlx_stack.core.process.httpx.get", side_effect=httpx.ConnectError("refused"))
    def test_connection_error(self, mock_get: MagicMock) -> None:
        result = http_health_check(8000)
        assert result.healthy is False
        assert result.response_time is None
        assert result.status_code is None

    @patch("mlx_stack.core.process.httpx.get", side_effect=httpx.TimeoutException("timeout"))
    def test_timeout_error(self, mock_get: MagicMock) -> None:
        result = http_health_check(8000)
        assert result.healthy is False

    @patch("mlx_stack.core.process.httpx.get", side_effect=OSError("network error"))
    def test_os_error(self, mock_get: MagicMock) -> None:
        result = http_health_check(8000)
        assert result.healthy is False

    @patch("mlx_stack.core.process.httpx.get")
    def test_custom_path_and_host(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        http_health_check(8000, path="/health", host="localhost")
        mock_get.assert_called_once()
        call_url = mock_get.call_args[0][0]
        assert call_url == "http://localhost:8000/health"


class TestWaitForHealthy:
    """Tests for wait_for_healthy."""

    @patch("mlx_stack.core.process.http_health_check")
    def test_immediately_healthy(self, mock_check: MagicMock) -> None:
        mock_check.return_value = HealthCheckResult(
            healthy=True, response_time=0.1, status_code=200
        )
        result = wait_for_healthy(8000, total_timeout=5)
        assert result.healthy is True

    @patch("mlx_stack.core.process.time.sleep")
    @patch("mlx_stack.core.process.http_health_check")
    def test_healthy_after_retries(
        self, mock_check: MagicMock, mock_sleep: MagicMock
    ) -> None:
        # First 2 calls fail, third succeeds
        mock_check.side_effect = [
            HealthCheckResult(healthy=False, response_time=None, status_code=None),
            HealthCheckResult(healthy=False, response_time=None, status_code=None),
            HealthCheckResult(healthy=True, response_time=0.2, status_code=200),
        ]
        result = wait_for_healthy(
            8000,
            total_timeout=30,
            initial_delay=0.1,
        )
        assert result.healthy is True
        assert mock_check.call_count == 3

    @patch("mlx_stack.core.process.time.sleep")
    @patch("mlx_stack.core.process.time.monotonic")
    @patch("mlx_stack.core.process.http_health_check")
    def test_timeout_raises(
        self,
        mock_check: MagicMock,
        mock_monotonic: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        mock_check.return_value = HealthCheckResult(
            healthy=False, response_time=None, status_code=None
        )
        # Simulate time passing past the deadline
        mock_monotonic.side_effect = [
            0.0,   # deadline = 0 + 2 = 2
            0.5,   # first iteration check
            0.5,   # per_request_timeout check
            1.0,   # after first check
            1.5,   # sleep calculation
            1.8,   # second iteration check
            1.8,   # per_request_timeout check
            2.5,   # after second check - past deadline
        ]

        with pytest.raises(HealthCheckError, match="timed out"):
            wait_for_healthy(8000, total_timeout=2, initial_delay=0.5)


# =========================================================================== #
# Port conflict detection
# =========================================================================== #


class TestSocketBindCheck:
    """Tests for _socket_bind_check."""

    def test_available_port(self) -> None:
        """Available port returns False (not in use)."""
        from mlx_stack.core.process import _socket_bind_check
        import socket

        # Find a free port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        _, port = sock.getsockname()
        sock.close()

        assert _socket_bind_check(port) is False

    def test_occupied_port(self) -> None:
        """Occupied port returns True (in use)."""
        from mlx_stack.core.process import _socket_bind_check
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        sock.bind(("127.0.0.1", 0))
        _, port = sock.getsockname()
        sock.listen(1)
        try:
            assert _socket_bind_check(port) is True
        finally:
            sock.close()


class TestFindPidOnPort:
    """Tests for _find_pid_on_port."""

    @patch("mlx_stack.core.process.psutil.net_connections")
    def test_finds_pid(self, mock_conns: MagicMock) -> None:
        from mlx_stack.core.process import _find_pid_on_port

        mock_conn = MagicMock()
        mock_conn.laddr = MagicMock()
        mock_conn.laddr.port = 8000
        mock_conn.status = "LISTEN"
        mock_conn.pid = 42
        mock_conns.return_value = [mock_conn]

        with patch("mlx_stack.core.process.psutil.Process") as mock_proc:
            mock_proc.return_value.name.return_value = "python"
            result = _find_pid_on_port(8000)
            assert result == (42, "python")

    @patch("mlx_stack.core.process.psutil.net_connections")
    def test_no_match(self, mock_conns: MagicMock) -> None:
        from mlx_stack.core.process import _find_pid_on_port

        mock_conns.return_value = []
        assert _find_pid_on_port(8000) is None

    @patch("mlx_stack.core.process.psutil.net_connections", side_effect=psutil.AccessDenied(0))
    def test_access_denied(self, mock_conns: MagicMock) -> None:
        from mlx_stack.core.process import _find_pid_on_port

        assert _find_pid_on_port(8000) is None


class TestCheckPortConflict:
    """Tests for check_port_conflict."""

    @patch("mlx_stack.core.process._find_pid_on_port", return_value=(42, "python"))
    @patch("mlx_stack.core.process._socket_bind_check", return_value=True)
    def test_port_in_use_with_owner(
        self, mock_bind: MagicMock, mock_find: MagicMock,
    ) -> None:
        """Port occupied, owner identified via psutil."""
        result = check_port_conflict(8000)
        assert result is not None
        assert result == (42, "python")

    @patch("mlx_stack.core.process._socket_bind_check", return_value=False)
    def test_port_available(self, mock_bind: MagicMock) -> None:
        """Port available via socket bind."""
        result = check_port_conflict(8000)
        assert result is None

    @patch("mlx_stack.core.process._find_pid_on_port", return_value=None)
    @patch("mlx_stack.core.process._socket_bind_check", return_value=True)
    def test_port_in_use_unknown_owner(
        self, mock_bind: MagicMock, mock_find: MagicMock,
    ) -> None:
        """Port occupied but owner can't be identified (e.g., macOS permission)."""
        result = check_port_conflict(8000)
        assert result is not None
        assert result == (0, "<unknown>")

    @patch("mlx_stack.core.process._find_pid_on_port", return_value=(42, "<unknown>"))
    @patch("mlx_stack.core.process._socket_bind_check", return_value=True)
    def test_process_vanished(
        self, mock_bind: MagicMock, mock_find: MagicMock,
    ) -> None:
        """Process vanished between detection and lookup."""
        result = check_port_conflict(8000)
        assert result is not None
        assert result == (42, "<unknown>")


class TestDetectPortConflict:
    """Tests for detect_port_conflict."""

    @patch("mlx_stack.core.process.check_port_conflict", return_value=None)
    def test_no_conflict(self, mock_check: MagicMock) -> None:
        detect_port_conflict(8000)  # Should not raise

    @patch("mlx_stack.core.process.check_port_conflict", return_value=(42, "python"))
    def test_conflict_raises(self, mock_check: MagicMock) -> None:
        with pytest.raises(PortConflictError) as exc_info:
            detect_port_conflict(8000)
        assert exc_info.value.port == 8000
        assert exc_info.value.pid == 42
        assert exc_info.value.name == "python"


class TestPortConflictError:
    """Tests for PortConflictError attributes."""

    def test_with_all_info(self) -> None:
        err = PortConflictError(port=8000, pid=42, name="python")
        assert "Port 8000" in str(err)
        assert "PID 42" in str(err)
        assert "python" in str(err)
        assert err.port == 8000
        assert err.pid == 42
        assert err.name == "python"

    def test_with_no_details(self) -> None:
        err = PortConflictError(port=8000)
        assert "Port 8000" in str(err)
        assert err.pid is None
        assert err.name is None


# =========================================================================== #
# Subprocess management
# =========================================================================== #


class TestStartService:
    """Tests for start_service."""

    @patch("mlx_stack.core.process.subprocess.Popen")
    def test_starts_process(self, mock_popen: MagicMock, mlx_stack_home: Path) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        info = start_service(
            "fast",
            cmd=["vllm-mlx", "--port", "8000"],
            port=8000,
        )

        assert info.name == "fast"
        assert info.pid == 12345
        assert info.port == 8000
        assert info.log_path is not None
        assert info.log_path.name == "fast.log"
        assert info.pid_path is not None

        # PID file should be written
        pid = read_pid_file("fast")
        assert pid == 12345

    @patch("mlx_stack.core.process.subprocess.Popen")
    def test_log_file_created(self, mock_popen: MagicMock, mlx_stack_home: Path) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 100
        mock_popen.return_value = mock_proc

        info = start_service("test-svc", cmd=["test"], port=9000)
        assert info.log_path is not None
        assert info.log_path.exists()

    @patch("mlx_stack.core.process.subprocess.Popen")
    def test_custom_env(self, mock_popen: MagicMock, mlx_stack_home: Path) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 100
        mock_popen.return_value = mock_proc

        start_service(
            "svc",
            cmd=["test"],
            port=9000,
            env={"OPENROUTER_API_KEY": "sk-test"},
        )

        call_kwargs = mock_popen.call_args[1]
        assert "OPENROUTER_API_KEY" in call_kwargs["env"]
        assert call_kwargs["env"]["OPENROUTER_API_KEY"] == "sk-test"

    @patch("mlx_stack.core.process.subprocess.Popen")
    def test_start_new_session(self, mock_popen: MagicMock, mlx_stack_home: Path) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 100
        mock_popen.return_value = mock_proc

        start_service("svc", cmd=["test"], port=9000)
        call_kwargs = mock_popen.call_args[1]
        assert call_kwargs["start_new_session"] is True

    @patch("mlx_stack.core.process.subprocess.Popen", side_effect=OSError("not found"))
    def test_popen_failure(self, mock_popen: MagicMock, mlx_stack_home: Path) -> None:
        with pytest.raises(ProcessError, match="Could not start service"):
            start_service("bad", cmd=["nonexistent"], port=9000)

    @patch("mlx_stack.core.process.subprocess.Popen")
    def test_custom_log_dir(self, mock_popen: MagicMock, mlx_stack_home: Path) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 100
        mock_popen.return_value = mock_proc

        custom_log_dir = mlx_stack_home / "custom-logs"
        info = start_service("svc", cmd=["test"], port=9000, log_dir=custom_log_dir)
        assert info.log_path is not None
        assert info.log_path.parent == custom_log_dir
        assert custom_log_dir.exists()


# =========================================================================== #
# Service shutdown
# =========================================================================== #


class TestStopService:
    """Tests for stop_service."""

    @patch("mlx_stack.core.process._terminate_process", return_value=(True, True))
    @patch("mlx_stack.core.process.is_process_alive", return_value=True)
    def test_graceful_shutdown(
        self,
        mock_alive: MagicMock,
        mock_terminate: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        write_pid_file("fast", 12345)
        result = stop_service("fast")
        assert result is not None
        assert result.graceful is True
        assert result.pid == 12345
        assert result.name == "fast"
        # PID file should be removed (termination confirmed)
        assert read_pid_file("fast") is None

    @patch("mlx_stack.core.process._terminate_process", return_value=(False, True))
    @patch("mlx_stack.core.process.is_process_alive", return_value=True)
    def test_forced_shutdown(
        self,
        mock_alive: MagicMock,
        mock_terminate: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        write_pid_file("fast", 12345)
        result = stop_service("fast")
        assert result is not None
        assert result.graceful is False

    def test_no_pid_file(self, mlx_stack_home: Path) -> None:
        result = stop_service("nonexistent")
        assert result is None

    def test_stale_pid_cleanup(self, mlx_stack_home: Path) -> None:
        write_pid_file("dead", 999999999)
        result = stop_service("dead")
        assert result is None  # Already dead, nothing to stop
        assert read_pid_file("dead") is None  # PID file cleaned

    def test_corrupt_pid_file(self, mlx_stack_home: Path) -> None:
        (mlx_stack_home / "pids").mkdir(parents=True, exist_ok=True)
        (mlx_stack_home / "pids" / "bad.pid").write_text("garbage")
        result = stop_service("bad")
        assert result is None
        # Corrupt file should be removed
        assert not (mlx_stack_home / "pids" / "bad.pid").exists()


class TestTerminateProcess:
    """Tests for _terminate_process."""

    @patch("mlx_stack.core.process.time.sleep")
    @patch("mlx_stack.core.process.is_process_alive")
    @patch("mlx_stack.core.process.os.kill")
    def test_graceful_sigterm(
        self,
        mock_kill: MagicMock,
        mock_alive: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        from mlx_stack.core.process import _terminate_process

        # Process dies after first alive check
        mock_alive.return_value = False
        graceful, confirmed = _terminate_process(123, grace_period=10)
        assert graceful is True
        assert confirmed is True
        mock_kill.assert_called_once_with(123, signal.SIGTERM)

    @patch("mlx_stack.core.process.time.sleep")
    @patch("mlx_stack.core.process.time.monotonic")
    @patch("mlx_stack.core.process.is_process_alive")
    @patch("mlx_stack.core.process.os.kill")
    def test_escalate_to_sigkill(
        self,
        mock_kill: MagicMock,
        mock_alive: MagicMock,
        mock_monotonic: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        from mlx_stack.core.process import _terminate_process

        # Process stays alive through grace period, then dies after SIGKILL
        mock_alive.side_effect = [True, True, True, False]
        mock_monotonic.side_effect = [
            0.0,   # deadline = 10.0
            5.0,   # still within grace
            11.0,  # past grace → SIGKILL
        ]

        graceful, confirmed = _terminate_process(123, grace_period=10)
        assert graceful is False
        assert confirmed is True
        # Should have been called with SIGTERM then SIGKILL
        assert mock_kill.call_count == 2
        mock_kill.assert_any_call(123, signal.SIGTERM)
        mock_kill.assert_any_call(123, signal.SIGKILL)

    @patch("mlx_stack.core.process.os.kill", side_effect=OSError("no such process"))
    def test_already_dead_on_sigterm(self, mock_kill: MagicMock) -> None:
        from mlx_stack.core.process import _terminate_process

        graceful, confirmed = _terminate_process(123, grace_period=10)
        assert graceful is True
        assert confirmed is True


class TestShutdownResult:
    """Tests for ShutdownResult."""

    def test_repr_graceful(self) -> None:
        r = ShutdownResult(name="fast", pid=123, graceful=True)
        assert "graceful" in repr(r)
        assert "fast" in repr(r)

    def test_repr_forced(self) -> None:
        r = ShutdownResult(name="fast", pid=123, graceful=False)
        assert "forced" in repr(r)


# =========================================================================== #
# Service status
# =========================================================================== #


class TestGetServiceStatus:
    """Tests for get_service_status."""

    def test_stopped_no_pid_file(self, mlx_stack_home: Path) -> None:
        result = get_service_status("nonexistent", port=8000)
        assert result["status"] == "stopped"
        assert result["pid"] is None
        assert result["uptime"] is None

    @patch("mlx_stack.core.process.http_health_check")
    @patch("mlx_stack.core.process.is_process_alive", return_value=True)
    def test_healthy(
        self,
        mock_alive: MagicMock,
        mock_check: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        write_pid_file("fast", 12345)
        mock_check.return_value = HealthCheckResult(
            healthy=True, response_time=0.5, status_code=200
        )

        result = get_service_status("fast", port=8000)
        assert result["status"] == "healthy"
        assert result["pid"] == 12345

    @patch("mlx_stack.core.process.http_health_check")
    @patch("mlx_stack.core.process.is_process_alive", return_value=True)
    def test_degraded(
        self,
        mock_alive: MagicMock,
        mock_check: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        write_pid_file("slow", 12345)
        mock_check.return_value = HealthCheckResult(
            healthy=True, response_time=3.5, status_code=200
        )

        result = get_service_status("slow", port=8000)
        assert result["status"] == "degraded"

    @patch("mlx_stack.core.process.http_health_check")
    @patch("mlx_stack.core.process.is_process_alive", return_value=True)
    def test_down_no_response(
        self,
        mock_alive: MagicMock,
        mock_check: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        write_pid_file("unreachable", 12345)
        mock_check.return_value = HealthCheckResult(
            healthy=False, response_time=None, status_code=None
        )

        result = get_service_status("unreachable", port=8000)
        assert result["status"] == "down"

    def test_crashed_dead_process(self, mlx_stack_home: Path) -> None:
        write_pid_file("crashed", 999999999)
        result = get_service_status("crashed", port=8000)
        assert result["status"] == "crashed"
        assert result["pid"] == 999999999

    def test_corrupt_pid_file(self, mlx_stack_home: Path) -> None:
        (mlx_stack_home / "pids").mkdir(parents=True, exist_ok=True)
        (mlx_stack_home / "pids" / "bad.pid").write_text("garbage")
        result = get_service_status("bad", port=8000)
        assert result["status"] == "crashed"

    @patch("mlx_stack.core.process.http_health_check")
    @patch("mlx_stack.core.process.is_process_alive", return_value=True)
    def test_includes_uptime(
        self,
        mock_alive: MagicMock,
        mock_check: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        write_pid_file("timed", 12345)
        mock_check.return_value = HealthCheckResult(
            healthy=True, response_time=0.1, status_code=200
        )

        result = get_service_status("timed", port=8000)
        assert result["uptime"] is not None
        assert result["uptime"] >= 0

    @patch("mlx_stack.core.process.http_health_check")
    @patch("mlx_stack.core.process.is_process_alive", return_value=True)
    def test_includes_response_time(
        self,
        mock_alive: MagicMock,
        mock_check: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        write_pid_file("fast", 12345)
        mock_check.return_value = HealthCheckResult(
            healthy=True, response_time=0.42, status_code=200
        )

        result = get_service_status("fast", port=8000)
        assert result["response_time"] == 0.42


# =========================================================================== #
# Uptime formatting
# =========================================================================== #


class TestFormatUptime:
    """Tests for format_uptime."""

    def test_none(self) -> None:
        assert format_uptime(None) == "-"

    def test_seconds(self) -> None:
        assert format_uptime(45) == "45s"

    def test_minutes(self) -> None:
        assert format_uptime(120) == "2m"

    def test_minutes_and_seconds(self) -> None:
        assert format_uptime(150) == "2m 30s"

    def test_hours(self) -> None:
        assert format_uptime(7200) == "2h"

    def test_hours_and_minutes(self) -> None:
        assert format_uptime(8100) == "2h 15m"

    def test_days(self) -> None:
        assert format_uptime(172800) == "2d"

    def test_days_and_hours(self) -> None:
        assert format_uptime(176400) == "2d 1h"

    def test_zero_seconds(self) -> None:
        assert format_uptime(0) == "0s"

    def test_just_under_a_minute(self) -> None:
        assert format_uptime(59) == "59s"

    def test_exactly_one_minute(self) -> None:
        assert format_uptime(60) == "1m"


# =========================================================================== #
# Constants and data classes
# =========================================================================== #


class TestConstants:
    """Tests for module constants."""

    def test_shutdown_grace_period(self) -> None:
        assert SHUTDOWN_GRACE_PERIOD == 10

    def test_health_check_timeout(self) -> None:
        assert HEALTH_CHECK_TIMEOUT == 120

    def test_status_check_timeout(self) -> None:
        assert STATUS_CHECK_TIMEOUT == 5.0

    def test_status_degraded_threshold(self) -> None:
        assert STATUS_DEGRADED_THRESHOLD == 2.0


class TestServiceInfo:
    """Tests for ServiceInfo data class."""

    def test_creation(self) -> None:
        info = ServiceInfo(
            name="fast",
            pid=123,
            port=8000,
            log_path=Path("/tmp/fast.log"),
            pid_path=Path("/tmp/fast.pid"),
        )
        assert info.name == "fast"
        assert info.pid == 123
        assert info.port == 8000


class TestHealthCheckResult:
    """Tests for HealthCheckResult data class."""

    def test_healthy(self) -> None:
        r = HealthCheckResult(healthy=True, response_time=0.5, status_code=200)
        assert r.healthy is True
        assert r.response_time == 0.5
        assert r.status_code == 200

    def test_unhealthy(self) -> None:
        r = HealthCheckResult(healthy=False, response_time=None, status_code=None)
        assert r.healthy is False
        assert r.response_time is None
