"""Process management module for mlx-stack.

Handles starting, stopping, and health-checking of vllm-mlx and LiteLLM
subprocesses. Manages PID files in ~/.mlx-stack/pids/, log file redirection
to ~/.mlx-stack/logs/, HTTP health checks with exponential backoff,
lockfile (fcntl.flock) for concurrent invocation prevention,
SIGTERM/SIGKILL shutdown with grace period, stale PID detection and cleanup,
and port conflict detection.

This is the infrastructure module used by up, down, and status commands.
"""

from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import httpx
import psutil

from mlx_stack.core.paths import (
    ensure_data_home,
    get_lock_path,
    get_logs_dir,
    get_pids_dir,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Default grace period (seconds) for SIGTERM before SIGKILL
SHUTDOWN_GRACE_PERIOD = 10

# Health check defaults
HEALTH_CHECK_TIMEOUT = 120  # total timeout in seconds
HEALTH_CHECK_INITIAL_DELAY = 0.5  # initial retry delay in seconds
HEALTH_CHECK_MAX_DELAY = 10.0  # maximum retry delay in seconds
HEALTH_CHECK_BACKOFF_FACTOR = 2.0  # exponential backoff multiplier

# Status health check timeout for the status command
STATUS_CHECK_TIMEOUT = 5.0  # per-service HTTP timeout in seconds
STATUS_DEGRADED_THRESHOLD = 2.0  # response time > 2s = degraded

# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class ProcessError(Exception):
    """Raised when a process management operation fails."""


class LockError(ProcessError):
    """Raised when the lockfile cannot be acquired."""


class HealthCheckError(ProcessError):
    """Raised when a health check fails after all retries."""


class PortConflictError(ProcessError):
    """Raised when a port is already in use by another process."""

    def __init__(self, port: int, pid: int | None = None, name: str | None = None) -> None:
        self.port = port
        self.pid = pid
        self.name = name
        parts = [f"Port {port} is already in use"]
        if pid is not None:
            parts.append(f"by PID {pid}")
        if name is not None:
            parts.append(f"({name})")
        super().__init__(" ".join(parts))


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ServiceInfo:
    """Information about a managed service."""

    name: str
    pid: int | None
    port: int
    log_path: Path | None
    pid_path: Path | None


class ShutdownResult:
    """Result of shutting down a single service."""

    def __init__(self, name: str, pid: int, graceful: bool) -> None:
        self.name = name
        self.pid = pid
        self.graceful = graceful

    def __repr__(self) -> str:
        method = "graceful" if self.graceful else "forced"
        return f"ShutdownResult(name={self.name!r}, pid={self.pid}, method={method!r})"


@dataclass(frozen=True)
class HealthCheckResult:
    """Result of a health check against a service."""

    healthy: bool
    response_time: float | None  # seconds, None if no response
    status_code: int | None


# --------------------------------------------------------------------------- #
# PID file management
# --------------------------------------------------------------------------- #


def _ensure_pids_dir() -> Path:
    """Ensure the PID directory exists and return its path."""
    pids_dir = get_pids_dir()
    pids_dir.mkdir(parents=True, exist_ok=True)
    return pids_dir


def _ensure_logs_dir() -> Path:
    """Ensure the logs directory exists and return its path."""
    logs_dir = get_logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def write_pid_file(service_name: str, pid: int) -> Path:
    """Write a PID file for a service.

    The PID file contains exactly the integer PID with no trailing
    whitespace or newline.

    Args:
        service_name: Name of the service (e.g. 'fast', 'litellm').
        pid: The process ID to write.

    Returns:
        Path to the created PID file.

    Raises:
        ProcessError: If the PID file cannot be written.
    """
    pids_dir = _ensure_pids_dir()
    pid_path = pids_dir / f"{service_name}.pid"
    try:
        pid_path.write_text(str(pid))
    except OSError as exc:
        msg = f"Could not write PID file for '{service_name}': {exc}"
        raise ProcessError(msg) from None
    return pid_path


def read_pid_file(service_name: str) -> int | None:
    """Read a PID from a service's PID file.

    Args:
        service_name: Name of the service.

    Returns:
        The PID as an integer, or None if the file doesn't exist.

    Raises:
        ProcessError: If the PID file exists but contains non-numeric content.
    """
    pid_path = get_pids_dir() / f"{service_name}.pid"
    if not pid_path.exists():
        return None

    content = pid_path.read_text().strip()
    if not content:
        return None

    try:
        return int(content)
    except ValueError:
        msg = (
            f"PID file for '{service_name}' contains non-numeric content: "
            f"{content!r}"
        )
        raise ProcessError(msg) from None


def remove_pid_file(service_name: str) -> bool:
    """Remove a service's PID file.

    Args:
        service_name: Name of the service.

    Returns:
        True if the file was removed, False if it didn't exist.
    """
    pid_path = get_pids_dir() / f"{service_name}.pid"
    if pid_path.exists():
        try:
            pid_path.unlink()
        except OSError:
            pass  # Best-effort removal
        return True
    return False


def list_pid_files() -> dict[str, Path]:
    """List all PID files in the pids directory.

    Returns:
        A dict mapping service name to PID file path.
    """
    pids_dir = get_pids_dir()
    if not pids_dir.exists():
        return {}

    result: dict[str, Path] = {}
    for pid_file in pids_dir.glob("*.pid"):
        service_name = pid_file.stem
        result[service_name] = pid_file
    return result


# --------------------------------------------------------------------------- #
# Process state checks
# --------------------------------------------------------------------------- #


def is_process_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running.

    Args:
        pid: The process ID to check.

    Returns:
        True if the process is alive, False otherwise.
    """
    try:
        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def is_stale_pid(service_name: str) -> bool:
    """Check if a service has a stale PID file (file exists, process dead).

    Args:
        service_name: Name of the service.

    Returns:
        True if the PID file exists but the process is not running.
    """
    try:
        pid = read_pid_file(service_name)
    except ProcessError:
        # Corrupt PID file — treat as stale
        return True

    if pid is None:
        return False  # No PID file at all

    return not is_process_alive(pid)


def cleanup_stale_pid(service_name: str) -> bool:
    """Clean up a stale PID file if the process is dead.

    Args:
        service_name: Name of the service.

    Returns:
        True if a stale PID was cleaned up, False if process is alive
        or no PID file exists.
    """
    if is_stale_pid(service_name):
        remove_pid_file(service_name)
        return True
    return False


# --------------------------------------------------------------------------- #
# Lockfile management
# --------------------------------------------------------------------------- #


@contextmanager
def acquire_lock() -> Iterator[None]:
    """Acquire an exclusive lock to prevent concurrent operations.

    Uses ``fcntl.flock`` on ``~/.mlx-stack/lock``. The lock is released
    when the context manager exits (or on process termination via OS-level
    FD cleanup).

    Yields:
        None when the lock is acquired.

    Raises:
        LockError: If the lock is already held by another process.
    """
    ensure_data_home()
    lock_path = get_lock_path()

    # Open in write mode, creating if it doesn't exist
    fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        msg = (
            "Another mlx-stack operation is already running. "
            "Wait for it to finish or remove the lock file: "
            f"{lock_path}"
        )
        raise LockError(msg) from None

    try:
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


# --------------------------------------------------------------------------- #
# Health checks
# --------------------------------------------------------------------------- #


def http_health_check(
    port: int,
    path: str = "/v1/models",
    timeout: float = STATUS_CHECK_TIMEOUT,
    host: str = "127.0.0.1",
) -> HealthCheckResult:
    """Perform a single HTTP health check against a service.

    Args:
        port: The port to check.
        path: The HTTP path to request.
        timeout: Request timeout in seconds.
        host: The host to connect to.

    Returns:
        A HealthCheckResult with the outcome.
    """
    url = f"http://{host}:{port}{path}"
    try:
        start = time.monotonic()
        response = httpx.get(url, timeout=timeout)
        elapsed = time.monotonic() - start
        return HealthCheckResult(
            healthy=response.status_code == 200,
            response_time=elapsed,
            status_code=response.status_code,
        )
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError, OSError):
        return HealthCheckResult(
            healthy=False,
            response_time=None,
            status_code=None,
        )


def wait_for_healthy(
    port: int,
    path: str = "/v1/models",
    total_timeout: float = HEALTH_CHECK_TIMEOUT,
    initial_delay: float = HEALTH_CHECK_INITIAL_DELAY,
    max_delay: float = HEALTH_CHECK_MAX_DELAY,
    backoff_factor: float = HEALTH_CHECK_BACKOFF_FACTOR,
    host: str = "127.0.0.1",
) -> HealthCheckResult:
    """Wait for a service to become healthy with exponential backoff.

    Polls the service's health endpoint repeatedly with increasing delay
    between attempts until either the service responds with HTTP 200 or
    the total timeout is exceeded.

    Args:
        port: The port to check.
        path: The HTTP path to request.
        total_timeout: Maximum total time to wait in seconds.
        initial_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        backoff_factor: Multiplier for exponential backoff.
        host: The host to connect to.

    Returns:
        A HealthCheckResult from the final check.

    Raises:
        HealthCheckError: If the service does not become healthy
            within the total timeout.
    """
    deadline = time.monotonic() + total_timeout
    delay = initial_delay
    last_result: HealthCheckResult | None = None

    while time.monotonic() < deadline:
        per_request_timeout = min(5.0, deadline - time.monotonic())
        if per_request_timeout <= 0:
            break

        result = http_health_check(
            port=port,
            path=path,
            timeout=per_request_timeout,
            host=host,
        )

        if result.healthy:
            return result

        last_result = result

        # Wait before next retry, respecting the deadline
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        sleep_time = min(delay, remaining)
        time.sleep(sleep_time)
        delay = min(delay * backoff_factor, max_delay)

    # Timed out
    if last_result is None:
        last_result = HealthCheckResult(healthy=False, response_time=None, status_code=None)

    msg = (
        f"Health check timed out after {total_timeout}s waiting for "
        f"http://{host}:{port}{path}"
    )
    raise HealthCheckError(msg)


# --------------------------------------------------------------------------- #
# Port conflict detection
# --------------------------------------------------------------------------- #


def _socket_bind_check(port: int) -> bool:
    """Check if a port is available by attempting a socket bind.

    This is more reliable than psutil.net_connections on macOS where
    the latter can fail with AccessDenied.

    Args:
        port: The TCP port to check.

    Returns:
        True if the port is in use (bind failed), False if available.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        sock.bind(("127.0.0.1", port))
        return False  # Port is available
    except OSError:
        return True  # Port is in use
    finally:
        sock.close()


def _find_pid_on_port(port: int) -> tuple[int, str] | None:
    """Find the PID and process name listening on a port via psutil.

    Args:
        port: The TCP port to look up.

    Returns:
        A tuple of (pid, process_name) if found, or None.
    """
    try:
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr and conn.laddr.port == port and conn.status == "LISTEN":
                pid = conn.pid
                if pid is not None:
                    try:
                        proc = psutil.Process(pid)
                        return (pid, proc.name())
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        return (pid, "<unknown>")
    except (psutil.AccessDenied, OSError):
        pass
    return None


def check_port_conflict(port: int) -> tuple[int, str] | None:
    """Check if a port is in use and identify the owning process.

    Uses a two-phase approach for reliability:
    1. Attempt a socket bind to definitively check port availability.
    2. If the port is in use, look up the owning PID/process via psutil.

    This ensures detection works even when psutil.net_connections is
    restricted by macOS permissions.

    Args:
        port: The TCP port to check.

    Returns:
        A tuple of (pid, process_name) if the port is in use,
        or None if the port is available.
    """
    # Phase 1: Socket bind check (most reliable)
    if not _socket_bind_check(port):
        return None  # Port is available

    # Phase 2: Port is in use — try to identify the owner
    owner = _find_pid_on_port(port)
    if owner is not None:
        return owner

    # Port is occupied but we can't identify the owner
    return (0, "<unknown>")



def detect_port_conflict(port: int) -> None:
    """Raise PortConflictError if a port is already in use.

    Args:
        port: The TCP port to check.

    Raises:
        PortConflictError: If the port is in use.
    """
    conflict = check_port_conflict(port)
    if conflict is not None:
        pid, name = conflict
        raise PortConflictError(port=port, pid=pid, name=name)


# --------------------------------------------------------------------------- #
# Subprocess management
# --------------------------------------------------------------------------- #


def start_service(
    service_name: str,
    cmd: list[str],
    port: int,
    env: dict[str, str] | None = None,
    log_dir: Path | None = None,
) -> ServiceInfo:
    """Start a subprocess for a service with log redirection and PID tracking.

    Starts the process detached from the CLI's lifecycle. Stdout and stderr
    are redirected to ``<log_dir>/<service_name>.log``.

    Args:
        service_name: Name for the service (used for PID/log files).
        cmd: The command to execute as a list of strings.
        port: The port the service will listen on.
        env: Optional environment variables for the subprocess.
            Merged with the current environment.
        log_dir: Directory for log files. Defaults to ``~/.mlx-stack/logs/``.

    Returns:
        A ServiceInfo with the started service's details.

    Raises:
        ProcessError: If the service cannot be started.
        PortConflictError: If the port is already in use.
    """
    # Ensure directories exist
    if log_dir is None:
        log_dir = _ensure_logs_dir()
    else:
        log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{service_name}.log"

    # Build environment
    process_env: dict[str, str] = dict(os.environ)
    if env:
        process_env.update(env)

    try:
        log_file = open(log_path, "w")  # noqa: SIM115
    except OSError as exc:
        msg = f"Could not open log file for '{service_name}': {exc}"
        raise ProcessError(msg) from None

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            env=process_env,
            start_new_session=True,
        )
    except OSError as exc:
        log_file.close()
        msg = f"Could not start service '{service_name}': {exc}"
        raise ProcessError(msg) from None

    # Write PID file — if this fails, kill the spawned process to prevent
    # leaked unmanaged subprocesses (scrutiny fix: orphan prevention).
    try:
        pid_path = write_pid_file(service_name, proc.pid)
    except ProcessError:
        # Kill the orphaned process before re-raising
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        log_file.close()
        msg = (
            f"Could not write PID file for '{service_name}' after "
            f"starting process (PID {proc.pid}). "
            f"The process has been terminated to prevent orphans."
        )
        raise ProcessError(msg) from None

    # Detach: close the log file handle in the parent process
    # The child process has its own file descriptors
    log_file.close()

    return ServiceInfo(
        name=service_name,
        pid=proc.pid,
        port=port,
        log_path=log_path,
        pid_path=pid_path,
    )


# --------------------------------------------------------------------------- #
# Service shutdown
# --------------------------------------------------------------------------- #


def stop_service(
    service_name: str,
    grace_period: float = SHUTDOWN_GRACE_PERIOD,
) -> ShutdownResult | None:
    """Stop a managed service by its PID file.

    Sends SIGTERM first with a grace period. If the process hasn't exited
    after the grace period, sends SIGKILL. Only removes the PID file once
    process termination is confirmed (scrutiny fix: verified termination).

    Args:
        service_name: Name of the service to stop.
        grace_period: Seconds to wait after SIGTERM before SIGKILL.

    Returns:
        A ShutdownResult if a process was stopped, or None if no process
        was found (PID file missing or process already dead).

    Raises:
        ProcessError: If the PID file is corrupt (non-numeric content).
    """
    try:
        pid = read_pid_file(service_name)
    except ProcessError:
        # Corrupt PID file — remove it and report
        remove_pid_file(service_name)
        return None

    if pid is None:
        return None

    if not is_process_alive(pid):
        # Stale PID — clean up
        remove_pid_file(service_name)
        return None

    # Send SIGTERM, escalate to SIGKILL if needed
    graceful, confirmed = _terminate_process(pid, grace_period)

    # Only remove PID file once termination is confirmed
    if confirmed:
        remove_pid_file(service_name)

    return ShutdownResult(name=service_name, pid=pid, graceful=graceful)


def _terminate_process(pid: int, grace_period: float) -> tuple[bool, bool]:
    """Terminate a process with SIGTERM, escalating to SIGKILL if needed.

    Verifies process termination after SIGKILL before returning
    (scrutiny fix: confirmed termination).

    Args:
        pid: Process ID to terminate.
        grace_period: Seconds to wait after SIGTERM.

    Returns:
        A tuple of (graceful, confirmed):
        - graceful: True if SIGTERM was sufficient, False if SIGKILL used.
        - confirmed: True if the process is confirmed dead, False if it
          may still be running after SIGKILL.
    """
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        # Process may have already exited
        return True, True

    # Wait for process to exit
    deadline = time.monotonic() + grace_period
    while time.monotonic() < deadline:
        if not is_process_alive(pid):
            return True, True
        time.sleep(0.2)

    # Grace period expired — send SIGKILL
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        # Process may have exited between check and kill
        return True, True

    # Wait for SIGKILL to take effect — verify process is actually dead
    for _ in range(25):
        if not is_process_alive(pid):
            return False, True
        time.sleep(0.1)

    # Process is still alive after SIGKILL — not confirmed dead
    return False, False


# --------------------------------------------------------------------------- #
# Service status
# --------------------------------------------------------------------------- #


def get_service_status(
    service_name: str,
    port: int,
    health_path: str = "/v1/models",
) -> dict[str, Any]:
    """Get the current status of a managed service.

    Implements 5-state reporting:
    - healthy: PID alive and HTTP 200 within 2s
    - degraded: PID alive and HTTP 200 but response time > 2s
    - down: PID alive but no HTTP response within 5s
    - crashed: PID file exists but process is dead
    - stopped: No PID file

    Args:
        service_name: Name of the service.
        port: The port the service listens on.
        health_path: The HTTP path for health checks.

    Returns:
        A dict with keys: status, pid, uptime (seconds or None),
        response_time (seconds or None).
    """
    pid_path = get_pids_dir() / f"{service_name}.pid"

    # No PID file → stopped
    if not pid_path.exists():
        return {
            "status": "stopped",
            "pid": None,
            "uptime": None,
            "response_time": None,
        }

    # Read PID
    try:
        pid = read_pid_file(service_name)
    except ProcessError:
        # Corrupt PID file
        return {
            "status": "crashed",
            "pid": None,
            "uptime": None,
            "response_time": None,
        }

    if pid is None:
        return {
            "status": "stopped",
            "pid": None,
            "uptime": None,
            "response_time": None,
        }

    # PID file exists but process is dead → crashed
    if not is_process_alive(pid):
        return {
            "status": "crashed",
            "pid": pid,
            "uptime": None,
            "response_time": None,
        }

    # PID alive → check HTTP health
    uptime = _get_uptime_from_pid_file(service_name)
    result = http_health_check(
        port=port,
        path=health_path,
        timeout=STATUS_CHECK_TIMEOUT,
    )

    if result.healthy and result.response_time is not None:
        if result.response_time <= STATUS_DEGRADED_THRESHOLD:
            status = "healthy"
        else:
            status = "degraded"
    else:
        status = "down"

    return {
        "status": status,
        "pid": pid,
        "uptime": uptime,
        "response_time": result.response_time,
    }


def _get_uptime_from_pid_file(service_name: str) -> float | None:
    """Calculate uptime from the PID file's modification timestamp.

    Args:
        service_name: Name of the service.

    Returns:
        Uptime in seconds, or None if the file doesn't exist.
    """
    pid_path = get_pids_dir() / f"{service_name}.pid"
    if not pid_path.exists():
        return None
    try:
        mtime = pid_path.stat().st_mtime
        return time.time() - mtime
    except OSError:
        return None


def format_uptime(seconds: float | None) -> str:
    """Format uptime seconds into a human-readable string.

    Args:
        seconds: Uptime in seconds, or None.

    Returns:
        Human-readable uptime string (e.g. '2h 15m') or '-'.
    """
    if seconds is None:
        return "-"

    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"

    minutes = seconds // 60
    if minutes < 60:
        remaining_s = seconds % 60
        if remaining_s > 0:
            return f"{minutes}m {remaining_s}s"
        return f"{minutes}m"

    hours = minutes // 60
    remaining_m = minutes % 60
    if hours < 24:
        if remaining_m > 0:
            return f"{hours}h {remaining_m}m"
        return f"{hours}h"

    days = hours // 24
    remaining_h = hours % 24
    if remaining_h > 0:
        return f"{days}d {remaining_h}h"
    return f"{days}d"
