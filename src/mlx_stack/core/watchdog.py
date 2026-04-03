"""Watchdog health monitor for mlx-stack.

Implements a long-running health monitor that:
- Polls service health at configurable intervals (default 30s)
- Auto-restarts crashed services (PID file exists, process dead)
- Does NOT restart stopped services (no PID file)
- Tracks flap detection with configurable max-restarts threshold
- Uses exponential backoff on restart delay
- Acquires lock only during restart operations
- Supports daemon mode (os.fork + os.setsid + PID file)
- Handles SIGTERM/SIGINT for clean shutdown
- Triggers log rotation each poll cycle
"""

from __future__ import annotations

import logging
import os
import signal
import time
from dataclasses import dataclass, field
from typing import Any

from mlx_stack.core.config import ConfigCorruptError, get_value
from mlx_stack.core.log_rotation import rotate_log
from mlx_stack.core.paths import get_data_home, get_logs_dir
from mlx_stack.core.process import (
    LockError,
    acquire_lock,
    is_process_alive,
    read_pid_file,
    remove_pid_file,
    start_service,
    write_pid_file,
)
from mlx_stack.core.stack_status import (
    ServiceHealth,
    ServiceStatus,
    run_status,
)
from mlx_stack.core.stack_up import (
    LITELLM_SERVICE_NAME,
    build_litellm_command,
    build_vllm_command,
    load_stack_definition,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

WATCHDOG_SERVICE_NAME = "watchdog"
WATCHDOG_PID_FILE = "watchdog.pid"

DEFAULT_INTERVAL = 30
DEFAULT_MAX_RESTARTS = 5
DEFAULT_RESTART_DELAY = 10


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class WatchdogError(Exception):
    """Raised when the watchdog encounters a fatal error."""


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class RestartRecord:
    """Record of a service restart event."""

    service: str
    timestamp: float
    reason: str
    attempt: int
    success: bool


@dataclass
class ServiceTracker:
    """Tracks restart history and flap state per service."""

    restart_timestamps: list[float] = field(default_factory=list)
    restart_count: int = 0
    is_flapping: bool = False
    last_restart_time: float = 0.0
    consecutive_failures: int = 0


@dataclass
class WatchdogState:
    """Internal state of the watchdog monitor."""

    shutdown_requested: bool = False
    service_trackers: dict[str, ServiceTracker] = field(default_factory=dict)
    restart_log: list[RestartRecord] = field(default_factory=list)
    is_daemon: bool = False
    cycle_count: int = 0


@dataclass
class PollResult:
    """Result of a single poll cycle."""

    statuses: list[ServiceStatus]
    restarts_attempted: int = 0
    restarts_succeeded: int = 0
    rotations_performed: int = 0
    flapping_services: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Stack definition helpers
# --------------------------------------------------------------------------- #


def _load_stack_for_watchdog(stack_name: str = "default") -> dict[str, Any]:
    """Load the stack definition for watchdog use.

    Args:
        stack_name: Stack definition name.

    Returns:
        The parsed stack definition dict.

    Raises:
        WatchdogError: If the stack cannot be loaded.
    """
    try:
        return load_stack_definition(stack_name)
    except Exception as exc:
        msg = f"No stack configuration found. Run 'mlx-stack init' first.\n{exc}"
        raise WatchdogError(msg) from None


def _get_tier_by_name(
    stack: dict[str, Any],
    service_name: str,
) -> dict[str, Any] | None:
    """Find a tier definition by service name.

    Args:
        stack: The stack definition dict.
        service_name: The service/tier name to look up.

    Returns:
        The tier dict if found, else None.
    """
    for tier in stack.get("tiers", []):
        if tier.get("name") == service_name:
            return tier
    return None


# --------------------------------------------------------------------------- #
# Flap detection
# --------------------------------------------------------------------------- #


def check_flapping(
    tracker: ServiceTracker,
    max_restarts: int,
    window_seconds: float = 600.0,
) -> bool:
    """Check if a service is flapping (restarting too frequently).

    A service is considered flapping if it has been restarted more than
    ``max_restarts`` times within the rolling window.

    Args:
        tracker: The service's restart tracker.
        max_restarts: Maximum restarts allowed in the window.
        window_seconds: The rolling window size in seconds (default 10 min).

    Returns:
        True if the service is flapping.
    """
    now = time.monotonic()
    cutoff = now - window_seconds

    # Prune old timestamps outside the window
    tracker.restart_timestamps = [ts for ts in tracker.restart_timestamps if ts > cutoff]

    if len(tracker.restart_timestamps) >= max_restarts:
        tracker.is_flapping = True
        return True

    return False


def reset_flap_state(
    tracker: ServiceTracker,
    stable_period: float = 300.0,
) -> bool:
    """Reset flap state if the service has been stable.

    A service is considered stable if no restarts have occurred in the
    last ``stable_period`` seconds.

    Args:
        tracker: The service's restart tracker.
        stable_period: Time in seconds of no restarts to consider stable.

    Returns:
        True if the flap state was reset.
    """
    if not tracker.is_flapping:
        return False

    now = time.monotonic()
    if tracker.last_restart_time == 0.0:
        return False

    if (now - tracker.last_restart_time) >= stable_period:
        tracker.is_flapping = False
        tracker.restart_timestamps.clear()
        tracker.restart_count = 0
        tracker.consecutive_failures = 0
        return True

    return False


# --------------------------------------------------------------------------- #
# Restart delay with exponential backoff
# --------------------------------------------------------------------------- #


def calculate_restart_delay(
    base_delay: float,
    consecutive_failures: int,
    max_delay: float = 300.0,
) -> float:
    """Calculate the restart delay with exponential backoff.

    Args:
        base_delay: The base restart delay in seconds.
        consecutive_failures: Number of consecutive failures.
        max_delay: Maximum delay cap in seconds.

    Returns:
        The delay in seconds before the next restart attempt.
    """
    if consecutive_failures <= 0:
        return base_delay

    delay = base_delay * (2 ** (consecutive_failures - 1))
    return min(delay, max_delay)


# --------------------------------------------------------------------------- #
# Service restart
# --------------------------------------------------------------------------- #


def restart_service(
    service_name: str,
    stack: dict[str, Any],
    tracker: ServiceTracker,
    vllm_binary: str | None = None,
    litellm_binary: str | None = None,
) -> bool:
    """Restart a crashed service using the stack definition.

    Acquires the lock only during the restart operation and releases
    it immediately after.

    Args:
        service_name: Name of the service to restart.
        stack: The stack definition dict.
        tracker: The service's restart tracker.
        vllm_binary: Path to vllm-mlx binary (resolved if None).
        litellm_binary: Path to litellm binary (resolved if None).

    Returns:
        True if the restart succeeded, False otherwise.
    """
    import shutil

    # Resolve binaries
    if vllm_binary is None:
        vllm_binary = shutil.which("vllm-mlx") or "vllm-mlx"
    if litellm_binary is None:
        litellm_binary = shutil.which("litellm") or "litellm"

    # Acquire lock BEFORE removing PID file.  If the lock cannot be
    # obtained, leave the PID file intact so the service stays in
    # "crashed" state for the next poll cycle.
    try:
        with acquire_lock():
            # Clean up old PID file only after lock is held
            remove_pid_file(service_name)

            if service_name == LITELLM_SERVICE_NAME:
                return _restart_litellm(service_name, stack, litellm_binary)
            return _restart_tier(service_name, stack, vllm_binary)
    except LockError:
        logger.warning(
            "Could not acquire lock to restart '%s' — another operation is in progress.",
            service_name,
        )
        return False
    except Exception:
        logger.exception("Failed to restart service '%s'.", service_name)
        return False


def _restart_tier(
    service_name: str,
    stack: dict[str, Any],
    vllm_binary: str,
) -> bool:
    """Restart a vllm-mlx tier service.

    Args:
        service_name: The tier name.
        stack: The stack definition dict.
        vllm_binary: Path to vllm-mlx binary.

    Returns:
        True if the restart succeeded.
    """
    tier = _get_tier_by_name(stack, service_name)
    if tier is None:
        logger.error("Tier '%s' not found in stack definition.", service_name)
        return False

    cmd = build_vllm_command(tier, vllm_binary)
    port = tier["port"]

    try:
        start_service(
            service_name=service_name,
            cmd=cmd,
            port=port,
        )
        return True
    except Exception:
        logger.exception("Failed to start tier '%s'.", service_name)
        return False


def _restart_litellm(
    service_name: str,
    stack: dict[str, Any],
    litellm_binary: str,
) -> bool:
    """Restart the LiteLLM proxy service.

    Args:
        service_name: The service name (litellm).
        stack: The stack definition dict.
        litellm_binary: Path to litellm binary.

    Returns:
        True if the restart succeeded.
    """
    try:
        litellm_port = int(get_value("litellm-port"))
    except (ConfigCorruptError, ValueError):
        litellm_port = 4000

    litellm_config_path = get_data_home() / "litellm.yaml"

    cmd = build_litellm_command(litellm_binary, litellm_port, litellm_config_path)

    # Build env with OpenRouter key if configured
    env: dict[str, str] | None = None
    try:
        openrouter_key = str(get_value("openrouter-key"))
        if openrouter_key:
            env = {"OPENROUTER_API_KEY": openrouter_key}
    except (ConfigCorruptError, Exception):
        pass

    try:
        start_service(
            service_name=service_name,
            cmd=cmd,
            port=litellm_port,
            env=env,
        )
        return True
    except Exception:
        logger.exception("Failed to start LiteLLM.")
        return False


# --------------------------------------------------------------------------- #
# Log rotation during poll
# --------------------------------------------------------------------------- #


def rotate_service_logs() -> int:
    """Rotate all service log files that exceed the configured threshold.

    Returns:
        Number of files that were rotated.
    """
    try:
        max_size_mb = int(get_value("log-max-size-mb"))
    except (ConfigCorruptError, ValueError):
        max_size_mb = 50

    try:
        max_files = int(get_value("log-max-files"))
    except (ConfigCorruptError, ValueError):
        max_files = 5

    logs_dir = get_logs_dir()
    if not logs_dir.exists():
        return 0

    rotated_count = 0
    for log_path in logs_dir.iterdir():
        if log_path.suffix == ".log" and log_path.is_file():
            try:
                if rotate_log(log_path, max_size_mb=max_size_mb, max_files=max_files):
                    rotated_count += 1
            except Exception:
                logger.exception("Error rotating log %s.", log_path)

    return rotated_count


# --------------------------------------------------------------------------- #
# Daemon mode
# --------------------------------------------------------------------------- #


def check_existing_watchdog() -> int | None:
    """Check if a watchdog is already running.

    Returns:
        The PID of the running watchdog, or None.
    """
    try:
        pid = read_pid_file(WATCHDOG_SERVICE_NAME)
    except Exception:
        return None

    if pid is None:
        return None

    if is_process_alive(pid):
        return pid

    # Stale PID file — clean up
    remove_pid_file(WATCHDOG_SERVICE_NAME)
    return None


def daemonize() -> None:
    """Fork the process to run as a daemon.

    Uses the double-fork technique with os.setsid() to fully detach
    from the terminal. Writes the daemon PID to the watchdog PID file.

    Raises:
        WatchdogError: If daemonization fails.
    """
    try:
        pid = os.fork()
        if pid > 0:
            # Parent exits
            os._exit(0)
    except OSError as exc:
        msg = f"First fork failed: {exc}"
        raise WatchdogError(msg) from None

    # Become session leader
    os.setsid()

    try:
        pid = os.fork()
        if pid > 0:
            # First child exits
            os._exit(0)
    except OSError as exc:
        msg = f"Second fork failed: {exc}"
        raise WatchdogError(msg) from None

    # Redirect stdin/stdout/stderr to /dev/null
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)

    # Write daemon PID
    write_pid_file(WATCHDOG_SERVICE_NAME, os.getpid())


def remove_watchdog_pid() -> None:
    """Remove the watchdog PID file."""
    remove_pid_file(WATCHDOG_SERVICE_NAME)


# --------------------------------------------------------------------------- #
# Signal handling
# --------------------------------------------------------------------------- #


def setup_signal_handlers(state: WatchdogState) -> None:
    """Register SIGTERM and SIGINT handlers for clean shutdown.

    Args:
        state: The watchdog state — sets shutdown_requested flag.
    """

    def handler(signum: int, frame: Any) -> None:
        state.shutdown_requested = True

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


# --------------------------------------------------------------------------- #
# Main poll cycle
# --------------------------------------------------------------------------- #


def poll_cycle(
    state: WatchdogState,
    stack: dict[str, Any],
    interval: int,
    max_restarts: int,
    restart_delay: int,
    vllm_binary: str | None = None,
    litellm_binary: str | None = None,
) -> PollResult:
    """Execute a single watchdog poll cycle.

    1. Get service status for all services.
    2. For each crashed service, attempt restart (with flap/backoff checks).
    3. Rotate logs.
    4. Return the poll result.

    Args:
        state: The watchdog state.
        stack: The stack definition.
        interval: Poll interval (for display).
        max_restarts: Maximum restarts before flap detection.
        restart_delay: Base restart delay in seconds.
        vllm_binary: Path to vllm-mlx binary.
        litellm_binary: Path to litellm binary.

    Returns:
        PollResult with status and action details.
    """
    state.cycle_count += 1

    # Get current status
    status_result = run_status()
    result = PollResult(statuses=status_result.services)

    if status_result.no_stack:
        return result

    # Check each service for crashed state
    for svc in status_result.services:
        service_name = svc.tier
        tracker = state.service_trackers.setdefault(service_name, ServiceTracker())

        # Try to reset flap state if service has been stable
        reset_flap_state(tracker)

        if svc.status != ServiceHealth.CRASHED:
            # Service is not crashed — reset consecutive failure counter
            # if it was previously restarted and is now healthy
            if svc.status == ServiceHealth.HEALTHY and tracker.consecutive_failures > 0:
                tracker.consecutive_failures = 0
            continue

        # Service is crashed — decide whether to restart
        if tracker.is_flapping:
            result.flapping_services.append(service_name)
            continue

        # Check flap detection before restarting
        if check_flapping(tracker, max_restarts):
            result.flapping_services.append(service_name)
            logger.warning(
                "Service '%s' marked as flapping after %d restarts. Stopping auto-restart.",
                service_name,
                max_restarts,
            )
            continue

        # Calculate restart delay with backoff
        delay = calculate_restart_delay(
            base_delay=float(restart_delay),
            consecutive_failures=tracker.consecutive_failures,
        )

        # Check if enough time has passed since last restart
        now = time.monotonic()
        if tracker.last_restart_time > 0:
            elapsed = now - tracker.last_restart_time
            if elapsed < delay:
                continue  # Not enough time elapsed; skip this cycle

        # Attempt restart
        result.restarts_attempted += 1
        tracker.restart_count += 1

        success = restart_service(
            service_name=service_name,
            stack=stack,
            tracker=tracker,
            vllm_binary=vllm_binary,
            litellm_binary=litellm_binary,
        )

        record = RestartRecord(
            service=service_name,
            timestamp=time.time(),
            reason="crashed (PID file exists, process dead)",
            attempt=tracker.restart_count,
            success=success,
        )
        state.restart_log.append(record)

        tracker.last_restart_time = time.monotonic()
        tracker.restart_timestamps.append(time.monotonic())

        if success:
            result.restarts_succeeded += 1
            tracker.consecutive_failures = 0
        else:
            tracker.consecutive_failures += 1

    # Rotate logs
    result.rotations_performed = rotate_service_logs()

    return result


# --------------------------------------------------------------------------- #
# Main watchdog loop
# --------------------------------------------------------------------------- #


def run_watchdog(
    interval: int = DEFAULT_INTERVAL,
    max_restarts: int = DEFAULT_MAX_RESTARTS,
    restart_delay: int = DEFAULT_RESTART_DELAY,
    daemon: bool = False,
    stack_name: str = "default",
    status_callback: Any = None,
    restart_callback: Any = None,
) -> WatchdogState:
    """Run the watchdog health monitor main loop.

    Args:
        interval: Seconds between health polls.
        max_restarts: Max restarts before marking as flapping.
        restart_delay: Base delay in seconds before restart.
        daemon: Whether to daemonize.
        stack_name: Stack definition name.
        status_callback: Called with (PollResult, WatchdogState) each cycle.
        restart_callback: Called with (RestartRecord) for each restart event.

    Returns:
        The final WatchdogState.

    Raises:
        WatchdogError: On fatal errors (no stack, already running, etc.).
    """
    # Check stack prerequisite
    stack = _load_stack_for_watchdog(stack_name)

    # Check for existing watchdog
    existing_pid = check_existing_watchdog()
    if existing_pid is not None:
        msg = (
            f"A watchdog is already running (PID {existing_pid}). "
            "Stop it before starting a new one."
        )
        raise WatchdogError(msg)

    state = WatchdogState(is_daemon=daemon)

    # Daemon mode
    if daemon:
        daemonize()

    # Set up signal handlers
    setup_signal_handlers(state)

    # Write PID for non-daemon mode too (for single-instance check)
    if not daemon:
        write_pid_file(WATCHDOG_SERVICE_NAME, os.getpid())

    # Resolve binaries
    import shutil

    vllm_binary = shutil.which("vllm-mlx") or "vllm-mlx"
    litellm_binary = shutil.which("litellm") or "litellm"

    try:
        while not state.shutdown_requested:
            result = poll_cycle(
                state=state,
                stack=stack,
                interval=interval,
                max_restarts=max_restarts,
                restart_delay=restart_delay,
                vllm_binary=vllm_binary,
                litellm_binary=litellm_binary,
            )

            # Callbacks
            if status_callback is not None:
                status_callback(result, state)

            if restart_callback is not None and result.restarts_attempted > 0:
                for record in state.restart_log[-result.restarts_attempted :]:
                    restart_callback(record)

            # Sleep in small increments so we can check shutdown flag
            sleep_end = time.monotonic() + interval
            while time.monotonic() < sleep_end:
                if state.shutdown_requested:
                    break
                time.sleep(min(0.5, sleep_end - time.monotonic()))

    finally:
        # Clean shutdown
        remove_watchdog_pid()

    return state
