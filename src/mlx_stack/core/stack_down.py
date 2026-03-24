"""Stack shutdown logic for mlx-stack.

Orchestrates stopping all managed services: terminates processes in
correct order (LiteLLM first, then model servers in reverse startup
order), SIGTERM with 10s grace period then SIGKILL, cleans up PID
files, acquires lockfile during operation. Supports --tier for
selective stop. Handles stale/corrupt PID files gracefully.
Reports 'Nothing to stop' when idle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlx_stack.core.catalog import get_entry_by_id, load_catalog
from mlx_stack.core.process import (
    LockError,
    ProcessError,
    acquire_lock,
    is_process_alive,
    list_pid_files,
    read_pid_file,
    remove_pid_file,
    stop_service,
)
from mlx_stack.core.stack_up import (
    LITELLM_SERVICE_NAME,
    load_stack_definition,
)

# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class DownError(Exception):
    """Raised when the down command encounters a fatal error."""


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class ServiceStopResult:
    """Result of stopping a single service."""

    name: str
    pid: int | None
    status: str  # "stopped", "stale", "corrupt", "not-running"
    graceful: bool | None = None  # None if not applicable
    error: str | None = None


@dataclass
class DownResult:
    """Result of the down command execution."""

    services: list[ServiceStopResult] = field(default_factory=list)
    nothing_to_stop: bool = False
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Tier ordering
# --------------------------------------------------------------------------- #


def _get_tier_names_from_stack(stack_name: str = "default") -> list[str]:
    """Get tier names from the stack definition in startup order.

    Returns tier names sorted by model size descending (largest first),
    matching the startup order used by ``mlx-stack up``.

    Args:
        stack_name: Stack definition name.

    Returns:
        List of tier names in startup order (largest model first).
        Empty list if the stack cannot be loaded.
    """
    try:
        stack = load_stack_definition(stack_name)
    except Exception:
        return []

    tiers = stack.get("tiers", [])

    # Try to sort by params_b from catalog (same as up command)
    try:
        catalog = load_catalog()
    except Exception:
        catalog = None

    if catalog is not None:
        def sort_key(tier: dict[str, Any]) -> tuple[float, str]:
            model_id = tier.get("model", "")
            entry = get_entry_by_id(catalog, model_id)
            params_b = entry.params_b if entry else 0.0
            return (-params_b, tier.get("name", ""))

        tiers = sorted(tiers, key=sort_key)

    return [t["name"] for t in tiers]


def _get_valid_tier_names(stack_name: str = "default") -> list[str]:
    """Get valid tier names from the stack definition.

    Args:
        stack_name: Stack definition name.

    Returns:
        List of tier names (unsorted). Empty if stack can't be loaded.
    """
    try:
        stack = load_stack_definition(stack_name)
    except Exception:
        return []

    tiers = stack.get("tiers", [])
    return [t["name"] for t in tiers]


# --------------------------------------------------------------------------- #
# Single service shutdown with stale/corrupt handling
# --------------------------------------------------------------------------- #


def _stop_single_service(service_name: str) -> ServiceStopResult:
    """Stop a single service, handling stale and corrupt PID files.

    Args:
        service_name: Name of the service to stop.

    Returns:
        A ServiceStopResult describing the outcome.
    """
    # Try to read the PID file
    try:
        pid = read_pid_file(service_name)
    except ProcessError:
        # Corrupt PID file — remove and report
        remove_pid_file(service_name)
        return ServiceStopResult(
            name=service_name,
            pid=None,
            status="corrupt",
            error="PID file contained non-numeric content; removed.",
        )

    if pid is None:
        # No PID file exists
        return ServiceStopResult(
            name=service_name,
            pid=None,
            status="not-running",
        )

    if not is_process_alive(pid):
        # Stale PID — process already dead
        remove_pid_file(service_name)
        return ServiceStopResult(
            name=service_name,
            pid=pid,
            status="stale",
            error=f"Process {pid} already dead; cleaned up stale PID file.",
        )

    # Process is alive — stop it
    shutdown = stop_service(service_name)

    if shutdown is not None:
        return ServiceStopResult(
            name=service_name,
            pid=shutdown.pid,
            status="stopped",
            graceful=shutdown.graceful,
        )

    # Shouldn't reach here, but handle gracefully
    return ServiceStopResult(
        name=service_name,
        pid=pid,
        status="stopped",
        graceful=True,
    )


# --------------------------------------------------------------------------- #
# Main shutdown orchestration
# --------------------------------------------------------------------------- #


def run_down(
    tier_filter: str | None = None,
    stack_name: str = "default",
) -> DownResult:
    """Execute the full stack shutdown flow.

    1. Acquire lockfile.
    2. Enumerate PID files to determine what's running.
    3. If --tier specified, stop only that tier.
    4. Otherwise stop LiteLLM first, then model servers in reverse
       startup order (smallest model first = reverse of largest-first).
    5. Clean up PID files.
    6. Release lockfile.

    Args:
        tier_filter: If set, stop only this tier.
        stack_name: Stack definition name.

    Returns:
        A DownResult with the outcome.

    Raises:
        DownError: On fatal errors (invalid tier filter).
        LockError: If the lockfile is held by another process.
    """
    result = DownResult()

    # --- Validate --tier filter ---
    if tier_filter is not None:
        valid_tiers = _get_valid_tier_names(stack_name)
        if valid_tiers and tier_filter not in valid_tiers:
            valid_list = ", ".join(sorted(valid_tiers))
            msg = (
                f"Unknown tier '{tier_filter}'. "
                f"Valid tiers: {valid_list}"
            )
            raise DownError(msg)

    # --- Check if anything is running ---
    pid_files = list_pid_files()
    if not pid_files:
        result.nothing_to_stop = True
        return result

    # --- Acquire lockfile and shut down ---
    try:
        with acquire_lock():
            return _run_shutdown(
                tier_filter=tier_filter,
                stack_name=stack_name,
                result=result,
            )
    except LockError:
        raise


def _run_shutdown(
    tier_filter: str | None,
    stack_name: str,
    result: DownResult,
) -> DownResult:
    """Execute the actual shutdown sequence under the lock.

    Args:
        tier_filter: If set, stop only this tier.
        stack_name: Stack definition name.
        result: The DownResult to populate.

    Returns:
        The populated DownResult.
    """
    pid_files = list_pid_files()

    if not pid_files:
        result.nothing_to_stop = True
        return result

    # --- Selective tier stop ---
    if tier_filter is not None:
        if tier_filter in pid_files:
            svc_result = _stop_single_service(tier_filter)
            result.services.append(svc_result)
        else:
            result.services.append(ServiceStopResult(
                name=tier_filter,
                pid=None,
                status="not-running",
            ))
        return result

    # --- Full shutdown: determine order ---
    # Get startup order from stack definition
    startup_order = _get_tier_names_from_stack(stack_name)

    # Reverse startup order for shutdown of model servers
    # (smallest first, since startup is largest first)
    shutdown_order_tiers = list(reversed(startup_order))

    # Collect all PID file service names
    all_services = set(pid_files.keys())

    # Step 1: Stop LiteLLM first (if running)
    if LITELLM_SERVICE_NAME in all_services:
        svc_result = _stop_single_service(LITELLM_SERVICE_NAME)
        result.services.append(svc_result)
        all_services.discard(LITELLM_SERVICE_NAME)

    # Step 2: Stop model servers in reverse startup order
    for tier_name in shutdown_order_tiers:
        if tier_name in all_services:
            svc_result = _stop_single_service(tier_name)
            result.services.append(svc_result)
            all_services.discard(tier_name)

    # Step 3: Stop any remaining services not in the stack definition
    # (orphaned PID files from previous stacks, etc.)
    for service_name in sorted(all_services):
        svc_result = _stop_single_service(service_name)
        result.services.append(svc_result)

    return result
