"""Stack status logic for mlx-stack.

Orchestrates health-checking all managed services and producing a
unified status report. Read-only: does not modify any files, clean up
PID files, restart services, or acquire the lockfile. Can run
concurrently with ``up`` or ``down``.

Implements 5-state reporting per service:
- healthy: PID alive and HTTP 200 within 2s
- degraded: PID alive and HTTP 200 but response time > 2s and <= 5s
- down: PID alive but no HTTP response within 5s
- crashed: PID file exists but process is dead
- stopped: No PID file
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlx_stack.core.paths import get_stacks_dir
from mlx_stack.core.process import (
    format_uptime,
    get_service_status,
)
from mlx_stack.core.stack_up import LITELLM_HEALTH_PATH, LITELLM_SERVICE_NAME

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Health check path for vllm-mlx model servers
VLLM_HEALTH_PATH = "/v1/models"


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ServiceStatus:
    """Status of a single managed service."""

    tier: str
    model: str
    port: int
    status: str  # "healthy", "degraded", "down", "crashed", "stopped"
    uptime: float | None  # seconds, None for stopped/crashed
    uptime_display: str  # human-readable string or "-"
    response_time: float | None  # seconds, None if no HTTP response
    pid: int | None


@dataclass
class StatusResult:
    """Result of the status command execution."""

    services: list[ServiceStatus] = field(default_factory=list)
    no_stack: bool = False
    message: str | None = None


# --------------------------------------------------------------------------- #
# Stack definition reading (read-only, no exceptions propagated)
# --------------------------------------------------------------------------- #


def _load_stack_for_status(stack_name: str = "default") -> dict[str, Any] | None:
    """Load a stack definition for status reporting.

    Unlike ``stack_up.load_stack_definition``, this never raises
    exceptions — it returns None when the stack cannot be loaded,
    allowing the status command to report "no stack configured".

    Args:
        stack_name: Name of the stack to load.

    Returns:
        The parsed stack definition dict, or None if unavailable.
    """
    import yaml

    stack_path = get_stacks_dir() / f"{stack_name}.yaml"
    if not stack_path.exists():
        return None

    try:
        content = stack_path.read_text(encoding="utf-8")
        stack = yaml.safe_load(content)
        if isinstance(stack, dict) and isinstance(stack.get("tiers"), list):
            return stack
    except Exception:
        pass

    return None


def _get_litellm_port() -> int:
    """Read the configured LiteLLM port.

    Returns:
        The configured port, or 4000 as default.
    """
    try:
        from mlx_stack.core.config import get_value

        port = get_value("litellm-port")
        if isinstance(port, int):
            return port
        return int(port)
    except Exception:
        return 4000


# --------------------------------------------------------------------------- #
# Main status orchestration
# --------------------------------------------------------------------------- #


def run_status(stack_name: str = "default") -> StatusResult:
    """Execute the full status check flow.

    1. Load the stack definition (read-only).
    2. For each tier, check PID file and HTTP health.
    3. Check LiteLLM service.
    4. Return a StatusResult with all service statuses.

    This function is entirely read-only: it does not modify PID files,
    restart services, acquire the lockfile, or write any files.

    Args:
        stack_name: Stack definition name.

    Returns:
        A StatusResult with the outcome.
    """
    result = StatusResult()

    # --- Load stack definition ---
    stack = _load_stack_for_status(stack_name)

    if stack is None:
        result.no_stack = True
        result.message = (
            "No stack configured — run 'mlx-stack init' to create a stack configuration."
        )
        return result

    tiers = stack.get("tiers", [])
    if not tiers:
        result.no_stack = True
        result.message = (
            "No stack configured — run 'mlx-stack init' to create a stack configuration."
        )
        return result

    # --- Check each tier ---
    for tier in tiers:
        tier_name = tier.get("name", "unknown")
        model = tier.get("model", "unknown")
        port = tier.get("port", 0)

        svc_status = get_service_status(
            service_name=tier_name,
            port=port,
            health_path=VLLM_HEALTH_PATH,
        )

        result.services.append(ServiceStatus(
            tier=tier_name,
            model=model,
            port=port,
            status=svc_status["status"],
            uptime=svc_status["uptime"],
            uptime_display=format_uptime(svc_status["uptime"]),
            response_time=svc_status["response_time"],
            pid=svc_status["pid"],
        ))

    # --- Check LiteLLM ---
    litellm_port = _get_litellm_port()
    litellm_status = get_service_status(
        service_name=LITELLM_SERVICE_NAME,
        port=litellm_port,
        health_path=LITELLM_HEALTH_PATH,
    )

    result.services.append(ServiceStatus(
        tier="litellm",
        model="proxy",
        port=litellm_port,
        status=litellm_status["status"],
        uptime=litellm_status["uptime"],
        uptime_display=format_uptime(litellm_status["uptime"]),
        response_time=litellm_status["response_time"],
        pid=litellm_status["pid"],
    ))

    return result


def status_to_dict(result: StatusResult) -> dict[str, Any]:
    """Convert a StatusResult to a JSON-serialisable dict.

    Args:
        result: The StatusResult to convert.

    Returns:
        A dict suitable for ``json.dumps``.
    """
    services_list: list[dict[str, Any]] = []
    for svc in result.services:
        services_list.append({
            "tier": svc.tier,
            "model": svc.model,
            "port": svc.port,
            "status": svc.status,
            "uptime": svc.uptime,
            "uptime_display": svc.uptime_display,
            "pid": svc.pid,
            "response_time": svc.response_time,
        })

    return {
        "services": services_list,
        "no_stack": result.no_stack,
        "message": result.message,
    }
