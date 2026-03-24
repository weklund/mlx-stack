"""Stack startup logic for mlx-stack.

Orchestrates starting all services defined in a stack definition:
reads default.yaml, starts vllm-mlx subprocesses sequentially (largest
model first), performs HTTP health checks with 120s timeout, starts
LiteLLM after all healthy servers, creates PID files, and produces a
summary. Supports dry-run, selective tier start, lockfile, port
conflict detection, stale PID cleanup, memory warnings, auto-install
of dependencies, and localhost-only binding. Propagates the OpenRouter
API key securely via env var.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil
import yaml

from mlx_stack.core.catalog import CatalogEntry, get_entry_by_id, load_catalog
from mlx_stack.core.config import ConfigCorruptError, get_value
from mlx_stack.core.deps import (
    DependencyError,
    DependencyInstallError,
    ensure_dependency,
)
from mlx_stack.core.paths import get_data_home, get_stacks_dir
from mlx_stack.core.process import (
    HealthCheckError,
    LockError,
    acquire_lock,
    check_port_conflict,
    cleanup_stale_pid,
    is_process_alive,
    read_pid_file,
    start_service,
    wait_for_healthy,
)
from mlx_stack.core.stack_init import STACK_SCHEMA_VERSION

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Health check path for vllm-mlx
VLLM_HEALTH_PATH = "/v1/models"

# Health check path for LiteLLM
LITELLM_HEALTH_PATH = "/health/liveliness"

# LiteLLM service name for PID files
LITELLM_SERVICE_NAME = "litellm"


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class UpError(Exception):
    """Raised when the up command encounters a fatal error."""


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class TierStatus:
    """Status of a single tier after startup attempt."""

    name: str
    model: str
    port: int
    status: str  # "healthy", "failed", "skipped", "dry-run", "already-running"
    error: str | None = None


@dataclass
class UpResult:
    """Result of the up command execution."""

    tiers: list[TierStatus] = field(default_factory=list)
    litellm: TierStatus | None = None
    dry_run: bool = False
    dry_run_commands: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    already_running: bool = False


# --------------------------------------------------------------------------- #
# Stack definition loading & validation
# --------------------------------------------------------------------------- #


def load_stack_definition(stack_name: str = "default") -> dict[str, Any]:
    """Load and validate a stack definition from disk.

    Args:
        stack_name: Name of the stack to load.

    Returns:
        The parsed stack definition dict.

    Raises:
        UpError: If the stack file is missing, invalid YAML, or has
            an unsupported schema version.
    """
    stack_path = get_stacks_dir() / f"{stack_name}.yaml"

    if not stack_path.exists():
        msg = (
            f"No stack definition found at {stack_path}.\n"
            "Run 'mlx-stack init' to create a stack configuration."
        )
        raise UpError(msg)

    try:
        content = stack_path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"Could not read stack file: {exc}"
        raise UpError(msg) from None

    try:
        stack = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        msg = f"Invalid YAML in stack file {stack_path}: {exc}"
        raise UpError(msg) from None

    if not isinstance(stack, dict):
        msg = f"Stack file {stack_path} has invalid format: expected a mapping."
        raise UpError(msg) from None

    # Validate schema version
    schema_version = stack.get("schema_version")
    if schema_version != STACK_SCHEMA_VERSION:
        msg = (
            f"Unsupported stack schema_version: {schema_version} "
            f"(expected {STACK_SCHEMA_VERSION}). "
            "Re-run 'mlx-stack init --force' to regenerate."
        )
        raise UpError(msg)

    # Validate tiers exist
    tiers = stack.get("tiers")
    if not tiers or not isinstance(tiers, list):
        msg = "Stack definition has no tiers."
        raise UpError(msg)

    return stack


# --------------------------------------------------------------------------- #
# Memory estimation
# --------------------------------------------------------------------------- #


def estimate_memory_usage(
    tiers: list[dict[str, Any]],
    catalog: list[CatalogEntry] | None = None,
) -> float:
    """Estimate total memory usage for all tiers.

    Uses catalog benchmark data to look up memory_gb per model+quant.
    Falls back to a rough params_b-based estimate if no benchmark data.

    Args:
        tiers: List of tier dicts from the stack definition.
        catalog: The loaded catalog. If None, attempts to load.

    Returns:
        Estimated total memory in GB.
    """
    if catalog is None:
        try:
            catalog = load_catalog()
        except Exception:
            return 0.0

    total = 0.0
    for tier in tiers:
        model_id = tier.get("model", "")
        entry = get_entry_by_id(catalog, model_id)
        if entry is None:
            continue

        # Look for memory_gb in any benchmark entry
        memory_gb = 0.0
        for _hw_key, bench in entry.benchmarks.items():
            memory_gb = bench.memory_gb
            break  # Take the first available benchmark's memory

        if memory_gb <= 0:
            # Rough estimate: ~1 GB per billion parameters for int4
            memory_gb = entry.params_b * 1.0

        total += memory_gb

    return total


def check_memory_warning(estimated_gb: float) -> str | None:
    """Check if estimated memory usage exceeds available system memory.

    Args:
        estimated_gb: Estimated total memory usage in GB.

    Returns:
        A warning string if memory is likely insufficient, or None.
    """
    try:
        vmem = psutil.virtual_memory()
        available_gb = vmem.available / (1024**3)
    except Exception:
        return None

    if estimated_gb > available_gb:
        return (
            f"Estimated memory usage ({estimated_gb:.1f} GB) exceeds "
            f"available system memory ({available_gb:.1f} GB). "
            "Performance may be degraded."
        )
    return None


# --------------------------------------------------------------------------- #
# vllm-mlx command building
# --------------------------------------------------------------------------- #


def build_vllm_command(
    tier: dict[str, Any],
    vllm_binary: str,
) -> list[str]:
    """Build the vllm-mlx command for a tier.

    Args:
        tier: Tier dict from the stack definition.
        vllm_binary: Path to the vllm-mlx binary.

    Returns:
        The command as a list of strings.
    """
    model_source = tier.get("source", "")
    port = tier["port"]

    cmd = [
        vllm_binary,
        "--model", model_source,
        "--port", str(port),
        "--host", "127.0.0.1",
    ]

    # Add vllm_flags
    vllm_flags = tier.get("vllm_flags", {})
    for flag_name, flag_value in vllm_flags.items():
        flag_key = f"--{flag_name.replace('_', '-')}"
        if isinstance(flag_value, bool):
            if flag_value:
                cmd.append(flag_key)
        else:
            cmd.extend([flag_key, str(flag_value)])

    return cmd


def build_litellm_command(
    litellm_binary: str,
    litellm_port: int,
    litellm_config_path: Path,
) -> list[str]:
    """Build the litellm command.

    Args:
        litellm_binary: Path to the litellm binary.
        litellm_port: Port for LiteLLM.
        litellm_config_path: Path to litellm.yaml config.

    Returns:
        The command as a list of strings.
    """
    return [
        litellm_binary,
        "--config", str(litellm_config_path),
        "--port", str(litellm_port),
        "--host", "127.0.0.1",
    ]


# --------------------------------------------------------------------------- #
# Dry-run command formatting
# --------------------------------------------------------------------------- #


def format_dry_run_command(
    cmd: list[str],
    env_vars: dict[str, str] | None = None,
) -> str:
    """Format a command for dry-run display.

    Hides any sensitive environment variable values.

    Args:
        cmd: The command as a list of strings.
        env_vars: Optional environment variables (values hidden).

    Returns:
        A human-readable command string.
    """
    parts: list[str] = []

    if env_vars:
        for key in sorted(env_vars.keys()):
            # Mask all env var values in dry-run
            parts.append(f"{key}=***")

    parts.extend(cmd)
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Sort tiers by params_b descending (largest model first)
# --------------------------------------------------------------------------- #


def sort_tiers_by_size(
    tiers: list[dict[str, Any]],
    catalog: list[CatalogEntry] | None = None,
) -> list[dict[str, Any]]:
    """Sort tiers by model size descending (largest first).

    Uses catalog params_b for ordering. Falls back to tier name if
    catalog entry is not found.

    Args:
        tiers: Tier entries from the stack definition.
        catalog: Loaded catalog for params_b lookup.

    Returns:
        Tiers sorted largest model first.
    """
    if catalog is None:
        return list(tiers)

    def sort_key(tier: dict[str, Any]) -> tuple[float, str]:
        model_id = tier.get("model", "")
        entry = get_entry_by_id(catalog, model_id)
        params_b = entry.params_b if entry else 0.0
        return (-params_b, tier.get("name", ""))

    return sorted(tiers, key=sort_key)


# --------------------------------------------------------------------------- #
# Main startup orchestration
# --------------------------------------------------------------------------- #


def run_up(
    dry_run: bool = False,
    tier_filter: str | None = None,
    stack_name: str = "default",
) -> UpResult:
    """Execute the full stack startup flow.

    1. Load and validate stack definition.
    2. Auto-install missing dependencies.
    3. Check for stale PIDs and clean up.
    4. Check for already-running services.
    5. Estimate memory and warn if needed.
    6. Start vllm-mlx instances sequentially (largest first).
    7. Health check each instance with exponential backoff.
    8. Start LiteLLM after all healthy model servers.
    9. Return summary result.

    Args:
        dry_run: If True, show commands without executing.
        tier_filter: If set, start only this tier (plus LiteLLM).
        stack_name: Stack definition name.

    Returns:
        An UpResult with the outcome.

    Raises:
        UpError: On fatal errors (missing stack, schema mismatch, etc.).
        LockError: If the lockfile is held by another process.
    """
    result = UpResult(dry_run=dry_run)

    # --- Load stack definition ---
    stack = load_stack_definition(stack_name)
    tiers = stack["tiers"]

    # --- Read config ---
    try:
        litellm_port = int(get_value("litellm-port"))
    except (ConfigCorruptError, ValueError):
        litellm_port = 4000

    try:
        openrouter_key = str(get_value("openrouter-key"))
    except (ConfigCorruptError, Exception):
        openrouter_key = ""

    litellm_config_path = get_data_home() / "litellm.yaml"

    # --- Validate --tier filter ---
    valid_tier_names = [t["name"] for t in tiers]
    if tier_filter is not None:
        if tier_filter not in valid_tier_names:
            valid_list = ", ".join(sorted(valid_tier_names))
            msg = (
                f"Unknown tier '{tier_filter}'. "
                f"Valid tiers: {valid_list}"
            )
            raise UpError(msg)
        tiers = [t for t in tiers if t["name"] == tier_filter]

    # --- Load catalog for sorting and memory estimation ---
    try:
        catalog = load_catalog()
    except Exception:
        catalog = None

    # --- Sort tiers by model size (largest first) ---
    tiers = sort_tiers_by_size(tiers, catalog)

    # --- Dry-run mode ---
    if dry_run:
        return _run_dry_run(
            tiers=tiers,
            litellm_port=litellm_port,
            litellm_config_path=litellm_config_path,
            openrouter_key=openrouter_key,
            catalog=catalog,
            result=result,
        )

    # --- Acquire lockfile ---
    # The context manager ensures the lock is released on exit, failure,
    # or crash (OS-level FD cleanup).
    try:
        with acquire_lock():
            return _run_startup(
                tiers=tiers,
                litellm_port=litellm_port,
                litellm_config_path=litellm_config_path,
                openrouter_key=openrouter_key,
                catalog=catalog,
                tier_filter=tier_filter,
                result=result,
            )
    except LockError:
        raise


def _run_dry_run(
    tiers: list[dict[str, Any]],
    litellm_port: int,
    litellm_config_path: Path,
    openrouter_key: str,
    catalog: list[CatalogEntry] | None,
    result: UpResult,
) -> UpResult:
    """Execute a dry-run — show commands without starting processes.

    Args:
        tiers: Tiers to start.
        litellm_port: LiteLLM port.
        litellm_config_path: Path to litellm.yaml.
        openrouter_key: OpenRouter API key (masked in output).
        catalog: Loaded catalog.
        result: The UpResult to populate.

    Returns:
        The populated UpResult.
    """
    vllm_binary = shutil.which("vllm-mlx") or "vllm-mlx"
    litellm_binary = shutil.which("litellm") or "litellm"

    for tier in tiers:
        cmd = build_vllm_command(tier, vllm_binary)
        cmd_str = format_dry_run_command(cmd)

        result.dry_run_commands.append({
            "service": tier["name"],
            "command": cmd_str,
            "type": "vllm-mlx",
        })

        result.tiers.append(TierStatus(
            name=tier["name"],
            model=tier.get("model", ""),
            port=tier["port"],
            status="dry-run",
        ))

    # LiteLLM command
    litellm_cmd = build_litellm_command(litellm_binary, litellm_port, litellm_config_path)

    env_display: dict[str, str] | None = None
    if openrouter_key:
        env_display = {"OPENROUTER_API_KEY": "***"}

    litellm_cmd_str = format_dry_run_command(litellm_cmd, env_display)

    result.dry_run_commands.append({
        "service": LITELLM_SERVICE_NAME,
        "command": litellm_cmd_str,
        "type": "litellm",
    })

    result.litellm = TierStatus(
        name=LITELLM_SERVICE_NAME,
        model="proxy",
        port=litellm_port,
        status="dry-run",
    )

    return result


def _run_startup(
    tiers: list[dict[str, Any]],
    litellm_port: int,
    litellm_config_path: Path,
    openrouter_key: str,
    catalog: list[CatalogEntry] | None,
    tier_filter: str | None,
    result: UpResult,
) -> UpResult:
    """Execute the actual startup sequence.

    Args:
        tiers: Tiers to start.
        litellm_port: LiteLLM port.
        litellm_config_path: Path to litellm.yaml.
        openrouter_key: OpenRouter API key.
        catalog: Loaded catalog.
        tier_filter: If set, only start this tier.
        result: The UpResult to populate.

    Returns:
        The populated UpResult.
    """
    # --- Auto-install dependencies ---
    try:
        ensure_dependency("vllm-mlx")
        ensure_dependency("litellm")
    except (DependencyError, DependencyInstallError) as exc:
        raise UpError(f"Dependency installation failed: {exc}") from None

    # --- Resolve binary paths ---
    vllm_binary = shutil.which("vllm-mlx")
    if vllm_binary is None:
        raise UpError(
            "vllm-mlx not found on PATH after installation. "
            "Install manually: uv tool install vllm-mlx"
        )

    litellm_binary = shutil.which("litellm")
    if litellm_binary is None:
        raise UpError(
            "litellm not found on PATH after installation. "
            "Install manually: uv tool install litellm"
        )

    # --- Check for already-running / stale PIDs ---
    any_stale = False

    for tier in tiers:
        tier_name = tier["name"]
        pid = read_pid_file(tier_name)

        if pid is not None:
            if is_process_alive(pid):
                # Already running
                result.tiers.append(TierStatus(
                    name=tier_name,
                    model=tier.get("model", ""),
                    port=tier["port"],
                    status="already-running",
                ))
                continue
            else:
                # Stale PID — clean up
                cleanup_stale_pid(tier_name)
                any_stale = True
        else:
            pass  # Tier needs to be started

    # Check LiteLLM
    litellm_pid = read_pid_file(LITELLM_SERVICE_NAME)
    litellm_already_running = False
    if litellm_pid is not None:
        if is_process_alive(litellm_pid):
            litellm_already_running = True
        else:
            cleanup_stale_pid(LITELLM_SERVICE_NAME)
            any_stale = True

    # If all tiers + LiteLLM are already running, report and return
    tiers_already_running = [
        t for t in result.tiers if t.status == "already-running"
    ]
    if len(tiers_already_running) == len(tiers) and litellm_already_running:
        result.already_running = True
        result.litellm = TierStatus(
            name=LITELLM_SERVICE_NAME,
            model="proxy",
            port=litellm_port,
            status="already-running",
        )
        return result

    if any_stale:
        result.warnings.append("Cleaned up stale PID files from previously crashed services.")

    # --- Memory warning ---
    estimated_gb = estimate_memory_usage(tiers, catalog)
    if estimated_gb > 0:
        warning = check_memory_warning(estimated_gb)
        if warning:
            result.warnings.append(warning)

    # --- Start vllm-mlx instances sequentially ---
    healthy_count = 0
    tiers_needing_start = [
        t for t in tiers
        if t["name"] not in {ts.name for ts in result.tiers}
    ]

    for tier in tiers_needing_start:
        tier_name = tier["name"]
        port = tier["port"]

        # Check port conflict
        conflict = check_port_conflict(port)
        if conflict is not None:
            conflict_pid, conflict_name = conflict
            result.tiers.append(TierStatus(
                name=tier_name,
                model=tier.get("model", ""),
                port=port,
                status="skipped",
                error=(
                    f"Port {port} already in use by "
                    f"PID {conflict_pid} ({conflict_name})"
                ),
            ))
            continue

        # Start the vllm-mlx subprocess
        cmd = build_vllm_command(tier, vllm_binary)

        try:
            start_service(
                service_name=tier_name,
                cmd=cmd,
                port=port,
            )
        except Exception as exc:
            result.tiers.append(TierStatus(
                name=tier_name,
                model=tier.get("model", ""),
                port=port,
                status="failed",
                error=str(exc),
            ))
            continue

        # Health check with exponential backoff
        try:
            wait_for_healthy(port=port, path=VLLM_HEALTH_PATH)
            result.tiers.append(TierStatus(
                name=tier_name,
                model=tier.get("model", ""),
                port=port,
                status="healthy",
            ))
            healthy_count += 1
        except HealthCheckError as exc:
            result.tiers.append(TierStatus(
                name=tier_name,
                model=tier.get("model", ""),
                port=port,
                status="failed",
                error=str(exc),
            ))

    # --- Count total healthy (including already-running) ---
    total_healthy = sum(
        1 for t in result.tiers if t.status in ("healthy", "already-running")
    )

    # --- Start LiteLLM if any healthy tiers and not already running ---
    if litellm_already_running:
        result.litellm = TierStatus(
            name=LITELLM_SERVICE_NAME,
            model="proxy",
            port=litellm_port,
            status="already-running",
        )
    elif total_healthy == 0:
        result.litellm = TierStatus(
            name=LITELLM_SERVICE_NAME,
            model="proxy",
            port=litellm_port,
            status="skipped",
            error="All model servers failed; LiteLLM not started.",
        )
    else:
        # Check LiteLLM port conflict
        litellm_conflict = check_port_conflict(litellm_port)
        if litellm_conflict is not None:
            conflict_pid, conflict_name = litellm_conflict
            result.litellm = TierStatus(
                name=LITELLM_SERVICE_NAME,
                model="proxy",
                port=litellm_port,
                status="skipped",
                error=(
                    f"Port {litellm_port} already in use by "
                    f"PID {conflict_pid} ({conflict_name})"
                ),
            )
        else:
            litellm_cmd = build_litellm_command(
                litellm_binary, litellm_port, litellm_config_path,
            )

            # Build env with OpenRouter key if configured
            litellm_env: dict[str, str] | None = None
            if openrouter_key:
                litellm_env = {"OPENROUTER_API_KEY": openrouter_key}

            try:
                start_service(
                    service_name=LITELLM_SERVICE_NAME,
                    cmd=litellm_cmd,
                    port=litellm_port,
                    env=litellm_env,
                )

                # Health check LiteLLM
                try:
                    wait_for_healthy(
                        port=litellm_port,
                        path=LITELLM_HEALTH_PATH,
                    )
                    result.litellm = TierStatus(
                        name=LITELLM_SERVICE_NAME,
                        model="proxy",
                        port=litellm_port,
                        status="healthy",
                    )
                except HealthCheckError as exc:
                    result.litellm = TierStatus(
                        name=LITELLM_SERVICE_NAME,
                        model="proxy",
                        port=litellm_port,
                        status="failed",
                        error=str(exc),
                    )
            except Exception as exc:
                result.litellm = TierStatus(
                    name=LITELLM_SERVICE_NAME,
                    model="proxy",
                    port=litellm_port,
                    status="failed",
                    error=str(exc),
                )

    return result
