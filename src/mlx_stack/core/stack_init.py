"""Stack initialization logic for mlx-stack.

Generates stack definition YAML and LiteLLM config files from a
recommendation result. Handles port allocation, vllm_flags generation,
cloud fallback, missing model detection, and overwrite protection.
"""

from __future__ import annotations

import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlx_stack.core.catalog import CatalogEntry, get_entry_by_id, load_catalog
from mlx_stack.core.config import ConfigCorruptError, get_value
from mlx_stack.core.hardware import detect_hardware, load_profile, save_profile
from mlx_stack.core.litellm_gen import generate_litellm_config, render_litellm_yaml
from mlx_stack.core.paths import ensure_data_home, get_data_home, get_stacks_dir
from mlx_stack.core.scoring import (
    VALID_INTENTS,
    RecommendationResult,
    ScoringError,
    TierAssignment,
)
from mlx_stack.core.scoring import recommend as run_recommend

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Default starting port for vllm-mlx instances
_VLLM_BASE_PORT = 8000

# Schema version for stack definition files
STACK_SCHEMA_VERSION = 1

# Default stack name
DEFAULT_STACK_NAME = "default"


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class InitError(Exception):
    """Raised when stack initialization fails."""


# --------------------------------------------------------------------------- #
# Port allocation
# --------------------------------------------------------------------------- #


def _is_port_available(port: int) -> bool:
    """Check if a TCP port is available for binding.

    Args:
        port: The port number to check.

    Returns:
        True if the port is available, False otherwise.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            sock.bind(("127.0.0.1", port))
            return True
    except OSError:
        return False


def allocate_ports(
    num_tiers: int,
    litellm_port: int = 4000,
    base_port: int = _VLLM_BASE_PORT,
) -> list[int]:
    """Allocate unique ports for vllm-mlx instances.

    Ensures no port conflicts with the LiteLLM port and skips ports
    that are already in use (detected via socket binding). Selects
    deterministic alternates by incrementing the port number.

    Args:
        num_tiers: Number of tiers needing ports.
        litellm_port: The LiteLLM proxy port to avoid.
        base_port: Starting port for allocation.

    Returns:
        List of unique port numbers, one per tier.

    Raises:
        InitError: If not enough ports can be allocated within a
            reasonable range (base_port .. base_port + 100).
    """
    ports: list[int] = []
    port = base_port
    max_port = base_port + 100  # Safety limit to prevent infinite loops

    for _ in range(num_tiers):
        # Skip the LiteLLM port and ports already in use
        while port == litellm_port or not _is_port_available(port):
            port += 1
            if port > max_port:
                msg = (
                    f"Could not allocate {num_tiers} free ports starting "
                    f"from {base_port}. All ports in range "
                    f"{base_port}–{max_port} are in use or reserved."
                )
                raise InitError(msg)
        ports.append(port)
        port += 1

    return ports


# --------------------------------------------------------------------------- #
# vllm flags generation
# --------------------------------------------------------------------------- #


def build_vllm_flags(entry: CatalogEntry) -> dict[str, Any]:
    """Build vllm_flags for a model based on its catalog capabilities.

    All models get:
    - continuous_batching: true
    - use_paged_cache: true

    Tool-calling models additionally get:
    - enable_auto_tool_choice: true
    - tool_call_parser: <parser from catalog>

    Thinking models additionally get:
    - reasoning_parser: <parser from catalog>

    Args:
        entry: The catalog entry for the model.

    Returns:
        A dict of vllm flags.
    """
    flags: dict[str, Any] = {
        "continuous_batching": True,
        "use_paged_cache": True,
    }

    if entry.capabilities.tool_calling:
        flags["enable_auto_tool_choice"] = True
        if entry.capabilities.tool_call_parser:
            flags["tool_call_parser"] = entry.capabilities.tool_call_parser

    if entry.capabilities.thinking and entry.capabilities.reasoning_parser:
        flags["reasoning_parser"] = entry.capabilities.reasoning_parser

    return flags


# --------------------------------------------------------------------------- #
# Stack definition generation
# --------------------------------------------------------------------------- #


def _build_tier_entry(
    assignment: TierAssignment,
    port: int,
    catalog: list[CatalogEntry],
) -> dict[str, Any]:
    """Build a single tier entry for the stack definition.

    Args:
        assignment: The tier assignment from the scoring engine.
        port: The allocated port for this tier.
        catalog: The full catalog (for capability lookup).

    Returns:
        A dict representing the tier in the stack YAML.
    """
    entry = assignment.model.entry
    quant = assignment.quant

    # Get the source HF repo for this quant
    source = ""
    if quant in entry.sources:
        source = entry.sources[quant].hf_repo

    return {
        "name": assignment.tier,
        "model": entry.id,
        "quant": quant,
        "source": source,
        "port": port,
        "vllm_flags": build_vllm_flags(entry),
    }


def generate_stack_definition(
    recommendation: RecommendationResult,
    ports: list[int],
    catalog: list[CatalogEntry],
    stack_name: str = DEFAULT_STACK_NAME,
    cloud_fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a stack definition YAML structure.

    Args:
        recommendation: The recommendation result with tier assignments.
        ports: Allocated ports, one per tier.
        catalog: The full catalog for capability lookups.
        stack_name: Name of the stack (default: 'default').
        cloud_fallback: Optional cloud fallback configuration.

    Returns:
        A dict representing the full stack definition.

    Raises:
        InitError: If generation fails.
    """
    if len(ports) != len(recommendation.tiers):
        msg = f"Port count ({len(ports)}) doesn't match tier count ({len(recommendation.tiers)})"
        raise InitError(msg)

    tiers: list[dict[str, Any]] = []
    for assignment, port in zip(recommendation.tiers, ports):
        tiers.append(_build_tier_entry(assignment, port, catalog))

    stack: dict[str, Any] = {
        "schema_version": STACK_SCHEMA_VERSION,
        "name": stack_name,
        "hardware_profile": recommendation.hardware_profile.profile_id,
        "intent": recommendation.intent,
        "created": datetime.now(timezone.utc).isoformat(),
        "tiers": tiers,
    }

    if cloud_fallback:
        stack["cloud_fallback"] = cloud_fallback

    return stack


# --------------------------------------------------------------------------- #
# Missing model detection
# --------------------------------------------------------------------------- #


def detect_missing_models(
    tiers: list[dict[str, Any]],
    models_dir: Path | None = None,
) -> list[str]:
    """Detect models referenced in the stack that are not locally available.

    Args:
        tiers: List of tier entries from the stack definition.
        models_dir: The models directory to check. If None, uses config.

    Returns:
        List of model IDs that are not found locally.
    """
    if models_dir is None:
        try:
            models_dir = Path(str(get_value("model-dir"))).expanduser()
        except (ConfigCorruptError, Exception):
            models_dir = get_data_home() / "models"

    missing: list[str] = []
    for tier in tiers:
        model_id = tier["model"]
        # Check if model directory exists (simple heuristic)
        # Models would be stored in subdirectories matching the source repo pattern
        # or the model ID
        model_path = models_dir / model_id
        source = tier.get("source", "")
        # Also check by HF repo name (directory name from hf_repo)
        source_dir_name = source.rsplit("/", 1)[-1] if "/" in source else source
        source_path = models_dir / source_dir_name if source_dir_name else None

        if not model_path.exists() and (source_path is None or not source_path.exists()):
            missing.append(model_id)

    return missing


# --------------------------------------------------------------------------- #
# Main init entry point
# --------------------------------------------------------------------------- #


def run_init(
    intent: str = "balanced",
    budget_pct: int | None = None,
    add_models: list[str] | None = None,
    remove_tiers: list[str] | None = None,
    force: bool = False,
    stack_name: str = DEFAULT_STACK_NAME,
) -> dict[str, Any]:
    """Run the full init flow: profile -> recommend -> generate configs.

    Args:
        intent: Recommendation intent (balanced or agent-fleet).
        budget_pct: Memory budget percentage override (uses config default if None).
        add_models: Additional model IDs to add as tiers.
        remove_tiers: Tier names to remove from recommendation.
        force: Whether to overwrite existing stack files.
        stack_name: Name for the stack definition.

    Returns:
        A dict with keys:
        - stack_path: Path to the generated stack YAML
        - litellm_path: Path to the generated LiteLLM config
        - stack: The stack definition dict
        - litellm_config: The LiteLLM config dict
        - missing_models: List of models not found locally

    Raises:
        InitError: If initialization fails.
    """
    # --- Validate intent ---
    if intent not in VALID_INTENTS:
        valid = ", ".join(sorted(VALID_INTENTS))
        msg = f"Invalid intent '{intent}'. Valid intents: {valid}"
        raise InitError(msg)

    # --- Resolve hardware profile ---
    profile = load_profile()
    if profile is None:
        try:
            profile = detect_hardware()
            save_profile(profile)
        except Exception as exc:
            msg = f"Hardware detection failed: {exc}"
            raise InitError(msg) from None

    # --- Read config values ---
    try:
        litellm_port = int(get_value("litellm-port"))
    except (ConfigCorruptError, ValueError):
        litellm_port = 4000

    if budget_pct is None:
        try:
            budget_pct = int(get_value("memory-budget-pct"))
        except (ConfigCorruptError, ValueError):
            budget_pct = 40

    try:
        openrouter_key = str(get_value("openrouter-key"))
    except (ConfigCorruptError, Exception):
        openrouter_key = ""

    # --- Check for existing stack ---
    stacks_dir = get_stacks_dir()
    stack_path = stacks_dir / f"{stack_name}.yaml"
    litellm_path = get_data_home() / "litellm.yaml"

    if stack_path.exists() and not force:
        msg = (
            f"Stack '{stack_name}' already exists at {stack_path}. "
            f"Use --force to overwrite."
        )
        raise InitError(msg)

    # --- Load catalog ---
    try:
        catalog = load_catalog()
    except Exception as exc:
        msg = f"Could not load model catalog: {exc}"
        raise InitError(msg) from None

    # --- Run recommendation ---
    try:
        recommendation = run_recommend(
            catalog=catalog,
            profile=profile,
            intent=intent,
            budget_pct=budget_pct,
            exclude_gated=True,
        )
    except ScoringError as exc:
        msg = f"Recommendation failed: {exc}"
        raise InitError(msg) from None

    if not recommendation.tiers:
        msg = (
            f"No models fit within the {recommendation.memory_budget_gb:.1f} GB budget. "
            f"Try increasing memory-budget-pct in config."
        )
        raise InitError(msg)

    # --- Apply --add/--remove customizations ---
    tiers = list(recommendation.tiers)

    if remove_tiers:
        valid_tier_names = {t.tier for t in tiers}
        for tier_name in remove_tiers:
            if tier_name not in valid_tier_names:
                valid = ", ".join(sorted(valid_tier_names))
                msg = (
                    f"Cannot remove tier '{tier_name}': not in the current stack. "
                    f"Valid tiers: {valid}"
                )
                raise InitError(msg)
        tiers = [t for t in tiers if t.tier not in set(remove_tiers)]

    warnings: list[str] = []

    if add_models:
        for model_id in add_models:
            entry = get_entry_by_id(catalog, model_id)
            if entry is None:
                msg = (
                    f"Unknown model '{model_id}'. "
                    f"Run 'mlx-stack models --catalog' to see available models."
                )
                raise InitError(msg)

            # Check if model already assigned
            assigned_ids = {t.model.entry.id for t in tiers}
            if model_id in assigned_ids:
                continue  # Skip duplicates silently

            # Create a tier name like 'added-<model_id>'
            from mlx_stack.core.scoring import INTENT_WEIGHTS, TierAssignment, score_model

            weights = INTENT_WEIGHTS.get(intent, INTENT_WEIGHTS["balanced"])
            try:
                scored = score_model(
                    entry, profile, weights, recommendation.memory_budget_gb,
                )
            except ScoringError as exc:
                msg = f"Cannot add model '{model_id}': {exc}"
                raise InitError(msg) from None

            # Warn if the model is gated (requires HuggingFace auth)
            if entry.gated:
                warnings.append(
                    f"Model '{model_id}' is gated and requires HuggingFace "
                    f"authentication. Set HF_TOKEN or run 'huggingface-cli login' "
                    f"before pulling."
                )

            # Warn if exceeding budget (per spec: warn, not block)
            total_memory = sum(t.model.memory_gb for t in tiers) + scored.memory_gb
            if total_memory > recommendation.memory_budget_gb:
                warnings.append(
                    f"Adding '{model_id}' exceeds memory budget "
                    f"({total_memory:.1f} GB > {recommendation.memory_budget_gb:.1f} GB)."
                )

            tier_name = f"added-{model_id}"
            tiers.append(TierAssignment(
                tier=tier_name,
                model=scored,
                quant="int4",
            ))

    if not tiers:
        msg = "No tiers remaining after customization. Cannot generate stack."
        raise InitError(msg)

    # --- Allocate ports ---
    ports = allocate_ports(len(tiers), litellm_port=litellm_port)

    # --- Generate stack definition ---
    cloud_fallback: dict[str, Any] | None = None
    if openrouter_key:
        cloud_fallback = {
            "provider": "openrouter",
            "models": ["openai/gpt-4o", "anthropic/claude-sonnet-4-20250514"],
        }

    stack = generate_stack_definition(
        recommendation=_with_tiers(recommendation, tiers),
        ports=ports,
        catalog=catalog,
        stack_name=stack_name,
        cloud_fallback=cloud_fallback,
    )

    # --- Generate LiteLLM config ---
    tier_entries = [
        {"name": t["name"], "model": t["model"], "port": t["port"]}
        for t in stack["tiers"]
    ]
    litellm_config = generate_litellm_config(
        tiers=tier_entries,
        litellm_port=litellm_port,
        openrouter_key=openrouter_key,
    )

    # --- Write files ---
    ensure_data_home()
    stacks_dir.mkdir(parents=True, exist_ok=True)

    stack_yaml = yaml.dump(stack, default_flow_style=False, sort_keys=False)
    stack_path.write_text(stack_yaml, encoding="utf-8")

    litellm_yaml = render_litellm_yaml(litellm_config)
    litellm_path.write_text(litellm_yaml, encoding="utf-8")

    # --- Detect missing models ---
    missing_models = detect_missing_models(stack["tiers"])

    # --- Compute total estimated memory for selected tiers ---
    total_memory_gb = sum(t.model.memory_gb for t in tiers)

    return {
        "stack_path": stack_path,
        "litellm_path": litellm_path,
        "stack": stack,
        "litellm_config": litellm_config,
        "missing_models": missing_models,
        "warnings": warnings,
        "profile": profile,
        "memory_budget_gb": recommendation.memory_budget_gb,
        "total_memory_gb": total_memory_gb,
    }


def _with_tiers(result: RecommendationResult, tiers: list[TierAssignment]) -> RecommendationResult:
    """Create a new RecommendationResult with different tiers.

    RecommendationResult is a frozen dataclass, so we create a new instance.
    """
    return RecommendationResult(
        tiers=tiers,
        all_scored=result.all_scored,
        memory_budget_gb=result.memory_budget_gb,
        intent=result.intent,
        hardware_profile=result.hardware_profile,
    )
