"""Onboarding orchestration for mlx-stack.

Drives the ``mlx-stack setup`` flow: hardware detection, model scoring,
tier assignment, config generation, model download, and stack startup.

Business logic with minimal display dependency (Rich Console for download
progress). The CLI layer in ``cli/setup.py`` handles all other user
interaction and display.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

from mlx_stack.core.config import get_value
from mlx_stack.core.discovery import DiscoveredModel
from mlx_stack.core.hardware import HardwareProfile, detect_hardware, save_profile
from mlx_stack.core.litellm_gen import generate_litellm_config
from mlx_stack.core.paths import ensure_data_home, get_data_home
from mlx_stack.core.pull import (
    ModelInventoryEntry,
    PullResult,
    add_to_inventory,
    download_model,
)
from mlx_stack.core.scoring import INTENT_WEIGHTS
from mlx_stack.core.stack_up import run_up

# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class OnboardingError(Exception):
    """Raised when onboarding fails."""


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ScoredDiscoveredModel:
    """A discovered model with computed scores."""

    model: DiscoveredModel
    composite_score: float
    speed_score: float
    quality_score: float
    tool_calling_score: float
    memory_efficiency_score: float
    is_recommended: bool = False


@dataclass(frozen=True)
class TierMapping:
    """A model assigned to a tier."""

    tier_name: str
    model: DiscoveredModel


@dataclass(frozen=True)
class SetupResult:
    """Final result of the setup flow."""

    stack_path: Path
    litellm_path: Path
    tiers: list[TierMapping]
    pull_results: list[PullResult]
    warnings: list[str]


# --------------------------------------------------------------------------- #
# Step 1: Hardware detection
# --------------------------------------------------------------------------- #


def detect_and_budget(
    budget_pct: int = 40,
) -> tuple[HardwareProfile, float]:
    """Detect hardware and compute memory budget.

    Returns:
        (profile, memory_budget_gb)
    """
    try:
        profile = detect_hardware()
        save_profile(profile)
    except Exception as exc:
        msg = f"Hardware detection failed: {exc}"
        raise OnboardingError(msg) from None

    budget_gb = profile.memory_gb * budget_pct / 100.0
    return profile, budget_gb


# --------------------------------------------------------------------------- #
# Step 2: Scoring (replicated from scoring.py for DiscoveredModel)
# --------------------------------------------------------------------------- #


def _normalize_gen_tps_log(gen_tps: float) -> float:
    """Log-scaled normalization of generation speed to [0, 1]."""
    max_ref = 200.0
    if gen_tps <= 0:
        return 0.0
    return min(math.log(1.0 + gen_tps) / math.log(1.0 + max_ref), 1.0)


def _normalize_quality(quality: float | None) -> float:
    """Normalize quality (0-1 pass rate) to [0, 1]. None → 0.5 (neutral)."""
    if quality is None:
        return 0.5
    return min(max(quality, 0.0), 1.0)


def _normalize_memory_efficiency(memory_gb: float, budget_gb: float) -> float:
    """Normalize memory efficiency to [0, 1]."""
    if budget_gb <= 0 or memory_gb <= 0:
        return 0.0
    return min(max((budget_gb - memory_gb) / budget_gb, 0.0), 1.0)


def score_and_filter(
    models: list[DiscoveredModel],
    intent: str,
    budget_gb: float,
) -> list[ScoredDiscoveredModel]:
    """Score and filter models for the given intent and budget.

    Filters out models that exceed memory budget, then scores remaining
    using intent-weighted composite scoring.

    Returns:
        Sorted list (highest composite_score first).
    """
    weights = INTENT_WEIGHTS.get(intent, INTENT_WEIGHTS["balanced"])
    scored: list[ScoredDiscoveredModel] = []

    for model in models:
        # Filter by memory budget
        mem = model.memory_gb
        if mem is not None and mem > budget_gb:
            continue
        # Skip models with unknown memory and large params (likely too big)
        if mem is None and model.params_b > budget_gb * 1.5:
            continue

        speed = _normalize_gen_tps_log(model.gen_tps or 0.0)
        quality = _normalize_quality(model.quality_overall)
        tool = 1.0 if model.tool_calling else 0.0
        mem_eff = _normalize_memory_efficiency(mem or 0.0, budget_gb)

        composite = (
            weights.speed * speed
            + weights.quality * quality
            + weights.tool_calling * tool
            + weights.memory_efficiency * mem_eff
        )

        scored.append(
            ScoredDiscoveredModel(
                model=model,
                composite_score=round(composite, 4),
                speed_score=round(speed, 4),
                quality_score=round(quality, 4),
                tool_calling_score=tool,
                memory_efficiency_score=round(mem_eff, 4),
            )
        )

    scored.sort(key=lambda s: s.composite_score, reverse=True)
    return scored


# --------------------------------------------------------------------------- #
# Step 3: Default selection
# --------------------------------------------------------------------------- #


def select_defaults(
    scored_models: list[ScoredDiscoveredModel],
    budget_gb: float,
) -> list[ScoredDiscoveredModel]:
    """Select the default recommended models that fit within budget.

    Greedy selection:
    1. Top composite_score → standard tier
    2. Top gen_tps (different model) → fast tier
    3. Stop when adding another model would exceed budget

    Returns:
        New list with is_recommended=True for selected models.
    """
    if not scored_models:
        return []

    selected_indices: set[int] = set()
    total_mem = 0.0

    # Pick standard: highest composite
    std_idx = 0
    std_mem = scored_models[std_idx].model.memory_gb or 0.0
    if std_mem <= budget_gb:
        selected_indices.add(std_idx)
        total_mem += std_mem

    # Pick fast: highest gen_tps that isn't the standard pick
    if len(scored_models) > 1:
        fast_candidates = [(i, s) for i, s in enumerate(scored_models) if i not in selected_indices]
        if fast_candidates:
            fast_candidates.sort(
                key=lambda x: x[1].model.gen_tps or 0.0,
                reverse=True,
            )
            fast_idx, fast_model = fast_candidates[0]
            fast_mem = fast_model.model.memory_gb or 0.0
            if total_mem + fast_mem <= budget_gb:
                selected_indices.add(fast_idx)

    # Mark selected
    result: list[ScoredDiscoveredModel] = []
    for i, s in enumerate(scored_models):
        if i in selected_indices:
            result.append(
                ScoredDiscoveredModel(
                    model=s.model,
                    composite_score=s.composite_score,
                    speed_score=s.speed_score,
                    quality_score=s.quality_score,
                    tool_calling_score=s.tool_calling_score,
                    memory_efficiency_score=s.memory_efficiency_score,
                    is_recommended=True,
                )
            )
        else:
            result.append(s)

    return result


# --------------------------------------------------------------------------- #
# Step 4: Tier assignment
# --------------------------------------------------------------------------- #


def assign_tiers(
    selected_models: list[ScoredDiscoveredModel],
) -> list[TierMapping]:
    """Assign selected models to tiers.

    Rules:
    - Highest composite → "standard"
    - Highest gen_tps (if different) → "fast"
    - Additional models → "added-N"
    """
    if not selected_models:
        return []

    # Sort by composite_score descending
    by_score = sorted(selected_models, key=lambda s: s.composite_score, reverse=True)
    mappings: list[TierMapping] = []
    used_names: set[str] = set()

    # Standard = highest composite
    mappings.append(TierMapping(tier_name="standard", model=by_score[0].model))
    used_names.add("standard")

    if len(by_score) > 1:
        # Fast = highest gen_tps among remaining
        remaining = by_score[1:]
        remaining.sort(key=lambda s: s.model.gen_tps or 0.0, reverse=True)
        mappings.append(TierMapping(tier_name="fast", model=remaining[0].model))
        used_names.add("fast")

        # Additional models
        for extra in remaining[1:]:
            n = len(mappings) - 1
            mappings.append(
                TierMapping(
                    tier_name=f"added-{n}",
                    model=extra.model,
                )
            )

    return mappings


# --------------------------------------------------------------------------- #
# Step 5: Config generation
# --------------------------------------------------------------------------- #

# Starting port for vllm-mlx instances
_BASE_PORT = 8000
# LiteLLM proxy port
_LITELLM_PORT_DEFAULT = 4000


def generate_config(
    profile: HardwareProfile,
    intent: str,
    tier_mappings: list[TierMapping],
    budget_gb: float,
) -> tuple[Path, Path]:
    """Generate stack YAML and LiteLLM config from setup selections.

    Writes files to ~/.mlx-stack/stacks/default.yaml and ~/.mlx-stack/litellm.yaml.

    Returns:
        (stack_path, litellm_path)
    """
    home = ensure_data_home()
    litellm_port = int(get_value("litellm-port") or _LITELLM_PORT_DEFAULT)

    # Allocate ports
    port = _BASE_PORT
    tiers_config: list[dict[str, Any]] = []
    for mapping in tier_mappings:
        # Skip LiteLLM port
        if port == litellm_port:
            port += 1

        # TODO(#17): re-enable continuous_batching (waybarrios/vllm-mlx#211).
        vllm_flags: dict[str, Any] = {
            "use_paged_cache": True,
        }
        if mapping.model.tool_calling:
            vllm_flags["enable_auto_tool_choice"] = True

        tiers_config.append(
            {
                "name": mapping.tier_name,
                "model": mapping.model.display_name,
                "quant": mapping.model.quant,
                "source": mapping.model.hf_repo,
                "port": port,
                "vllm_flags": vllm_flags,
            }
        )
        port += 1

    # Build stack YAML
    stack_def = {
        "schema_version": 1,
        "name": "default",
        "hardware_profile": profile.profile_id,
        "intent": intent,
        "created": datetime.now(UTC).isoformat(),
        "tiers": tiers_config,
    }

    stacks_dir = home / "stacks"
    stacks_dir.mkdir(parents=True, exist_ok=True)
    stack_path = stacks_dir / "default.yaml"
    stack_path.write_text(
        yaml.dump(stack_def, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    # Build LiteLLM config
    litellm_tiers = [
        {"name": t["name"], "model": t["model"], "port": t["port"]} for t in tiers_config
    ]
    openrouter_key = str(get_value("openrouter-key") or "")
    litellm_config = generate_litellm_config(
        tiers=litellm_tiers,
        litellm_port=litellm_port,
        openrouter_key=openrouter_key,
    )

    litellm_path = home / "litellm.yaml"
    litellm_path.write_text(
        yaml.dump(litellm_config, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    return stack_path, litellm_path


# --------------------------------------------------------------------------- #
# Step 6: Pull models
# --------------------------------------------------------------------------- #


def pull_setup_models(
    models: list[DiscoveredModel],
    console: Console | None = None,
) -> list[PullResult]:
    """Download each selected model.

    Uses download_model() from core/pull.py directly. Skips models
    that are already downloaded.

    Returns:
        List of PullResult for each model.
    """
    if console is None:
        console = Console()

    home = get_data_home()
    models_dir = Path(str(get_value("model-dir") or str(home / "models")))
    models_dir.mkdir(parents=True, exist_ok=True)

    results: list[PullResult] = []

    for model in models:
        repo_name = model.hf_repo.rsplit("/", 1)[-1] if "/" in model.hf_repo else model.hf_repo
        local_path = models_dir / repo_name

        already_exists = local_path.exists() and any(local_path.iterdir())

        if already_exists:
            console.print(f"[dim]  {model.display_name} already downloaded.[/dim]")
        else:
            download_model(model.hf_repo, local_path, console)

            # Add to inventory
            entry = ModelInventoryEntry(
                model_id=model.display_name,
                name=model.display_name,
                quant=model.quant,
                source_type="mlx-community",
                hf_repo=model.hf_repo,
                local_path=str(local_path),
                disk_size_gb=model.memory_gb or 0.0,
                downloaded_at=datetime.now(UTC).isoformat(),
            )
            add_to_inventory(entry)

        results.append(
            PullResult(
                model_id=model.display_name,
                name=model.display_name,
                quant=model.quant,
                source_type="mlx-community",
                local_path=local_path,
                already_existed=already_exists,
                disk_size_gb=model.memory_gb or 0.0,
            )
        )

    return results


# --------------------------------------------------------------------------- #
# Step 7: Start stack
# --------------------------------------------------------------------------- #


def start_stack() -> Any:
    """Start the stack using run_up().

    Returns:
        UpResult from core/stack_up.py.
    """
    return run_up()
