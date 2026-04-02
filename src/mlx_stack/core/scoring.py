"""Recommendation scoring engine for mlx-stack.

Scores and ranks catalog models for a given hardware profile and intent.
Supports 2 intents (balanced, agent-fleet) with weighted composite scoring
across speed, quality, tool_calling, and memory_efficiency dimensions.

Uses log-scaled gen_tps normalization to prevent extreme speed models from
dominating composite scores. Filters models by memory budget (default 40%
of unified memory). Assigns models to tiers (fast, standard, longctx).

Handles bandwidth-ratio estimation for unknown hardware profiles where
catalog benchmark data is not available.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from mlx_stack.core.catalog import BenchmarkResult, CatalogEntry
from mlx_stack.core.hardware import HardwareProfile

logger = logging.getLogger("mlx_stack")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Default memory budget percentage of unified memory
DEFAULT_MEMORY_BUDGET_PCT = 40

# Valid intents
VALID_INTENTS: set[str] = {"balanced", "agent-fleet"}

# Tier names
TIER_FAST = "fast"
TIER_STANDARD = "standard"
TIER_LONGCTX = "longctx"

# Architecture that qualifies for longctx tier
_LONGCTX_ARCHITECTURES: set[str] = {"mamba2-hybrid"}

# Default quantization for memory lookup
_DEFAULT_QUANT = "int4"

# Reference hardware profiles used in catalog benchmarks and their bandwidths.
# Used for bandwidth-ratio estimation when a model has no benchmark data
# for the user's hardware.
_REFERENCE_PROFILES: dict[str, float] = {
    "m4-pro-48": 273.0,
    "m4-max-128": 546.0,
    "m5-max-128": 546.0,
}


# --------------------------------------------------------------------------- #
# Intent weight configurations
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IntentWeights:
    """Weight configuration for a recommendation intent.

    Weights should sum to 1.0 for normalized scoring.
    """

    speed: float
    quality: float
    tool_calling: float
    memory_efficiency: float

    def __post_init__(self) -> None:
        total = self.speed + self.quality + self.tool_calling + self.memory_efficiency
        if not math.isclose(total, 1.0, abs_tol=0.001):
            msg = f"Intent weights must sum to 1.0, got {total}"
            raise ValueError(msg)


# Weight configurations for each intent
INTENT_WEIGHTS: dict[str, IntentWeights] = {
    "balanced": IntentWeights(
        speed=0.25,
        quality=0.40,
        tool_calling=0.15,
        memory_efficiency=0.20,
    ),
    "agent-fleet": IntentWeights(
        speed=0.30,
        quality=0.20,
        tool_calling=0.35,
        memory_efficiency=0.15,
    ),
}


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class ScoringError(Exception):
    """Raised when scoring fails."""


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ScoredModel:
    """A catalog model with computed scores for a given hardware/intent."""

    entry: CatalogEntry
    composite_score: float
    speed_score: float
    quality_score: float
    tool_calling_score: float
    memory_efficiency_score: float
    gen_tps: float
    memory_gb: float
    is_estimated: bool  # True if gen_tps was bandwidth-ratio estimated


@dataclass(frozen=True)
class TierAssignment:
    """A model assigned to a specific tier."""

    tier: str
    model: ScoredModel
    quant: str


@dataclass(frozen=True)
class RecommendationResult:
    """The full recommendation result with tier assignments and scored models."""

    tiers: list[TierAssignment]
    all_scored: list[ScoredModel]
    memory_budget_gb: float
    intent: str
    hardware_profile: HardwareProfile


# --------------------------------------------------------------------------- #
# Memory budget
# --------------------------------------------------------------------------- #


def compute_memory_budget(memory_gb: int, budget_pct: int = DEFAULT_MEMORY_BUDGET_PCT) -> float:
    """Compute the memory budget in GB from total memory and percentage.

    Args:
        memory_gb: Total unified memory in GB.
        budget_pct: Budget percentage (1-100).

    Returns:
        Memory budget in GB.
    """
    return memory_gb * (budget_pct / 100.0)


# --------------------------------------------------------------------------- #
# Benchmark resolution
# --------------------------------------------------------------------------- #


def _resolve_benchmark(
    entry: CatalogEntry,
    profile: HardwareProfile,
    quant: str = _DEFAULT_QUANT,
    saved_benchmarks: dict[str, Any] | None = None,
) -> tuple[float, float, bool]:
    """Resolve gen_tps and memory_gb for a model on the given hardware.

    Tries in order:
    1. Saved benchmark data (from bench --save)
    2. Direct catalog benchmark match for profile_id
    3. Bandwidth-ratio estimation from a reference profile

    Args:
        entry: The catalog entry.
        profile: The hardware profile.
        quant: Quantization level (for saved benchmarks key).
        saved_benchmarks: Optional dict from bench --save JSON files,
            keyed by model_id with gen_tps, prompt_tps, memory_gb.

    Returns:
        Tuple of (gen_tps, memory_gb, is_estimated).
    """
    # 1. Check saved benchmarks
    if saved_benchmarks and entry.id in saved_benchmarks:
        saved = saved_benchmarks[entry.id]
        try:
            return (
                float(saved.get("gen_tps", 0.0)),
                float(saved.get("memory_gb", 0.0)),
                False,
            )
        except (ValueError, TypeError):
            # Malformed saved benchmark data — fall through to catalog lookup
            logger.warning(
                "Ignoring malformed saved benchmark for model '%s': "
                "invalid numeric values",
                entry.id,
            )

    profile_id = profile.profile_id

    # 2. Direct match in catalog benchmarks
    if profile_id in entry.benchmarks:
        bench = entry.benchmarks[profile_id]
        return bench.gen_tps, bench.memory_gb, False

    # 3. Bandwidth-ratio estimation from a reference profile
    return _estimate_from_bandwidth_ratio(entry, profile)


def _estimate_from_bandwidth_ratio(
    entry: CatalogEntry,
    profile: HardwareProfile,
) -> tuple[float, float, bool]:
    """Estimate gen_tps using bandwidth ratio from a reference profile.

    Picks the best available reference profile from the catalog benchmarks
    and scales gen_tps by the ratio of actual bandwidth to reference bandwidth.

    Args:
        entry: The catalog entry with benchmark data for reference profiles.
        profile: The user's hardware profile.

    Returns:
        Tuple of (estimated_gen_tps, memory_gb, True).
        memory_gb is taken directly from the reference (memory usage is
        hardware-independent for the same quant).

    Raises:
        ScoringError: If no reference benchmark data is available for this model.
    """
    if not entry.benchmarks:
        msg = f"Model '{entry.id}' has no benchmark data for estimation"
        raise ScoringError(msg)

    # Pick the reference profile: prefer one with known bandwidth in our table,
    # otherwise use the first available
    ref_profile_id: str | None = None
    ref_bench: BenchmarkResult | None = None

    for pid, bench in entry.benchmarks.items():
        if pid in _REFERENCE_PROFILES:
            ref_profile_id = pid
            ref_bench = bench
            break

    if ref_profile_id is None or ref_bench is None:
        # Fall back to the first available benchmark
        ref_profile_id = next(iter(entry.benchmarks))
        ref_bench = entry.benchmarks[ref_profile_id]

    # Get reference bandwidth
    ref_bandwidth = _REFERENCE_PROFILES.get(ref_profile_id)
    if ref_bandwidth is None:
        # Unknown reference — use the estimate_bandwidth heuristic based on
        # memory size. For the reference, we don't know the memory, so assume
        # a middle-ground bandwidth.
        ref_bandwidth = 400.0  # Conservative middle estimate

    # Scale gen_tps by bandwidth ratio
    actual_bandwidth = profile.bandwidth_gbps
    ratio = actual_bandwidth / ref_bandwidth if ref_bandwidth > 0 else 1.0
    estimated_gen_tps = ref_bench.gen_tps * ratio

    # Memory usage is the same regardless of hardware
    return estimated_gen_tps, ref_bench.memory_gb, True


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def _normalize_gen_tps_log(gen_tps: float) -> float:
    """Normalize gen_tps using log scaling.

    Log scaling prevents very fast models (e.g., 0.8B at 195 tps) from
    having disproportionately high speed scores compared to larger models
    (e.g., 72B at 12 tps).

    Maps gen_tps to a 0-1 range using log(1 + gen_tps) / log(1 + max_reference).
    The max reference is set to 200 tps (reasonable upper bound for current hardware).

    Args:
        gen_tps: Generation tokens per second.

    Returns:
        Normalized score in [0, 1].
    """
    max_ref = 200.0  # Upper bound reference for normalization
    if gen_tps <= 0:
        return 0.0
    score = math.log(1.0 + gen_tps) / math.log(1.0 + max_ref)
    return min(score, 1.0)


def _normalize_quality(quality_overall: int) -> float:
    """Normalize quality score to [0, 1].

    Quality scores are on a 0-100 scale.

    Args:
        quality_overall: Overall quality score (0-100).

    Returns:
        Normalized score in [0, 1].
    """
    return min(max(quality_overall / 100.0, 0.0), 1.0)


def _normalize_tool_calling(has_tool_calling: bool) -> float:
    """Normalize tool calling capability to a score.

    Args:
        has_tool_calling: Whether the model supports tool calling.

    Returns:
        1.0 if tool calling is supported, 0.0 otherwise.
    """
    return 1.0 if has_tool_calling else 0.0


def _normalize_memory_efficiency(memory_gb: float, budget_gb: float) -> float:
    """Normalize memory efficiency to [0, 1].

    Models using less of the budget score higher, encouraging efficient use
    of available memory. Uses (budget - memory) / budget ratio.

    Args:
        memory_gb: Model memory usage in GB.
        budget_gb: Available memory budget in GB.

    Returns:
        Normalized score in [0, 1]. Higher means more memory-efficient.
    """
    if budget_gb <= 0 or memory_gb <= 0:
        return 0.0
    efficiency = (budget_gb - memory_gb) / budget_gb
    return min(max(efficiency, 0.0), 1.0)


def score_model(
    entry: CatalogEntry,
    profile: HardwareProfile,
    weights: IntentWeights,
    budget_gb: float,
    quant: str = _DEFAULT_QUANT,
    saved_benchmarks: dict[str, Any] | None = None,
) -> ScoredModel:
    """Score a single model for the given hardware profile and intent weights.

    Args:
        entry: The catalog entry to score.
        profile: The hardware profile.
        weights: The intent weight configuration.
        budget_gb: Memory budget in GB.
        quant: Quantization level.
        saved_benchmarks: Optional saved benchmark data.

    Returns:
        A ScoredModel with all computed scores.

    Raises:
        ScoringError: If benchmark data cannot be resolved.
    """
    gen_tps, memory_gb, is_estimated = _resolve_benchmark(
        entry, profile, quant, saved_benchmarks
    )

    speed_score = _normalize_gen_tps_log(gen_tps)
    quality_score = _normalize_quality(entry.quality.overall)
    tool_calling_score = _normalize_tool_calling(entry.capabilities.tool_calling)
    memory_eff_score = _normalize_memory_efficiency(memory_gb, budget_gb)

    composite = (
        weights.speed * speed_score
        + weights.quality * quality_score
        + weights.tool_calling * tool_calling_score
        + weights.memory_efficiency * memory_eff_score
    )

    return ScoredModel(
        entry=entry,
        composite_score=composite,
        speed_score=speed_score,
        quality_score=quality_score,
        tool_calling_score=tool_calling_score,
        memory_efficiency_score=memory_eff_score,
        gen_tps=gen_tps,
        memory_gb=memory_gb,
        is_estimated=is_estimated,
    )


def score_and_filter(
    catalog: list[CatalogEntry],
    profile: HardwareProfile,
    intent: str,
    budget_gb: float,
    quant: str = _DEFAULT_QUANT,
    saved_benchmarks: dict[str, Any] | None = None,
    exclude_gated: bool = False,
) -> list[ScoredModel]:
    """Score all catalog models and filter by memory budget.

    Models whose memory_gb exceeds the budget are excluded.
    Results are sorted by composite_score descending.

    Args:
        catalog: List of catalog entries to score.
        profile: The hardware profile.
        intent: The recommendation intent (balanced or agent-fleet).
        budget_gb: Memory budget in GB.
        quant: Quantization level.
        saved_benchmarks: Optional saved benchmark data.
        exclude_gated: If True, exclude gated models that require
            HuggingFace authentication.

    Returns:
        List of ScoredModel instances within budget, sorted by score descending.

    Raises:
        ScoringError: If the intent is invalid.
    """
    if intent not in VALID_INTENTS:
        valid = ", ".join(sorted(VALID_INTENTS))
        msg = f"Invalid intent '{intent}'. Valid intents: {valid}"
        raise ScoringError(msg)

    weights = INTENT_WEIGHTS[intent]
    scored: list[ScoredModel] = []

    for entry in catalog:
        if exclude_gated and entry.gated:
            continue

        try:
            model = score_model(entry, profile, weights, budget_gb, quant, saved_benchmarks)
        except ScoringError:
            # Skip models that can't be scored (no benchmark data at all)
            continue

        # Filter by memory budget
        if model.memory_gb <= budget_gb:
            scored.append(model)

    # Sort by composite score descending (deterministic: break ties by model id)
    scored.sort(key=lambda m: (-m.composite_score, m.entry.id))

    return scored


# --------------------------------------------------------------------------- #
# Tier assignment
# --------------------------------------------------------------------------- #


def assign_tiers(
    scored_models: list[ScoredModel],
    memory_budget_gb: float,
) -> list[TierAssignment]:
    """Assign scored models to tiers: standard, fast, longctx.

    Assignment rules:
    - standard: highest intent-weighted composite score from budget-eligible
      candidates (varies by intent — agent-fleet favours tool_calling,
      balanced favours quality)
    - fast: highest gen_tps model that is different from standard
    - longctx: architecturally diverse model (e.g., mamba2-hybrid) if available
      and different from standard and fast

    Small-memory systems (budget < 16 GB) get 1-2 tiers.
    Large-memory systems get up to 3 tiers.

    Args:
        scored_models: Pre-filtered, scored models within budget.
            Must already be scored with intent-specific weights.
        memory_budget_gb: The memory budget in GB (used for tier count decisions).

    Returns:
        List of TierAssignment instances, ordered: standard, fast, longctx.
    """
    if not scored_models:
        return []

    assignments: list[TierAssignment] = []
    used_model_ids: set[str] = set()

    # --- Standard tier: highest intent-weighted composite score ---
    # The composite_score already reflects intent weights (quality-heavy for
    # balanced, tool_calling-heavy for agent-fleet), so sorting by it
    # naturally produces different tier assignments per intent.
    standard_candidates = sorted(
        scored_models,
        key=lambda m: (-m.composite_score, m.entry.id),
    )
    standard_model = standard_candidates[0]
    assignments.append(TierAssignment(
        tier=TIER_STANDARD,
        model=standard_model,
        quant=_DEFAULT_QUANT,
    ))
    used_model_ids.add(standard_model.entry.id)

    # --- Fast tier: highest gen_tps, different from standard ---
    fast_candidates = sorted(
        [m for m in scored_models if m.entry.id not in used_model_ids],
        key=lambda m: (-m.gen_tps, m.entry.id),
    )
    if fast_candidates:
        fast_model = fast_candidates[0]
        assignments.append(TierAssignment(
            tier=TIER_FAST,
            model=fast_model,
            quant=_DEFAULT_QUANT,
        ))
        used_model_ids.add(fast_model.entry.id)

    # --- Longctx tier: architecturally diverse, only for larger budgets ---
    # Only assign longctx if budget >= 16 GB (small systems get 1-2 tiers)
    if memory_budget_gb >= 16.0:
        longctx_candidates = sorted(
            [
                m
                for m in scored_models
                if m.entry.id not in used_model_ids
                and m.entry.architecture in _LONGCTX_ARCHITECTURES
            ],
            key=lambda m: (-m.composite_score, m.entry.id),
        )
        if longctx_candidates:
            longctx_model = longctx_candidates[0]
            assignments.append(TierAssignment(
                tier=TIER_LONGCTX,
                model=longctx_model,
                quant=_DEFAULT_QUANT,
            ))
            used_model_ids.add(longctx_model.entry.id)

    return assignments


# --------------------------------------------------------------------------- #
# Main recommendation entry point
# --------------------------------------------------------------------------- #


def recommend(
    catalog: list[CatalogEntry],
    profile: HardwareProfile,
    intent: str = "balanced",
    budget_pct: int = DEFAULT_MEMORY_BUDGET_PCT,
    budget_gb_override: float | None = None,
    quant: str = _DEFAULT_QUANT,
    saved_benchmarks: dict[str, Any] | None = None,
    exclude_gated: bool = False,
) -> RecommendationResult:
    """Generate a recommendation for the given hardware and intent.

    This is the main entry point for the scoring engine. It:
    1. Computes the memory budget
    2. Scores and filters all catalog models
    3. Assigns models to tiers

    Args:
        catalog: Full catalog of model entries.
        profile: The hardware profile.
        intent: Recommendation intent (balanced or agent-fleet).
        budget_pct: Memory budget percentage (1-100). Ignored if
            budget_gb_override is set.
        budget_gb_override: Explicit memory budget in GB, overriding
            percentage-based calculation.
        quant: Default quantization level.
        saved_benchmarks: Optional saved benchmark data from bench --save.
        exclude_gated: If True, exclude gated models from recommendations.

    Returns:
        A RecommendationResult with tier assignments and all scored models.

    Raises:
        ScoringError: If the intent is invalid or scoring fails.
    """
    if intent not in VALID_INTENTS:
        valid = ", ".join(sorted(VALID_INTENTS))
        msg = f"Invalid intent '{intent}'. Valid intents: {valid}"
        raise ScoringError(msg)

    # Compute memory budget
    if budget_gb_override is not None:
        budget_gb = budget_gb_override
    else:
        budget_gb = compute_memory_budget(profile.memory_gb, budget_pct)

    # Score and filter
    scored = score_and_filter(
        catalog, profile, intent, budget_gb, quant, saved_benchmarks,
        exclude_gated=exclude_gated,
    )

    # Assign tiers
    tiers = assign_tiers(scored, budget_gb)

    return RecommendationResult(
        tiers=tiers,
        all_scored=scored,
        memory_budget_gb=budget_gb,
        intent=intent,
        hardware_profile=profile,
    )
