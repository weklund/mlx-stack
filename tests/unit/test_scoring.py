"""Tests for the recommendation scoring engine (core/scoring.py).

Covers:
- Memory budget computation
- Benchmark resolution (direct, saved, bandwidth-ratio estimation)
- Log-scaled gen_tps normalization
- Quality, tool-calling, and memory-efficiency normalization
- Composite scoring with intent weights
- Memory budget filtering
- Tier assignment (standard, fast, longctx)
- Tier count based on memory size
- Deterministic scoring
- Edge cases (empty catalog, single model, all models too large)
"""

from __future__ import annotations

import math

import pytest

from mlx_stack.core.catalog import (
    BenchmarkResult,
    CatalogEntry,
)
from mlx_stack.core.hardware import HardwareProfile
from mlx_stack.core.scoring import (
    INTENT_WEIGHTS,
    TIER_FAST,
    TIER_LONGCTX,
    TIER_STANDARD,
    VALID_INTENTS,
    IntentWeights,
    RecommendationResult,
    ScoredModel,
    ScoringError,
    _normalize_gen_tps_log,
    _normalize_memory_efficiency,
    _normalize_quality,
    _normalize_tool_calling,
    _resolve_benchmark,
    assign_tiers,
    compute_memory_budget,
    recommend,
    score_and_filter,
    score_model,
)
from tests.factories import make_entry, make_profile

# --------------------------------------------------------------------------- #
# Fixtures — reusable test data
# --------------------------------------------------------------------------- #

# Default benchmarks for this test module's basic_entry and most calls that
# rely on the 3-profile convention (m4-pro-48, m4-max-128, m5-max-128).
_SCORING_BENCHMARKS: dict[str, BenchmarkResult] = {
    "m4-pro-48": BenchmarkResult(prompt_tps=95.0, gen_tps=52.0, memory_gb=5.5),
    "m4-max-128": BenchmarkResult(prompt_tps=140.0, gen_tps=77.0, memory_gb=5.5),
    "m5-max-128": BenchmarkResult(prompt_tps=155.0, gen_tps=85.0, memory_gb=5.5),
}


@pytest.fixture
def m4_max_128_profile() -> HardwareProfile:
    """M4 Max 128 GB profile — matches catalog benchmark key 'm4-max-128'."""
    return make_profile()


@pytest.fixture
def m4_pro_48_profile() -> HardwareProfile:
    """M4 Pro 48 GB profile — matches catalog benchmark key 'm4-pro-48'."""
    return make_profile(
        chip="Apple M4 Pro",
        gpu_cores=18,
        memory_gb=48,
        bandwidth_gbps=273.0,
    )


@pytest.fixture
def unknown_profile() -> HardwareProfile:
    """Unknown hardware profile — no catalog benchmark match."""
    return make_profile(
        chip="Apple M6",
        gpu_cores=60,
        memory_gb=256,
        bandwidth_gbps=1000.0,
        is_estimate=True,
    )


@pytest.fixture
def small_memory_profile() -> HardwareProfile:
    """Small memory profile (32 GB) for tier count tests."""
    return make_profile(
        chip="Apple M4 Pro",
        gpu_cores=18,
        memory_gb=32,
        bandwidth_gbps=273.0,
    )


@pytest.fixture
def basic_entry() -> CatalogEntry:
    """A basic catalog entry with standard benchmarks."""
    return make_entry(benchmarks=_SCORING_BENCHMARKS, tags=["balanced"])


@pytest.fixture
def sample_catalog() -> list[CatalogEntry]:
    """A representative catalog for testing scoring and tier assignment."""
    return [
        # High quality model
        make_entry(
            model_id="premium-72b",
            name="Premium 72B",
            params_b=72.0,
            quality_overall=91,
            quality_coding=90,
            quality_reasoning=92,
            quality_instruction=92,
            tool_calling=True,
            benchmarks={
                "m4-max-128": BenchmarkResult(prompt_tps=18.0, gen_tps=11.0, memory_gb=42.0),
                "m5-max-128": BenchmarkResult(prompt_tps=20.0, gen_tps=12.5, memory_gb=42.0),
            },
            tags=["premium", "quality"],
        ),
        # Fast small model
        make_entry(
            model_id="fast-0.8b",
            name="Fast 0.8B",
            params_b=0.8,
            quality_overall=42,
            quality_coding=38,
            quality_reasoning=35,
            quality_instruction=48,
            tool_calling=True,
            benchmarks={
                "m4-pro-48": BenchmarkResult(prompt_tps=320.0, gen_tps=125.0, memory_gb=0.8),
                "m4-max-128": BenchmarkResult(prompt_tps=480.0, gen_tps=185.0, memory_gb=0.8),
                "m5-max-128": BenchmarkResult(prompt_tps=510.0, gen_tps=195.0, memory_gb=0.8),
            },
            tags=["fast-inference"],
        ),
        # Mid-tier model
        make_entry(
            model_id="mid-8b",
            name="Mid 8B",
            params_b=8.0,
            quality_overall=68,
            quality_coding=65,
            quality_reasoning=62,
            quality_instruction=72,
            tool_calling=True,
            benchmarks={
                "m4-pro-48": BenchmarkResult(prompt_tps=95.0, gen_tps=52.0, memory_gb=5.5),
                "m4-max-128": BenchmarkResult(prompt_tps=140.0, gen_tps=77.0, memory_gb=5.5),
                "m5-max-128": BenchmarkResult(prompt_tps=155.0, gen_tps=85.0, memory_gb=5.5),
            },
            tags=["balanced", "agent-ready"],
        ),
        # Longctx model with mamba2-hybrid architecture
        make_entry(
            model_id="longctx-32b",
            name="LongCtx 32B",
            params_b=32.0,
            architecture="mamba2-hybrid",
            quality_overall=85,
            quality_coding=86,
            quality_reasoning=90,
            quality_instruction=80,
            tool_calling=False,
            benchmarks={
                "m4-pro-48": BenchmarkResult(prompt_tps=26.0, gen_tps=15.0, memory_gb=20.0),
                "m4-max-128": BenchmarkResult(prompt_tps=40.0, gen_tps=23.0, memory_gb=20.0),
                "m5-max-128": BenchmarkResult(prompt_tps=44.0, gen_tps=25.0, memory_gb=20.0),
            },
            tags=["reasoning", "long-context"],
        ),
        # Model without tool calling
        make_entry(
            model_id="no-tools-27b",
            name="No Tools 27B",
            params_b=27.0,
            quality_overall=80,
            quality_coding=76,
            quality_reasoning=78,
            quality_instruction=83,
            tool_calling=False,
            benchmarks={
                "m4-pro-48": BenchmarkResult(prompt_tps=30.0, gen_tps=18.0, memory_gb=17.0),
                "m4-max-128": BenchmarkResult(prompt_tps=45.0, gen_tps=26.0, memory_gb=17.0),
                "m5-max-128": BenchmarkResult(prompt_tps=50.0, gen_tps=29.0, memory_gb=17.0),
            },
            tags=["vision", "quality"],
        ),
        # Very large model that exceeds most budgets
        make_entry(
            model_id="huge-100b",
            name="Huge 100B",
            params_b=100.0,
            quality_overall=95,
            quality_coding=94,
            quality_reasoning=96,
            quality_instruction=95,
            tool_calling=True,
            benchmarks={
                "m4-max-128": BenchmarkResult(prompt_tps=8.0, gen_tps=5.0, memory_gb=60.0),
            },
            tags=["premium"],
        ),
    ]


# =========================================================================== #
# Tests: Memory budget computation
# =========================================================================== #


class TestComputeMemoryBudget:
    """Tests for compute_memory_budget."""

    def test_default_40_pct(self) -> None:
        assert compute_memory_budget(128) == pytest.approx(51.2)

    def test_default_40_pct_of_64(self) -> None:
        assert compute_memory_budget(64) == pytest.approx(25.6)

    def test_default_40_pct_of_32(self) -> None:
        assert compute_memory_budget(32) == pytest.approx(12.8)

    def test_custom_pct(self) -> None:
        assert compute_memory_budget(128, 60) == pytest.approx(76.8)

    def test_100_pct(self) -> None:
        assert compute_memory_budget(128, 100) == pytest.approx(128.0)

    def test_1_pct(self) -> None:
        assert compute_memory_budget(128, 1) == pytest.approx(1.28)


# =========================================================================== #
# Tests: Normalization functions
# =========================================================================== #


class TestNormalization:
    """Tests for normalization functions."""

    def test_log_gen_tps_zero(self) -> None:
        assert _normalize_gen_tps_log(0.0) == 0.0

    def test_log_gen_tps_negative(self) -> None:
        assert _normalize_gen_tps_log(-5.0) == 0.0

    def test_log_gen_tps_positive(self) -> None:
        result = _normalize_gen_tps_log(85.0)
        assert 0.0 < result < 1.0

    def test_log_gen_tps_high(self) -> None:
        result = _normalize_gen_tps_log(195.0)
        assert 0.9 < result <= 1.0

    def test_log_gen_tps_clamped_above_max_ref(self) -> None:
        """Regression: gen_tps above max_ref (200) must be clamped to 1.0.

        High-bandwidth unknown hardware can produce gen_tps > 200 via
        bandwidth-ratio estimation, which would otherwise push the
        normalized score above 1.0 and distort composite rankings.
        """
        result = _normalize_gen_tps_log(500.0)
        assert result == 1.0

    def test_log_gen_tps_clamped_extreme_bandwidth(self) -> None:
        """Regression: extreme gen_tps (e.g., 1000 tok/s) stays clamped."""
        result = _normalize_gen_tps_log(1000.0)
        assert result == 1.0

    def test_log_gen_tps_monotonic(self) -> None:
        """Faster models get higher scores but log scaling compresses differences."""
        slow = _normalize_gen_tps_log(12.0)
        medium = _normalize_gen_tps_log(52.0)
        fast = _normalize_gen_tps_log(185.0)
        assert slow < medium < fast

    def test_log_gen_tps_compression(self) -> None:
        """Log scaling compresses the difference between fast models."""
        v12 = _normalize_gen_tps_log(12.0)
        v52 = _normalize_gen_tps_log(52.0)
        v185 = _normalize_gen_tps_log(185.0)
        # The difference between 12 and 52 should be smaller in log than linear
        # but still significant
        ratio_slow_to_mid = v52 - v12
        ratio_mid_to_fast = v185 - v52
        # Log compression means the gap narrows at higher values
        assert ratio_mid_to_fast < ratio_slow_to_mid

    def test_quality_normalization(self) -> None:
        assert _normalize_quality(0) == 0.0
        assert _normalize_quality(50) == 0.5
        assert _normalize_quality(100) == 1.0

    def test_quality_clamped(self) -> None:
        assert _normalize_quality(-10) == 0.0
        assert _normalize_quality(150) == 1.0

    def test_tool_calling_true(self) -> None:
        assert _normalize_tool_calling(True) == 1.0

    def test_tool_calling_false(self) -> None:
        assert _normalize_tool_calling(False) == 0.0

    def test_memory_efficiency_half(self) -> None:
        # 5.5 GB out of 51.2 GB budget
        result = _normalize_memory_efficiency(5.5, 51.2)
        expected = (51.2 - 5.5) / 51.2
        assert result == pytest.approx(expected)

    def test_memory_efficiency_full_budget(self) -> None:
        result = _normalize_memory_efficiency(51.2, 51.2)
        assert result == pytest.approx(0.0)

    def test_memory_efficiency_zero_budget(self) -> None:
        assert _normalize_memory_efficiency(5.0, 0.0) == 0.0

    def test_memory_efficiency_zero_memory(self) -> None:
        assert _normalize_memory_efficiency(0.0, 50.0) == 0.0

    def test_memory_efficiency_exceeds_budget(self) -> None:
        # Clamped to 0 if memory > budget (shouldn't normally happen after filtering)
        assert _normalize_memory_efficiency(60.0, 50.0) == 0.0


# =========================================================================== #
# Tests: Intent weights
# =========================================================================== #


class TestIntentWeights:
    """Tests for intent weight configurations."""

    def test_balanced_weights_sum_to_one(self) -> None:
        w = INTENT_WEIGHTS["balanced"]
        total = w.speed + w.quality + w.tool_calling + w.memory_efficiency
        assert math.isclose(total, 1.0)

    def test_agent_fleet_weights_sum_to_one(self) -> None:
        w = INTENT_WEIGHTS["agent-fleet"]
        total = w.speed + w.quality + w.tool_calling + w.memory_efficiency
        assert math.isclose(total, 1.0)

    def test_balanced_quality_highest(self) -> None:
        w = INTENT_WEIGHTS["balanced"]
        assert w.quality > w.speed
        assert w.quality > w.tool_calling
        assert w.quality > w.memory_efficiency

    def test_agent_fleet_tool_calling_highest(self) -> None:
        w = INTENT_WEIGHTS["agent-fleet"]
        assert w.tool_calling > w.quality
        assert w.tool_calling > w.memory_efficiency

    def test_different_intents_have_different_weights(self) -> None:
        balanced = INTENT_WEIGHTS["balanced"]
        fleet = INTENT_WEIGHTS["agent-fleet"]
        assert balanced.quality != fleet.quality
        assert balanced.tool_calling != fleet.tool_calling

    def test_valid_intents(self) -> None:
        assert {"balanced", "agent-fleet"} == VALID_INTENTS

    def test_all_valid_intents_have_weights(self) -> None:
        for intent in VALID_INTENTS:
            assert intent in INTENT_WEIGHTS

    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match=r"must sum to 1\.0"):
            IntentWeights(speed=0.5, quality=0.5, tool_calling=0.5, memory_efficiency=0.5)


# =========================================================================== #
# Tests: Benchmark resolution
# =========================================================================== #


class TestBenchmarkResolution:
    """Tests for _resolve_benchmark."""

    def test_direct_match(
        self, basic_entry: CatalogEntry, m4_max_128_profile: HardwareProfile
    ) -> None:
        # Act
        gen_tps, memory_gb, is_estimated = _resolve_benchmark(basic_entry, m4_max_128_profile)

        # Assert
        assert gen_tps == 77.0
        assert memory_gb == 5.5
        assert is_estimated is False

    def test_direct_match_different_profile(
        self, basic_entry: CatalogEntry, m4_pro_48_profile: HardwareProfile
    ) -> None:
        # Act
        gen_tps, memory_gb, is_estimated = _resolve_benchmark(basic_entry, m4_pro_48_profile)

        # Assert
        assert gen_tps == 52.0
        assert memory_gb == 5.5
        assert is_estimated is False

    def test_saved_benchmarks_override(
        self, basic_entry: CatalogEntry, m4_max_128_profile: HardwareProfile
    ) -> None:
        # Arrange
        saved = {
            "test-model": {"gen_tps": 90.0, "memory_gb": 5.8},
        }

        # Act
        gen_tps, memory_gb, is_estimated = _resolve_benchmark(
            basic_entry, m4_max_128_profile, saved_benchmarks=saved
        )

        # Assert
        assert gen_tps == 90.0
        assert memory_gb == 5.8
        assert is_estimated is False

    def test_bandwidth_ratio_estimation(
        self, basic_entry: CatalogEntry, unknown_profile: HardwareProfile
    ) -> None:
        gen_tps, memory_gb, is_estimated = _resolve_benchmark(basic_entry, unknown_profile)
        # Unknown M6 with 1000 GB/s, reference m4-pro-48 has 273 GB/s, gen_tps=52.0
        # Expected: 52.0 * (1000 / 273) ≈ 190.5
        assert is_estimated is True
        assert gen_tps > 0
        assert memory_gb == 5.5  # Memory doesn't change with hardware

    def test_bandwidth_ratio_scales_correctly(self) -> None:
        """Bandwidth-ratio estimation should scale gen_tps proportionally."""
        entry = make_entry(
            benchmarks={
                "m4-max-128": BenchmarkResult(prompt_tps=100.0, gen_tps=50.0, memory_gb=5.0),
            },
        )
        # Profile with double the bandwidth of reference
        profile_2x = make_profile(
            chip="Apple M99",
            bandwidth_gbps=1092.0,  # 2x m4-max-128's 546
        )
        gen_tps, _, is_estimated = _resolve_benchmark(entry, profile_2x)
        assert is_estimated is True
        assert gen_tps == pytest.approx(100.0, rel=0.01)  # 50 * (1092/546)

    def test_malformed_saved_benchmark_falls_through(
        self, basic_entry: CatalogEntry, m4_max_128_profile: HardwareProfile
    ) -> None:
        """Regression: malformed saved benchmark values fall through gracefully.

        If saved benchmark JSON has non-numeric values (e.g., 'NaN-string'),
        _resolve_benchmark should log a warning and fall through to catalog
        data instead of raising a ValueError traceback.
        """
        saved = {
            "test-model": {"gen_tps": "not_a_number", "memory_gb": 5.5},
        }
        gen_tps, memory_gb, is_estimated = _resolve_benchmark(
            basic_entry, m4_max_128_profile, saved_benchmarks=saved
        )
        # Should fall through to direct catalog match for m4-max-128
        assert gen_tps == 77.0
        assert memory_gb == 5.5
        assert is_estimated is False

    def test_malformed_saved_benchmark_none_value(
        self, basic_entry: CatalogEntry, m4_max_128_profile: HardwareProfile
    ) -> None:
        """Regression: saved benchmark with None value falls through."""
        saved = {
            "test-model": {"gen_tps": None, "memory_gb": None},
        }
        gen_tps, memory_gb, is_estimated = _resolve_benchmark(
            basic_entry, m4_max_128_profile, saved_benchmarks=saved
        )
        # Should fall through to direct catalog match for m4-max-128
        assert gen_tps == 77.0
        assert memory_gb == 5.5
        assert is_estimated is False

    def test_model_without_benchmarks_raises(self, unknown_profile: HardwareProfile) -> None:
        entry = make_entry(benchmarks={})
        with pytest.raises(ScoringError, match="no benchmark data"):
            _resolve_benchmark(entry, unknown_profile)


# =========================================================================== #
# Tests: score_model
# =========================================================================== #


class TestScoreModel:
    """Tests for score_model."""

    def test_basic_scoring(
        self, basic_entry: CatalogEntry, m4_max_128_profile: HardwareProfile
    ) -> None:
        # Arrange
        weights = INTENT_WEIGHTS["balanced"]
        budget_gb = 51.2

        # Act
        scored = score_model(basic_entry, m4_max_128_profile, weights, budget_gb)

        # Assert
        assert isinstance(scored, ScoredModel)
        assert scored.entry.id == "test-model"
        assert 0 < scored.composite_score <= 1.0
        assert scored.gen_tps == 77.0
        assert scored.memory_gb == 5.5
        assert scored.is_estimated is False

    def test_all_scores_in_range(
        self, basic_entry: CatalogEntry, m4_max_128_profile: HardwareProfile
    ) -> None:
        weights = INTENT_WEIGHTS["balanced"]
        scored = score_model(basic_entry, m4_max_128_profile, weights, 51.2)

        assert 0.0 <= scored.speed_score <= 1.0
        assert 0.0 <= scored.quality_score <= 1.0
        assert 0.0 <= scored.tool_calling_score <= 1.0
        assert 0.0 <= scored.memory_efficiency_score <= 1.0
        assert 0.0 <= scored.composite_score <= 1.0

    def test_composite_is_weighted_sum(
        self, basic_entry: CatalogEntry, m4_max_128_profile: HardwareProfile
    ) -> None:
        weights = INTENT_WEIGHTS["balanced"]
        scored = score_model(basic_entry, m4_max_128_profile, weights, 51.2)

        expected = (
            weights.speed * scored.speed_score
            + weights.quality * scored.quality_score
            + weights.tool_calling * scored.tool_calling_score
            + weights.memory_efficiency * scored.memory_efficiency_score
        )
        assert scored.composite_score == pytest.approx(expected)

    def test_high_bandwidth_hardware_speed_score_clamped(self) -> None:
        """Regression: bandwidth-ratio estimation on high-bandwidth hardware.

        When hardware has much higher bandwidth than reference profiles,
        estimated gen_tps can exceed 200 (the normalization reference).
        The speed_score must still be clamped to [0, 1].
        """
        entry = make_entry(
            model_id="fast-model",
            benchmarks={
                "m4-max-128": BenchmarkResult(prompt_tps=480.0, gen_tps=185.0, memory_gb=0.8),
            },
        )
        # Very high bandwidth hardware — 4x the reference m4-max-128 (546 GB/s)
        high_bw_profile = make_profile(
            chip="Apple M99 Ultra",
            gpu_cores=128,
            memory_gb=512,
            bandwidth_gbps=2184.0,  # 4x m4-max-128
            is_estimate=True,
        )
        weights = INTENT_WEIGHTS["balanced"]
        scored = score_model(entry, high_bw_profile, weights, 200.0)

        # gen_tps should be estimated as ~740 (185 * 2184/546)
        assert scored.gen_tps > 200.0
        assert scored.is_estimated is True
        # Speed score must be clamped to 1.0
        assert scored.speed_score == 1.0
        # Composite score must remain in [0, 1]
        assert 0.0 <= scored.composite_score <= 1.0

    def test_tool_calling_model_scores_higher_for_agent_fleet(
        self, m4_max_128_profile: HardwareProfile
    ) -> None:
        """Models with tool_calling should score higher under agent-fleet intent."""
        with_tools = make_entry(model_id="with-tools", tool_calling=True)
        without_tools = make_entry(
            model_id="no-tools",
            tool_calling=False,
            tool_call_parser=None,
        )

        weights = INTENT_WEIGHTS["agent-fleet"]
        scored_with = score_model(with_tools, m4_max_128_profile, weights, 51.2)
        scored_without = score_model(without_tools, m4_max_128_profile, weights, 51.2)

        assert scored_with.composite_score > scored_without.composite_score


# =========================================================================== #
# Tests: score_and_filter
# =========================================================================== #


class TestScoreAndFilter:
    """Tests for score_and_filter."""

    def test_filters_by_memory_budget(
        self,
        sample_catalog: list[CatalogEntry],
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        """Models exceeding budget should be excluded."""
        budget_gb = 25.0  # Should exclude premium-72b (42GB) and huge-100b (60GB)
        scored = score_and_filter(sample_catalog, m4_max_128_profile, "balanced", budget_gb)
        model_ids = {m.entry.id for m in scored}
        assert "premium-72b" not in model_ids
        assert "huge-100b" not in model_ids
        # These should be included
        assert "fast-0.8b" in model_ids
        assert "mid-8b" in model_ids
        assert "longctx-32b" in model_ids

    def test_sorted_by_composite_score_descending(
        self,
        sample_catalog: list[CatalogEntry],
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        scored = score_and_filter(sample_catalog, m4_max_128_profile, "balanced", 51.2)
        for i in range(len(scored) - 1):
            assert scored[i].composite_score >= scored[i + 1].composite_score

    def test_invalid_intent_raises(
        self,
        sample_catalog: list[CatalogEntry],
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        with pytest.raises(ScoringError, match="Invalid intent"):
            score_and_filter(sample_catalog, m4_max_128_profile, "invalid", 51.2)

    def test_empty_catalog(self, m4_max_128_profile: HardwareProfile) -> None:
        scored = score_and_filter([], m4_max_128_profile, "balanced", 51.2)
        assert scored == []

    def test_all_models_exceed_budget(
        self,
        sample_catalog: list[CatalogEntry],
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        scored = score_and_filter(sample_catalog, m4_max_128_profile, "balanced", 0.1)
        assert scored == []

    def test_different_intents_produce_different_scores(
        self,
        sample_catalog: list[CatalogEntry],
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        balanced = score_and_filter(sample_catalog, m4_max_128_profile, "balanced", 51.2)
        fleet = score_and_filter(sample_catalog, m4_max_128_profile, "agent-fleet", 51.2)

        # Get scores for the same model under different intents
        balanced_scores = {m.entry.id: m.composite_score for m in balanced}
        fleet_scores = {m.entry.id: m.composite_score for m in fleet}

        # At least one model should have a different score
        common_ids = set(balanced_scores) & set(fleet_scores)
        assert any(not math.isclose(balanced_scores[mid], fleet_scores[mid]) for mid in common_ids)

    def test_deterministic_scoring(
        self,
        sample_catalog: list[CatalogEntry],
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        """Same inputs must always produce same outputs."""
        result1 = score_and_filter(sample_catalog, m4_max_128_profile, "balanced", 51.2)
        result2 = score_and_filter(sample_catalog, m4_max_128_profile, "balanced", 51.2)
        assert len(result1) == len(result2)
        for m1, m2 in zip(result1, result2, strict=False):
            assert m1.entry.id == m2.entry.id
            assert m1.composite_score == m2.composite_score

    def test_saved_benchmarks_used(
        self,
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        entry = make_entry()
        saved = {"test-model": {"gen_tps": 200.0, "memory_gb": 5.5}}
        scored = score_and_filter(
            [entry],
            m4_max_128_profile,
            "balanced",
            51.2,
            saved_benchmarks=saved,
        )
        assert len(scored) == 1
        assert scored[0].gen_tps == 200.0
        assert scored[0].is_estimated is False


# =========================================================================== #
# Tests: Tier assignment
# =========================================================================== #


class TestAssignTiers:
    """Tests for assign_tiers."""

    def test_standard_is_highest_composite_score(
        self,
        sample_catalog: list[CatalogEntry],
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        """Standard tier gets the model with the highest composite score.

        The composite score is intent-weighted, so standard tier reflects
        the intent-specific best model rather than the raw quality leader.
        """
        scored = score_and_filter(sample_catalog, m4_max_128_profile, "balanced", 51.2)
        tiers = assign_tiers(scored, 51.2)
        standard = next(t for t in tiers if t.tier == TIER_STANDARD)
        # Standard tier should be the model with the highest composite score
        max_composite = max(m.composite_score for m in scored)
        assert standard.model.composite_score == pytest.approx(max_composite)

    def test_fast_is_highest_gen_tps(
        self,
        sample_catalog: list[CatalogEntry],
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        scored = score_and_filter(sample_catalog, m4_max_128_profile, "balanced", 51.2)
        tiers = assign_tiers(scored, 51.2)
        fast = next(t for t in tiers if t.tier == TIER_FAST)
        # fast-0.8b should be the fast tier (185 gen_tps on m4-max-128)
        assert fast.model.entry.id == "fast-0.8b"

    def test_longctx_is_architecturally_diverse(
        self,
        sample_catalog: list[CatalogEntry],
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        scored = score_and_filter(sample_catalog, m4_max_128_profile, "balanced", 51.2)
        tiers = assign_tiers(scored, 51.2)
        longctx_tiers = [t for t in tiers if t.tier == TIER_LONGCTX]
        assert len(longctx_tiers) == 1
        assert longctx_tiers[0].model.entry.architecture == "mamba2-hybrid"

    def test_tiers_use_distinct_models(
        self,
        sample_catalog: list[CatalogEntry],
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        scored = score_and_filter(sample_catalog, m4_max_128_profile, "balanced", 51.2)
        tiers = assign_tiers(scored, 51.2)
        model_ids = [t.model.entry.id for t in tiers]
        assert len(model_ids) == len(set(model_ids))

    def test_small_memory_fewer_tiers(self, small_memory_profile: HardwareProfile) -> None:
        """Small memory systems (budget < 16 GB) should get 1-2 tiers."""
        # 32 GB * 40% = 12.8 GB budget — below 16 GB threshold
        catalog = [
            make_entry(
                model_id="small-3b",
                params_b=3.0,
                quality_overall=55,
                benchmarks={
                    "m4-pro-48": BenchmarkResult(prompt_tps=180.0, gen_tps=88.0, memory_gb=2.5),
                },
            ),
            make_entry(
                model_id="mid-8b",
                params_b=8.0,
                quality_overall=68,
                benchmarks={
                    "m4-pro-48": BenchmarkResult(prompt_tps=95.0, gen_tps=52.0, memory_gb=5.5),
                },
            ),
            make_entry(
                model_id="longctx-8b",
                architecture="mamba2-hybrid",
                params_b=8.0,
                quality_overall=70,
                tool_calling=False,
                benchmarks={
                    "m4-pro-48": BenchmarkResult(prompt_tps=90.0, gen_tps=50.0, memory_gb=5.5),
                },
            ),
        ]
        budget_gb = compute_memory_budget(32)  # 12.8 GB
        scored = score_and_filter(catalog, small_memory_profile, "balanced", budget_gb)
        tiers = assign_tiers(scored, budget_gb)
        # Should have at most 2 tiers (no longctx for budget < 16)
        tier_names = [t.tier for t in tiers]
        assert TIER_LONGCTX not in tier_names
        assert len(tiers) <= 2

    def test_large_memory_up_to_3_tiers(
        self,
        sample_catalog: list[CatalogEntry],
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        """Large memory systems should get up to 3 tiers."""
        budget_gb = compute_memory_budget(128)  # 51.2 GB
        scored = score_and_filter(sample_catalog, m4_max_128_profile, "balanced", budget_gb)
        tiers = assign_tiers(scored, budget_gb)
        assert len(tiers) == 3
        tier_names = [t.tier for t in tiers]
        assert TIER_STANDARD in tier_names
        assert TIER_FAST in tier_names
        assert TIER_LONGCTX in tier_names

    def test_empty_scored_models(self) -> None:
        tiers = assign_tiers([], 51.2)
        assert tiers == []

    def test_single_model(self) -> None:
        entry = make_entry()
        profile = make_profile()
        weights = INTENT_WEIGHTS["balanced"]
        scored = [score_model(entry, profile, weights, 51.2)]
        tiers = assign_tiers(scored, 51.2)
        assert len(tiers) == 1
        assert tiers[0].tier == TIER_STANDARD

    def test_no_longctx_architecture_available(
        self,
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        """If no mamba2-hybrid models exist, only 2 tiers are assigned."""
        catalog = [
            make_entry(model_id="high-q", quality_overall=91),
            make_entry(
                model_id="fast-model",
                quality_overall=42,
                benchmarks={
                    "m4-max-128": BenchmarkResult(prompt_tps=480.0, gen_tps=185.0, memory_gb=0.8),
                },
            ),
        ]
        scored = score_and_filter(catalog, m4_max_128_profile, "balanced", 51.2)
        tiers = assign_tiers(scored, 51.2)
        tier_names = [t.tier for t in tiers]
        assert TIER_LONGCTX not in tier_names
        assert len(tiers) == 2

    def test_tier_order(
        self,
        sample_catalog: list[CatalogEntry],
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        """Tiers should be ordered: standard, fast, longctx."""
        scored = score_and_filter(sample_catalog, m4_max_128_profile, "balanced", 51.2)
        tiers = assign_tiers(scored, 51.2)
        tier_names = [t.tier for t in tiers]
        assert tier_names[0] == TIER_STANDARD
        if len(tier_names) > 1:
            assert tier_names[1] == TIER_FAST
        if len(tier_names) > 2:
            assert tier_names[2] == TIER_LONGCTX

    def test_quant_is_int4(
        self,
        sample_catalog: list[CatalogEntry],
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        scored = score_and_filter(sample_catalog, m4_max_128_profile, "balanced", 51.2)
        tiers = assign_tiers(scored, 51.2)
        for tier in tiers:
            assert tier.quant == "int4"


# =========================================================================== #
# Tests: Full recommend()
# =========================================================================== #


class TestRecommend:
    """Tests for the recommend() entry point."""

    def test_basic_recommendation(
        self,
        sample_catalog: list[CatalogEntry],
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        result = recommend(sample_catalog, m4_max_128_profile)
        assert isinstance(result, RecommendationResult)
        assert result.intent == "balanced"
        assert result.memory_budget_gb == pytest.approx(51.2)
        assert len(result.tiers) > 0
        assert len(result.all_scored) > 0

    def test_budget_excludes_large_models(
        self,
        sample_catalog: list[CatalogEntry],
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        result = recommend(sample_catalog, m4_max_128_profile)
        for scored in result.all_scored:
            assert scored.memory_gb <= result.memory_budget_gb

    def test_budget_gb_override(
        self,
        sample_catalog: list[CatalogEntry],
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        result = recommend(sample_catalog, m4_max_128_profile, budget_gb_override=30.0)
        assert result.memory_budget_gb == 30.0
        for scored in result.all_scored:
            assert scored.memory_gb <= 30.0

    def test_budget_pct(
        self,
        sample_catalog: list[CatalogEntry],
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        result = recommend(sample_catalog, m4_max_128_profile, budget_pct=60)
        assert result.memory_budget_gb == pytest.approx(76.8)

    def test_agent_fleet_intent(
        self,
        sample_catalog: list[CatalogEntry],
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        result = recommend(sample_catalog, m4_max_128_profile, intent="agent-fleet")
        assert result.intent == "agent-fleet"

    def test_different_intents_produce_different_results(
        self,
        sample_catalog: list[CatalogEntry],
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        balanced = recommend(sample_catalog, m4_max_128_profile, intent="balanced")
        fleet = recommend(sample_catalog, m4_max_128_profile, intent="agent-fleet")

        # Get scoring for a model with tool_calling to compare
        balanced_scores = {m.entry.id: m.composite_score for m in balanced.all_scored}
        fleet_scores = {m.entry.id: m.composite_score for m in fleet.all_scored}

        # At least one model should have different scores
        common_ids = set(balanced_scores) & set(fleet_scores)
        assert any(not math.isclose(balanced_scores[mid], fleet_scores[mid]) for mid in common_ids)

    def test_invalid_intent_raises(
        self,
        sample_catalog: list[CatalogEntry],
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        with pytest.raises(ScoringError, match="Invalid intent"):
            recommend(sample_catalog, m4_max_128_profile, intent="coding")

    def test_hardware_profile_in_result(
        self,
        sample_catalog: list[CatalogEntry],
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        result = recommend(sample_catalog, m4_max_128_profile)
        assert result.hardware_profile == m4_max_128_profile

    def test_unknown_hardware_uses_estimation(
        self,
        sample_catalog: list[CatalogEntry],
        unknown_profile: HardwareProfile,
    ) -> None:
        result = recommend(sample_catalog, unknown_profile)
        # At least some models should have estimated benchmarks
        estimated_models = [m for m in result.all_scored if m.is_estimated]
        assert len(estimated_models) > 0

    def test_saved_benchmarks_override_catalog(
        self,
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        entry = make_entry()
        saved = {"test-model": {"gen_tps": 200.0, "memory_gb": 5.5}}
        result = recommend([entry], m4_max_128_profile, saved_benchmarks=saved)
        assert len(result.all_scored) == 1
        assert result.all_scored[0].gen_tps == 200.0
        assert result.all_scored[0].is_estimated is False

    def test_deterministic(
        self,
        sample_catalog: list[CatalogEntry],
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        """Same inputs always produce same outputs."""
        r1 = recommend(sample_catalog, m4_max_128_profile)
        r2 = recommend(sample_catalog, m4_max_128_profile)

        assert len(r1.tiers) == len(r2.tiers)
        for t1, t2 in zip(r1.tiers, r2.tiers, strict=False):
            assert t1.tier == t2.tier
            assert t1.model.entry.id == t2.model.entry.id
            assert t1.model.composite_score == t2.model.composite_score

        assert len(r1.all_scored) == len(r2.all_scored)
        for m1, m2 in zip(r1.all_scored, r2.all_scored, strict=False):
            assert m1.entry.id == m2.entry.id
            assert m1.composite_score == m2.composite_score

    def test_small_budget_gives_fewer_tiers(self, small_memory_profile: HardwareProfile) -> None:
        """On small memory, recommendation produces fewer tiers."""
        catalog = [
            make_entry(
                model_id="small-model",
                quality_overall=65,
                benchmarks={
                    "m4-pro-48": BenchmarkResult(prompt_tps=180.0, gen_tps=88.0, memory_gb=2.5),
                },
            ),
            make_entry(
                model_id="mid-model",
                quality_overall=70,
                benchmarks={
                    "m4-pro-48": BenchmarkResult(prompt_tps=95.0, gen_tps=52.0, memory_gb=5.5),
                },
            ),
        ]
        budget_gb = compute_memory_budget(32)  # 12.8 GB
        result = recommend(
            catalog,
            small_memory_profile,
            budget_gb_override=budget_gb,
        )
        # Should get at most 2 tiers for small memory
        assert len(result.tiers) <= 2
        tier_names = [t.tier for t in result.tiers]
        assert TIER_LONGCTX not in tier_names

    def test_zero_models_fit(
        self,
        sample_catalog: list[CatalogEntry],
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        """When no models fit the budget, result has empty tiers."""
        result = recommend(sample_catalog, m4_max_128_profile, budget_gb_override=0.1)
        assert result.tiers == []
        assert result.all_scored == []


# =========================================================================== #
# Tests: Real catalog integration
# =========================================================================== #


class TestWithRealCatalog:
    """Integration tests using the real shipped catalog data."""

    def test_real_catalog_recommendation(self) -> None:
        """Test scoring against real catalog entries."""
        from mlx_stack.core.catalog import load_catalog

        catalog = load_catalog()
        assert len(catalog) == 15

        profile = make_profile()  # M4 Max 128 GB
        result = recommend(catalog, profile)

        # Should have some models and tiers
        assert len(result.all_scored) > 0
        assert len(result.tiers) >= 1

        # All scored models should be within budget
        for scored in result.all_scored:
            assert scored.memory_gb <= result.memory_budget_gb

    def test_real_catalog_balanced_vs_agent_fleet(self) -> None:
        """Different intents should produce at least one different score."""
        from mlx_stack.core.catalog import load_catalog

        catalog = load_catalog()
        profile = make_profile()

        balanced = recommend(catalog, profile, intent="balanced")
        fleet = recommend(catalog, profile, intent="agent-fleet")

        balanced_scores = {m.entry.id: m.composite_score for m in balanced.all_scored}
        fleet_scores = {m.entry.id: m.composite_score for m in fleet.all_scored}

        common = set(balanced_scores) & set(fleet_scores)
        differences = [
            mid for mid in common if not math.isclose(balanced_scores[mid], fleet_scores[mid])
        ]
        assert len(differences) > 0, "Balanced and agent-fleet should differ"

    def test_real_catalog_m4_pro_48(self) -> None:
        """Verify recommendations for M4 Pro 48 GB."""
        from mlx_stack.core.catalog import load_catalog

        catalog = load_catalog()
        profile = make_profile(
            chip="Apple M4 Pro",
            gpu_cores=18,
            memory_gb=48,
            bandwidth_gbps=273.0,
        )
        result = recommend(catalog, profile)

        # Budget: 48 * 0.4 = 19.2 GB
        assert result.memory_budget_gb == pytest.approx(19.2)

        # No model should exceed budget
        for scored in result.all_scored:
            assert scored.memory_gb <= 19.2

        # Should have tiers
        assert len(result.tiers) >= 1

    def test_real_catalog_unknown_hardware(self) -> None:
        """Verify recommendations for unknown hardware use estimation."""
        from mlx_stack.core.catalog import load_catalog

        catalog = load_catalog()
        profile = make_profile(
            chip="Apple M6",
            memory_gb=256,
            bandwidth_gbps=1000.0,
            is_estimate=True,
        )
        result = recommend(catalog, profile)

        # All models should have estimated benchmarks
        estimated = [m for m in result.all_scored if m.is_estimated]
        assert len(estimated) == len(result.all_scored)

    def test_real_catalog_deterministic(self) -> None:
        """Same inputs produce same outputs with real catalog."""
        from mlx_stack.core.catalog import load_catalog

        catalog = load_catalog()
        profile = make_profile()

        r1 = recommend(catalog, profile)
        r2 = recommend(catalog, profile)

        assert len(r1.tiers) == len(r2.tiers)
        for t1, t2 in zip(r1.tiers, r2.tiers, strict=False):
            assert t1.tier == t2.tier
            assert t1.model.entry.id == t2.model.entry.id

    def test_real_catalog_intents_produce_different_tier_assignments(self) -> None:
        """Regression: balanced and agent-fleet produce different tier model assignments.

        This was broken when assign_tiers() used hardcoded quality.overall and
        gen_tps instead of the intent-weighted composite score. Now that
        assign_tiers() uses composite_score for the standard tier, different
        intents (which weight quality vs. tool_calling differently) should
        produce at least one different tier assignment on the same hardware.
        """
        from mlx_stack.core.catalog import load_catalog

        catalog = load_catalog()
        profile = make_profile()  # M4 Max 128 GB

        balanced = recommend(catalog, profile, intent="balanced")
        fleet = recommend(catalog, profile, intent="agent-fleet")

        balanced_tier_models = {t.tier: t.model.entry.id for t in balanced.tiers}
        fleet_tier_models = {t.tier: t.model.entry.id for t in fleet.tiers}

        # The two intents must produce at least one different tier assignment
        common_tiers = set(balanced_tier_models) & set(fleet_tier_models)
        has_difference = any(
            balanced_tier_models[tier] != fleet_tier_models[tier] for tier in common_tiers
        )
        assert has_difference, (
            f"balanced and agent-fleet should differ in at least one tier, "
            f"but both produced identical assignments: "
            f"balanced={balanced_tier_models}, fleet={fleet_tier_models}"
        )


# =========================================================================== #
# Tests: Intent differentiation in tier assignment
# =========================================================================== #


class TestIntentDifferentiation:
    """Regression tests: different intents produce different tier assignments.

    Prior to the fix, assign_tiers() used hardcoded quality.overall for the
    standard tier and gen_tps for the fast tier, ignoring the composite score
    entirely. This meant that balanced and agent-fleet produced identical
    tier assignments despite having very different weight configurations.
    """

    def test_different_intents_different_standard_tier(
        self,
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        """Standard tier should differ between balanced and agent-fleet.

        Construct a catalog where:
        - Model A: high quality, no tool_calling, moderate speed, same memory
          → favoured by balanced (quality weight 0.40)
        - Model B: moderate quality, has tool_calling, same speed, same memory
          → favoured by agent-fleet (tool_calling weight 0.35)

        Models are carefully balanced so that speed and memory_efficiency
        are similar, making the quality vs. tool_calling weight difference
        the decisive factor.
        """
        model_a = make_entry(
            model_id="quality-leader",
            name="Quality Leader",
            quality_overall=95,
            quality_coding=93,
            quality_reasoning=94,
            quality_instruction=95,
            tool_calling=False,
            tool_call_parser=None,
            benchmarks={
                "m4-max-128": BenchmarkResult(prompt_tps=40.0, gen_tps=25.0, memory_gb=20.0),
            },
        )
        model_b = make_entry(
            model_id="tool-caller",
            name="Tool Caller",
            quality_overall=50,
            quality_coding=48,
            quality_reasoning=45,
            quality_instruction=52,
            tool_calling=True,
            benchmarks={
                "m4-max-128": BenchmarkResult(prompt_tps=42.0, gen_tps=26.0, memory_gb=20.0),
            },
        )
        catalog = [model_a, model_b]
        budget_gb = 51.2

        balanced_scored = score_and_filter(catalog, m4_max_128_profile, "balanced", budget_gb)
        fleet_scored = score_and_filter(catalog, m4_max_128_profile, "agent-fleet", budget_gb)

        balanced_tiers = assign_tiers(balanced_scored, budget_gb)
        fleet_tiers = assign_tiers(fleet_scored, budget_gb)

        balanced_standard = next(t for t in balanced_tiers if t.tier == TIER_STANDARD)
        fleet_standard = next(t for t in fleet_tiers if t.tier == TIER_STANDARD)

        # Under balanced (quality=0.40, tool_calling=0.15), quality-leader
        # should win due to its much higher quality score.
        # Under agent-fleet (quality=0.20, tool_calling=0.35), tool-caller
        # should win due to the large tool_calling bonus outweighing quality.
        assert balanced_standard.model.entry.id != fleet_standard.model.entry.id, (
            "Standard tier should differ between balanced and agent-fleet intents"
        )
        assert balanced_standard.model.entry.id == "quality-leader"
        assert fleet_standard.model.entry.id == "tool-caller"

    def test_standard_tier_uses_composite_not_raw_quality(
        self,
        m4_max_128_profile: HardwareProfile,
    ) -> None:
        """Standard tier should pick the highest composite score, not raw quality."""
        # Model with highest quality but terrible speed and efficiency
        high_q = make_entry(
            model_id="slow-quality",
            quality_overall=95,
            benchmarks={
                "m4-max-128": BenchmarkResult(prompt_tps=5.0, gen_tps=3.0, memory_gb=48.0),
            },
        )
        # Model with moderate quality but good speed and efficiency
        balanced_model = make_entry(
            model_id="balanced-model",
            quality_overall=70,
            benchmarks={
                "m4-max-128": BenchmarkResult(prompt_tps=120.0, gen_tps=70.0, memory_gb=5.0),
            },
        )

        scored = score_and_filter([high_q, balanced_model], m4_max_128_profile, "balanced", 51.2)
        tiers = assign_tiers(scored, 51.2)
        standard = next(t for t in tiers if t.tier == TIER_STANDARD)

        # The balanced-model should win because its composite score is higher
        # even though slow-quality has higher raw quality
        assert standard.model.entry.id == "balanced-model"


# =========================================================================== #
# Gated model filtering tests
# =========================================================================== #


class TestExcludeGated:
    """Tests for exclude_gated parameter in score_and_filter."""

    def test_exclude_gated_filters_gated_models(self, m4_max_128_profile: HardwareProfile) -> None:
        """Gated models are excluded when exclude_gated=True."""
        open_model = make_entry(model_id="open-model", name="Open Model")
        gated_model = make_entry(model_id="gated-model", name="Gated Model", gated=True)

        scored = score_and_filter(
            [open_model, gated_model],
            m4_max_128_profile,
            "balanced",
            51.2,
            exclude_gated=True,
        )
        scored_ids = {m.entry.id for m in scored}
        assert "open-model" in scored_ids
        assert "gated-model" not in scored_ids

    def test_exclude_gated_false_includes_all(self, m4_max_128_profile: HardwareProfile) -> None:
        """All models included when exclude_gated=False (default)."""
        open_model = make_entry(model_id="open-model", name="Open Model")
        gated_model = make_entry(model_id="gated-model", name="Gated Model", gated=True)

        scored = score_and_filter(
            [open_model, gated_model],
            m4_max_128_profile,
            "balanced",
            51.2,
            exclude_gated=False,
        )
        scored_ids = {m.entry.id for m in scored}
        assert "open-model" in scored_ids
        assert "gated-model" in scored_ids

    def test_recommend_exclude_gated(self, m4_max_128_profile: HardwareProfile) -> None:
        """Gated models excluded from tier assignments via recommend()."""
        open_model = make_entry(model_id="open-model", name="Open Model")
        gated_model = make_entry(
            model_id="gated-model",
            name="Gated Model",
            quality_overall=99,
            gated=True,
        )

        result = recommend(
            [open_model, gated_model],
            m4_max_128_profile,
            exclude_gated=True,
        )
        tier_ids = {t.model.entry.id for t in result.tiers}
        assert "gated-model" not in tier_ids
        assert "open-model" in tier_ids
