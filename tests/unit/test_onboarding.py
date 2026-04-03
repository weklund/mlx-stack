"""Behavioral tests for the onboarding orchestration module.

Tests assert outcomes — which models are selected, how tiers are
assigned, what scores mean — not which internal functions are called.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from mlx_stack.core.discovery import DiscoveredModel
from mlx_stack.core.onboarding import (
    ScoredDiscoveredModel,
    TierMapping,
    assign_tiers,
    generate_config,
    score_and_filter,
    select_defaults,
)

# --------------------------------------------------------------------------- #
# Fixtures: realistic model data
# --------------------------------------------------------------------------- #


def _model(
    name: str = "TestModel-8B",
    hf_repo: str = "mlx-community/TestModel-8B-4bit",
    params_b: float = 8.0,
    quant: str = "int4",
    downloads: int = 10_000,
    gen_tps: float | None = 50.0,
    memory_gb: float | None = 5.0,
    quality_overall: float | None = 0.80,
    tool_calling: bool = False,
    thinking: bool = False,
    has_benchmark: bool = True,
) -> DiscoveredModel:
    return DiscoveredModel(
        hf_repo=hf_repo,
        display_name=name,
        params_b=params_b,
        quant=quant,
        downloads=downloads,
        gen_tps=gen_tps,
        memory_gb=memory_gb,
        quality_overall=quality_overall,
        tool_calling=tool_calling,
        thinking=thinking,
        has_benchmark=has_benchmark,
    )


# A realistic set of models for testing scoring and selection
SMALL_FAST = _model(
    "SmallFast-2B",
    "mlx-community/SmallFast-2B-4bit",
    params_b=2.0,
    gen_tps=120.0,
    memory_gb=1.5,
    quality_overall=0.60,
    tool_calling=True,
)
MEDIUM_QUALITY = _model(
    "MedQuality-9B",
    "mlx-community/MedQuality-9B-4bit",
    params_b=9.0,
    gen_tps=50.0,
    memory_gb=5.5,
    quality_overall=0.95,
    tool_calling=True,
)
LARGE_SLOW = _model(
    "LargeSlow-27B",
    "mlx-community/LargeSlow-27B-4bit",
    params_b=27.0,
    gen_tps=20.0,
    memory_gb=16.0,
    quality_overall=0.98,
)
TINY_MODEL = _model(
    "Tiny-0.5B",
    "mlx-community/Tiny-0.5B-4bit",
    params_b=0.5,
    gen_tps=180.0,
    memory_gb=0.5,
    quality_overall=0.40,
)
NO_BENCHMARK = _model(
    "Unknown-7B",
    "mlx-community/Unknown-7B-4bit",
    params_b=7.0,
    gen_tps=None,
    memory_gb=4.2,
    quality_overall=None,
    has_benchmark=False,
)

ALL_MODELS = [SMALL_FAST, MEDIUM_QUALITY, LARGE_SLOW, TINY_MODEL, NO_BENCHMARK]


# --------------------------------------------------------------------------- #
# Scoring and filtering
# --------------------------------------------------------------------------- #


class TestScoreAndFilter:
    """score_and_filter produces correct scoring behavior."""

    def test_models_exceeding_budget_are_excluded(self) -> None:
        """Models that use more memory than the budget don't appear."""
        budget_gb = 6.0  # excludes LARGE_SLOW (16 GB)

        scored = score_and_filter(ALL_MODELS, "balanced", budget_gb)
        repos = {s.model.hf_repo for s in scored}

        assert LARGE_SLOW.hf_repo not in repos
        assert MEDIUM_QUALITY.hf_repo in repos  # 5.5 GB fits in 6 GB

    def test_all_models_within_budget_are_included(self) -> None:
        """Every model under the budget appears in results."""
        budget_gb = 20.0

        scored = score_and_filter(ALL_MODELS, "balanced", budget_gb)

        assert len(scored) == len(ALL_MODELS)

    def test_results_sorted_by_composite_score_descending(self) -> None:
        """The first model has the highest composite score."""
        scored = score_and_filter(ALL_MODELS, "balanced", budget_gb=20.0)

        scores = [s.composite_score for s in scored]
        assert scores == sorted(scores, reverse=True)

    def test_high_quality_model_ranks_higher_in_balanced(self) -> None:
        """In balanced intent, a high-quality model outranks a fast-but-low-quality one."""
        scored = score_and_filter(
            [SMALL_FAST, MEDIUM_QUALITY],
            "balanced",
            budget_gb=20.0,
        )

        # MEDIUM_QUALITY has 0.95 quality vs SMALL_FAST's 0.60
        # balanced weights quality at 0.40 — should rank higher
        assert scored[0].model.hf_repo == MEDIUM_QUALITY.hf_repo

    def test_tool_calling_model_ranks_higher_in_agent_fleet(self) -> None:
        """In agent-fleet intent, tool calling capability boosts ranking."""
        no_tools = _model(
            "NoTools-8B",
            "mlx-community/NoTools-8B-4bit",
            gen_tps=50.0,
            memory_gb=5.0,
            quality_overall=0.90,
            tool_calling=False,
        )
        with_tools = _model(
            "WithTools-8B",
            "mlx-community/WithTools-8B-4bit",
            gen_tps=50.0,
            memory_gb=5.0,
            quality_overall=0.90,
            tool_calling=True,
        )

        scored = score_and_filter([no_tools, with_tools], "agent-fleet", budget_gb=20.0)

        # agent-fleet weights tool_calling at 0.35 — should win
        assert scored[0].model.tool_calling is True

    def test_models_without_benchmarks_get_neutral_quality(self) -> None:
        """Models with no quality data get a neutral (0.5) quality score, not 0."""
        scored = score_and_filter([NO_BENCHMARK], "balanced", budget_gb=20.0)

        assert len(scored) == 1
        assert scored[0].quality_score == 0.5

    def test_empty_input_returns_empty(self) -> None:
        assert score_and_filter([], "balanced", budget_gb=20.0) == []


# --------------------------------------------------------------------------- #
# Default selection
# --------------------------------------------------------------------------- #


class TestSelectDefaults:
    """select_defaults picks models that fit within budget."""

    def test_selects_up_to_two_models_within_budget(self) -> None:
        """Standard + fast are selected if both fit."""
        scored = score_and_filter(
            [SMALL_FAST, MEDIUM_QUALITY, TINY_MODEL],
            "balanced",
            budget_gb=10.0,
        )

        result = select_defaults(scored, budget_gb=10.0)

        recommended = [s for s in result if s.is_recommended]
        assert len(recommended) == 2

    def test_total_memory_within_budget(self) -> None:
        """Combined memory of selected models stays within budget."""
        scored = score_and_filter(ALL_MODELS, "balanced", budget_gb=8.0)

        result = select_defaults(scored, budget_gb=8.0)

        recommended = [s for s in result if s.is_recommended]
        total_mem = sum(s.model.memory_gb or 0.0 for s in recommended)
        assert total_mem <= 8.0

    def test_single_model_when_budget_tight(self) -> None:
        """Only one model selected if budget only fits one."""
        scored = score_and_filter(
            [MEDIUM_QUALITY, SMALL_FAST],
            "balanced",
            budget_gb=6.0,
        )

        result = select_defaults(scored, budget_gb=6.0)

        recommended = [s for s in result if s.is_recommended]
        # 5.5 + 1.5 = 7.0 > 6.0, so only one fits
        assert len(recommended) == 1

    def test_highest_quality_is_recommended(self) -> None:
        """The top composite_score model is always recommended."""
        scored = score_and_filter(
            [SMALL_FAST, MEDIUM_QUALITY],
            "balanced",
            budget_gb=20.0,
        )

        result = select_defaults(scored, budget_gb=20.0)

        # First in scored list (highest composite) should be recommended
        assert result[0].is_recommended is True

    def test_fastest_model_also_recommended_when_different(self) -> None:
        """If highest quality != fastest, both are recommended."""
        scored = score_and_filter(
            [SMALL_FAST, MEDIUM_QUALITY],
            "balanced",
            budget_gb=20.0,
        )

        result = select_defaults(scored, budget_gb=20.0)

        recommended = [s for s in result if s.is_recommended]
        recommended_repos = {s.model.hf_repo for s in recommended}
        # SMALL_FAST has highest gen_tps, MEDIUM_QUALITY has highest composite
        assert SMALL_FAST.hf_repo in recommended_repos
        assert MEDIUM_QUALITY.hf_repo in recommended_repos

    def test_empty_input_returns_empty(self) -> None:
        assert select_defaults([], budget_gb=20.0) == []


# --------------------------------------------------------------------------- #
# Tier assignment
# --------------------------------------------------------------------------- #


class TestAssignTiers:
    """assign_tiers maps models to named tiers based on their strengths."""

    def _scored(self, model: DiscoveredModel, score: float = 0.5) -> ScoredDiscoveredModel:
        return ScoredDiscoveredModel(
            model=model,
            composite_score=score,
            speed_score=0.5,
            quality_score=0.5,
            tool_calling_score=0.0,
            memory_efficiency_score=0.5,
            is_recommended=True,
        )

    def test_single_model_gets_standard_tier(self) -> None:
        """One model → one tier named 'standard'."""
        tiers = assign_tiers([self._scored(MEDIUM_QUALITY, score=0.8)])

        assert len(tiers) == 1
        assert tiers[0].tier_name == "standard"
        assert tiers[0].model.hf_repo == MEDIUM_QUALITY.hf_repo

    def test_two_models_get_standard_and_fast(self) -> None:
        """Two models → 'standard' (highest composite) and 'fast' (highest speed)."""
        tiers = assign_tiers(
            [
                self._scored(MEDIUM_QUALITY, score=0.9),
                self._scored(SMALL_FAST, score=0.7),
            ]
        )

        assert len(tiers) == 2
        tier_names = {t.tier_name for t in tiers}
        assert tier_names == {"standard", "fast"}

        standard = next(t for t in tiers if t.tier_name == "standard")
        assert standard.model.hf_repo == MEDIUM_QUALITY.hf_repo

    def test_three_models_third_gets_added_tier(self) -> None:
        """Third model gets 'added-N' tier name."""
        tiers = assign_tiers(
            [
                self._scored(MEDIUM_QUALITY, score=0.9),
                self._scored(SMALL_FAST, score=0.7),
                self._scored(TINY_MODEL, score=0.5),
            ]
        )

        assert len(tiers) == 3
        tier_names = [t.tier_name for t in tiers]
        assert "standard" in tier_names
        assert "fast" in tier_names
        assert any(t.startswith("added-") for t in tier_names)

    def test_empty_input_returns_empty(self) -> None:
        assert assign_tiers([]) == []


# --------------------------------------------------------------------------- #
# Config generation
# --------------------------------------------------------------------------- #


class TestGenerateConfig:
    """generate_config writes valid stack and LiteLLM YAML files."""

    @pytest.fixture
    def _mock_home(self, mlx_stack_home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Use isolated mlx_stack_home and mock config reads."""
        monkeypatch.setattr(
            "mlx_stack.core.onboarding.get_value",
            lambda key: {"litellm-port": 4000, "openrouter-key": ""}.get(key),
        )
        return mlx_stack_home

    def _profile(self) -> Any:
        from mlx_stack.core.hardware import HardwareProfile

        return HardwareProfile(
            chip="Apple M4 Pro",
            gpu_cores=20,
            memory_gb=64,
            bandwidth_gbps=273.0,
            is_estimate=False,
        )

    def test_generates_valid_stack_yaml(self, _mock_home: Path) -> None:
        """Stack YAML has correct schema version, tiers, and ports."""
        tiers = [
            TierMapping(tier_name="standard", model=MEDIUM_QUALITY),
            TierMapping(tier_name="fast", model=SMALL_FAST),
        ]

        stack_path, _litellm_path = generate_config(
            self._profile(),
            "balanced",
            tiers,
            budget_gb=25.0,
        )

        assert stack_path.exists()
        stack = yaml.safe_load(stack_path.read_text())

        assert stack["schema_version"] == 1
        assert len(stack["tiers"]) == 2
        assert stack["intent"] == "balanced"

        # Ports are unique
        ports = [t["port"] for t in stack["tiers"]]
        assert len(ports) == len(set(ports))

    def test_generates_valid_litellm_yaml(self, _mock_home: Path) -> None:
        """LiteLLM config has model_list entries matching tiers."""
        tiers = [
            TierMapping(tier_name="standard", model=MEDIUM_QUALITY),
            TierMapping(tier_name="fast", model=SMALL_FAST),
        ]

        _, litellm_path = generate_config(
            self._profile(),
            "balanced",
            tiers,
            budget_gb=25.0,
        )

        assert litellm_path.exists()
        config = yaml.safe_load(litellm_path.read_text())

        model_names = [m["model_name"] for m in config["model_list"]]
        assert "standard" in model_names
        assert "fast" in model_names

    def test_tool_calling_model_gets_auto_tool_choice(self, _mock_home: Path) -> None:
        """Models with tool_calling=True get enable_auto_tool_choice flag."""
        tiers = [TierMapping(tier_name="standard", model=SMALL_FAST)]  # has tool_calling=True

        stack_path, _ = generate_config(
            self._profile(),
            "balanced",
            tiers,
            budget_gb=25.0,
        )

        stack = yaml.safe_load(stack_path.read_text())
        vllm_flags = stack["tiers"][0]["vllm_flags"]
        assert vllm_flags.get("enable_auto_tool_choice") is True

    def test_non_tool_calling_model_no_auto_tool_choice(self, _mock_home: Path) -> None:
        """Models without tool_calling don't get the flag."""
        no_tools = _model("NoTools-8B", "mlx-community/NoTools-8B-4bit", tool_calling=False)
        tiers = [TierMapping(tier_name="standard", model=no_tools)]

        stack_path, _ = generate_config(
            self._profile(),
            "balanced",
            tiers,
            budget_gb=25.0,
        )

        stack = yaml.safe_load(stack_path.read_text())
        vllm_flags = stack["tiers"][0]["vllm_flags"]
        assert "enable_auto_tool_choice" not in vllm_flags

    def test_ports_skip_litellm_port(self, _mock_home: Path) -> None:
        """Port allocation skips the LiteLLM port (4000)."""
        # Create enough tiers that port 4000 would be hit if counting from 3999
        tiers = [TierMapping(tier_name=f"tier-{i}", model=TINY_MODEL) for i in range(5)]

        stack_path, _ = generate_config(
            self._profile(),
            "balanced",
            tiers,
            budget_gb=25.0,
        )

        stack = yaml.safe_load(stack_path.read_text())
        ports = [t["port"] for t in stack["tiers"]]
        assert 4000 not in ports
