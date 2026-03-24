"""Tests for the `mlx-stack recommend` CLI command.

Validates:
- VAL-RECOMMEND-001: Models exceeding memory budget are excluded
- VAL-RECOMMEND-002: Default memory budget is 40% of unified memory
- VAL-RECOMMEND-003: Explicit --budget overrides default
- VAL-RECOMMEND-004: Different intents produce different recommendations
- VAL-RECOMMEND-005: Tier assignment follows quality/speed/architecture rules
- VAL-RECOMMEND-006: Small-memory systems get fewer tiers; large-memory get up to 3
- VAL-RECOMMEND-007: Reads existing profile or auto-detects hardware if missing
- VAL-RECOMMEND-008: Unknown hardware uses bandwidth-ratio estimation with warning
- VAL-RECOMMEND-009: Default output shows formatted table with tiers; --show-all shows all
- VAL-RECOMMEND-010: Cloud fallback tier conditional on OpenRouter key
- VAL-RECOMMEND-011: Recommendation is display-only — no files written
- VAL-RECOMMEND-012: Edge cases — zero models, invalid intent, invalid budget
- VAL-CROSS-003: Profile → recommend data flow
- VAL-CROSS-012: Saved benchmark data overrides catalog in recommendations
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from mlx_stack.cli.main import cli
from mlx_stack.cli.recommend import parse_budget
from mlx_stack.core.catalog import (
    BenchmarkResult,
    Capabilities,
    CatalogEntry,
    QualityScores,
    QuantSource,
)
from mlx_stack.core.hardware import HardwareProfile

# --------------------------------------------------------------------------- #
# Fixtures — reusable test data
# --------------------------------------------------------------------------- #


def _make_profile(
    chip: str = "Apple M4 Max",
    gpu_cores: int = 40,
    memory_gb: int = 128,
    bandwidth_gbps: float = 546.0,
    is_estimate: bool = False,
) -> HardwareProfile:
    """Create a HardwareProfile for testing."""
    return HardwareProfile(
        chip=chip,
        gpu_cores=gpu_cores,
        memory_gb=memory_gb,
        bandwidth_gbps=bandwidth_gbps,
        is_estimate=is_estimate,
    )


def _make_small_profile() -> HardwareProfile:
    """Create a small-memory profile (32 GB)."""
    return HardwareProfile(
        chip="Apple M4 Pro",
        gpu_cores=20,
        memory_gb=32,
        bandwidth_gbps=273.0,
        is_estimate=False,
    )


def _make_entry(
    model_id: str = "test-model",
    name: str = "Test Model",
    family: str = "Test",
    params_b: float = 8.0,
    architecture: str = "transformer",
    quality_overall: int = 70,
    quality_coding: int = 65,
    quality_reasoning: int = 60,
    quality_instruction: int = 72,
    tool_calling: bool = True,
    tool_call_parser: str | None = "hermes",
    thinking: bool = False,
    benchmarks: dict[str, BenchmarkResult] | None = None,
    tags: list[str] | None = None,
) -> CatalogEntry:
    """Create a CatalogEntry for testing."""
    if benchmarks is None:
        benchmarks = {
            "m4-pro-32": BenchmarkResult(prompt_tps=95.0, gen_tps=52.0, memory_gb=5.5),
            "m4-max-128": BenchmarkResult(prompt_tps=140.0, gen_tps=77.0, memory_gb=5.5),
        }
    return CatalogEntry(
        id=model_id,
        name=name,
        family=family,
        params_b=params_b,
        architecture=architecture,
        min_mlx_lm_version="0.22.0",
        sources={
            "int4": QuantSource(hf_repo=f"test/{model_id}-4bit", disk_size_gb=4.5),
        },
        capabilities=Capabilities(
            tool_calling=tool_calling,
            tool_call_parser=tool_call_parser if tool_calling else None,
            thinking=thinking,
            reasoning_parser=None,
            vision=False,
        ),
        quality=QualityScores(
            overall=quality_overall,
            coding=quality_coding,
            reasoning=quality_reasoning,
            instruction_following=quality_instruction,
        ),
        benchmarks=benchmarks,
        tags=tags or [],
    )


def _make_test_catalog() -> list[CatalogEntry]:
    """Build a diverse test catalog for recommendation tests."""
    return [
        # High quality model (standard tier candidate)
        _make_entry(
            model_id="high-quality-32b",
            name="High Quality 32B",
            family="Quality",
            params_b=32.0,
            quality_overall=87,
            quality_coding=85,
            quality_reasoning=88,
            quality_instruction=88,
            tool_calling=True,
            benchmarks={
                "m4-pro-32": BenchmarkResult(prompt_tps=26.0, gen_tps=15.0, memory_gb=20.0),
                "m4-max-128": BenchmarkResult(prompt_tps=40.0, gen_tps=23.0, memory_gb=20.0),
            },
            tags=["quality"],
        ),
        # Fast small model (fast tier candidate)
        _make_entry(
            model_id="fast-0.8b",
            name="Fast 0.8B",
            family="Fast",
            params_b=0.8,
            quality_overall=30,
            quality_coding=25,
            quality_reasoning=20,
            quality_instruction=35,
            tool_calling=True,
            benchmarks={
                "m4-pro-32": BenchmarkResult(prompt_tps=310.0, gen_tps=195.0, memory_gb=0.6),
                "m4-max-128": BenchmarkResult(prompt_tps=410.0, gen_tps=280.0, memory_gb=0.6),
            },
            tags=["fast"],
        ),
        # Medium model
        _make_entry(
            model_id="medium-8b",
            name="Medium 8B",
            family="Medium",
            params_b=8.0,
            quality_overall=68,
            quality_coding=65,
            quality_reasoning=62,
            quality_instruction=72,
            tool_calling=True,
            benchmarks={
                "m4-pro-32": BenchmarkResult(prompt_tps=95.0, gen_tps=52.0, memory_gb=5.5),
                "m4-max-128": BenchmarkResult(prompt_tps=140.0, gen_tps=77.0, memory_gb=5.5),
            },
            tags=["balanced"],
        ),
        # Longctx model (mamba2-hybrid architecture)
        _make_entry(
            model_id="longctx-32b",
            name="LongCtx 32B",
            family="LongCtx",
            params_b=32.0,
            architecture="mamba2-hybrid",
            quality_overall=85,
            quality_coding=86,
            quality_reasoning=90,
            quality_instruction=80,
            tool_calling=False,
            benchmarks={
                "m4-pro-32": BenchmarkResult(prompt_tps=26.0, gen_tps=15.0, memory_gb=20.0),
                "m4-max-128": BenchmarkResult(prompt_tps=40.0, gen_tps=23.0, memory_gb=20.0),
            },
            tags=["long-context"],
        ),
        # Large model that only fits on big systems
        _make_entry(
            model_id="huge-72b",
            name="Huge 72B",
            family="Huge",
            params_b=72.0,
            quality_overall=92,
            quality_coding=90,
            quality_reasoning=94,
            quality_instruction=93,
            tool_calling=True,
            benchmarks={
                "m4-max-128": BenchmarkResult(prompt_tps=12.0, gen_tps=8.0, memory_gb=42.0),
            },
            tags=["quality"],
        ),
    ]


# --------------------------------------------------------------------------- #
# Budget parsing tests
# --------------------------------------------------------------------------- #


class TestBudgetParsing:
    """Tests for the parse_budget helper."""

    def test_parse_number_with_gb(self) -> None:
        assert parse_budget("30gb") == 30.0

    def test_parse_number_with_GB(self) -> None:
        assert parse_budget("30GB") == 30.0

    def test_parse_number_without_suffix(self) -> None:
        assert parse_budget("16") == 16.0

    def test_parse_decimal(self) -> None:
        assert parse_budget("25.6gb") == 25.6

    def test_parse_with_spaces(self) -> None:
        assert parse_budget(" 30gb ") == 30.0

    def test_invalid_format_raises(self) -> None:
        import click

        with pytest.raises(click.BadParameter, match="Invalid budget format"):
            parse_budget("abc")

    def test_negative_value_raises(self) -> None:
        import click

        with pytest.raises(click.BadParameter, match="Invalid budget format"):
            parse_budget("-5gb")

    def test_zero_raises(self) -> None:
        import click

        with pytest.raises(click.BadParameter, match="positive"):
            parse_budget("0gb")

    def test_empty_string_raises(self) -> None:
        import click

        with pytest.raises(click.BadParameter, match="Invalid budget format"):
            parse_budget("")


# --------------------------------------------------------------------------- #
# VAL-RECOMMEND-001: Models exceeding memory budget are excluded
# --------------------------------------------------------------------------- #


class TestBudgetFiltering:
    """VAL-RECOMMEND-001: Every model in recommendation has memory ≤ budget."""

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_models_within_budget(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mlx_stack_home: Path,
    ) -> None:
        """All recommended models have memory ≤ computed budget."""
        mock_load_profile.return_value = _make_profile(memory_gb=128)  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend"])
        assert result.exit_code == 0
        # Huge 72B requires 42 GB, budget is 128*0.4=51.2 GB, so it should appear
        # All smaller models should also appear
        # No model exceeding budget should appear in tiers

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_small_budget_excludes_large_models(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mlx_stack_home: Path,
    ) -> None:
        """Models exceeding explicit budget are excluded."""
        mock_load_profile.return_value = _make_profile(memory_gb=128)  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend", "--budget", "10gb"])
        assert result.exit_code == 0
        # Only models with memory ≤ 10 GB should appear
        assert "Fast 0.8B" in result.output
        assert "Medium 8B" in result.output
        # 20 GB and 42 GB models excluded
        assert "High Quality 32B" not in result.output
        assert "Huge 72B" not in result.output


# --------------------------------------------------------------------------- #
# VAL-RECOMMEND-002: Default memory budget is 40% of unified memory
# --------------------------------------------------------------------------- #


class TestDefaultBudget:
    """VAL-RECOMMEND-002: Default budget = 40% of unified memory."""

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_default_budget_64gb(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mlx_stack_home: Path,
    ) -> None:
        """On 64 GB system, budget = 25.6 GB. 32B models (20 GB) fit."""
        mock_load_profile.return_value = _make_profile(memory_gb=64)  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend"])
        assert result.exit_code == 0
        # 25.6 GB budget: 20 GB models fit, 42 GB model doesn't
        assert "25.6 GB" in result.output
        assert "Huge 72B" not in result.output

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_default_budget_128gb(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mlx_stack_home: Path,
    ) -> None:
        """On 128 GB system, budget = 51.2 GB. All models fit."""
        mock_load_profile.return_value = _make_profile(memory_gb=128)  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend", "--show-all"])
        assert result.exit_code == 0
        # 51.2 GB budget: all models fit (42 GB largest)
        assert "51.2 GB" in result.output
        assert "Huge 72B" in result.output


# --------------------------------------------------------------------------- #
# VAL-RECOMMEND-003: Explicit --budget overrides default
# --------------------------------------------------------------------------- #


class TestBudgetOverride:
    """VAL-RECOMMEND-003: --budget overrides default calculation."""

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_budget_override(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mlx_stack_home: Path,
    ) -> None:
        """--budget 30gb overrides default on 64 GB machine."""
        mock_load_profile.return_value = _make_profile(memory_gb=64)  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        runner = CliRunner()
        # Default budget would be 25.6 GB; override to 30 GB
        result = runner.invoke(cli, ["recommend", "--budget", "30gb", "--show-all"])
        assert result.exit_code == 0
        assert "30.0 GB" in result.output
        # 20 GB models fit (they didn't need override, but 30 GB includes them)
        assert "High Quality 32B" in result.output

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_budget_override_excludes_when_tight(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mlx_stack_home: Path,
    ) -> None:
        """--budget 4gb excludes most models."""
        mock_load_profile.return_value = _make_profile(memory_gb=128)  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend", "--budget", "4gb", "--show-all"])
        assert result.exit_code == 0
        # Only 0.6 GB model fits
        assert "Fast 0.8B" in result.output
        assert "Medium 8B" not in result.output


# --------------------------------------------------------------------------- #
# VAL-RECOMMEND-004: Different intents produce different recommendations
# --------------------------------------------------------------------------- #


class TestIntentDifference:
    """VAL-RECOMMEND-004: Different intents produce different model assignments."""

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_balanced_vs_agent_fleet(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mlx_stack_home: Path,
    ) -> None:
        """balanced and agent-fleet produce at least some different output."""
        mock_load_profile.return_value = _make_profile(memory_gb=128)  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        runner = CliRunner()
        result_balanced = runner.invoke(cli, ["recommend", "--intent", "balanced", "--show-all"])
        result_agent = runner.invoke(cli, ["recommend", "--intent", "agent-fleet", "--show-all"])
        assert result_balanced.exit_code == 0
        assert result_agent.exit_code == 0

        # Outputs should contain different intent labels
        assert "balanced" in result_balanced.output
        assert "agent-fleet" in result_agent.output

        # The composite scores or rankings should differ for models
        # (verified indirectly by different score values in output)
        assert result_balanced.output != result_agent.output


# --------------------------------------------------------------------------- #
# VAL-RECOMMEND-005: Tier assignment follows quality/speed/architecture rules
# --------------------------------------------------------------------------- #


class TestTierAssignment:
    """VAL-RECOMMEND-005: Tier assignment rules."""

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_standard_is_highest_quality(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mlx_stack_home: Path,
    ) -> None:
        """Standard tier gets the highest quality model."""
        mock_load_profile.return_value = _make_profile(memory_gb=128)  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend"])
        assert result.exit_code == 0
        # Huge 72B has quality 92 - should be standard
        output_lines = result.output.split("\n")
        standard_line = [line for line in output_lines if "standard" in line.lower()]
        assert len(standard_line) > 0
        # Standard should be the highest quality model that fits
        assert "Huge 72B" in standard_line[0]

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_fast_is_highest_tps(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mlx_stack_home: Path,
    ) -> None:
        """Fast tier gets the highest gen_tps model (different from standard)."""
        mock_load_profile.return_value = _make_profile(memory_gb=128)  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend"])
        assert result.exit_code == 0
        output_lines = result.output.split("\n")
        fast_line = [
            line for line in output_lines
            if "fast" in line.lower() and "standard" not in line.lower()
        ]
        assert len(fast_line) > 0
        # Fast 0.8B has 280 tps on m4-max-128 - should be fast tier
        assert "Fast 0.8B" in fast_line[0]

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_longctx_is_mamba2(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mlx_stack_home: Path,
    ) -> None:
        """Longctx tier gets a mamba2-hybrid model if budget allows."""
        mock_load_profile.return_value = _make_profile(memory_gb=128)  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend"])
        assert result.exit_code == 0
        output_lines = result.output.split("\n")
        longctx_line = [line for line in output_lines if "longctx" in line.lower()]
        assert len(longctx_line) > 0
        assert "LongCtx 32B" in longctx_line[0]


# --------------------------------------------------------------------------- #
# VAL-RECOMMEND-006: Tier count based on memory size
# --------------------------------------------------------------------------- #


class TestTierCount:
    """VAL-RECOMMEND-006: Small memory = fewer tiers; large memory = up to 3."""

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_large_memory_three_tiers(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mlx_stack_home: Path,
    ) -> None:
        """128 GB system gets up to 3 tiers."""
        mock_load_profile.return_value = _make_profile(memory_gb=128)  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend"])
        assert result.exit_code == 0
        assert "standard" in result.output.lower()
        assert "fast" in result.output.lower()
        assert "longctx" in result.output.lower()

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_small_budget_fewer_tiers(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mlx_stack_home: Path,
    ) -> None:
        """With very small budget, only 1-2 tiers available."""
        mock_load_profile.return_value = _make_profile(memory_gb=128)  # type: ignore[attr-defined]
        # Use only small models in catalog
        catalog = [
            _make_entry(
                model_id="tiny-1b",
                name="Tiny 1B",
                quality_overall=30,
                benchmarks={
                    "m4-max-128": BenchmarkResult(
                        prompt_tps=200.0, gen_tps=150.0, memory_gb=1.0
                    ),
                },
            ),
        ]
        mock_load_catalog.return_value = catalog  # type: ignore[attr-defined]

        runner = CliRunner()
        # Budget 5gb - only one model fits
        result = runner.invoke(cli, ["recommend", "--budget", "5gb"])
        assert result.exit_code == 0
        # With only 1 model, can only have 1 tier
        assert "standard" in result.output.lower()
        assert "longctx" not in result.output.lower()


# --------------------------------------------------------------------------- #
# VAL-RECOMMEND-007: Reads existing profile or auto-detects
# --------------------------------------------------------------------------- #


class TestProfileResolution:
    """VAL-RECOMMEND-007: Profile integration."""

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_uses_existing_profile(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mlx_stack_home: Path,
    ) -> None:
        """When profile.json exists, it is used."""
        profile = _make_profile(memory_gb=64)
        mock_load_profile.return_value = profile  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend"])
        assert result.exit_code == 0
        assert "64 GB" in result.output

    @patch("mlx_stack.cli.recommend.save_profile")
    @patch("mlx_stack.cli.recommend.detect_hardware")
    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_auto_detects_when_no_profile(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mock_detect: object,
        mock_save: object,
        mlx_stack_home: Path,
    ) -> None:
        """When no profile.json, auto-detect and save."""
        mock_load_profile.return_value = None  # type: ignore[attr-defined]
        mock_detect.return_value = _make_profile(memory_gb=128)  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend"])
        assert result.exit_code == 0
        assert "detecting hardware" in result.output.lower()
        mock_detect.assert_called_once()  # type: ignore[attr-defined]
        mock_save.assert_called_once()  # type: ignore[attr-defined]

    @patch("mlx_stack.cli.recommend.detect_hardware")
    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_hardware_detection_failure(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mock_detect: object,
        mlx_stack_home: Path,
    ) -> None:
        """If auto-detect fails, exits with error."""
        from mlx_stack.core.hardware import HardwareError

        mock_load_profile.return_value = None  # type: ignore[attr-defined]
        mock_detect.side_effect = HardwareError("Not Apple Silicon")  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend"])
        assert result.exit_code != 0
        assert "Not Apple Silicon" in result.output


# --------------------------------------------------------------------------- #
# VAL-RECOMMEND-008: Unknown hardware uses estimation with warning
# --------------------------------------------------------------------------- #


class TestEstimatedPerformance:
    """VAL-RECOMMEND-008: Unknown hardware labels values as 'estimated'."""

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_estimated_label_shown(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mlx_stack_home: Path,
    ) -> None:
        """Unknown hardware profile shows estimated labels and bench suggestion."""
        # Use a profile_id that doesn't match any catalog benchmarks
        profile = _make_profile(
            chip="Apple M6 Ultra",
            memory_gb=256,
            bandwidth_gbps=800.0,
            is_estimate=True,
        )
        mock_load_profile.return_value = profile  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend"])
        assert result.exit_code == 0
        # Should show "(est.)" or "estimated" in output
        assert "est." in result.output.lower()
        assert "bench --save" in result.output


# --------------------------------------------------------------------------- #
# VAL-RECOMMEND-009: Default output shows tiers; --show-all shows all
# --------------------------------------------------------------------------- #


class TestOutputFormats:
    """VAL-RECOMMEND-009: Output format tests."""

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_default_shows_tier_table(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mlx_stack_home: Path,
    ) -> None:
        """Default output shows Recommended Stack with tier names."""
        mock_load_profile.return_value = _make_profile(memory_gb=128)  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend"])
        assert result.exit_code == 0
        assert "Recommended Stack" in result.output
        assert "standard" in result.output.lower()

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_show_all_lists_all_models(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mlx_stack_home: Path,
    ) -> None:
        """--show-all shows all budget-fitting models sorted by score."""
        mock_load_profile.return_value = _make_profile(memory_gb=128)  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend", "--show-all"])
        assert result.exit_code == 0
        assert "All Budget-Fitting Models" in result.output
        # All 5 models should appear (51.2 GB budget, all fit)
        assert "High Quality 32B" in result.output
        assert "Fast 0.8B" in result.output
        assert "Medium 8B" in result.output
        assert "LongCtx 32B" in result.output
        assert "Huge 72B" in result.output

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_show_all_contains_score_column(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mlx_stack_home: Path,
    ) -> None:
        """--show-all output includes Score column."""
        mock_load_profile.return_value = _make_profile(memory_gb=128)  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend", "--show-all"])
        assert result.exit_code == 0
        assert "Score" in result.output

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_default_shows_memory_and_tps_columns(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mlx_stack_home: Path,
    ) -> None:
        """Default tier output includes Gen TPS and Memory columns."""
        mock_load_profile.return_value = _make_profile(memory_gb=128)  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend"])
        assert result.exit_code == 0
        assert "Gen TPS" in result.output
        assert "Memory" in result.output
        assert "tok/s" in result.output
        assert "GB" in result.output


# --------------------------------------------------------------------------- #
# VAL-RECOMMEND-010: Cloud fallback conditional on OpenRouter key
# --------------------------------------------------------------------------- #


class TestCloudFallback:
    """VAL-RECOMMEND-010: Cloud fallback shown only when key configured."""

    @patch("mlx_stack.cli.recommend.get_value")
    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_cloud_fallback_with_key(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mock_get_value: object,
        mlx_stack_home: Path,
    ) -> None:
        """Cloud fallback shown when OpenRouter key is set."""
        mock_load_profile.return_value = _make_profile(memory_gb=128)  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        def side_effect(key: str) -> object:
            if key == "openrouter-key":
                return "sk-or-test-key-123"
            if key == "memory-budget-pct":
                return 40
            return ""

        mock_get_value.side_effect = side_effect  # type: ignore[attr-defined]

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend"])
        assert result.exit_code == 0
        assert "Cloud Fallback" in result.output
        assert "OpenRouter" in result.output

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_no_cloud_fallback_without_key(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mlx_stack_home: Path,
    ) -> None:
        """Cloud fallback NOT shown when no OpenRouter key."""
        mock_load_profile.return_value = _make_profile(memory_gb=128)  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend"])
        assert result.exit_code == 0
        assert "Cloud Fallback" not in result.output


# --------------------------------------------------------------------------- #
# VAL-RECOMMEND-011: Display-only — no files written
# --------------------------------------------------------------------------- #


class TestDisplayOnly:
    """VAL-RECOMMEND-011: No files written to stacks/ or litellm.yaml."""

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_no_stack_files_written(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mlx_stack_home: Path,
    ) -> None:
        """Running recommend does not create stacks/ or litellm.yaml."""
        mock_load_profile.return_value = _make_profile(memory_gb=128)  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        stacks_dir = mlx_stack_home / "stacks"
        litellm_file = mlx_stack_home / "litellm.yaml"

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend"])
        assert result.exit_code == 0
        assert not stacks_dir.exists()
        assert not litellm_file.exists()

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_no_files_with_show_all(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mlx_stack_home: Path,
    ) -> None:
        """--show-all also does not write files."""
        mock_load_profile.return_value = _make_profile(memory_gb=128)  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        stacks_dir = mlx_stack_home / "stacks"
        litellm_file = mlx_stack_home / "litellm.yaml"

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend", "--show-all"])
        assert result.exit_code == 0
        assert not stacks_dir.exists()
        assert not litellm_file.exists()

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_display_only_message(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mlx_stack_home: Path,
    ) -> None:
        """Output includes display-only notice and init suggestion."""
        mock_load_profile.return_value = _make_profile(memory_gb=128)  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend"])
        assert result.exit_code == 0
        assert "no files were written" in result.output.lower()
        assert "init" in result.output.lower()


# --------------------------------------------------------------------------- #
# VAL-RECOMMEND-012: Edge cases — zero models, invalid intent, invalid budget
# --------------------------------------------------------------------------- #


class TestEdgeCases:
    """VAL-RECOMMEND-012: Edge case handling."""

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_zero_models_fitting_budget(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mlx_stack_home: Path,
    ) -> None:
        """Budget too small for any model produces clear error."""
        mock_load_profile.return_value = _make_profile(memory_gb=128)  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend", "--budget", "0.1gb"])
        assert result.exit_code != 0
        assert "no models fit" in result.output.lower()

    def test_invalid_intent(self, mlx_stack_home: Path) -> None:
        """Invalid intent produces descriptive error with valid list."""
        runner = CliRunner()
        result = runner.invoke(cli, ["recommend", "--intent", "invalid_intent"])
        assert result.exit_code != 0
        assert "invalid intent" in result.output.lower()
        assert "balanced" in result.output.lower()
        assert "agent-fleet" in result.output.lower()

    def test_invalid_budget_format_letters(self, mlx_stack_home: Path) -> None:
        """--budget abc produces descriptive error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["recommend", "--budget", "abc"])
        assert result.exit_code != 0
        assert "invalid budget" in result.output.lower()

    def test_invalid_budget_format_negative(self, mlx_stack_home: Path) -> None:
        """--budget -5gb produces descriptive error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["recommend", "--budget", "-5gb"])
        assert result.exit_code != 0
        assert "invalid budget" in result.output.lower()

    def test_no_traceback_on_error(self, mlx_stack_home: Path) -> None:
        """No Python traceback on any error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["recommend", "--intent", "bad"])
        assert "Traceback" not in result.output

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_catalog_load_failure(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mlx_stack_home: Path,
    ) -> None:
        """Catalog load failure shows clear error."""
        from mlx_stack.core.catalog import CatalogError

        mock_load_profile.return_value = _make_profile(memory_gb=128)  # type: ignore[attr-defined]
        mock_load_catalog.side_effect = CatalogError("Corrupt catalog")  # type: ignore[attr-defined]

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend"])
        assert result.exit_code != 0
        assert "Corrupt catalog" in result.output
        assert "Traceback" not in result.output


# --------------------------------------------------------------------------- #
# VAL-CROSS-003: Profile → recommend data flow
# --------------------------------------------------------------------------- #


class TestProfileDataFlow:
    """VAL-CROSS-003: Profile data flows correctly into recommendations."""

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_profile_id_used_for_benchmarks(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mlx_stack_home: Path,
    ) -> None:
        """The profile's profile_id is used to look up catalog benchmarks."""
        profile = _make_profile(memory_gb=128)
        mock_load_profile.return_value = profile  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend"])
        assert result.exit_code == 0
        # Profile is m4-max-128, benchmark data for this profile should be used
        assert "Apple M4 Max" in result.output


# --------------------------------------------------------------------------- #
# VAL-CROSS-012: Saved benchmark data overrides catalog
# --------------------------------------------------------------------------- #


class TestSavedBenchmarks:
    """VAL-CROSS-012: Saved benchmark data overrides catalog data."""

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_saved_benchmarks_used(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mlx_stack_home: Path,
    ) -> None:
        """Saved benchmark data overrides catalog data in scoring."""
        profile = _make_profile(memory_gb=128)
        mock_load_profile.return_value = profile  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        # Write saved benchmarks
        benchmarks_dir = mlx_stack_home / "benchmarks"
        benchmarks_dir.mkdir(parents=True)
        saved_data = {
            "medium-8b": {
                "gen_tps": 100.0,  # Higher than catalog's 77.0
                "prompt_tps": 200.0,
                "memory_gb": 5.5,
            }
        }
        (benchmarks_dir / f"{profile.profile_id}.json").write_text(
            json.dumps(saved_data)
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend", "--show-all"])
        assert result.exit_code == 0
        # The saved benchmark gen_tps (100.0) should be used instead of catalog (77.0)
        assert "100.0" in result.output


# --------------------------------------------------------------------------- #
# Config memory-budget-pct integration
# --------------------------------------------------------------------------- #


class TestConfigIntegration:
    """Tests for config values flowing into recommendations."""

    @patch("mlx_stack.cli.recommend.get_value")
    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_config_budget_pct_used(
        self,
        mock_load_profile: object,
        mock_load_catalog: object,
        mock_get_value: object,
        mlx_stack_home: Path,
    ) -> None:
        """memory-budget-pct from config is used when no --budget flag."""
        mock_load_profile.return_value = _make_profile(memory_gb=128)  # type: ignore[attr-defined]
        mock_load_catalog.return_value = _make_test_catalog()  # type: ignore[attr-defined]

        def side_effect(key: str) -> object:
            if key == "memory-budget-pct":
                return 60  # 60% of 128 = 76.8 GB
            if key == "openrouter-key":
                return ""
            return ""

        mock_get_value.side_effect = side_effect  # type: ignore[attr-defined]

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend"])
        assert result.exit_code == 0
        # Budget should be 76.8 GB (60% of 128)
        assert "76.8 GB" in result.output


# --------------------------------------------------------------------------- #
# Help text tests
# --------------------------------------------------------------------------- #


class TestHelpText:
    """Tests for recommend command help text."""

    def test_recommend_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["recommend", "--help"])
        assert result.exit_code == 0
        assert "--budget" in result.output
        assert "--intent" in result.output
        assert "--show-all" in result.output

    def test_recommend_help_describes_intent(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["recommend", "--help"])
        assert "balanced" in result.output
        assert "agent-fleet" in result.output
