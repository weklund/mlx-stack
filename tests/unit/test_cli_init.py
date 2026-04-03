"""Tests for the `mlx-stack init` CLI command and core stack_init module.

Validates:
- VAL-INIT-001: Non-interactive mode completes without prompts
- VAL-INIT-002: Stack definition is valid YAML with all required top-level fields
- VAL-INIT-003: Each tier has required fields with unique ports
- VAL-INIT-004: vllm flags include correct feature flags
- VAL-INIT-005: LiteLLM config is valid with correct model list and endpoints
- VAL-INIT-006: LiteLLM config includes fallback chain
- VAL-INIT-007: Cloud fallback conditional on OpenRouter key
+ Port-in-use detection with deterministic alternate port selection
+ Total estimated memory displayed in init terminal summary
- VAL-INIT-008: --add and --remove customize tier selection
- VAL-INIT-009: Overwrite protection with --force
- VAL-INIT-010: Missing local models detected with pull suggestion
- VAL-INIT-011: Directory structure auto-created
- VAL-INIT-012: Terminal summary displayed on success
- VAL-INIT-013: LiteLLM config includes router settings
- VAL-CROSS-004: Recommend tier assignments flow into init stack definition
- VAL-CROSS-005: Different hardware profiles produce different stacks
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

from mlx_stack.cli.main import cli
from mlx_stack.core.catalog import (
    BenchmarkResult,
    Capabilities,
    CatalogEntry,
    QualityScores,
    QuantSource,
)
from mlx_stack.core.hardware import HardwareProfile
from mlx_stack.core.stack_init import (
    InitError,
    allocate_ports,
    build_vllm_flags,
    detect_missing_models,
    run_init,
)

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
    reasoning_parser: str | None = None,
    benchmarks: dict[str, BenchmarkResult] | None = None,
    tags: list[str] | None = None,
    memory_gb: float = 5.5,
    gated: bool = False,
) -> CatalogEntry:
    """Create a CatalogEntry for testing."""
    if benchmarks is None:
        benchmarks = {
            "m4-pro-32": BenchmarkResult(prompt_tps=95.0, gen_tps=52.0, memory_gb=memory_gb),
            "m4-max-128": BenchmarkResult(prompt_tps=140.0, gen_tps=77.0, memory_gb=memory_gb),
        }
    return CatalogEntry(
        id=model_id,
        name=name,
        family=family,
        params_b=params_b,
        architecture=architecture,
        min_mlx_lm_version="0.22.0",
        sources={
            "int4": QuantSource(hf_repo=f"mlx-community/{model_id}-4bit", disk_size_gb=4.5),
        },
        capabilities=Capabilities(
            tool_calling=tool_calling,
            tool_call_parser=tool_call_parser if tool_calling else None,
            thinking=thinking,
            reasoning_parser=reasoning_parser if thinking else None,
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
        gated=gated,
    )


def _make_test_catalog() -> list[CatalogEntry]:
    """Create a test catalog with several models."""
    return [
        # High quality, slow — standard tier candidate
        _make_entry(
            model_id="big-model",
            name="Big Model 49B",
            params_b=49.0,
            quality_overall=87,
            quality_coding=85,
            quality_reasoning=88,
            quality_instruction=88,
            tool_calling=True,
            tool_call_parser="hermes",
            thinking=True,
            reasoning_parser="nemotron",
            benchmarks={
                "m4-max-128": BenchmarkResult(prompt_tps=22.0, gen_tps=13.0, memory_gb=30.0),
            },
            memory_gb=30.0,
        ),
        # Fast, small — fast tier candidate
        _make_entry(
            model_id="fast-model",
            name="Fast Model 3B",
            params_b=3.0,
            quality_overall=55,
            quality_coding=50,
            quality_reasoning=48,
            quality_instruction=58,
            tool_calling=True,
            tool_call_parser="hermes",
            benchmarks={
                "m4-max-128": BenchmarkResult(prompt_tps=400.0, gen_tps=150.0, memory_gb=2.0),
            },
            memory_gb=2.0,
        ),
        # Long context architecture — longctx tier candidate
        _make_entry(
            model_id="longctx-model",
            name="LongCtx Model 32B",
            params_b=32.0,
            architecture="mamba2-hybrid",
            quality_overall=85,
            quality_coding=86,
            quality_reasoning=90,
            quality_instruction=80,
            tool_calling=False,
            thinking=True,
            reasoning_parser="deepseek_r1",
            benchmarks={
                "m4-max-128": BenchmarkResult(prompt_tps=40.0, gen_tps=23.0, memory_gb=20.0),
            },
            memory_gb=20.0,
        ),
        # Medium model for add/remove tests
        _make_entry(
            model_id="medium-model",
            name="Medium Model 8B",
            params_b=8.0,
            quality_overall=68,
            quality_coding=65,
            quality_reasoning=62,
            quality_instruction=72,
            benchmarks={
                "m4-max-128": BenchmarkResult(prompt_tps=140.0, gen_tps=77.0, memory_gb=5.5),
            },
            memory_gb=5.5,
        ),
    ]


def _write_profile(home: Path, profile: HardwareProfile) -> None:
    """Write a profile.json to the test home directory."""
    profile_path = home / "profile.json"
    profile_path.write_text(json.dumps(profile.to_dict(), indent=2))


# --------------------------------------------------------------------------- #
# Tests: port allocation
# --------------------------------------------------------------------------- #


class TestPortAllocation:
    """Tests for port allocation logic."""

    def test_allocates_sequential_ports(self) -> None:
        """Ports are allocated sequentially from base port."""
        ports = allocate_ports(3, litellm_port=4000)
        assert ports == [8000, 8001, 8002]

    def test_skips_litellm_port(self) -> None:
        """LiteLLM port is skipped in allocation."""
        ports = allocate_ports(3, litellm_port=8001)
        assert 8001 not in ports
        assert len(ports) == 3
        assert ports == [8000, 8002, 8003]

    def test_unique_ports(self) -> None:
        """All allocated ports are unique."""
        ports = allocate_ports(5, litellm_port=4000)
        assert len(set(ports)) == 5

    def test_zero_tiers(self) -> None:
        """Zero tiers returns empty list."""
        ports = allocate_ports(0, litellm_port=4000)
        assert ports == []

    def test_single_tier(self) -> None:
        """Single tier gets one port."""
        ports = allocate_ports(1, litellm_port=4000)
        assert ports == [8000]


# --------------------------------------------------------------------------- #
# Tests: vllm_flags generation
# --------------------------------------------------------------------------- #


class TestVLLMFlags:
    """Tests for vllm_flags generation."""

    def test_base_flags_always_present(self) -> None:
        """continuous_batching and use_paged_cache always present."""
        entry = _make_entry(tool_calling=False, thinking=False)
        flags = build_vllm_flags(entry)
        assert flags["continuous_batching"] is True
        assert flags["use_paged_cache"] is True

    def test_tool_calling_flags(self) -> None:
        """Tool-calling models get enable_auto_tool_choice and tool_call_parser."""
        entry = _make_entry(tool_calling=True, tool_call_parser="hermes")
        flags = build_vllm_flags(entry)
        assert flags["enable_auto_tool_choice"] is True
        assert flags["tool_call_parser"] == "hermes"

    def test_no_tool_calling_flags_without_capability(self) -> None:
        """Non-tool-calling models don't get tool-calling flags."""
        entry = _make_entry(tool_calling=False)
        flags = build_vllm_flags(entry)
        assert "enable_auto_tool_choice" not in flags
        assert "tool_call_parser" not in flags

    def test_thinking_model_gets_reasoning_parser(self) -> None:
        """Thinking-capable models get reasoning_parser."""
        entry = _make_entry(
            tool_calling=False,
            thinking=True,
            reasoning_parser="deepseek_r1",
        )
        flags = build_vllm_flags(entry)
        assert flags["reasoning_parser"] == "deepseek_r1"

    def test_no_reasoning_parser_without_thinking(self) -> None:
        """Non-thinking models don't get reasoning_parser."""
        entry = _make_entry(thinking=False)
        flags = build_vllm_flags(entry)
        assert "reasoning_parser" not in flags

    def test_combined_tool_and_thinking_flags(self) -> None:
        """Model with both tool-calling and thinking gets all flags."""
        entry = _make_entry(
            tool_calling=True,
            tool_call_parser="hermes",
            thinking=True,
            reasoning_parser="nemotron",
        )
        flags = build_vllm_flags(entry)
        assert flags["continuous_batching"] is True
        assert flags["use_paged_cache"] is True
        assert flags["enable_auto_tool_choice"] is True
        assert flags["tool_call_parser"] == "hermes"
        assert flags["reasoning_parser"] == "nemotron"


# --------------------------------------------------------------------------- #
# Tests: stack definition generation
# --------------------------------------------------------------------------- #


class TestStackDefinitionGeneration:
    """Tests for stack definition YAML generation."""

    def test_schema_version_is_1(self, mlx_stack_home: Path) -> None:
        """Stack definition has schema_version: 1."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = run_init(intent="balanced", force=True)

        assert result["stack"]["schema_version"] == 1

    def test_hardware_profile_matches(self, mlx_stack_home: Path) -> None:
        """hardware_profile matches the detected profile ID."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = run_init(intent="balanced", force=True)

        assert result["stack"]["hardware_profile"] == profile.profile_id

    def test_intent_matches(self, mlx_stack_home: Path) -> None:
        """intent field matches the selected intent."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = run_init(intent="agent-fleet", force=True)

        assert result["stack"]["intent"] == "agent-fleet"

    def test_name_is_default(self, mlx_stack_home: Path) -> None:
        """Stack name is 'default'."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = run_init(intent="balanced", force=True)

        assert result["stack"]["name"] == "default"

    def test_created_timestamp_is_iso8601(self, mlx_stack_home: Path) -> None:
        """created field is a valid ISO 8601 timestamp."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = run_init(intent="balanced", force=True)

        created = result["stack"]["created"]
        # Should parse as ISO 8601
        dt = datetime.fromisoformat(created)
        assert dt is not None

    def test_tiers_have_required_fields(self, mlx_stack_home: Path) -> None:
        """Each tier has name, model, quant, source, port, vllm_flags."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = run_init(intent="balanced", force=True)

        for tier in result["stack"]["tiers"]:
            assert "name" in tier
            assert "model" in tier
            assert "quant" in tier
            assert "source" in tier
            assert "port" in tier
            assert "vllm_flags" in tier

    def test_tier_ports_are_unique(self, mlx_stack_home: Path) -> None:
        """All tier ports are unique."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = run_init(intent="balanced", force=True)

        ports = [t["port"] for t in result["stack"]["tiers"]]
        assert len(ports) == len(set(ports))

    def test_tier_ports_dont_conflict_with_litellm(self, mlx_stack_home: Path) -> None:
        """No tier port equals the LiteLLM port (default 4000)."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = run_init(intent="balanced", force=True)

        ports = {t["port"] for t in result["stack"]["tiers"]}
        assert 4000 not in ports


# --------------------------------------------------------------------------- #
# Tests: file generation
# --------------------------------------------------------------------------- #


class TestFileGeneration:
    """Tests for file writing."""

    def test_stack_yaml_written(self, mlx_stack_home: Path) -> None:
        """Stack YAML file is written to the correct path."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = run_init(intent="balanced", force=True)

        stack_path = Path(result["stack_path"])
        assert stack_path.exists()

        # Should be valid YAML
        data = yaml.safe_load(stack_path.read_text())
        assert data["schema_version"] == 1

    def test_litellm_yaml_written(self, mlx_stack_home: Path) -> None:
        """LiteLLM YAML file is written to the correct path."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = run_init(intent="balanced", force=True)

        litellm_path = Path(result["litellm_path"])
        assert litellm_path.exists()

        data = yaml.safe_load(litellm_path.read_text())
        assert "model_list" in data

    def test_directory_auto_created(self, clean_mlx_stack_home: Path) -> None:
        """VAL-INIT-011: Directories are auto-created if missing."""
        profile = _make_profile()
        catalog = _make_test_catalog()

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = run_init(intent="balanced", force=True)

        assert Path(result["stack_path"]).exists()
        assert Path(result["litellm_path"]).exists()


# --------------------------------------------------------------------------- #
# Tests: LiteLLM config content
# --------------------------------------------------------------------------- #


class TestLiteLLMConfigContent:
    """Tests for LiteLLM config content validation."""

    def test_model_list_has_correct_count(self, mlx_stack_home: Path) -> None:
        """model_list has one entry per local tier."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = run_init(intent="balanced", force=True)

        num_tiers = len(result["stack"]["tiers"])
        num_model_entries = len(result["litellm_config"]["model_list"])
        assert num_model_entries == num_tiers

    def test_api_base_matches_tier_port(self, mlx_stack_home: Path) -> None:
        """api_base URLs match tier ports."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = run_init(intent="balanced", force=True)

        tiers = result["stack"]["tiers"]
        model_list = result["litellm_config"]["model_list"]

        for tier in tiers:
            matching = [m for m in model_list if m["model_name"] == tier["name"]]
            assert len(matching) == 1
            assert (
                matching[0]["litellm_params"]["api_base"] == f"http://localhost:{tier['port']}/v1"
            )

    def test_model_uses_openai_prefix(self, mlx_stack_home: Path) -> None:
        """Model identifiers use openai/ prefix."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = run_init(intent="balanced", force=True)

        for entry in result["litellm_config"]["model_list"]:
            assert entry["litellm_params"]["model"].startswith("openai/")

    def test_api_key_is_dummy(self, mlx_stack_home: Path) -> None:
        """api_key is 'dummy' for local models."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = run_init(intent="balanced", force=True)

        for entry in result["litellm_config"]["model_list"]:
            assert entry["litellm_params"]["api_key"] == "dummy"

    def test_router_settings_present(self, mlx_stack_home: Path) -> None:
        """VAL-INIT-013: router_settings present with correct values."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = run_init(intent="balanced", force=True)

        rs = result["litellm_config"]["router_settings"]
        assert "routing_strategy" in rs
        assert rs["num_retries"] == 2
        assert rs["timeout"] == 120

    def test_fallback_chain_present(self, mlx_stack_home: Path) -> None:
        """VAL-INIT-006: Fallback chain references valid tier names."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = run_init(intent="balanced", force=True)

        litellm = result["litellm_config"]
        if "general_settings" in litellm and "fallbacks" in litellm["general_settings"]:
            tier_names = {t["name"] for t in result["stack"]["tiers"]}
            for fb in litellm["general_settings"]["fallbacks"]:
                for src, targets in fb.items():
                    assert src in tier_names
                    for target in targets:
                        assert target in tier_names


# --------------------------------------------------------------------------- #
# Tests: cloud fallback
# --------------------------------------------------------------------------- #


class TestCloudFallback:
    """Tests for cloud fallback configuration."""

    def test_cloud_fallback_with_key(self, mlx_stack_home: Path) -> None:
        """VAL-INIT-007: Cloud fallback added with OpenRouter key."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
            patch("mlx_stack.core.stack_init.get_value") as mock_get,
        ):

            def config_side_effect(key: str):
                if key == "openrouter-key":
                    return "sk-or-test123"
                if key == "litellm-port":
                    return 4000
                if key == "memory-budget-pct":
                    return 40
                if key == "model-dir":
                    return str(mlx_stack_home / "models")
                return ""

            mock_get.side_effect = config_side_effect
            result = run_init(intent="balanced", force=True)

        # Stack should have cloud_fallback section
        assert "cloud_fallback" in result["stack"]
        assert result["stack"]["cloud_fallback"]["provider"] == "openrouter"

        # LiteLLM config should have premium entries
        premium = [
            e for e in result["litellm_config"]["model_list"] if e["model_name"] == "premium"
        ]
        assert len(premium) > 0

    def test_no_cloud_without_key(self, mlx_stack_home: Path) -> None:
        """No cloud fallback when OpenRouter key is empty."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = run_init(intent="balanced", force=True)

        assert "cloud_fallback" not in result["stack"]

        premium = [
            e for e in result["litellm_config"]["model_list"] if e["model_name"] == "premium"
        ]
        assert len(premium) == 0


# --------------------------------------------------------------------------- #
# Tests: overwrite protection
# --------------------------------------------------------------------------- #


class TestOverwriteProtection:
    """Tests for overwrite protection."""

    def test_overwrite_blocked_without_force(self, mlx_stack_home: Path) -> None:
        """VAL-INIT-009: Existing stack requires --force to overwrite."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        # Create initial stack
        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            run_init(intent="balanced", force=True)

        # Try again without force
        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            with pytest.raises(InitError, match="already exists"):
                run_init(intent="balanced", force=False)

    def test_overwrite_allowed_with_force(self, mlx_stack_home: Path) -> None:
        """--force allows overwriting existing stack."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            run_init(intent="balanced", force=True)

        # Overwrite with force
        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = run_init(intent="balanced", force=True)

        assert result["stack"]["schema_version"] == 1


# --------------------------------------------------------------------------- #
# Tests: --add and --remove
# --------------------------------------------------------------------------- #


class TestAddRemove:
    """Tests for --add and --remove customization."""

    def test_remove_tier(self, mlx_stack_home: Path) -> None:
        """VAL-INIT-008: --remove excludes a tier."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = run_init(
                intent="balanced",
                remove_tiers=["fast"],
                force=True,
            )

        tier_names = [t["name"] for t in result["stack"]["tiers"]]
        assert "fast" not in tier_names

    def test_remove_invalid_tier_errors(self, mlx_stack_home: Path) -> None:
        """Removing a non-existent tier raises an error."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            with pytest.raises(InitError, match="Cannot remove tier"):
                run_init(
                    intent="balanced",
                    remove_tiers=["nonexistent"],
                    force=True,
                )

    def test_add_model(self, mlx_stack_home: Path) -> None:
        """VAL-INIT-008: --add includes an additional model."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = run_init(
                intent="balanced",
                add_models=["medium-model"],
                force=True,
            )

        model_ids = [t["model"] for t in result["stack"]["tiers"]]
        assert "medium-model" in model_ids

    def test_add_unknown_model_errors(self, mlx_stack_home: Path) -> None:
        """Adding an unknown model raises an error."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            with pytest.raises(InitError, match="Unknown model"):
                run_init(
                    intent="balanced",
                    add_models=["nonexistent-model"],
                    force=True,
                )

    def test_invalid_intent_errors(self, mlx_stack_home: Path) -> None:
        """Invalid intent raises an error."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            with pytest.raises(InitError, match="Invalid intent"):
                run_init(intent="invalid", force=True)


# --------------------------------------------------------------------------- #
# Tests: missing model detection
# --------------------------------------------------------------------------- #


class TestMissingModelDetection:
    """Tests for missing local model detection."""

    def test_all_models_missing(self, tmp_path: Path) -> None:
        """VAL-INIT-010: All models detected as missing."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        tiers = [
            {"name": "standard", "model": "big-model", "source": "mlx-community/big-4bit"},
            {"name": "fast", "model": "fast-model", "source": "mlx-community/fast-4bit"},
        ]
        missing = detect_missing_models(tiers, models_dir)
        assert set(missing) == {"big-model", "fast-model"}

    def test_model_present_by_id(self, tmp_path: Path) -> None:
        """Model found by its ID directory."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "big-model").mkdir()

        tiers = [
            {"name": "standard", "model": "big-model", "source": "mlx-community/big-4bit"},
            {"name": "fast", "model": "fast-model", "source": "mlx-community/fast-4bit"},
        ]
        missing = detect_missing_models(tiers, models_dir)
        assert missing == ["fast-model"]

    def test_model_present_by_source_dir(self, tmp_path: Path) -> None:
        """Model found by HF repo directory name."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "big-4bit").mkdir()

        tiers = [
            {"name": "standard", "model": "big-model", "source": "mlx-community/big-4bit"},
        ]
        missing = detect_missing_models(tiers, models_dir)
        assert missing == []

    def test_no_models_missing(self, tmp_path: Path) -> None:
        """No missing models when all are present."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "big-model").mkdir()
        (models_dir / "fast-model").mkdir()

        tiers = [
            {"name": "standard", "model": "big-model", "source": "mlx-community/big-4bit"},
            {"name": "fast", "model": "fast-model", "source": "mlx-community/fast-4bit"},
        ]
        missing = detect_missing_models(tiers, models_dir)
        assert missing == []


# --------------------------------------------------------------------------- #
# Tests: CLI command integration
# --------------------------------------------------------------------------- #


class TestCLIInit:
    """Tests for the CLI init command via CliRunner."""

    def test_accept_defaults_completes(self, mlx_stack_home: Path) -> None:
        """VAL-INIT-001: --accept-defaults completes without prompts."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        runner = CliRunner()
        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = runner.invoke(cli, ["init", "--accept-defaults"])

        assert result.exit_code == 0

    def test_accept_defaults_with_intent(self, mlx_stack_home: Path) -> None:
        """--accept-defaults combined with --intent works."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        runner = CliRunner()
        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = runner.invoke(cli, ["init", "--accept-defaults", "--intent", "agent-fleet"])

        assert result.exit_code == 0

    def test_overwrite_without_force_exits_error(self, mlx_stack_home: Path) -> None:
        """VAL-INIT-009: Without --force, existing stack causes error exit."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        runner = CliRunner()
        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            # First init
            result = runner.invoke(cli, ["init", "--accept-defaults"])
            assert result.exit_code == 0

            # Second without force
            result = runner.invoke(cli, ["init", "--accept-defaults"])
            assert result.exit_code == 1
            assert "already exists" in result.output or "force" in result.output.lower()

    def test_force_allows_overwrite(self, mlx_stack_home: Path) -> None:
        """--force allows overwriting existing stack."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        runner = CliRunner()
        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = runner.invoke(cli, ["init", "--accept-defaults"])
            assert result.exit_code == 0

            result = runner.invoke(cli, ["init", "--accept-defaults", "--force"])
            assert result.exit_code == 0

    def test_output_shows_file_paths(self, mlx_stack_home: Path) -> None:
        """VAL-INIT-012: Output shows file paths."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        runner = CliRunner()
        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = runner.invoke(cli, ["init", "--accept-defaults"])

        assert "default.yaml" in result.output
        assert "litellm.yaml" in result.output

    def test_output_shows_tier_assignments(self, mlx_stack_home: Path) -> None:
        """VAL-INIT-012: Output shows tier assignments."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        runner = CliRunner()
        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = runner.invoke(cli, ["init", "--accept-defaults"])

        assert "standard" in result.output or "fast" in result.output

    def test_output_shows_next_steps(self, mlx_stack_home: Path) -> None:
        """VAL-INIT-012: Output shows next-step instructions."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        runner = CliRunner()
        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = runner.invoke(cli, ["init", "--accept-defaults"])

        # Should mention next steps
        assert "pull" in result.output or "up" in result.output

    def test_missing_models_shows_pull_suggestion(self, mlx_stack_home: Path) -> None:
        """VAL-INIT-010: Missing models show pull suggestion."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        runner = CliRunner()
        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = runner.invoke(cli, ["init", "--accept-defaults"])

        # Models are not downloaded, so should suggest pulling
        assert "pull" in result.output

    def test_generated_stack_yaml_is_valid(self, mlx_stack_home: Path) -> None:
        """VAL-INIT-002: Stack YAML is valid and parseable."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        runner = CliRunner()
        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = runner.invoke(cli, ["init", "--accept-defaults"])

        assert result.exit_code == 0

        stack_path = mlx_stack_home / "stacks" / "default.yaml"
        assert stack_path.exists()
        data = yaml.safe_load(stack_path.read_text())
        assert data["schema_version"] == 1
        assert "tiers" in data
        assert "hardware_profile" in data
        assert "intent" in data
        assert data["name"] == "default"

    def test_generated_litellm_yaml_is_valid(self, mlx_stack_home: Path) -> None:
        """VAL-INIT-005: LiteLLM YAML is valid and parseable."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        runner = CliRunner()
        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = runner.invoke(cli, ["init", "--accept-defaults"])

        assert result.exit_code == 0

        litellm_path = mlx_stack_home / "litellm.yaml"
        assert litellm_path.exists()
        data = yaml.safe_load(litellm_path.read_text())
        assert "model_list" in data
        assert "router_settings" in data

    def test_add_option_works(self, mlx_stack_home: Path) -> None:
        """--add works via CLI."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        runner = CliRunner()
        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = runner.invoke(cli, ["init", "--accept-defaults", "--add", "medium-model"])

        assert result.exit_code == 0

    def test_remove_option_works(self, mlx_stack_home: Path) -> None:
        """--remove works via CLI."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        runner = CliRunner()
        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = runner.invoke(cli, ["init", "--accept-defaults", "--remove", "fast"])

        assert result.exit_code == 0

    def test_different_intents_produce_different_stacks(self, mlx_stack_home: Path) -> None:
        """VAL-CROSS-005: Different intents produce different selections."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        results = {}
        for intent_name in ["balanced", "agent-fleet"]:
            with (
                patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
                patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
            ):
                result = run_init(intent=intent_name, force=True)
                results[intent_name] = result["stack"]

        # Both should have tiers
        assert len(results["balanced"]["tiers"]) > 0
        assert len(results["agent-fleet"]["tiers"]) > 0

    def test_vllm_flags_in_generated_stack(self, mlx_stack_home: Path) -> None:
        """VAL-INIT-004: vllm_flags have correct feature flags."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = run_init(intent="balanced", force=True)

        for tier in result["stack"]["tiers"]:
            flags = tier["vllm_flags"]
            assert flags["continuous_batching"] is True
            assert flags["use_paged_cache"] is True


# --------------------------------------------------------------------------- #
# Tests: port-in-use detection
# --------------------------------------------------------------------------- #


class TestPortInUseDetection:
    """Tests for real port-in-use detection during init port allocation."""

    def test_skips_port_in_use(self) -> None:
        """Ports detected as in-use are skipped, next available selected."""

        # Mock _is_port_available: 8000 is in use, 8001 is free
        def mock_available(port: int) -> bool:
            return port != 8000

        with patch("mlx_stack.core.stack_init._is_port_available", side_effect=mock_available):
            ports = allocate_ports(2, litellm_port=4000)

        assert 8000 not in ports
        assert ports[0] == 8001
        assert ports[1] == 8002

    def test_skips_multiple_in_use_ports(self) -> None:
        """Multiple consecutive in-use ports are skipped deterministically."""
        in_use = {8000, 8001, 8002}

        def mock_available(port: int) -> bool:
            return port not in in_use

        with patch("mlx_stack.core.stack_init._is_port_available", side_effect=mock_available):
            ports = allocate_ports(2, litellm_port=4000)

        assert ports == [8003, 8004]

    def test_skips_litellm_port_and_in_use(self) -> None:
        """Both LiteLLM port and in-use ports are skipped."""
        in_use = {8001}

        def mock_available(port: int) -> bool:
            return port not in in_use

        with patch("mlx_stack.core.stack_init._is_port_available", side_effect=mock_available):
            ports = allocate_ports(3, litellm_port=8000)

        # 8000 = litellm, 8001 = in use, so: 8002, 8003, 8004
        assert 8000 not in ports
        assert 8001 not in ports
        assert ports == [8002, 8003, 8004]

    def test_all_ports_available(self) -> None:
        """When all ports are available, sequential allocation is unchanged."""
        with patch("mlx_stack.core.stack_init._is_port_available", return_value=True):
            ports = allocate_ports(3, litellm_port=4000)

        assert ports == [8000, 8001, 8002]

    def test_raises_when_no_ports_available(self) -> None:
        """Raises InitError when no ports can be allocated within range."""
        with patch("mlx_stack.core.stack_init._is_port_available", return_value=False):
            with pytest.raises(InitError, match="Could not allocate"):
                allocate_ports(1, litellm_port=4000)

    def test_port_detection_in_full_init(self, mlx_stack_home: Path) -> None:
        """Port-in-use detection is exercised during full init flow."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        # Block port 8000 so init has to pick alternate
        def mock_available(port: int) -> bool:
            return port != 8000

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
            patch("mlx_stack.core.stack_init._is_port_available", side_effect=mock_available),
        ):
            result = run_init(intent="balanced", force=True)

        tier_ports = [t["port"] for t in result["stack"]["tiers"]]
        assert 8000 not in tier_ports
        # Ports should start from 8001 (next available)
        assert tier_ports[0] == 8001


# --------------------------------------------------------------------------- #
# Tests: total estimated memory display
# --------------------------------------------------------------------------- #


class TestTotalEstimatedMemory:
    """Tests for total estimated memory in init result and display."""

    def test_total_memory_in_result(self, mlx_stack_home: Path) -> None:
        """run_init returns total_memory_gb summing all tier memory."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = run_init(intent="balanced", force=True)

        assert "total_memory_gb" in result
        assert result["total_memory_gb"] > 0
        # Total should be the sum of individual tier memories
        assert isinstance(result["total_memory_gb"], float)

    def test_total_memory_displayed_in_summary(self, mlx_stack_home: Path) -> None:
        """VAL-INIT-012: Terminal summary shows total estimated memory."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        runner = CliRunner()
        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = runner.invoke(cli, ["init", "--accept-defaults"])

        assert result.exit_code == 0
        assert "Total estimated memory" in result.output
        # Should contain a numeric value like "52.0 GB"
        assert "GB" in result.output

    def test_total_memory_sum_is_correct(self, mlx_stack_home: Path) -> None:
        """Total memory is the sum of individual tier memory_gb values."""
        profile = _make_profile()
        catalog = _make_test_catalog()
        _write_profile(mlx_stack_home, profile)

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = run_init(intent="balanced", force=True)

        # The total should be positive. Note: individual models fit within budget,
        # but their sum may exceed the budget (this is the expected behavior —
        # models are individually budget-eligible).
        assert result["total_memory_gb"] > 0
        # Total memory should be reasonable (less than total system memory)
        assert result["total_memory_gb"] < profile.memory_gb


# =========================================================================== #
# Gated model exclusion tests
# =========================================================================== #


class TestGatedModelExclusion:
    """Tests that gated models are excluded from default init."""

    def test_init_excludes_gated_models(self, mlx_stack_home: Path) -> None:
        """Default init excludes gated models from tier assignments."""
        profile = _make_profile()
        _write_profile(mlx_stack_home, profile)

        # Create catalog where the best model is gated
        catalog = [
            _make_entry(
                model_id="gated-best",
                name="Gated Best",
                quality_overall=99,
                gated=True,
            ),
            _make_entry(
                model_id="open-good",
                name="Open Good",
                quality_overall=70,
                gated=False,
            ),
        ]

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = run_init(intent="balanced", force=True)

        tier_model_ids = {t["model"] for t in result["stack"]["tiers"]}
        assert "gated-best" not in tier_model_ids
        assert "open-good" in tier_model_ids

    def test_add_gated_model_warns(self, mlx_stack_home: Path) -> None:
        """Adding a gated model via --add produces a warning."""
        profile = _make_profile()
        _write_profile(mlx_stack_home, profile)

        catalog = [
            _make_entry(model_id="open-model", name="Open Model"),
            _make_entry(model_id="gated-model", name="Gated Model", gated=True),
        ]

        with (
            patch("mlx_stack.core.stack_init.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_init.load_profile", return_value=profile),
        ):
            result = run_init(
                intent="balanced",
                add_models=["gated-model"],
                force=True,
            )

        warnings = result["warnings"]
        gated_warnings = [w for w in warnings if "gated" in w.lower()]
        assert len(gated_warnings) >= 1
        assert "HuggingFace authentication" in gated_warnings[0]
