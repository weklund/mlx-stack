"""Unit tests for the LiteLLM configuration generation module.

Tests cover:
- Model list entry generation with correct api_base, model prefix, api_key
- Cloud fallback entry generation with OpenRouter prefix
- Fallback chain construction
- Router settings (routing_strategy, num_retries, timeout)
- Edge cases: empty tiers, single tier, cloud-only fallback
"""

from __future__ import annotations

import pytest
import yaml

from mlx_stack.core.litellm_gen import (
    DEFAULT_ROUTING_STRATEGY,
    LiteLLMGenError,
    generate_litellm_config,
    render_litellm_yaml,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _make_tiers(
    count: int = 2,
    base_port: int = 8000,
) -> list[dict]:
    """Create a list of tier dicts for testing."""
    tier_data = [
        {"name": "standard", "model": "nemotron-49b", "port": 8000},
        {"name": "fast", "model": "qwen3.5-8b", "port": 8001},
        {"name": "longctx", "model": "deepseek-r1-32b", "port": 8002},
    ]
    return tier_data[:count]


# --------------------------------------------------------------------------- #
# Tests: model_list generation
# --------------------------------------------------------------------------- #


class TestModelListGeneration:
    """Tests for model_list entries in the LiteLLM config."""

    def test_model_entry_has_correct_structure(self) -> None:
        """Each model entry has model_name and litellm_params."""
        config = generate_litellm_config(tiers=_make_tiers(1))
        entry = config["model_list"][0]

        assert "model_name" in entry
        assert "litellm_params" in entry
        assert "model" in entry["litellm_params"]
        assert "api_base" in entry["litellm_params"]
        assert "api_key" in entry["litellm_params"]

    def test_model_has_openai_prefix(self) -> None:
        """Model identifier uses openai/ prefix."""
        config = generate_litellm_config(tiers=_make_tiers(1))
        model = config["model_list"][0]["litellm_params"]["model"]
        assert model.startswith("openai/")
        assert model == "openai/nemotron-49b"

    def test_api_base_matches_tier_port(self) -> None:
        """api_base uses the correct port from the tier."""
        tiers = _make_tiers(2)
        config = generate_litellm_config(tiers=tiers)

        for i, tier in enumerate(tiers):
            entry = config["model_list"][i]
            expected_base = f"http://localhost:{tier['port']}/v1"
            assert entry["litellm_params"]["api_base"] == expected_base

    def test_api_key_is_dummy(self) -> None:
        """api_key is set to 'dummy' for local models."""
        config = generate_litellm_config(tiers=_make_tiers(1))
        assert config["model_list"][0]["litellm_params"]["api_key"] == "dummy"

    def test_model_name_matches_tier_name(self) -> None:
        """model_name matches the tier name."""
        tiers = _make_tiers(3)
        config = generate_litellm_config(tiers=tiers)

        for i, tier in enumerate(tiers):
            assert config["model_list"][i]["model_name"] == tier["name"]

    def test_one_entry_per_local_tier(self) -> None:
        """model_list has one entry per local tier (without cloud)."""
        config = generate_litellm_config(tiers=_make_tiers(3))
        assert len(config["model_list"]) == 3

    def test_two_tier_config(self) -> None:
        """Two-tier config generates two model_list entries."""
        config = generate_litellm_config(tiers=_make_tiers(2))
        assert len(config["model_list"]) == 2


# --------------------------------------------------------------------------- #
# Tests: cloud fallback
# --------------------------------------------------------------------------- #


class TestCloudFallback:
    """Tests for cloud fallback entries in the LiteLLM config."""

    def test_cloud_entries_added_with_key(self) -> None:
        """Cloud entries are added when openrouter_key is set."""
        config = generate_litellm_config(
            tiers=_make_tiers(2),
            openrouter_key="sk-or-test123",
        )
        # 2 local + 2 cloud (gpt-4o and claude)
        assert len(config["model_list"]) == 4

    def test_cloud_entries_use_premium_name(self) -> None:
        """Cloud entries use 'premium' as model_name."""
        config = generate_litellm_config(
            tiers=_make_tiers(1),
            openrouter_key="sk-or-test123",
        )
        cloud_entries = [
            e for e in config["model_list"]
            if e["model_name"] == "premium"
        ]
        assert len(cloud_entries) == 2

    def test_cloud_entries_use_openrouter_prefix(self) -> None:
        """Cloud model identifiers use openrouter/ prefix."""
        config = generate_litellm_config(
            tiers=_make_tiers(1),
            openrouter_key="sk-or-test123",
        )
        cloud_entries = [
            e for e in config["model_list"]
            if e["model_name"] == "premium"
        ]
        for entry in cloud_entries:
            assert entry["litellm_params"]["model"].startswith("openrouter/")

    def test_cloud_api_key_uses_env_reference(self) -> None:
        """Cloud entries use os.environ/OPENROUTER_API_KEY for the key."""
        config = generate_litellm_config(
            tiers=_make_tiers(1),
            openrouter_key="sk-or-test123",
        )
        cloud_entries = [
            e for e in config["model_list"]
            if e["model_name"] == "premium"
        ]
        for entry in cloud_entries:
            assert entry["litellm_params"]["api_key"] == "os.environ/OPENROUTER_API_KEY"

    def test_no_cloud_without_key(self) -> None:
        """No cloud entries when openrouter_key is empty."""
        config = generate_litellm_config(tiers=_make_tiers(2))
        cloud = [e for e in config["model_list"] if e["model_name"] == "premium"]
        assert len(cloud) == 0

    def test_no_cloud_with_empty_string(self) -> None:
        """No cloud entries when openrouter_key is empty string."""
        config = generate_litellm_config(
            tiers=_make_tiers(2),
            openrouter_key="",
        )
        cloud = [e for e in config["model_list"] if e["model_name"] == "premium"]
        assert len(cloud) == 0


# --------------------------------------------------------------------------- #
# Tests: fallback chain
# --------------------------------------------------------------------------- #


class TestFallbackChain:
    """Tests for the fallback chain in the LiteLLM config."""

    def test_fallback_chain_between_local_tiers(self) -> None:
        """Fallback chain links local tiers sequentially."""
        config = generate_litellm_config(tiers=_make_tiers(3))
        fallbacks = config.get("general_settings", {}).get("fallbacks", [])

        # standard -> fast, fast -> longctx
        assert len(fallbacks) == 2
        assert fallbacks[0] == {"standard": ["fast"]}
        assert fallbacks[1] == {"fast": ["longctx"]}

    def test_fallback_chain_includes_premium_with_cloud(self) -> None:
        """Last local tier falls back to premium when cloud is enabled."""
        config = generate_litellm_config(
            tiers=_make_tiers(2),
            openrouter_key="sk-or-test",
        )
        fallbacks = config.get("general_settings", {}).get("fallbacks", [])

        # standard -> fast, fast -> premium
        assert len(fallbacks) == 2
        assert fallbacks[-1] == {"fast": ["premium"]}

    def test_single_tier_no_fallback(self) -> None:
        """Single tier without cloud produces no fallback chain."""
        config = generate_litellm_config(tiers=_make_tiers(1))
        fallbacks = config.get("general_settings", {}).get("fallbacks", [])
        assert len(fallbacks) == 0

    def test_single_tier_with_cloud_has_fallback(self) -> None:
        """Single tier with cloud falls back to premium."""
        config = generate_litellm_config(
            tiers=_make_tiers(1),
            openrouter_key="sk-or-test",
        )
        fallbacks = config.get("general_settings", {}).get("fallbacks", [])
        assert len(fallbacks) == 1
        assert fallbacks[0] == {"standard": ["premium"]}

    def test_fallback_references_valid_tier_names(self) -> None:
        """All fallback references point to valid tier names in model_list."""
        config = generate_litellm_config(
            tiers=_make_tiers(3),
            openrouter_key="sk-or-test",
        )
        valid_names = {e["model_name"] for e in config["model_list"]}
        fallbacks = config.get("general_settings", {}).get("fallbacks", [])

        for fb in fallbacks:
            for src, targets in fb.items():
                assert src in valid_names, f"Fallback source '{src}' not in model_list"
                for target in targets:
                    assert target in valid_names, f"Fallback target '{target}' not in model_list"


# --------------------------------------------------------------------------- #
# Tests: router settings
# --------------------------------------------------------------------------- #


class TestRouterSettings:
    """Tests for router_settings in the LiteLLM config."""

    def test_router_settings_present(self) -> None:
        """router_settings is present in the config."""
        config = generate_litellm_config(tiers=_make_tiers(1))
        assert "router_settings" in config

    def test_routing_strategy(self) -> None:
        """routing_strategy has expected default value."""
        config = generate_litellm_config(tiers=_make_tiers(1))
        assert config["router_settings"]["routing_strategy"] == DEFAULT_ROUTING_STRATEGY

    def test_num_retries(self) -> None:
        """num_retries is 2."""
        config = generate_litellm_config(tiers=_make_tiers(1))
        assert config["router_settings"]["num_retries"] == 2

    def test_timeout(self) -> None:
        """timeout is 120."""
        config = generate_litellm_config(tiers=_make_tiers(1))
        assert config["router_settings"]["timeout"] == 120


# --------------------------------------------------------------------------- #
# Tests: YAML rendering
# --------------------------------------------------------------------------- #


class TestYAMLRendering:
    """Tests for YAML rendering of the config."""

    def test_render_produces_valid_yaml(self) -> None:
        """Rendered YAML can be parsed back to a dict."""
        config = generate_litellm_config(tiers=_make_tiers(2))
        yaml_str = render_litellm_yaml(config)
        parsed = yaml.safe_load(yaml_str)
        assert isinstance(parsed, dict)
        assert "model_list" in parsed

    def test_render_round_trips(self) -> None:
        """Config round-trips through YAML serialization."""
        config = generate_litellm_config(
            tiers=_make_tiers(2),
            openrouter_key="sk-or-test",
        )
        yaml_str = render_litellm_yaml(config)
        parsed = yaml.safe_load(yaml_str)

        # Check key structure is preserved
        assert len(parsed["model_list"]) == len(config["model_list"])
        assert parsed["router_settings"] == config["router_settings"]


# --------------------------------------------------------------------------- #
# Tests: error handling
# --------------------------------------------------------------------------- #


class TestErrorHandling:
    """Tests for error handling in LiteLLM config generation."""

    def test_empty_tiers_raises_error(self) -> None:
        """Empty tiers list raises LiteLLMGenError."""
        with pytest.raises(LiteLLMGenError, match="no tiers"):
            generate_litellm_config(tiers=[])
