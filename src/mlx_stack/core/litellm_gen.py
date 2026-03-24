"""LiteLLM configuration generation for mlx-stack.

Generates a valid LiteLLM proxy config YAML file from a stack definition.
Produces model_list entries with correct api_base, openai/ prefix, and
api_key. Includes router_settings and fallback chain between tiers.
Handles cloud fallback via OpenRouter when an API key is configured.
"""

from __future__ import annotations

from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Default router settings
DEFAULT_ROUTING_STRATEGY = "simple-shuffle"
DEFAULT_NUM_RETRIES = 2
DEFAULT_TIMEOUT = 120


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class LiteLLMGenError(Exception):
    """Raised when LiteLLM config generation fails."""


# --------------------------------------------------------------------------- #
# LiteLLM config generation
# --------------------------------------------------------------------------- #


def _build_model_entry(
    tier_name: str,
    model_id: str,
    port: int,
) -> dict[str, Any]:
    """Build a single model_list entry for LiteLLM.

    Args:
        tier_name: Tier name (e.g., 'standard', 'fast').
        model_id: The catalog model ID.
        port: The port the vllm-mlx instance serves on.

    Returns:
        A dict suitable for inclusion in model_list.
    """
    return {
        "model_name": tier_name,
        "litellm_params": {
            "model": f"openai/{model_id}",
            "api_base": f"http://localhost:{port}/v1",
            "api_key": "dummy",
        },
    }


def _build_cloud_entry(tier_name: str, cloud_model: str) -> dict[str, Any]:
    """Build a cloud fallback model_list entry for LiteLLM.

    Uses os.environ/OPENROUTER_API_KEY for the API key so LiteLLM
    reads it from the environment at runtime.

    Args:
        tier_name: Tier name for the cloud entry (e.g., 'premium').
        cloud_model: The cloud model identifier (e.g., 'openrouter/openai/gpt-4o').

    Returns:
        A dict suitable for inclusion in model_list.
    """
    return {
        "model_name": tier_name,
        "litellm_params": {
            "model": cloud_model,
            "api_key": "os.environ/OPENROUTER_API_KEY",
        },
    }


def _build_fallback_chain(
    tier_names: list[str],
    has_cloud: bool,
) -> list[dict[str, list[str]]]:
    """Build the fallback chain for LiteLLM routing.

    Creates a chain where each tier falls back to the next one.
    If cloud fallback is enabled, the last local tier falls back to premium.

    Args:
        tier_names: Ordered list of local tier names.
        has_cloud: Whether cloud fallback is enabled.

    Returns:
        A list of fallback mappings.
    """
    if not tier_names:
        return []

    all_tiers = list(tier_names)
    if has_cloud:
        all_tiers.append("premium")

    fallbacks: list[dict[str, list[str]]] = []
    for i, tier in enumerate(all_tiers[:-1]):
        fallbacks.append({tier: [all_tiers[i + 1]]})

    return fallbacks


def generate_litellm_config(
    tiers: list[dict[str, Any]],
    litellm_port: int = 4000,
    openrouter_key: str = "",
) -> dict[str, Any]:
    """Generate a complete LiteLLM proxy configuration.

    Args:
        tiers: List of tier dicts, each with 'name', 'model', 'port' keys.
        litellm_port: The port LiteLLM will serve on (used for validation only).
        openrouter_key: OpenRouter API key. If non-empty, cloud fallback is added.

    Returns:
        A dict representing the full LiteLLM YAML config.

    Raises:
        LiteLLMGenError: If generation fails.
    """
    if not tiers:
        msg = "Cannot generate LiteLLM config with no tiers"
        raise LiteLLMGenError(msg)

    model_list: list[dict[str, Any]] = []
    tier_names: list[str] = []

    for tier in tiers:
        name = tier["name"]
        model_id = tier["model"]
        port = tier["port"]

        model_list.append(_build_model_entry(name, model_id, port))
        tier_names.append(name)

    # Cloud fallback
    has_cloud = bool(openrouter_key)
    if has_cloud:
        model_list.append(
            _build_cloud_entry("premium", "openrouter/openai/gpt-4o")
        )
        model_list.append(
            _build_cloud_entry("premium", "openrouter/anthropic/claude-sonnet-4-20250514")
        )

    # Build fallback chain
    fallbacks = _build_fallback_chain(tier_names, has_cloud)

    config: dict[str, Any] = {
        "model_list": model_list,
        "router_settings": {
            "routing_strategy": DEFAULT_ROUTING_STRATEGY,
            "num_retries": DEFAULT_NUM_RETRIES,
            "timeout": DEFAULT_TIMEOUT,
        },
    }

    if fallbacks:
        config["general_settings"] = {
            "fallbacks": fallbacks,
        }

    return config


def render_litellm_yaml(config: dict[str, Any]) -> str:
    """Render a LiteLLM config dict as YAML string.

    Args:
        config: The LiteLLM configuration dict.

    Returns:
        A YAML string suitable for writing to a file.
    """
    return yaml.dump(config, default_flow_style=False, sort_keys=False)
