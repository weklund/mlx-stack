"""Model discovery for mlx-stack.

Merges two data sources into a unified model list:
- HuggingFace API: live query of mlx-community text-generation models
- benchmark_data.json: static overlay with performance and quality data

The HF API provides what exists (repo names, downloads). The benchmark
file provides how well it performs (speed, quality, capabilities).
Models discovered from HF but without benchmark data still appear —
they just show "--" for performance columns.
"""

from __future__ import annotations

import importlib.resources
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from huggingface_hub import HfApi

logger = logging.getLogger("mlx_stack")

# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class DiscoveryError(Exception):
    """Raised when model discovery fails."""


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DiscoveredModel:
    """A model discovered from HuggingFace and/or benchmark data."""

    hf_repo: str
    display_name: str
    params_b: float
    quant: str
    downloads: int
    gen_tps: float | None = None
    prompt_tps: float | None = None
    memory_gb: float | None = None
    quality_overall: float | None = None
    tool_calling: bool = False
    thinking: bool = False
    has_benchmark: bool = False


# --------------------------------------------------------------------------- #
# Quant and param inference
# --------------------------------------------------------------------------- #

_QUANT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"-4bit(?:s)?$", re.IGNORECASE), "int4"),
    (re.compile(r"-8bit(?:s)?$", re.IGNORECASE), "int8"),
    (re.compile(r"-bf16$", re.IGNORECASE), "bf16"),
    (re.compile(r"-fp16$", re.IGNORECASE), "fp16"),
    (re.compile(r"-6bit$", re.IGNORECASE), "int6"),
    (re.compile(r"-5bit$", re.IGNORECASE), "int5"),
]

_PARAMS_PATTERN = re.compile(r"(\d+(?:\.\d+)?)B(?!it)", re.IGNORECASE)


def infer_quant_from_repo(repo_name: str) -> str:
    """Infer quantization from a HuggingFace repo name.

    Examples:
        'Qwen3.5-9B-4bit' -> 'int4'
        'Qwen3.5-9B-8bit' -> 'int8'
        'Qwen3.5-9B-bf16' -> 'bf16'
    """
    name = repo_name.rsplit("/", 1)[-1]
    for pattern, quant in _QUANT_PATTERNS:
        if pattern.search(name):
            return quant
    return "unknown"


def infer_params_from_repo(repo_name: str) -> float | None:
    """Infer parameter count (in billions) from repo name.

    Examples:
        'Qwen3.5-9B-4bit' -> 9.0
        'Qwen3.5-0.8B-4bit' -> 0.8
    """
    name = repo_name.rsplit("/", 1)[-1]
    matches = _PARAMS_PATTERN.findall(name)
    if matches:
        return float(matches[-1])
    return None


def infer_display_name(repo_name: str) -> str:
    """Infer a human-readable display name from repo name.

    Strips the mlx-community/ prefix and quant suffix.

    Examples:
        'mlx-community/Qwen3.5-9B-4bit' -> 'Qwen3.5-9B'
    """
    name = repo_name.rsplit("/", 1)[-1]
    # Strip quant suffixes
    for pattern, _ in _QUANT_PATTERNS:
        name = pattern.sub("", name)
    return name


def estimate_memory_from_params(params_b: float, quant: str) -> float:
    """Estimate memory usage in GB from parameter count and quantization.

    Rough heuristics based on bytes-per-parameter + overhead.
    """
    bytes_per_param = {
        "int4": 0.6,
        "int5": 0.75,
        "int6": 0.9,
        "int8": 1.1,
        "fp16": 2.0,
        "bf16": 2.0,
    }
    bpp = bytes_per_param.get(quant, 0.6)
    return round(params_b * bpp, 1)


def get_base_model_name(repo_name: str) -> str:
    """Extract base model name without quant suffix for deduplication.

    Examples:
        'mlx-community/Qwen3.5-9B-4bit' -> 'mlx-community/Qwen3.5-9B'
    """
    name = repo_name
    for pattern, _ in _QUANT_PATTERNS:
        name = pattern.sub("", name)
    return name


# --------------------------------------------------------------------------- #
# Benchmark data loading
# --------------------------------------------------------------------------- #


def load_benchmark_data() -> dict[str, Any]:
    """Load benchmark_data.json from package data.

    Returns:
        Parsed JSON dict with 'models' key mapping repo names to benchmark data.

    Raises:
        DiscoveryError: If the file is missing or corrupt.
    """
    try:
        data_pkg = importlib.resources.files("mlx_stack.data")
        benchmark_file = data_pkg.joinpath("benchmark_data.json")
        content = benchmark_file.read_text(encoding="utf-8")
        return json.loads(content)
    except (FileNotFoundError, json.JSONDecodeError, TypeError) as exc:
        msg = f"Failed to load benchmark_data.json: {exc}"
        raise DiscoveryError(msg) from None


# --------------------------------------------------------------------------- #
# HuggingFace API query
# --------------------------------------------------------------------------- #


def query_hf_models(
    author: str = "mlx-community",
    pipeline_tag: str = "text-generation",
    limit: int = 200,
) -> list[Any]:
    """Query HuggingFace Hub for text-generation models from mlx-community.

    Args:
        author: HF organization to query.
        pipeline_tag: Pipeline tag filter.
        limit: Maximum models to return.

    Returns:
        List of ModelInfo objects from huggingface_hub.

    Raises:
        DiscoveryError: If the API call fails.
    """
    try:
        api = HfApi()
        return list(
            api.list_models(
                author=author,
                pipeline_tag=pipeline_tag,
                sort="downloads",
                limit=limit,
            )
        )
    except Exception as exc:
        msg = f"HuggingFace API query failed: {exc}"
        raise DiscoveryError(msg) from None


# --------------------------------------------------------------------------- #
# Quant variant lookup
# --------------------------------------------------------------------------- #


def get_quant_variants(
    base_name: str,
    all_hf_models: list[Any],
) -> dict[str, str]:
    """Find all quant variants for a base model name.

    Args:
        base_name: Base model name without quant (e.g., 'mlx-community/Qwen3.5-9B').
        all_hf_models: Full list of HF ModelInfo objects.

    Returns:
        Dict mapping quant string to full repo name.
        E.g., {'int4': 'mlx-community/Qwen3.5-9B-4bit', 'int8': 'mlx-community/Qwen3.5-9B-8bit'}
    """
    variants: dict[str, str] = {}
    for model in all_hf_models:
        repo = model.id if hasattr(model, "id") else str(model)
        if get_base_model_name(repo) == base_name:
            quant = infer_quant_from_repo(repo)
            if quant != "unknown":
                variants[quant] = repo
    return variants


# --------------------------------------------------------------------------- #
# Merging
# --------------------------------------------------------------------------- #


def _build_discovered_model(
    hf_repo: str,
    downloads: int,
    benchmark_data: dict[str, Any],
    hardware_profile_id: str | None,
) -> DiscoveredModel:
    """Build a DiscoveredModel from HF repo info + optional benchmark overlay."""
    quant = infer_quant_from_repo(hf_repo)
    params_b = infer_params_from_repo(hf_repo) or 0.0
    display_name = infer_display_name(hf_repo)

    # Look up benchmark data
    models_data = benchmark_data.get("models", {})
    bench = models_data.get(hf_repo)

    if bench:
        # Get hardware-specific benchmarks
        hw_bench = None
        if hardware_profile_id:
            hw_bench = bench.get("benchmarks", {}).get(hardware_profile_id)
        # Fall back to first available hardware profile
        if not hw_bench:
            benchmarks = bench.get("benchmarks", {})
            if benchmarks:
                hw_bench = next(iter(benchmarks.values()))

        gen_tps = hw_bench.get("generation_tps") if hw_bench else None
        prompt_tps = hw_bench.get("prompt_tps") if hw_bench else None
        memory_gb = hw_bench.get("peak_memory_gib") if hw_bench else None

        quality_data = bench.get("quality", {})
        quality_overall = quality_data.get("overall_pass_rate")

        params_b = bench.get("params_b", params_b)

        return DiscoveredModel(
            hf_repo=hf_repo,
            display_name=display_name,
            params_b=params_b,
            quant=quant,
            downloads=downloads,
            gen_tps=gen_tps,
            prompt_tps=prompt_tps,
            memory_gb=memory_gb,
            quality_overall=quality_overall,
            tool_calling=bench.get("tool_calling", False),
            thinking=bench.get("thinking", False),
            has_benchmark=True,
        )

    # No benchmark data — estimate memory from params
    memory_est = estimate_memory_from_params(params_b, quant) if params_b > 0 else None

    return DiscoveredModel(
        hf_repo=hf_repo,
        display_name=display_name,
        params_b=params_b,
        quant=quant,
        downloads=downloads,
        memory_gb=memory_est,
        has_benchmark=False,
    )


def discover_models(
    hardware_profile_id: str | None = None,
    author: str = "mlx-community",
    default_quant: str = "int4",
) -> list[DiscoveredModel]:
    """Discover available models from HF API + benchmark data overlay.

    Main entry point for model discovery. Queries HuggingFace for
    text-generation models from mlx-community, merges with benchmark
    data for performance/quality overlay, and returns a deduplicated
    list filtered to the default quantization.

    Args:
        hardware_profile_id: Hardware profile for benchmark lookup (e.g., 'm4-pro-64').
        author: HuggingFace organization to query.
        default_quant: Default quantization to show (default 'int4').

    Returns:
        List of DiscoveredModel sorted by: benchmarked first, then by downloads.
        If HF API fails, returns benchmark-only models.
    """
    benchmark_data = load_benchmark_data()

    # Try HF API
    hf_models: list[Any] = []
    try:
        hf_models = query_hf_models(author=author)
    except DiscoveryError:
        logger.warning("HuggingFace API unreachable, using benchmark data only")

    if hf_models:
        # Build models from HF results, filtered to default quant
        seen_bases: set[str] = set()
        models: list[DiscoveredModel] = []

        for hf_model in hf_models:
            repo = hf_model.id
            quant = infer_quant_from_repo(repo)

            # Only show default quant rows (deduplicate by base model)
            if quant != default_quant:
                continue
            base = get_base_model_name(repo)
            if base in seen_bases:
                continue
            seen_bases.add(base)

            downloads = getattr(hf_model, "downloads", 0) or 0
            model = _build_discovered_model(
                repo,
                downloads,
                benchmark_data,
                hardware_profile_id,
            )
            models.append(model)

        # Add benchmark-only models not found in HF results
        for repo in benchmark_data.get("models", {}):
            quant = infer_quant_from_repo(repo)
            if quant != default_quant:
                continue
            base = get_base_model_name(repo)
            if base in seen_bases:
                continue
            seen_bases.add(base)
            model = _build_discovered_model(repo, 0, benchmark_data, hardware_profile_id)
            models.append(model)

    else:
        # Fallback: benchmark-only models
        models = []
        seen_bases = set()
        for repo in benchmark_data.get("models", {}):
            quant = infer_quant_from_repo(repo)
            if quant != default_quant:
                continue
            base = get_base_model_name(repo)
            if base in seen_bases:
                continue
            seen_bases.add(base)
            model = _build_discovered_model(repo, 0, benchmark_data, hardware_profile_id)
            models.append(model)

    # Sort: benchmarked models first (by downloads), then non-benchmarked (by downloads)
    models.sort(key=lambda m: (not m.has_benchmark, -m.downloads))
    return models
