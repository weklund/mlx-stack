"""Shared test data factories for mlx-stack tests.

Consolidates all duplicate helper functions (_make_entry, _make_stack_yaml, etc.)
into a single module. Every test file imports from here instead of defining
its own copy.

Factory functions are plain functions (not pytest fixtures) so they can be
called with different arguments in the same test. Filesystem helpers accept
a ``Path`` parameter for the target directory.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import yaml

from mlx_stack.core.catalog import (
    BenchmarkResult,
    Capabilities,
    CatalogEntry,
    QualityScores,
    QuantSource,
)
from mlx_stack.core.hardware import HardwareProfile

# --------------------------------------------------------------------------- #
# Catalog data factories
# --------------------------------------------------------------------------- #


def make_entry(
    model_id: str = "test-model",
    name: str | None = None,
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
    vision: bool = False,
    benchmarks: dict[str, BenchmarkResult] | None = None,
    memory_gb: float = 5.5,
    tags: list[str] | None = None,
    gated: bool = False,
    disk_size_gb: float = 4.5,
    sources: dict[str, QuantSource] | None = None,
) -> CatalogEntry:
    """Create a ``CatalogEntry`` for testing.

    Unified superset of all former ``_make_entry`` helpers.  Every parameter
    has a sensible default so callers only override what they care about.
    """
    if name is None:
        name = f"Test {model_id}"

    if sources is None:
        sources = {
            "int4": QuantSource(
                hf_repo=f"mlx-community/{model_id}-4bit",
                disk_size_gb=disk_size_gb,
            ),
        }

    if benchmarks is None:
        benchmarks = {
            "m4-max-128": BenchmarkResult(
                prompt_tps=100.0, gen_tps=50.0, memory_gb=memory_gb,
            ),
        }

    return CatalogEntry(
        id=model_id,
        name=name,
        family=family,
        params_b=params_b,
        architecture=architecture,
        min_mlx_lm_version="0.22.0",
        sources=sources,
        capabilities=Capabilities(
            tool_calling=tool_calling,
            tool_call_parser=tool_call_parser if tool_calling else None,
            thinking=thinking,
            reasoning_parser=reasoning_parser if thinking else None,
            vision=vision,
        ),
        quality=QualityScores(
            overall=quality_overall,
            coding=quality_coding,
            reasoning=quality_reasoning,
            instruction_following=quality_instruction,
        ),
        benchmarks=benchmarks,
        tags=tags if tags is not None else [],
        gated=gated,
    )


def make_test_catalog() -> list[CatalogEntry]:
    """Standard two-model catalog: big-model (49B) + fast-model (3B).

    Matches the pattern used in ``test_cli_up``, ``test_lifecycle_fixes``,
    and ``test_cross_area``.
    """
    return [
        make_entry("big-model", params_b=49.0, memory_gb=30.0),
        make_entry("fast-model", params_b=3.0, memory_gb=2.0),
    ]


# --------------------------------------------------------------------------- #
# Hardware profile factory
# --------------------------------------------------------------------------- #


def make_profile(
    chip: str = "Apple M4 Max",
    gpu_cores: int = 40,
    memory_gb: int = 128,
    bandwidth_gbps: float = 546.0,
    is_estimate: bool = False,
) -> HardwareProfile:
    """Create a ``HardwareProfile`` for testing."""
    return HardwareProfile(
        chip=chip,
        gpu_cores=gpu_cores,
        memory_gb=memory_gb,
        bandwidth_gbps=bandwidth_gbps,
        is_estimate=is_estimate,
    )


def make_small_profile() -> HardwareProfile:
    """M4 Pro 24 GB — memory-constrained profile for budget tests."""
    return make_profile(
        chip="Apple M4 Pro",
        gpu_cores=20,
        memory_gb=24,
        bandwidth_gbps=273.0,
    )


# --------------------------------------------------------------------------- #
# Stack YAML factories
# --------------------------------------------------------------------------- #


def make_stack_yaml(
    tiers: list[dict[str, Any]] | None = None,
    schema_version: int = 1,
    litellm_port: int = 4000,
) -> dict[str, Any]:
    """Create a stack definition dict for testing.

    The ``litellm_port`` parameter is stored as top-level metadata (matches
    the ``test_cli_up`` variant).  The default two-tier stack uses
    ``standard`` (port 8000) and ``fast`` (port 8001).
    """
    if tiers is None:
        tiers = [
            {
                "name": "standard",
                "model": "big-model",
                "quant": "int4",
                "source": "mlx-community/big-model-4bit",
                "port": 8000,
                "vllm_flags": {
                    "continuous_batching": True,
                    "use_paged_cache": True,
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "hermes",
                },
            },
            {
                "name": "fast",
                "model": "fast-model",
                "quant": "int4",
                "source": "mlx-community/fast-model-4bit",
                "port": 8001,
                "vllm_flags": {
                    "continuous_batching": True,
                    "use_paged_cache": True,
                },
            },
        ]
    return {
        "schema_version": schema_version,
        "name": "default",
        "hardware_profile": "m4-max-128",
        "intent": "balanced",
        "created": "2026-03-24T00:00:00+00:00",
        "tiers": tiers,
    }


def write_stack_yaml(
    mlx_stack_home: Path,
    stack: dict[str, Any] | None = None,
) -> Path:
    """Write a stack YAML file to ``mlx_stack_home`` and return its path."""
    if stack is None:
        stack = make_stack_yaml()
    stacks_dir = mlx_stack_home / "stacks"
    stacks_dir.mkdir(parents=True, exist_ok=True)
    stack_path = stacks_dir / "default.yaml"
    stack_path.write_text(yaml.dump(stack, default_flow_style=False))
    return stack_path


def write_litellm_yaml(mlx_stack_home: Path) -> Path:
    """Write a minimal ``litellm.yaml`` config and return its path."""
    litellm_config = {
        "model_list": [
            {
                "model_name": "standard",
                "litellm_params": {
                    "model": "openai/big-model",
                    "api_base": "http://localhost:8000/v1",
                    "api_key": "dummy",
                },
            },
        ],
    }
    litellm_path = mlx_stack_home / "litellm.yaml"
    litellm_path.write_text(yaml.dump(litellm_config, default_flow_style=False))
    return litellm_path


# --------------------------------------------------------------------------- #
# PID file helpers
# --------------------------------------------------------------------------- #


def create_pid_file(
    mlx_stack_home: Path,
    service_name: str,
    pid: int | str = 12345,
) -> Path:
    """Create a PID file in ``mlx_stack_home/pids/`` and return its path."""
    pids_dir = mlx_stack_home / "pids"
    pids_dir.mkdir(parents=True, exist_ok=True)
    pid_path = pids_dir / f"{service_name}.pid"
    pid_path.write_text(str(pid))
    return pid_path


# --------------------------------------------------------------------------- #
# Log file helpers
# --------------------------------------------------------------------------- #


def create_log_file(
    logs_dir: Path,
    service: str,
    content: str = "",
    size_mb: float = 0,
) -> Path:
    """Create a log file.

    If *content* is provided it is written as-is.  Otherwise *size_mb* bytes
    of filler are written.  Returns the log file path.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{service}.log"
    if content:
        log_path.write_text(content)
    elif size_mb > 0:
        log_path.write_bytes(b"x" * int(size_mb * 1024 * 1024))
    else:
        log_path.touch()
    return log_path


def create_archive(
    logs_dir: Path,
    service: str,
    number: int,
    content: str,
) -> Path:
    """Create a gzip log archive and return its path."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    archive_path = logs_dir / f"{service}.log.{number}.gz"
    with gzip.open(str(archive_path), "wb") as f:
        f.write(content.encode("utf-8"))
    return archive_path


def read_gz(path: Path) -> bytes:
    """Read and decompress a gzip file."""
    with gzip.open(str(path), "rb") as f:
        return f.read()
