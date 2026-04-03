"""Behavioral tests for the model discovery module.

Tests assert what the discovery module produces — the shape and content
of DiscoveredModel lists — not which internal functions are called.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from mlx_stack.core.discovery import (
    DiscoveryError,
    discover_models,
    estimate_memory_from_params,
    get_base_model_name,
    infer_display_name,
    infer_params_from_repo,
    infer_quant_from_repo,
)

# --------------------------------------------------------------------------- #
# Fixtures: realistic data
# --------------------------------------------------------------------------- #

SAMPLE_BENCHMARK_DATA: dict[str, Any] = {
    "generated_at": "2026-04-02T00:00:00Z",
    "source": "test",
    "models": {
        "mlx-community/Qwen3.5-9B-4bit": {
            "benchmark_name": "qwen-3.5-9b",
            "params_b": 9.0,
            "thinking": True,
            "tool_calling": True,
            "benchmarks": {
                "m4-pro-64": {
                    "generation_tps": 62.3,
                    "prompt_tps": 337.4,
                    "peak_memory_gib": 5.2,
                },
            },
            "quality": {"overall_pass_rate": 0.98},
        },
        "mlx-community/Qwen3.5-4B-4bit": {
            "benchmark_name": "qwen-3.5-4b",
            "params_b": 4.0,
            "thinking": True,
            "tool_calling": False,
            "benchmarks": {
                "m4-pro-64": {
                    "generation_tps": 95.1,
                    "prompt_tps": 500.0,
                    "peak_memory_gib": 2.4,
                },
            },
            "quality": {"overall_pass_rate": 0.91},
        },
        "mlx-community/DeepSeek-R1-8B-4bit": {
            "benchmark_name": "deepseek-r1-8b",
            "params_b": 8.0,
            "thinking": True,
            "tool_calling": False,
            "benchmarks": {
                "m4-pro-64": {
                    "generation_tps": 55.0,
                    "prompt_tps": 300.0,
                    "peak_memory_gib": 4.8,
                },
            },
        },
    },
}


def _make_hf_model(repo_id: str, downloads: int = 0) -> SimpleNamespace:
    """Create a mock HuggingFace ModelInfo-like object."""
    return SimpleNamespace(id=repo_id, downloads=downloads)


SAMPLE_HF_MODELS = [
    _make_hf_model("mlx-community/Qwen3.5-9B-4bit", downloads=119_000),
    _make_hf_model("mlx-community/Qwen3.5-9B-8bit", downloads=5_000),
    _make_hf_model("mlx-community/Qwen3.5-4B-4bit", downloads=6_500),
    _make_hf_model("mlx-community/Qwen3.5-4B-8bit", downloads=1_200),
    _make_hf_model("mlx-community/DeepSeek-R1-8B-4bit", downloads=42_000),
    _make_hf_model("mlx-community/NewModel-7B-4bit", downloads=800),
]


# --------------------------------------------------------------------------- #
# Quant / params / name inference
# --------------------------------------------------------------------------- #


class TestInferQuant:
    """Quant is correctly inferred from various repo naming conventions."""

    @pytest.mark.parametrize(
        ("repo", "expected"),
        [
            ("mlx-community/Qwen3.5-9B-4bit", "int4"),
            ("mlx-community/Qwen3.5-9B-8bit", "int8"),
            ("mlx-community/Qwen3.5-9B-bf16", "bf16"),
            ("mlx-community/NVIDIA-Nemotron-Nano-9B-v2-4bits", "int4"),
            ("mlx-community/Model-6bit", "int6"),
            ("mlx-community/Model-5bit", "int5"),
            ("mlx-community/SomeModel", "unknown"),
        ],
    )
    def test_infer_quant(self, repo: str, expected: str) -> None:
        assert infer_quant_from_repo(repo) == expected


class TestInferParams:
    """Parameter count is correctly extracted from repo names."""

    @pytest.mark.parametrize(
        ("repo", "expected"),
        [
            ("mlx-community/Qwen3.5-9B-4bit", 9.0),
            ("mlx-community/Qwen3.5-0.8B-4bit", 0.8),
            ("mlx-community/Llama-3.3-70B-Instruct-4bit", 70.0),
            ("mlx-community/NoParams-Model", None),
        ],
    )
    def test_infer_params(self, repo: str, expected: float | None) -> None:
        assert infer_params_from_repo(repo) == expected


class TestInferDisplayName:
    """Display names strip org prefix and quant suffix."""

    @pytest.mark.parametrize(
        ("repo", "expected"),
        [
            ("mlx-community/Qwen3.5-9B-4bit", "Qwen3.5-9B"),
            ("mlx-community/Qwen3.5-9B-bf16", "Qwen3.5-9B"),
            ("mlx-community/NVIDIA-Nemotron-Nano-9B-v2-4bits", "NVIDIA-Nemotron-Nano-9B-v2"),
        ],
    )
    def test_display_name(self, repo: str, expected: str) -> None:
        assert infer_display_name(repo) == expected


class TestBaseModelName:
    """Base model name strips quant for deduplication."""

    def test_same_base_different_quants(self) -> None:
        base_4bit = get_base_model_name("mlx-community/Qwen3.5-9B-4bit")
        base_8bit = get_base_model_name("mlx-community/Qwen3.5-9B-8bit")
        base_bf16 = get_base_model_name("mlx-community/Qwen3.5-9B-bf16")
        assert base_4bit == base_8bit == base_bf16 == "mlx-community/Qwen3.5-9B"

    def test_different_models_different_bases(self) -> None:
        assert get_base_model_name("mlx-community/Qwen3.5-9B-4bit") != get_base_model_name(
            "mlx-community/Qwen3.5-4B-4bit"
        )


class TestMemoryEstimation:
    """Memory estimates scale with params and quant as expected."""

    def test_int4_smaller_than_int8(self) -> None:
        assert estimate_memory_from_params(8.0, "int4") < estimate_memory_from_params(8.0, "int8")

    def test_int8_smaller_than_bf16(self) -> None:
        assert estimate_memory_from_params(8.0, "int8") < estimate_memory_from_params(8.0, "bf16")

    def test_larger_model_uses_more_memory(self) -> None:
        assert estimate_memory_from_params(14.0, "int4") > estimate_memory_from_params(4.0, "int4")


# --------------------------------------------------------------------------- #
# Model discovery (the main behavior)
# --------------------------------------------------------------------------- #


class TestDiscoverModels:
    """discover_models() returns the right models with correct attributes."""

    @patch("mlx_stack.core.discovery.load_benchmark_data")
    @patch("mlx_stack.core.discovery.query_hf_models")
    def test_returns_only_default_quant(self, mock_hf, mock_bench) -> None:
        """Only int4 models appear by default — no int8/bf16 duplicates."""
        mock_hf.return_value = SAMPLE_HF_MODELS
        mock_bench.return_value = SAMPLE_BENCHMARK_DATA

        models = discover_models(hardware_profile_id="m4-pro-64")

        quants = {m.quant for m in models}
        assert quants == {"int4"}, f"Expected only int4, got {quants}"

    @patch("mlx_stack.core.discovery.load_benchmark_data")
    @patch("mlx_stack.core.discovery.query_hf_models")
    def test_no_duplicate_base_models(self, mock_hf, mock_bench) -> None:
        """Each base model appears at most once."""
        mock_hf.return_value = SAMPLE_HF_MODELS
        mock_bench.return_value = SAMPLE_BENCHMARK_DATA

        models = discover_models(hardware_profile_id="m4-pro-64")

        display_names = [m.display_name for m in models]
        assert len(display_names) == len(set(display_names)), (
            f"Duplicate models found: {display_names}"
        )

    @patch("mlx_stack.core.discovery.load_benchmark_data")
    @patch("mlx_stack.core.discovery.query_hf_models")
    def test_benchmarked_models_have_performance_data(self, mock_hf, mock_bench) -> None:
        """Models with benchmark data get gen_tps, memory, quality populated."""
        mock_hf.return_value = SAMPLE_HF_MODELS
        mock_bench.return_value = SAMPLE_BENCHMARK_DATA

        models = discover_models(hardware_profile_id="m4-pro-64")

        qwen9b = next(m for m in models if "9B" in m.display_name and "Qwen3.5" in m.display_name)
        assert qwen9b.has_benchmark is True
        assert qwen9b.gen_tps is not None
        assert qwen9b.gen_tps > 0
        assert qwen9b.memory_gb is not None
        assert qwen9b.memory_gb > 0
        assert qwen9b.quality_overall is not None

    @patch("mlx_stack.core.discovery.load_benchmark_data")
    @patch("mlx_stack.core.discovery.query_hf_models")
    def test_unbenchmarked_models_still_appear(self, mock_hf, mock_bench) -> None:
        """HF models without benchmark data are included with estimated memory."""
        mock_hf.return_value = SAMPLE_HF_MODELS
        mock_bench.return_value = SAMPLE_BENCHMARK_DATA

        models = discover_models(hardware_profile_id="m4-pro-64")

        new_model = next((m for m in models if "NewModel" in m.display_name), None)
        assert new_model is not None, "NewModel-7B should appear even without benchmarks"
        assert new_model.has_benchmark is False
        assert new_model.gen_tps is None
        assert new_model.memory_gb is not None  # estimated from params

    @patch("mlx_stack.core.discovery.load_benchmark_data")
    @patch("mlx_stack.core.discovery.query_hf_models")
    def test_benchmarked_models_sorted_first(self, mock_hf, mock_bench) -> None:
        """Benchmarked models appear before unbenchmarked ones."""
        mock_hf.return_value = SAMPLE_HF_MODELS
        mock_bench.return_value = SAMPLE_BENCHMARK_DATA

        models = discover_models(hardware_profile_id="m4-pro-64")

        benchmarked = [m for m in models if m.has_benchmark]
        unbenchmarked = [m for m in models if not m.has_benchmark]

        if benchmarked and unbenchmarked:
            last_benchmarked_idx = max(models.index(m) for m in benchmarked)
            first_unbenchmarked_idx = min(models.index(m) for m in unbenchmarked)
            assert last_benchmarked_idx < first_unbenchmarked_idx

    @patch("mlx_stack.core.discovery.load_benchmark_data")
    @patch("mlx_stack.core.discovery.query_hf_models")
    def test_downloads_preserved_from_hf(self, mock_hf, mock_bench) -> None:
        """Download counts from HuggingFace are preserved on the model."""
        mock_hf.return_value = SAMPLE_HF_MODELS
        mock_bench.return_value = SAMPLE_BENCHMARK_DATA

        models = discover_models(hardware_profile_id="m4-pro-64")

        qwen9b = next(m for m in models if "9B" in m.display_name and "Qwen3.5" in m.display_name)
        assert qwen9b.downloads == 119_000

    @patch("mlx_stack.core.discovery.load_benchmark_data")
    @patch("mlx_stack.core.discovery.query_hf_models")
    def test_capabilities_from_benchmark_data(self, mock_hf, mock_bench) -> None:
        """Tool calling and thinking flags come from benchmark data."""
        mock_hf.return_value = SAMPLE_HF_MODELS
        mock_bench.return_value = SAMPLE_BENCHMARK_DATA

        models = discover_models(hardware_profile_id="m4-pro-64")

        qwen9b = next(m for m in models if "9B" in m.display_name and "Qwen3.5" in m.display_name)
        assert qwen9b.tool_calling is True
        assert qwen9b.thinking is True

        qwen4b = next(m for m in models if "4B" in m.display_name and "Qwen3.5" in m.display_name)
        assert qwen4b.tool_calling is False

    @patch("mlx_stack.core.discovery.load_benchmark_data")
    @patch("mlx_stack.core.discovery.query_hf_models")
    def test_fallback_when_hf_unreachable(self, mock_hf, mock_bench) -> None:
        """When HF API fails, returns benchmark-only models."""
        mock_hf.side_effect = DiscoveryError("network down")
        mock_bench.return_value = SAMPLE_BENCHMARK_DATA

        models = discover_models(hardware_profile_id="m4-pro-64")

        assert len(models) > 0, "Should have models from benchmark data"
        assert all(m.has_benchmark for m in models)
        assert all(m.downloads == 0 for m in models)

    @patch("mlx_stack.core.discovery.load_benchmark_data")
    @patch("mlx_stack.core.discovery.query_hf_models")
    def test_benchmark_only_models_included(self, mock_hf, mock_bench) -> None:
        """Models in benchmark_data.json but not in HF results still appear."""
        # HF only returns Qwen models, but benchmark has DeepSeek too
        hf_qwen_only = [m for m in SAMPLE_HF_MODELS if "Qwen" in m.id]
        mock_hf.return_value = hf_qwen_only
        mock_bench.return_value = SAMPLE_BENCHMARK_DATA

        models = discover_models(hardware_profile_id="m4-pro-64")

        deepseek = next((m for m in models if "DeepSeek" in m.display_name), None)
        assert deepseek is not None, "DeepSeek from benchmark data should be included"

    @patch("mlx_stack.core.discovery.load_benchmark_data")
    @patch("mlx_stack.core.discovery.query_hf_models")
    def test_hardware_profile_selects_correct_benchmarks(self, mock_hf, mock_bench) -> None:
        """Benchmark data for the specified hardware profile is used."""
        mock_hf.return_value = SAMPLE_HF_MODELS
        mock_bench.return_value = SAMPLE_BENCHMARK_DATA

        models = discover_models(hardware_profile_id="m4-pro-64")
        qwen9b = next(m for m in models if "9B" in m.display_name and "Qwen3.5" in m.display_name)

        # Should use m4-pro-64 benchmark data
        assert qwen9b.gen_tps == 62.3
        assert qwen9b.memory_gb == 5.2
