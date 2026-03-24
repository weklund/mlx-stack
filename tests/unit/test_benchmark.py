"""Tests for the benchmark execution engine (core/benchmark.py).

Tests cover:
- Prompt generation
- Statistics computation
- Metric classification (PASS/WARN/FAIL)
- Catalog comparison
- Tool-calling benchmark
- Temporary instance management
- Target resolution (running tiers and catalog models)
- Benchmark save/load
- Error handling
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlx_stack.core.benchmark import (
    CLASSIFICATION_FAIL,
    CLASSIFICATION_PASS,
    CLASSIFICATION_WARN,
    BenchmarkError,
    BenchmarkResult_,
    BenchmarkRunError,
    BenchmarkTargetError,
    IterationResult,
    ToolCallResult,
    _classify_metric,
    _compare_against_catalog,
    _compute_stats,
    _find_temp_port,
    _generate_prompt,
    _get_all_tier_names,
    _get_running_tier_names,
    _get_used_ports,
    _run_single_iteration,
    save_benchmark_results,
)
from mlx_stack.core.catalog import (
    BenchmarkResult as CatalogBenchmarkResult,
)
from mlx_stack.core.catalog import (
    Capabilities,
    CatalogEntry,
    QualityScores,
    QuantSource,
)
from mlx_stack.core.hardware import HardwareProfile

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def sample_entry() -> CatalogEntry:
    """A sample catalog entry for testing."""
    return CatalogEntry(
        id="qwen3.5-8b",
        name="Qwen 3.5 8B",
        family="Qwen 3.5",
        params_b=8.0,
        architecture="transformer",
        min_mlx_lm_version="0.22.0",
        sources={
            "int4": QuantSource(hf_repo="mlx-community/Qwen3.5-8B-4bit", disk_size_gb=4.5),
            "int8": QuantSource(hf_repo="mlx-community/Qwen3.5-8B-8bit", disk_size_gb=8.5),
        },
        capabilities=Capabilities(
            tool_calling=True,
            tool_call_parser="hermes",
            thinking=True,
            reasoning_parser="qwen3",
            vision=False,
        ),
        quality=QualityScores(overall=68, coding=65, reasoning=62, instruction_following=72),
        benchmarks={
            "m5-max-128": CatalogBenchmarkResult(
                prompt_tps=155.0, gen_tps=85.0, memory_gb=5.5
            ),
            "m4-pro-48": CatalogBenchmarkResult(
                prompt_tps=95.0, gen_tps=52.0, memory_gb=5.5
            ),
        },
        tags=["balanced", "agent-ready"],
    )


@pytest.fixture()
def sample_entry_no_tool_calling() -> CatalogEntry:
    """A sample catalog entry without tool-calling."""
    return CatalogEntry(
        id="deepseek-r1-8b",
        name="DeepSeek R1 8B",
        family="DeepSeek R1",
        params_b=8.0,
        architecture="transformer",
        min_mlx_lm_version="0.22.0",
        sources={
            "int4": QuantSource(hf_repo="mlx-community/DeepSeek-R1-8B-4bit", disk_size_gb=4.5),
        },
        capabilities=Capabilities(
            tool_calling=False,
            tool_call_parser=None,
            thinking=True,
            reasoning_parser="deepseek",
            vision=False,
        ),
        quality=QualityScores(overall=65, coding=60, reasoning=70, instruction_following=60),
        benchmarks={
            "m5-max-128": CatalogBenchmarkResult(
                prompt_tps=150.0, gen_tps=80.0, memory_gb=5.0
            ),
        },
        tags=["reasoning"],
    )


@pytest.fixture()
def sample_profile() -> HardwareProfile:
    """A sample hardware profile for testing."""
    return HardwareProfile(
        chip="Apple M5 Max",
        gpu_cores=40,
        memory_gb=128,
        bandwidth_gbps=546.0,
        is_estimate=False,
    )


# --------------------------------------------------------------------------- #
# Test: Prompt generation
# --------------------------------------------------------------------------- #


class TestPromptGeneration:
    """Tests for _generate_prompt."""

    def test_generates_nonempty_string(self) -> None:
        prompt = _generate_prompt(1024)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_longer_token_count_produces_longer_prompt(self) -> None:
        short = _generate_prompt(100)
        long = _generate_prompt(1000)
        assert len(long) > len(short)

    def test_minimal_token_count(self) -> None:
        prompt = _generate_prompt(1)
        assert isinstance(prompt, str)
        assert len(prompt) > 0


# --------------------------------------------------------------------------- #
# Test: Statistics computation
# --------------------------------------------------------------------------- #


class TestComputeStats:
    """Tests for _compute_stats."""

    def test_empty_list(self) -> None:
        mean, std = _compute_stats([])
        assert mean == 0.0
        assert std == 0.0

    def test_single_value(self) -> None:
        mean, std = _compute_stats([42.0])
        assert mean == 42.0
        assert std == 0.0

    def test_known_values(self) -> None:
        values = [10.0, 20.0, 30.0]
        mean, std = _compute_stats(values)
        assert mean == pytest.approx(20.0)
        assert std == pytest.approx(10.0)

    def test_identical_values(self) -> None:
        values = [5.0, 5.0, 5.0]
        mean, std = _compute_stats(values)
        assert mean == 5.0
        assert std == 0.0

    def test_two_values(self) -> None:
        values = [10.0, 20.0]
        mean, std = _compute_stats(values)
        assert mean == pytest.approx(15.0)
        # Sample std dev of [10, 20] = sqrt(50) ≈ 7.07
        assert std == pytest.approx(7.0710678, rel=1e-4)


# --------------------------------------------------------------------------- #
# Test: Metric classification
# --------------------------------------------------------------------------- #


class TestClassifyMetric:
    """Tests for _classify_metric."""

    def test_pass_within_15_percent(self) -> None:
        result = _classify_metric("gen_tps", 90.0, 100.0)
        assert result.classification == CLASSIFICATION_PASS
        assert result.metric == "gen_tps"
        assert result.measured == 90.0
        assert result.catalog == 100.0

    def test_pass_above_catalog(self) -> None:
        result = _classify_metric("gen_tps", 110.0, 100.0)
        assert result.classification == CLASSIFICATION_PASS

    def test_warn_between_15_and_30_percent(self) -> None:
        result = _classify_metric("gen_tps", 75.0, 100.0)
        assert result.classification == CLASSIFICATION_WARN

    def test_fail_below_30_percent(self) -> None:
        result = _classify_metric("gen_tps", 60.0, 100.0)
        assert result.classification == CLASSIFICATION_FAIL

    def test_exact_threshold_15_percent(self) -> None:
        """At exactly 15% below (measured=85), should still be PASS."""
        result = _classify_metric("gen_tps", 85.0, 100.0)
        assert result.classification == CLASSIFICATION_PASS

    def test_exact_threshold_30_percent(self) -> None:
        """At exactly 30% below (measured=70), should still be WARN."""
        result = _classify_metric("gen_tps", 70.0, 100.0)
        assert result.classification == CLASSIFICATION_WARN

    def test_zero_catalog_value(self) -> None:
        result = _classify_metric("gen_tps", 50.0, 0.0)
        assert result.classification == CLASSIFICATION_PASS

    def test_delta_percentage_negative_means_below(self) -> None:
        result = _classify_metric("gen_tps", 75.0, 100.0)
        # 25% below catalog => delta_pct = 25.0
        assert result.delta_pct == pytest.approx(25.0)


# --------------------------------------------------------------------------- #
# Test: Catalog comparison
# --------------------------------------------------------------------------- #


class TestCatalogComparison:
    """Tests for _compare_against_catalog."""

    def test_matching_profile_returns_classifications(
        self, sample_entry: CatalogEntry, sample_profile: HardwareProfile
    ) -> None:
        classifications = _compare_against_catalog(
            prompt_tps_mean=150.0,
            gen_tps_mean=80.0,
            entry=sample_entry,
            profile=sample_profile,
        )
        assert len(classifications) == 2
        assert classifications[0].metric == "prompt_tps"
        assert classifications[1].metric == "gen_tps"

    def test_all_pass_when_near_catalog(
        self, sample_entry: CatalogEntry, sample_profile: HardwareProfile
    ) -> None:
        classifications = _compare_against_catalog(
            prompt_tps_mean=155.0,
            gen_tps_mean=85.0,
            entry=sample_entry,
            profile=sample_profile,
        )
        for cls in classifications:
            assert cls.classification == CLASSIFICATION_PASS

    def test_fail_when_far_below(
        self, sample_entry: CatalogEntry, sample_profile: HardwareProfile
    ) -> None:
        classifications = _compare_against_catalog(
            prompt_tps_mean=50.0,
            gen_tps_mean=30.0,
            entry=sample_entry,
            profile=sample_profile,
        )
        for cls in classifications:
            assert cls.classification == CLASSIFICATION_FAIL

    def test_no_matching_profile_returns_empty(
        self, sample_entry: CatalogEntry
    ) -> None:
        unknown_profile = HardwareProfile(
            chip="Apple M99",
            gpu_cores=100,
            memory_gb=256,
            bandwidth_gbps=2000.0,
            is_estimate=True,
        )
        classifications = _compare_against_catalog(
            prompt_tps_mean=150.0,
            gen_tps_mean=80.0,
            entry=sample_entry,
            profile=unknown_profile,
        )
        assert classifications == []


# --------------------------------------------------------------------------- #
# Test: BenchmarkResult_ serialization
# --------------------------------------------------------------------------- #


class TestBenchmarkResultSerialization:
    """Tests for BenchmarkResult_ to_save_dict."""

    def test_to_save_dict(self) -> None:
        result = BenchmarkResult_(
            model_id="qwen3.5-8b",
            quant="int4",
            prompt_tps_mean=150.0,
            prompt_tps_std=5.0,
            gen_tps_mean=80.0,
            gen_tps_std=2.0,
        )
        d = result.to_save_dict()
        assert d["model_id"] == "qwen3.5-8b"
        assert d["quant"] == "int4"
        assert d["prompt_tps"] == 150.0
        assert d["gen_tps"] == 80.0

    def test_to_save_dict_has_memory_gb(self) -> None:
        result = BenchmarkResult_(
            model_id="test",
            quant="int4",
        )
        d = result.to_save_dict()
        assert "memory_gb" in d


# --------------------------------------------------------------------------- #
# Test: Save benchmark results
# --------------------------------------------------------------------------- #


class TestSaveBenchmarkResults:
    """Tests for save_benchmark_results."""

    def test_saves_to_correct_path(
        self, mlx_stack_home: Path, sample_profile: HardwareProfile
    ) -> None:
        result = BenchmarkResult_(
            model_id="qwen3.5-8b",
            quant="int4",
            prompt_tps_mean=150.0,
            gen_tps_mean=80.0,
        )
        saved_path = save_benchmark_results(result, sample_profile)
        assert saved_path.exists()
        assert saved_path.name == f"{sample_profile.profile_id}.json"

    def test_saved_data_is_valid_json(
        self, mlx_stack_home: Path, sample_profile: HardwareProfile
    ) -> None:
        result = BenchmarkResult_(
            model_id="qwen3.5-8b",
            quant="int4",
            prompt_tps_mean=150.0,
            gen_tps_mean=80.0,
        )
        saved_path = save_benchmark_results(result, sample_profile)
        data = json.loads(saved_path.read_text())
        assert isinstance(data, dict)
        assert "qwen3.5-8b" in data
        assert data["qwen3.5-8b"]["prompt_tps"] == 150.0
        assert data["qwen3.5-8b"]["gen_tps"] == 80.0

    def test_merges_with_existing_data(
        self, mlx_stack_home: Path, sample_profile: HardwareProfile
    ) -> None:
        # Save first result
        result1 = BenchmarkResult_(
            model_id="model-a",
            quant="int4",
            prompt_tps_mean=100.0,
            gen_tps_mean=50.0,
        )
        save_benchmark_results(result1, sample_profile)

        # Save second result
        result2 = BenchmarkResult_(
            model_id="model-b",
            quant="int4",
            prompt_tps_mean=200.0,
            gen_tps_mean=100.0,
        )
        saved_path = save_benchmark_results(result2, sample_profile)

        data = json.loads(saved_path.read_text())
        assert "model-a" in data
        assert "model-b" in data

    def test_overwrites_existing_model(
        self, mlx_stack_home: Path, sample_profile: HardwareProfile
    ) -> None:
        result1 = BenchmarkResult_(
            model_id="qwen3.5-8b",
            quant="int4",
            prompt_tps_mean=100.0,
            gen_tps_mean=50.0,
        )
        save_benchmark_results(result1, sample_profile)

        result2 = BenchmarkResult_(
            model_id="qwen3.5-8b",
            quant="int4",
            prompt_tps_mean=200.0,
            gen_tps_mean=100.0,
        )
        saved_path = save_benchmark_results(result2, sample_profile)

        data = json.loads(saved_path.read_text())
        assert data["qwen3.5-8b"]["prompt_tps"] == 200.0


# --------------------------------------------------------------------------- #
# Test: Find temp port
# --------------------------------------------------------------------------- #


class TestFindTempPort:
    """Tests for _find_temp_port."""

    @patch("mlx_stack.core.benchmark.check_port_conflict", return_value=None)
    def test_finds_first_available_port(self, mock_check: MagicMock) -> None:
        used = {8000, 8001, 8002, 4000}
        port = _find_temp_port(used)
        assert port >= 8100
        assert port < 8200

    @patch("mlx_stack.core.benchmark.check_port_conflict", return_value=None)
    def test_skips_used_ports(self, mock_check: MagicMock) -> None:
        used = {8100, 8101, 8102}
        port = _find_temp_port(used)
        assert port not in used

    @patch("mlx_stack.core.benchmark.check_port_conflict", return_value=(1234, "test"))
    def test_all_ports_in_use_raises_error(self, mock_check: MagicMock) -> None:
        used = set(range(8100, 8200))
        with pytest.raises(BenchmarkError, match="Could not find an available port"):
            _find_temp_port(used)


# --------------------------------------------------------------------------- #
# Test: Stack tier resolution
# --------------------------------------------------------------------------- #


class TestTierResolution:
    """Tests for tier resolution functions."""

    def test_get_all_tier_names_no_stack(self, mlx_stack_home: Path) -> None:
        names = _get_all_tier_names()
        assert names == []

    def test_get_running_tier_names_no_stack(self, mlx_stack_home: Path) -> None:
        running = _get_running_tier_names()
        assert running == []

    def test_get_used_ports_defaults(self, mlx_stack_home: Path) -> None:
        ports = _get_used_ports()
        # Should include default LiteLLM port
        assert 4000 in ports

    def test_get_all_tier_names_with_stack(self, mlx_stack_home: Path) -> None:
        import yaml

        stacks_dir = mlx_stack_home / "stacks"
        stacks_dir.mkdir(parents=True, exist_ok=True)
        stack_def = {
            "schema_version": 1,
            "tiers": [
                {"name": "fast", "model": "qwen3.5-3b", "port": 8000},
                {"name": "standard", "model": "qwen3.5-8b", "port": 8001},
            ],
        }
        (stacks_dir / "default.yaml").write_text(yaml.dump(stack_def))

        names = _get_all_tier_names()
        assert "fast" in names
        assert "standard" in names

    def test_get_used_ports_with_stack(self, mlx_stack_home: Path) -> None:
        import yaml

        stacks_dir = mlx_stack_home / "stacks"
        stacks_dir.mkdir(parents=True, exist_ok=True)
        stack_def = {
            "schema_version": 1,
            "tiers": [
                {"name": "fast", "port": 8000},
                {"name": "standard", "port": 8001},
            ],
        }
        (stacks_dir / "default.yaml").write_text(yaml.dump(stack_def))

        ports = _get_used_ports()
        assert 8000 in ports
        assert 8001 in ports
        assert 4000 in ports  # default LiteLLM


# --------------------------------------------------------------------------- #
# Test: Single iteration (mocked HTTP)
# --------------------------------------------------------------------------- #


class TestSingleIteration:
    """Tests for _run_single_iteration with mocked HTTP."""

    @patch("mlx_stack.core.benchmark.httpx.post")
    def test_successful_iteration(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
            },
        }
        mock_post.return_value = mock_response

        result = _run_single_iteration(
            port=8000,
            model_name="test-model",
            prompt="test prompt",
            max_tokens=100,
        )

        assert isinstance(result, IterationResult)
        assert result.prompt_tokens == 1000
        assert result.completion_tokens == 100
        assert result.prompt_tps > 0
        assert result.gen_tps > 0

    @patch("mlx_stack.core.benchmark.httpx.post")
    def test_api_error_raises(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        with pytest.raises(BenchmarkRunError, match="status 500"):
            _run_single_iteration(
                port=8000,
                model_name="test-model",
                prompt="test",
            )

    @patch("mlx_stack.core.benchmark.httpx.post")
    def test_timeout_raises(self, mock_post: MagicMock) -> None:
        import httpx

        mock_post.side_effect = httpx.TimeoutException("timeout")

        with pytest.raises(BenchmarkRunError, match="timed out"):
            _run_single_iteration(
                port=8000,
                model_name="test-model",
                prompt="test",
            )

    @patch("mlx_stack.core.benchmark.httpx.post")
    def test_zero_tokens_returns_zero_tps(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": ""}}],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
        }
        mock_post.return_value = mock_response

        result = _run_single_iteration(
            port=8000,
            model_name="test-model",
            prompt="test",
        )
        assert result.prompt_tps == 0.0
        assert result.gen_tps == 0.0


# --------------------------------------------------------------------------- #
# Test: Tool-calling benchmark (mocked HTTP)
# --------------------------------------------------------------------------- #


class TestToolCallBenchmark:
    """Tests for tool-calling benchmark with mocked HTTP."""

    @patch("mlx_stack.core.benchmark.httpx.post")
    def test_successful_tool_call(self, mock_post: MagicMock) -> None:
        from mlx_stack.core.benchmark import _run_tool_call_benchmark

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "San Francisco, CA"}',
                                }
                            }
                        ]
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        result = _run_tool_call_benchmark(port=8000, model_name="test")
        assert result.success is True
        assert result.round_trip_time > 0
        assert result.error is None

    @patch("mlx_stack.core.benchmark.httpx.post")
    def test_wrong_function_name(self, mock_post: MagicMock) -> None:
        from mlx_stack.core.benchmark import _run_tool_call_benchmark

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "wrong_function",
                                    "arguments": "{}",
                                }
                            }
                        ]
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        result = _run_tool_call_benchmark(port=8000, model_name="test")
        assert result.success is False
        assert "Wrong function name" in (result.error or "")

    @patch("mlx_stack.core.benchmark.httpx.post")
    def test_no_tool_calls_in_response(self, mock_post: MagicMock) -> None:
        from mlx_stack.core.benchmark import _run_tool_call_benchmark

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "It's sunny"}}]
        }
        mock_post.return_value = mock_response

        result = _run_tool_call_benchmark(port=8000, model_name="test")
        assert result.success is False
        assert result.error is not None

    @patch("mlx_stack.core.benchmark.httpx.post")
    def test_api_error(self, mock_post: MagicMock) -> None:
        from mlx_stack.core.benchmark import _run_tool_call_benchmark

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        result = _run_tool_call_benchmark(port=8000, model_name="test")
        assert result.success is False

    @patch("mlx_stack.core.benchmark.httpx.post")
    def test_missing_location_in_args(self, mock_post: MagicMock) -> None:
        from mlx_stack.core.benchmark import _run_tool_call_benchmark

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"unit": "celsius"}',
                                }
                            }
                        ]
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        result = _run_tool_call_benchmark(port=8000, model_name="test")
        assert result.success is False
        assert "location" in (result.error or "")


# --------------------------------------------------------------------------- #
# Test: Temp instance cleanup
# --------------------------------------------------------------------------- #


class TestTempInstanceCleanup:
    """Tests for temporary instance cleanup."""

    @patch("mlx_stack.core.benchmark.stop_service")
    @patch("mlx_stack.core.benchmark.read_pid_file", return_value=None)
    @patch("mlx_stack.core.benchmark.remove_pid_file")
    def test_cleanup_calls_stop_service(
        self,
        mock_remove: MagicMock,
        mock_read: MagicMock,
        mock_stop: MagicMock,
    ) -> None:
        from mlx_stack.core.benchmark import _cleanup_temp_instance

        _cleanup_temp_instance("bench-temp-test")
        mock_stop.assert_called_once()

    @patch("mlx_stack.core.benchmark.stop_service", side_effect=Exception("fail"))
    @patch("mlx_stack.core.benchmark.read_pid_file", return_value=None)
    @patch("mlx_stack.core.benchmark.remove_pid_file")
    def test_cleanup_handles_stop_failure(
        self,
        mock_remove: MagicMock,
        mock_read: MagicMock,
        mock_stop: MagicMock,
    ) -> None:
        from mlx_stack.core.benchmark import _cleanup_temp_instance

        # Should not raise
        _cleanup_temp_instance("bench-temp-test")


# --------------------------------------------------------------------------- #
# Test: resolve_target
# --------------------------------------------------------------------------- #


class TestResolveTarget:
    """Tests for resolve_target."""

    @patch("mlx_stack.core.benchmark._find_running_tier")
    @patch("mlx_stack.core.benchmark.load_catalog")
    @patch("mlx_stack.core.benchmark._get_all_tier_names", return_value=["fast", "standard"])
    @patch("mlx_stack.core.benchmark._get_running_tier_names", return_value=["fast"])
    def test_unknown_target_raises_with_suggestions(
        self,
        mock_running: MagicMock,
        mock_all: MagicMock,
        mock_catalog: MagicMock,
        mock_tier: MagicMock,
    ) -> None:
        from mlx_stack.core.benchmark import resolve_target

        mock_tier.return_value = None
        mock_catalog.return_value = []

        with pytest.raises(BenchmarkTargetError) as exc_info:
            resolve_target("nonexistent")

        error_msg = str(exc_info.value)
        assert "nonexistent" in error_msg
        assert "fast" in error_msg
        assert "standard" in error_msg
        assert "models --catalog" in error_msg

    @patch("mlx_stack.core.benchmark._find_running_tier")
    @patch("mlx_stack.core.benchmark.load_catalog")
    def test_resolves_running_tier(
        self,
        mock_catalog: MagicMock,
        mock_tier: MagicMock,
        sample_entry: CatalogEntry,
    ) -> None:
        from mlx_stack.core.benchmark import resolve_target

        mock_tier.return_value = {
            "name": "fast",
            "model": "qwen3.5-8b",
            "quant": "int4",
            "port": 8000,
            "source": "mlx-community/Qwen3.5-8B-4bit",
        }
        mock_catalog.return_value = [sample_entry]

        target = resolve_target("fast")
        assert target.model_id == "qwen3.5-8b"
        assert target.port == 8000
        assert target.is_running_tier is True
        assert target.temp_service_name is None


# --------------------------------------------------------------------------- #
# Test: run_benchmark (integration-level with mocks)
# --------------------------------------------------------------------------- #


class TestRunBenchmark:
    """Tests for the main run_benchmark function."""

    @patch("mlx_stack.core.benchmark.ensure_dependency")
    @patch("mlx_stack.core.benchmark.resolve_target")
    @patch("mlx_stack.core.benchmark._run_iterations")
    @patch("mlx_stack.core.benchmark.load_profile")
    @patch("mlx_stack.core.benchmark._compare_against_catalog", return_value=[])
    @patch("mlx_stack.core.benchmark._run_tool_call_benchmark")
    def test_successful_benchmark_running_tier(
        self,
        mock_tool: MagicMock,
        mock_compare: MagicMock,
        mock_profile: MagicMock,
        mock_iterations: MagicMock,
        mock_resolve: MagicMock,
        mock_deps: MagicMock,
        sample_entry: CatalogEntry,
        sample_profile: HardwareProfile,
    ) -> None:
        from mlx_stack.core.benchmark import BenchmarkTarget, run_benchmark

        mock_resolve.return_value = BenchmarkTarget(
            model_id="qwen3.5-8b",
            quant="int4",
            port=8000,
            model_name="mlx-community/Qwen3.5-8B-4bit",
            entry=sample_entry,
            is_running_tier=True,
        )
        mock_iterations.return_value = [
            IterationResult(
                prompt_tps=150.0, gen_tps=80.0,
                prompt_tokens=1000, completion_tokens=100, total_time=10.0,
            ),
            IterationResult(
                prompt_tps=160.0, gen_tps=85.0,
                prompt_tokens=1000, completion_tokens=100, total_time=9.5,
            ),
            IterationResult(
                prompt_tps=155.0, gen_tps=82.0,
                prompt_tokens=1000, completion_tokens=100, total_time=9.8,
            ),
        ]
        mock_profile.return_value = sample_profile
        mock_tool.return_value = ToolCallResult(success=True, round_trip_time=0.5)

        result = run_benchmark("fast")

        assert result.model_id == "qwen3.5-8b"
        assert result.prompt_tps_mean == pytest.approx(155.0)
        assert result.gen_tps_mean == pytest.approx(82.333, rel=1e-2)
        assert result.used_temporary_instance is False

    @patch("mlx_stack.core.benchmark.ensure_dependency")
    @patch("mlx_stack.core.benchmark.resolve_target")
    @patch("mlx_stack.core.benchmark._run_iterations")
    @patch("mlx_stack.core.benchmark.load_profile", return_value=None)
    @patch("mlx_stack.core.benchmark.detect_hardware")
    def test_benchmark_with_no_profile_detects_hardware(
        self,
        mock_detect: MagicMock,
        mock_profile: MagicMock,
        mock_iterations: MagicMock,
        mock_resolve: MagicMock,
        mock_deps: MagicMock,
        sample_entry: CatalogEntry,
        sample_profile: HardwareProfile,
    ) -> None:
        from mlx_stack.core.benchmark import BenchmarkTarget, run_benchmark

        # Non-tool-calling entry
        entry = CatalogEntry(
            id="test-model",
            name="Test",
            family="Test",
            params_b=8.0,
            architecture="transformer",
            min_mlx_lm_version="0.22.0",
            sources={"int4": QuantSource(hf_repo="test/test", disk_size_gb=4.0)},
            capabilities=Capabilities(
                tool_calling=False, tool_call_parser=None,
                thinking=False, reasoning_parser=None, vision=False,
            ),
            quality=QualityScores(overall=50, coding=50, reasoning=50, instruction_following=50),
            benchmarks={},
            tags=[],
        )
        mock_resolve.return_value = BenchmarkTarget(
            model_id="test-model",
            quant="int4",
            port=8000,
            model_name="test/test",
            entry=entry,
            is_running_tier=True,
        )
        mock_iterations.return_value = [
            IterationResult(
                prompt_tps=100.0, gen_tps=50.0,
                prompt_tokens=1000, completion_tokens=100, total_time=10.0,
            ),
        ]
        mock_detect.return_value = sample_profile

        result = run_benchmark("test-model")
        assert result.model_id == "test-model"
        mock_detect.assert_called_once()

    @patch("mlx_stack.core.benchmark.ensure_dependency")
    @patch("mlx_stack.core.benchmark.resolve_target")
    @patch("mlx_stack.core.benchmark._execute_benchmark", side_effect=BenchmarkRunError("fail"))
    @patch("mlx_stack.core.benchmark._cleanup_temp_instance")
    def test_cleanup_on_failure_with_temp_instance(
        self,
        mock_cleanup: MagicMock,
        mock_exec: MagicMock,
        mock_resolve: MagicMock,
        mock_deps: MagicMock,
        sample_entry: CatalogEntry,
    ) -> None:
        from mlx_stack.core.benchmark import BenchmarkTarget, run_benchmark

        mock_resolve.return_value = BenchmarkTarget(
            model_id="qwen3.5-8b",
            quant="int4",
            port=8100,
            model_name="test",
            entry=sample_entry,
            is_running_tier=False,
            temp_service_name="bench-temp-qwen3.5-8b",
        )

        with pytest.raises(BenchmarkRunError):
            run_benchmark("qwen3.5-8b")

        # Cleanup should be called at least once (in both except and finally)
        assert mock_cleanup.call_count >= 1

    @patch("mlx_stack.core.benchmark.ensure_dependency")
    @patch("mlx_stack.core.benchmark.resolve_target")
    @patch("mlx_stack.core.benchmark._run_iterations")
    @patch("mlx_stack.core.benchmark.load_profile")
    @patch("mlx_stack.core.benchmark._compare_against_catalog")
    @patch("mlx_stack.core.benchmark.save_benchmark_results")
    def test_save_flag_persists_results(
        self,
        mock_save: MagicMock,
        mock_compare: MagicMock,
        mock_profile: MagicMock,
        mock_iterations: MagicMock,
        mock_resolve: MagicMock,
        mock_deps: MagicMock,
        sample_entry: CatalogEntry,
        sample_profile: HardwareProfile,
    ) -> None:
        from mlx_stack.core.benchmark import BenchmarkTarget, run_benchmark

        entry = CatalogEntry(
            id="test",
            name="Test",
            family="Test",
            params_b=8.0,
            architecture="transformer",
            min_mlx_lm_version="0.22.0",
            sources={"int4": QuantSource(hf_repo="test/test", disk_size_gb=4.0)},
            capabilities=Capabilities(
                tool_calling=False, tool_call_parser=None,
                thinking=False, reasoning_parser=None, vision=False,
            ),
            quality=QualityScores(overall=50, coding=50, reasoning=50, instruction_following=50),
            benchmarks={},
            tags=[],
        )
        mock_resolve.return_value = BenchmarkTarget(
            model_id="test",
            quant="int4",
            port=8000,
            model_name="test",
            entry=entry,
            is_running_tier=True,
        )
        mock_iterations.return_value = [
            IterationResult(
                prompt_tps=100.0, gen_tps=50.0,
                prompt_tokens=1000, completion_tokens=100, total_time=10.0,
            ),
        ]
        mock_profile.return_value = sample_profile
        mock_compare.return_value = []
        mock_save.return_value = Path("/tmp/test.json")

        run_benchmark("test", save=True)
        mock_save.assert_called_once()
