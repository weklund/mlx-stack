"""Tests for the bench CLI command (cli/bench.py).

Tests cover:
- Help text output
- Successful benchmark display
- Error handling for missing targets
- Error handling for benchmark failures
- Dependency errors
- Catalog errors
- Display formatting (performance table, comparison, tool-calling)
- --save flag behavior
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from mlx_stack.cli.main import cli
from mlx_stack.core.benchmark import (
    BenchmarkError,
    BenchmarkResult_,
    BenchmarkRunError,
    BenchmarkTargetError,
    IterationResult,
    MetricClassification,
    ToolCallResult,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def runner() -> CliRunner:
    """Create a Click test runner."""
    return CliRunner()


@pytest.fixture
def sample_result() -> BenchmarkResult_:
    """A sample successful benchmark result."""
    return BenchmarkResult_(
        model_id="qwen3.5-8b",
        quant="int4",
        iterations=[
            IterationResult(
                prompt_tps=150.0,
                gen_tps=80.0,
                prompt_tokens=1000,
                completion_tokens=100,
                total_time=10.0,
            ),
            IterationResult(
                prompt_tps=160.0,
                gen_tps=85.0,
                prompt_tokens=1000,
                completion_tokens=100,
                total_time=9.5,
            ),
            IterationResult(
                prompt_tps=155.0,
                gen_tps=82.0,
                prompt_tokens=1000,
                completion_tokens=100,
                total_time=9.8,
            ),
        ],
        prompt_tps_mean=155.0,
        prompt_tps_std=5.0,
        gen_tps_mean=82.3,
        gen_tps_std=2.5,
        classifications=[
            MetricClassification(
                metric="prompt_tps",
                measured=155.0,
                catalog=155.0,
                delta_pct=0.0,
                classification="PASS",
            ),
            MetricClassification(
                metric="gen_tps",
                measured=82.3,
                catalog=85.0,
                delta_pct=3.2,
                classification="PASS",
            ),
        ],
        tool_call_result=ToolCallResult(success=True, round_trip_time=0.5),
        used_temporary_instance=False,
        catalog_data_available=True,
    )


@pytest.fixture
def sample_result_no_catalog() -> BenchmarkResult_:
    """A sample result with no catalog data for comparison."""
    return BenchmarkResult_(
        model_id="qwen3.5-8b",
        quant="int4",
        iterations=[
            IterationResult(
                prompt_tps=150.0,
                gen_tps=80.0,
                prompt_tokens=1000,
                completion_tokens=100,
                total_time=10.0,
            ),
        ],
        prompt_tps_mean=150.0,
        prompt_tps_std=0.0,
        gen_tps_mean=80.0,
        gen_tps_std=0.0,
        classifications=[],
        tool_call_result=None,
        used_temporary_instance=True,
        catalog_data_available=False,
    )


# --------------------------------------------------------------------------- #
# Test: Help text
# --------------------------------------------------------------------------- #


class TestBenchHelp:
    """Tests for bench command help text."""

    def test_bench_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["bench", "--help"])
        assert result.exit_code == 0
        assert "Benchmark a tier or model" in result.output
        assert "--save" in result.output
        assert "TARGET" in result.output

    def test_bench_in_main_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "bench" in result.output

    def test_bench_requires_target(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["bench"])
        assert result.exit_code != 0

    def test_bench_help_documents_hf_repo(self, runner: CliRunner) -> None:
        """Bench help text mentions HuggingFace repo string format."""
        result = runner.invoke(cli, ["bench", "--help"])
        assert result.exit_code == 0
        assert "HuggingFace" in result.output
        assert "mlx-community/Model-4bit" in result.output


# --------------------------------------------------------------------------- #
# Test: Successful benchmark display
# --------------------------------------------------------------------------- #


class TestBenchSuccess:
    """Tests for successful bench command output."""

    @patch("mlx_stack.core.benchmark.run_benchmark")
    def test_displays_performance_results(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
        sample_result: BenchmarkResult_,
    ) -> None:
        mock_bench.return_value = sample_result

        result = runner.invoke(cli, ["bench", "fast"])
        assert result.exit_code == 0
        assert "qwen3.5-8b" in result.output
        assert "int4" in result.output
        assert "155.0" in result.output  # prompt_tps_mean
        assert "82.3" in result.output  # gen_tps_mean

    @patch("mlx_stack.core.benchmark.run_benchmark")
    def test_displays_catalog_comparison(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
        sample_result: BenchmarkResult_,
    ) -> None:
        mock_bench.return_value = sample_result

        result = runner.invoke(cli, ["bench", "fast"])
        assert result.exit_code == 0
        assert "PASS" in result.output
        assert "Catalog Comparison" in result.output

    @patch("mlx_stack.core.benchmark.run_benchmark")
    def test_displays_tool_call_result(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
        sample_result: BenchmarkResult_,
    ) -> None:
        mock_bench.return_value = sample_result

        result = runner.invoke(cli, ["bench", "fast"])
        assert result.exit_code == 0
        assert "Tool Calling" in result.output
        assert "Valid tool call" in result.output

    @patch("mlx_stack.core.benchmark.run_benchmark")
    def test_displays_iteration_details(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
        sample_result: BenchmarkResult_,
    ) -> None:
        mock_bench.return_value = sample_result

        result = runner.invoke(cli, ["bench", "fast"])
        assert result.exit_code == 0
        assert "Iteration Details" in result.output

    @patch("mlx_stack.core.benchmark.run_benchmark")
    def test_shows_running_tier_indicator(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
        sample_result: BenchmarkResult_,
    ) -> None:
        mock_bench.return_value = sample_result

        result = runner.invoke(cli, ["bench", "fast"])
        assert result.exit_code == 0
        assert "running tier" in result.output.lower()

    @patch("mlx_stack.core.benchmark.run_benchmark")
    def test_shows_temp_instance_indicator(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
        sample_result_no_catalog: BenchmarkResult_,
    ) -> None:
        mock_bench.return_value = sample_result_no_catalog

        result = runner.invoke(cli, ["bench", "qwen3.5-8b"])
        assert result.exit_code == 0
        assert "temporary" in result.output.lower()


# --------------------------------------------------------------------------- #
# Test: No catalog data
# --------------------------------------------------------------------------- #


class TestBenchNoCatalog:
    """Tests for bench output when no catalog data is available."""

    @patch("mlx_stack.core.benchmark.run_benchmark")
    def test_shows_no_catalog_message(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
        sample_result_no_catalog: BenchmarkResult_,
    ) -> None:
        mock_bench.return_value = sample_result_no_catalog

        result = runner.invoke(cli, ["bench", "qwen3.5-8b"])
        assert result.exit_code == 0
        assert "No catalog benchmark data" in result.output
        assert "--save" in result.output


# --------------------------------------------------------------------------- #
# Test: --save flag
# --------------------------------------------------------------------------- #


class TestBenchSave:
    """Tests for --save flag behavior."""

    @patch("mlx_stack.core.benchmark.run_benchmark")
    def test_save_flag_passed_to_benchmark(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
        sample_result: BenchmarkResult_,
    ) -> None:
        mock_bench.return_value = sample_result

        result = runner.invoke(cli, ["bench", "fast", "--save"])
        assert result.exit_code == 0
        mock_bench.assert_called_once_with(target="fast", save=True)
        assert "Results saved" in result.output

    @patch("mlx_stack.core.benchmark.run_benchmark")
    def test_no_save_flag(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
        sample_result: BenchmarkResult_,
    ) -> None:
        mock_bench.return_value = sample_result

        result = runner.invoke(cli, ["bench", "fast"])
        assert result.exit_code == 0
        mock_bench.assert_called_once_with(target="fast", save=False)


# --------------------------------------------------------------------------- #
# Test: Error handling
# --------------------------------------------------------------------------- #


class TestBenchErrors:
    """Tests for bench command error handling."""

    @patch("mlx_stack.core.benchmark.run_benchmark")
    def test_target_error_shows_suggestions(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
    ) -> None:
        mock_bench.side_effect = BenchmarkTargetError(
            "'nonexistent' is not a running tier or known model.\n"
            "Valid tier names: fast, standard\n"
            "Use 'mlx-stack models --catalog' to see available models."
        )

        result = runner.invoke(cli, ["bench", "nonexistent"])
        assert result.exit_code == 1
        assert "nonexistent" in result.output

    @patch("mlx_stack.core.benchmark.run_benchmark")
    def test_benchmark_run_error(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
    ) -> None:
        mock_bench.side_effect = BenchmarkRunError("API request failed")

        result = runner.invoke(cli, ["bench", "fast"])
        assert result.exit_code == 1
        assert "Benchmark error" in result.output

    @patch("mlx_stack.core.benchmark.run_benchmark")
    def test_generic_benchmark_error(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
    ) -> None:
        mock_bench.side_effect = BenchmarkError("Something went wrong")

        result = runner.invoke(cli, ["bench", "fast"])
        assert result.exit_code == 1
        assert "Error" in result.output

    @patch("mlx_stack.core.benchmark.run_benchmark")
    def test_dependency_error(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
    ) -> None:
        from mlx_stack.core.deps import DependencyError

        mock_bench.side_effect = DependencyError("uv not found")

        result = runner.invoke(cli, ["bench", "fast"])
        assert result.exit_code == 1
        assert "Dependency error" in result.output

    @patch("mlx_stack.core.benchmark.run_benchmark")
    def test_dependency_install_error(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
    ) -> None:
        from mlx_stack.core.deps import DependencyInstallError

        mock_bench.side_effect = DependencyInstallError("install failed")

        result = runner.invoke(cli, ["bench", "fast"])
        assert result.exit_code == 1
        assert "Dependency error" in result.output

    @patch("mlx_stack.core.benchmark.run_benchmark")
    def test_catalog_error(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
    ) -> None:
        from mlx_stack.core.catalog import CatalogError

        mock_bench.side_effect = CatalogError("Corrupt catalog")

        result = runner.invoke(cli, ["bench", "fast"])
        assert result.exit_code == 1
        assert "Catalog error" in result.output

    def test_no_python_traceback_on_error(self, runner: CliRunner) -> None:
        """Verify no Python traceback appears in error output."""
        with patch("mlx_stack.core.benchmark.run_benchmark") as mock_bench:
            mock_bench.side_effect = BenchmarkTargetError("not found")
            result = runner.invoke(cli, ["bench", "nonexistent"])
            assert "Traceback" not in result.output


# --------------------------------------------------------------------------- #
# Test: WARN and FAIL classifications display
# --------------------------------------------------------------------------- #


class TestClassificationDisplay:
    """Tests for WARN and FAIL classification display."""

    @patch("mlx_stack.core.benchmark.run_benchmark")
    def test_warn_classification_displayed(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
    ) -> None:
        result_data = BenchmarkResult_(
            model_id="test-model",
            quant="int4",
            iterations=[
                IterationResult(
                    prompt_tps=80.0,
                    gen_tps=60.0,
                    prompt_tokens=1000,
                    completion_tokens=100,
                    total_time=10.0,
                ),
            ],
            prompt_tps_mean=80.0,
            prompt_tps_std=0.0,
            gen_tps_mean=60.0,
            gen_tps_std=0.0,
            classifications=[
                MetricClassification(
                    metric="gen_tps",
                    measured=60.0,
                    catalog=85.0,
                    delta_pct=29.4,
                    classification="WARN",
                ),
            ],
            catalog_data_available=True,
        )
        mock_bench.return_value = result_data

        result = runner.invoke(cli, ["bench", "fast"])
        assert result.exit_code == 0
        assert "WARN" in result.output

    @patch("mlx_stack.core.benchmark.run_benchmark")
    def test_fail_classification_displayed(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
    ) -> None:
        result_data = BenchmarkResult_(
            model_id="test-model",
            quant="int4",
            iterations=[
                IterationResult(
                    prompt_tps=50.0,
                    gen_tps=30.0,
                    prompt_tokens=1000,
                    completion_tokens=100,
                    total_time=10.0,
                ),
            ],
            prompt_tps_mean=50.0,
            prompt_tps_std=0.0,
            gen_tps_mean=30.0,
            gen_tps_std=0.0,
            classifications=[
                MetricClassification(
                    metric="gen_tps",
                    measured=30.0,
                    catalog=85.0,
                    delta_pct=64.7,
                    classification="FAIL",
                ),
            ],
            catalog_data_available=True,
        )
        mock_bench.return_value = result_data

        result = runner.invoke(cli, ["bench", "fast"])
        assert result.exit_code == 0
        assert "FAIL" in result.output


# --------------------------------------------------------------------------- #
# Test: Delta sign correctness
# --------------------------------------------------------------------------- #


class TestDeltaSignDisplay:
    """Tests that delta percentages display the correct sign.

    delta_pct is (catalog - measured) / catalog * 100, so:
    - positive delta = measured below catalog (slower)
    - negative delta = measured above catalog (faster)
    - zero delta = exact match
    """

    @patch("mlx_stack.core.benchmark.run_benchmark")
    def test_below_catalog_shows_positive_delta(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
    ) -> None:
        """Measured 60 vs catalog 85 → delta_pct=+29.4 → should display +29.4%."""
        result_data = BenchmarkResult_(
            model_id="test-model",
            quant="int4",
            iterations=[
                IterationResult(
                    prompt_tps=80.0,
                    gen_tps=60.0,
                    prompt_tokens=1000,
                    completion_tokens=100,
                    total_time=10.0,
                ),
            ],
            prompt_tps_mean=80.0,
            prompt_tps_std=0.0,
            gen_tps_mean=60.0,
            gen_tps_std=0.0,
            classifications=[
                MetricClassification(
                    metric="gen_tps",
                    measured=60.0,
                    catalog=85.0,
                    delta_pct=29.4,
                    classification="WARN",
                ),
            ],
            catalog_data_available=True,
        )
        mock_bench.return_value = result_data

        result = runner.invoke(cli, ["bench", "fast"])
        assert result.exit_code == 0
        assert "+29.4%" in result.output

    @patch("mlx_stack.core.benchmark.run_benchmark")
    def test_above_catalog_shows_negative_delta(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
    ) -> None:
        """Measured 90 vs catalog 85 → delta_pct=-5.9 → should display -5.9%."""
        result_data = BenchmarkResult_(
            model_id="test-model",
            quant="int4",
            iterations=[
                IterationResult(
                    prompt_tps=100.0,
                    gen_tps=90.0,
                    prompt_tokens=1000,
                    completion_tokens=100,
                    total_time=10.0,
                ),
            ],
            prompt_tps_mean=100.0,
            prompt_tps_std=0.0,
            gen_tps_mean=90.0,
            gen_tps_std=0.0,
            classifications=[
                MetricClassification(
                    metric="gen_tps",
                    measured=90.0,
                    catalog=85.0,
                    delta_pct=-5.9,
                    classification="PASS",
                ),
            ],
            catalog_data_available=True,
        )
        mock_bench.return_value = result_data

        result = runner.invoke(cli, ["bench", "fast"])
        assert result.exit_code == 0
        assert "-5.9%" in result.output

    @patch("mlx_stack.core.benchmark.run_benchmark")
    def test_exact_match_shows_zero_delta(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
    ) -> None:
        """Measured == catalog → delta_pct=0.0 → should display +0.0%."""
        result_data = BenchmarkResult_(
            model_id="test-model",
            quant="int4",
            iterations=[
                IterationResult(
                    prompt_tps=155.0,
                    gen_tps=85.0,
                    prompt_tokens=1000,
                    completion_tokens=100,
                    total_time=10.0,
                ),
            ],
            prompt_tps_mean=155.0,
            prompt_tps_std=0.0,
            gen_tps_mean=85.0,
            gen_tps_std=0.0,
            classifications=[
                MetricClassification(
                    metric="gen_tps",
                    measured=85.0,
                    catalog=85.0,
                    delta_pct=0.0,
                    classification="PASS",
                ),
            ],
            catalog_data_available=True,
        )
        mock_bench.return_value = result_data

        result = runner.invoke(cli, ["bench", "fast"])
        assert result.exit_code == 0
        assert "+0.0%" in result.output


# --------------------------------------------------------------------------- #
# Test: Tool-calling display variants
# --------------------------------------------------------------------------- #


class TestToolCallingDisplay:
    """Tests for tool-calling display variants."""

    @patch("mlx_stack.core.benchmark.run_benchmark")
    def test_failed_tool_call_displayed(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
    ) -> None:
        result_data = BenchmarkResult_(
            model_id="test-model",
            quant="int4",
            iterations=[
                IterationResult(
                    prompt_tps=100.0,
                    gen_tps=50.0,
                    prompt_tokens=1000,
                    completion_tokens=100,
                    total_time=10.0,
                ),
            ],
            prompt_tps_mean=100.0,
            prompt_tps_std=0.0,
            gen_tps_mean=50.0,
            gen_tps_std=0.0,
            tool_call_result=ToolCallResult(
                success=False,
                round_trip_time=1.0,
                error="No tool calls in response",
            ),
            catalog_data_available=True,
        )
        mock_bench.return_value = result_data

        result = runner.invoke(cli, ["bench", "fast"])
        assert result.exit_code == 0
        assert "Tool call failed" in result.output

    @patch("mlx_stack.core.benchmark.run_benchmark")
    def test_no_tool_calling_skip_message(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
    ) -> None:
        result_data = BenchmarkResult_(
            model_id="test-model",
            quant="int4",
            iterations=[
                IterationResult(
                    prompt_tps=100.0,
                    gen_tps=50.0,
                    prompt_tokens=1000,
                    completion_tokens=100,
                    total_time=10.0,
                ),
            ],
            prompt_tps_mean=100.0,
            prompt_tps_std=0.0,
            gen_tps_mean=50.0,
            gen_tps_std=0.0,
            tool_call_result=None,
            catalog_data_available=True,
        )
        mock_bench.return_value = result_data

        result = runner.invoke(cli, ["bench", "fast"])
        assert result.exit_code == 0
        assert "skipped" in result.output.lower() or "does not support" in result.output.lower()


# --------------------------------------------------------------------------- #
# Test: HF repo string as standalone bench command
# --------------------------------------------------------------------------- #


class TestBenchHfRepo:
    """Tests for mlx-stack bench with HuggingFace repo strings."""

    @patch("mlx_stack.core.benchmark.run_benchmark")
    def test_hf_repo_bench_standalone(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
    ) -> None:
        """mlx-stack bench mlx-community/Model-4bit works as standalone command."""
        mock_bench.return_value = BenchmarkResult_(
            model_id="mlx-community/Model-4bit",
            quant="int4",
            iterations=[
                IterationResult(
                    prompt_tps=120.0,
                    gen_tps=65.0,
                    prompt_tokens=1000,
                    completion_tokens=100,
                    total_time=12.0,
                ),
            ],
            prompt_tps_mean=120.0,
            prompt_tps_std=0.0,
            gen_tps_mean=65.0,
            gen_tps_std=0.0,
            used_temporary_instance=True,
            catalog_data_available=False,
        )

        result = runner.invoke(cli, ["bench", "mlx-community/Model-4bit"])
        assert result.exit_code == 0
        assert "mlx-community/Model-4bit" in result.output
        mock_bench.assert_called_once_with(target="mlx-community/Model-4bit", save=False)

    @patch("mlx_stack.core.benchmark.run_benchmark")
    def test_hf_repo_bench_with_save(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
    ) -> None:
        """mlx-stack bench mlx-community/Model-4bit --save saves results."""
        mock_bench.return_value = BenchmarkResult_(
            model_id="mlx-community/Model-4bit",
            quant="int4",
            iterations=[
                IterationResult(
                    prompt_tps=120.0,
                    gen_tps=65.0,
                    prompt_tokens=1000,
                    completion_tokens=100,
                    total_time=12.0,
                ),
            ],
            prompt_tps_mean=120.0,
            prompt_tps_std=0.0,
            gen_tps_mean=65.0,
            gen_tps_std=0.0,
            used_temporary_instance=True,
            catalog_data_available=False,
        )

        result = runner.invoke(cli, ["bench", "mlx-community/Model-4bit", "--save"])
        assert result.exit_code == 0
        mock_bench.assert_called_once_with(target="mlx-community/Model-4bit", save=True)
        assert "Results saved" in result.output

    @patch("mlx_stack.core.benchmark.run_benchmark")
    def test_hf_repo_bench_shows_temp_instance(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
    ) -> None:
        """HF repo bench shows 'temporary' instance indicator."""
        mock_bench.return_value = BenchmarkResult_(
            model_id="mlx-community/Model-4bit",
            quant="int4",
            iterations=[
                IterationResult(
                    prompt_tps=120.0,
                    gen_tps=65.0,
                    prompt_tokens=1000,
                    completion_tokens=100,
                    total_time=12.0,
                ),
            ],
            prompt_tps_mean=120.0,
            prompt_tps_std=0.0,
            gen_tps_mean=65.0,
            gen_tps_std=0.0,
            used_temporary_instance=True,
            catalog_data_available=False,
        )

        result = runner.invoke(cli, ["bench", "mlx-community/Model-4bit"])
        assert result.exit_code == 0
        assert "temporary" in result.output.lower()


# --------------------------------------------------------------------------- #
# Test: --save output references updated commands
# --------------------------------------------------------------------------- #


class TestBenchSaveOutputReferences:
    """Tests that --save output does not reference removed commands."""

    @patch("mlx_stack.core.benchmark.run_benchmark")
    def test_save_output_no_recommend_or_init(
        self,
        mock_bench: MagicMock,
        runner: CliRunner,
        sample_result: BenchmarkResult_,
    ) -> None:
        """VAL-CROSS-009: bench --save output does not reference 'recommend' or 'init'."""
        mock_bench.return_value = sample_result

        result = runner.invoke(cli, ["bench", "fast", "--save"])
        assert result.exit_code == 0
        # Should not reference removed commands
        lower_output = result.output.lower()
        # Check that "recommend" and "init" don't appear as standalone command references
        # (they may appear as substrings of other words, so check for command patterns)
        assert "'recommend'" not in lower_output
        assert "'init'" not in lower_output
