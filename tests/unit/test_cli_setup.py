"""Behavioral tests for the setup CLI command.

Tests the user-visible behavior: what the command outputs, what exit
codes it produces — not which internal functions are called.

Config generation, model pulling, and stack startup are tested
separately in test_onboarding.py. These tests verify the CLI layer:
display, prompts, error messages, exit codes.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from click.testing import CliRunner

from mlx_stack.cli.setup import setup

# --------------------------------------------------------------------------- #
# Mock data
# --------------------------------------------------------------------------- #

MOCK_PROFILE = SimpleNamespace(
    chip="Apple M4 Pro",
    gpu_cores=20,
    memory_gb=64,
    bandwidth_gbps=273.0,
    profile_id="m4-pro-64",
)

MOCK_UP_RESULT = SimpleNamespace(
    tiers=[
        SimpleNamespace(
            name="standard", model="Qwen3.5-9B",
            port=8000, status="healthy", error=None,
        ),
        SimpleNamespace(
            name="fast", model="SmallFast-4B",
            port=8001, status="healthy", error=None,
        ),
    ],
    litellm=SimpleNamespace(
        name="litellm", model="proxy",
        port=4000, status="healthy", error=None,
    ),
    dry_run=False,
    warnings=[],
    already_running=False,
)

MOCK_BENCHMARK_DATA = {
    "models": {
        "mlx-community/Qwen3.5-9B-4bit": {
            "params_b": 9.0, "thinking": True, "tool_calling": True,
            "benchmarks": {"m4-pro-64": {
                "generation_tps": 62.0, "prompt_tps": 337.0,
                "peak_memory_gib": 5.2,
            }},
            "quality": {"overall_pass_rate": 0.98},
        },
        "mlx-community/SmallFast-4B-4bit": {
            "params_b": 4.0, "thinking": False, "tool_calling": False,
            "benchmarks": {"m4-pro-64": {
                "generation_tps": 95.0, "prompt_tps": 500.0,
                "peak_memory_gib": 2.4,
            }},
            "quality": {"overall_pass_rate": 0.91},
        },
    },
}


def _run_setup(args: list[str], mlx_stack_home: Path) -> Any:
    """Run setup with all external deps mocked. Returns CliRunner Result."""
    runner = CliRunner()

    with (
        patch("mlx_stack.core.onboarding.detect_hardware", return_value=MOCK_PROFILE),
        patch("mlx_stack.core.onboarding.save_profile"),
        patch("mlx_stack.core.discovery.query_hf_models", return_value=[]),
        patch("mlx_stack.core.discovery.load_benchmark_data",
              return_value=MOCK_BENCHMARK_DATA),
        patch("mlx_stack.cli.setup.generate_config",
              return_value=(mlx_stack_home / "stacks" / "default.yaml",
                            mlx_stack_home / "litellm.yaml")),
        patch("mlx_stack.cli.setup.pull_setup_models", return_value=[]),
        patch("mlx_stack.cli.setup.start_stack", return_value=MOCK_UP_RESULT),
    ):
        result = runner.invoke(setup, args)

    return result


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestSetupAcceptDefaults:
    """--accept-defaults runs the full flow non-interactively."""

    def test_completes_successfully(self, mlx_stack_home: Path) -> None:
        """Setup with --accept-defaults exits 0."""
        result = _run_setup(["--accept-defaults"], mlx_stack_home)
        assert result.exit_code == 0, (
            f"Exit {result.exit_code}:\n{result.output}"
        )

    def test_shows_hardware_info(self, mlx_stack_home: Path) -> None:
        """Output includes detected hardware details."""
        result = _run_setup(["--accept-defaults"], mlx_stack_home)
        assert "Apple M4 Pro" in result.output
        assert "64 GB" in result.output

    def test_shows_model_names(self, mlx_stack_home: Path) -> None:
        """Output includes discovered model names."""
        result = _run_setup(["--accept-defaults"], mlx_stack_home)
        assert "Qwen3.5-9B" in result.output

    def test_shows_api_endpoint(self, mlx_stack_home: Path) -> None:
        """Output includes the API endpoint for connecting tools."""
        result = _run_setup(["--accept-defaults"], mlx_stack_home)
        assert "localhost" in result.output
        assert "/v1" in result.output

    def test_shows_tier_assignment(self, mlx_stack_home: Path) -> None:
        """Output shows which model is assigned to which tier."""
        result = _run_setup(["--accept-defaults"], mlx_stack_home)
        assert "standard" in result.output

    def test_does_not_prompt(self, mlx_stack_home: Path) -> None:
        """--accept-defaults never shows interactive prompts."""
        result = _run_setup(["--accept-defaults"], mlx_stack_home)
        assert "Install LaunchAgent?" not in result.output
        assert "How will you use your stack?" not in result.output


class TestSetupIntentFlag:
    """--intent flag controls which intent is used without prompting."""

    def test_intent_flag_accepted(self, mlx_stack_home: Path) -> None:
        """Providing --intent with --accept-defaults works."""
        result = _run_setup(
            ["--accept-defaults", "--intent", "agent-fleet"], mlx_stack_home,
        )
        assert result.exit_code == 0


class TestSetupErrorHandling:
    """Setup shows clear errors for failure cases."""

    def test_hardware_detection_failure(self, mlx_stack_home: Path) -> None:
        """Hardware detection failure shows error and exits 1."""
        runner = CliRunner()

        with (
            patch("mlx_stack.core.onboarding.detect_hardware",
                  side_effect=RuntimeError("no chip")),
            patch("mlx_stack.core.onboarding.save_profile"),
        ):
            result = runner.invoke(setup, ["--accept-defaults"])

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_no_models_found(self, mlx_stack_home: Path) -> None:
        """No models available shows error and exits 1."""
        runner = CliRunner()

        with (
            patch("mlx_stack.core.onboarding.detect_hardware",
                  return_value=MOCK_PROFILE),
            patch("mlx_stack.core.onboarding.save_profile"),
            patch("mlx_stack.core.discovery.query_hf_models",
                  return_value=[]),
            patch("mlx_stack.core.discovery.load_benchmark_data",
                  return_value={"models": {}}),
        ):
            result = runner.invoke(setup, ["--accept-defaults"])

        assert result.exit_code == 1
