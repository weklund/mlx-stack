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

import yaml
from click.testing import CliRunner

from mlx_stack.cli.setup import setup
from tests.factories import make_entry, make_stack_yaml, write_litellm_yaml, write_stack_yaml

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
            name="standard",
            model="Qwen3.5-9B",
            port=8000,
            status="healthy",
            error=None,
        ),
        SimpleNamespace(
            name="fast",
            model="SmallFast-4B",
            port=8001,
            status="healthy",
            error=None,
        ),
    ],
    litellm=SimpleNamespace(
        name="litellm",
        model="proxy",
        port=4000,
        status="healthy",
        error=None,
    ),
    dry_run=False,
    warnings=[],
    already_running=False,
)

MOCK_BENCHMARK_DATA = {
    "models": {
        "mlx-community/Qwen3.5-9B-4bit": {
            "params_b": 9.0,
            "thinking": True,
            "tool_calling": True,
            "benchmarks": {
                "m4-pro-64": {
                    "generation_tps": 62.0,
                    "prompt_tps": 337.0,
                    "peak_memory_gib": 5.2,
                }
            },
            "quality": {"overall_pass_rate": 0.98},
        },
        "mlx-community/SmallFast-4B-4bit": {
            "params_b": 4.0,
            "thinking": False,
            "tool_calling": False,
            "benchmarks": {
                "m4-pro-64": {
                    "generation_tps": 95.0,
                    "prompt_tps": 500.0,
                    "peak_memory_gib": 2.4,
                }
            },
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
        patch("mlx_stack.core.discovery.load_benchmark_data", return_value=MOCK_BENCHMARK_DATA),
        patch(
            "mlx_stack.cli.setup.generate_config",
            return_value=(
                mlx_stack_home / "stacks" / "default.yaml",
                mlx_stack_home / "litellm.yaml",
            ),
        ),
        patch("mlx_stack.cli.setup.pull_setup_models", return_value=[]),
        patch("mlx_stack.cli.setup.start_stack", return_value=MOCK_UP_RESULT),
    ):
        return runner.invoke(setup, args)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestSetupAcceptDefaults:
    """--accept-defaults runs the full flow non-interactively."""

    def test_completes_successfully(self, mlx_stack_home: Path) -> None:
        """Setup with --accept-defaults exits 0."""
        result = _run_setup(["--accept-defaults"], mlx_stack_home)
        assert result.exit_code == 0, f"Exit {result.exit_code}:\n{result.output}"

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
            ["--accept-defaults", "--intent", "agent-fleet"],
            mlx_stack_home,
        )
        assert result.exit_code == 0


class TestSetupErrorHandling:
    """Setup shows clear errors for failure cases."""

    def test_hardware_detection_failure(self, mlx_stack_home: Path) -> None:
        """Hardware detection failure shows error and exits 1."""
        runner = CliRunner()

        with (
            patch("mlx_stack.core.onboarding.detect_hardware", side_effect=RuntimeError("no chip")),
            patch("mlx_stack.core.onboarding.save_profile"),
        ):
            result = runner.invoke(setup, ["--accept-defaults"])

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_no_models_found(self, mlx_stack_home: Path) -> None:
        """No models available shows error and exits 1."""
        runner = CliRunner()

        with (
            patch("mlx_stack.core.onboarding.detect_hardware", return_value=MOCK_PROFILE),
            patch("mlx_stack.core.onboarding.save_profile"),
            patch("mlx_stack.core.discovery.query_hf_models", return_value=[]),
            patch("mlx_stack.core.discovery.load_benchmark_data", return_value={"models": {}}),
        ):
            result = runner.invoke(setup, ["--accept-defaults"])

        assert result.exit_code == 1


# --------------------------------------------------------------------------- #
# Helpers for stack modification tests
# --------------------------------------------------------------------------- #

# Standard two-tier stack for modification tests
_TWO_TIER_STACK = make_stack_yaml(
    tiers=[
        {
            "name": "standard",
            "model": "big-model",
            "quant": "int4",
            "source": "mlx-community/big-model-4bit",
            "port": 8000,
            "vllm_flags": {"continuous_batching": True, "use_paged_cache": True},
        },
        {
            "name": "fast",
            "model": "fast-model",
            "quant": "int4",
            "source": "mlx-community/fast-model-4bit",
            "port": 8001,
            "vllm_flags": {"continuous_batching": True, "use_paged_cache": True},
        },
    ],
)

_THREE_TIER_STACK = make_stack_yaml(
    tiers=[
        {
            "name": "standard",
            "model": "big-model",
            "quant": "int4",
            "source": "mlx-community/big-model-4bit",
            "port": 8000,
            "vllm_flags": {"continuous_batching": True, "use_paged_cache": True},
        },
        {
            "name": "fast",
            "model": "fast-model",
            "quant": "int4",
            "source": "mlx-community/fast-model-4bit",
            "port": 8001,
            "vllm_flags": {"continuous_batching": True, "use_paged_cache": True},
        },
        {
            "name": "reasoning",
            "model": "reason-model",
            "quant": "int4",
            "source": "mlx-community/reason-model-4bit",
            "port": 8002,
            "vllm_flags": {"continuous_batching": True, "use_paged_cache": True},
        },
    ],
)

_ONE_TIER_STACK = make_stack_yaml(
    tiers=[
        {
            "name": "standard",
            "model": "big-model",
            "quant": "int4",
            "source": "mlx-community/big-model-4bit",
            "port": 8000,
            "vllm_flags": {"continuous_batching": True, "use_paged_cache": True},
        },
    ],
)

# Catalog entry for resolving catalog IDs
_MOCK_CATALOG_ENTRY = make_entry(
    model_id="qwen3.5-8b",
    name="Qwen 3.5 8B",
    family="Qwen 3.5",
    params_b=8.0,
)


def _setup_existing_stack(
    mlx_stack_home: Path,
    stack: dict[str, Any] | None = None,
) -> Path:
    """Write an existing stack and litellm config. Returns stack path."""
    stack_path = write_stack_yaml(mlx_stack_home, stack)
    write_litellm_yaml(mlx_stack_home)
    return stack_path


# --------------------------------------------------------------------------- #
# Tests for --add flag
# --------------------------------------------------------------------------- #


class TestSetupAddHfRepo:
    """--add with HF repo string adds model to existing stack."""

    def test_add_hf_repo_adds_tier(self, mlx_stack_home: Path) -> None:
        """--add mlx-community/Model-4bit adds a new tier to existing stack."""
        _setup_existing_stack(mlx_stack_home, _TWO_TIER_STACK)
        runner = CliRunner()

        result = runner.invoke(
            setup,
            ["--add", "mlx-community/Phi-4-mini-instruct-4bit"],
        )

        assert result.exit_code == 0, f"Exit {result.exit_code}:\n{result.output}"
        stack = yaml.safe_load(
            (mlx_stack_home / "stacks" / "default.yaml").read_text()
        )
        assert len(stack["tiers"]) == 3
        new_tier = stack["tiers"][2]
        assert new_tier["source"] == "mlx-community/Phi-4-mini-instruct-4bit"

    def test_add_hf_repo_output_mentions_mlx_stack_up(self, mlx_stack_home: Path) -> None:
        """Output tells user to run 'mlx-stack up'."""
        _setup_existing_stack(mlx_stack_home, _TWO_TIER_STACK)
        runner = CliRunner()

        result = runner.invoke(
            setup,
            ["--add", "mlx-community/Phi-4-mini-instruct-4bit"],
        )

        assert result.exit_code == 0
        assert "mlx-stack up" in result.output

    def test_add_hf_repo_output_describes_change(self, mlx_stack_home: Path) -> None:
        """Output describes what was added."""
        _setup_existing_stack(mlx_stack_home, _TWO_TIER_STACK)
        runner = CliRunner()

        result = runner.invoke(
            setup,
            ["--add", "mlx-community/Phi-4-mini-instruct-4bit"],
        )

        assert result.exit_code == 0
        assert "Added" in result.output or "added" in result.output

    def test_add_hf_repo_updates_litellm(self, mlx_stack_home: Path) -> None:
        """--add also updates litellm.yaml with new tier."""
        _setup_existing_stack(mlx_stack_home, _TWO_TIER_STACK)
        runner = CliRunner()

        result = runner.invoke(
            setup,
            ["--add", "mlx-community/Phi-4-mini-instruct-4bit"],
        )

        assert result.exit_code == 0
        litellm = yaml.safe_load(
            (mlx_stack_home / "litellm.yaml").read_text()
        )
        model_names = [m["model_name"] for m in litellm["model_list"]]
        assert len(model_names) == 3

    def test_add_hf_repo_auto_assigns_tier_name(self, mlx_stack_home: Path) -> None:
        """--add without --as auto-generates a non-empty tier name."""
        _setup_existing_stack(mlx_stack_home, _TWO_TIER_STACK)
        runner = CliRunner()

        result = runner.invoke(
            setup,
            ["--add", "mlx-community/Phi-4-mini-instruct-4bit"],
        )

        assert result.exit_code == 0
        stack = yaml.safe_load(
            (mlx_stack_home / "stacks" / "default.yaml").read_text()
        )
        new_tier = stack["tiers"][2]
        assert new_tier["name"]  # non-empty


class TestSetupAddCatalogId:
    """--add with catalog ID resolves and adds model."""

    def test_add_catalog_id_resolves_and_adds(self, mlx_stack_home: Path) -> None:
        """--add qwen3.5-8b resolves catalog ID to HF repo and adds tier."""
        _setup_existing_stack(mlx_stack_home, _TWO_TIER_STACK)
        runner = CliRunner()

        with patch(
            "mlx_stack.cli.setup.get_entry_by_id",
            return_value=_MOCK_CATALOG_ENTRY,
        ):
            result = runner.invoke(setup, ["--add", "qwen3.5-8b"])

        assert result.exit_code == 0, f"Exit {result.exit_code}:\n{result.output}"
        stack = yaml.safe_load(
            (mlx_stack_home / "stacks" / "default.yaml").read_text()
        )
        assert len(stack["tiers"]) == 3
        new_tier = stack["tiers"][2]
        assert "mlx-community" in new_tier["source"]

    def test_add_invalid_catalog_id_shows_error(self, mlx_stack_home: Path) -> None:
        """--add with invalid catalog ID produces model-not-found error."""
        _setup_existing_stack(mlx_stack_home, _TWO_TIER_STACK)
        runner = CliRunner()

        with patch(
            "mlx_stack.cli.setup.get_entry_by_id",
            return_value=None,
        ):
            result = runner.invoke(setup, ["--add", "nonexistent-model"])

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()


class TestSetupAddAsFlag:
    """--as flag assigns custom tier name."""

    def test_add_with_as_sets_custom_name(self, mlx_stack_home: Path) -> None:
        """--add Model --as reasoning creates tier named 'reasoning'."""
        _setup_existing_stack(mlx_stack_home, _TWO_TIER_STACK)
        runner = CliRunner()

        result = runner.invoke(
            setup,
            ["--add", "mlx-community/SomeModel-4bit", "--as", "reasoning"],
        )

        assert result.exit_code == 0, f"Exit {result.exit_code}:\n{result.output}"
        stack = yaml.safe_load(
            (mlx_stack_home / "stacks" / "default.yaml").read_text()
        )
        tier_names = [t["name"] for t in stack["tiers"]]
        assert "reasoning" in tier_names

    def test_add_with_duplicate_as_errors(self, mlx_stack_home: Path) -> None:
        """--as with existing tier name produces duplicate error."""
        _setup_existing_stack(mlx_stack_home, _TWO_TIER_STACK)
        runner = CliRunner()

        result = runner.invoke(
            setup,
            ["--add", "mlx-community/SomeModel-4bit", "--as", "standard"],
        )

        assert result.exit_code != 0
        assert "duplicate" in result.output.lower() or "already exists" in result.output.lower()

    def test_as_without_add_errors(self, mlx_stack_home: Path) -> None:
        """--as without --add produces an error."""
        runner = CliRunner()

        result = runner.invoke(setup, ["--as", "custom-name"])

        assert result.exit_code != 0
        assert "--as" in result.output and "--add" in result.output


class TestSetupAddMultiple:
    """Multiple --add flags in one invocation."""

    def test_add_two_models(self, mlx_stack_home: Path) -> None:
        """Two --add flags add two tiers."""
        _setup_existing_stack(mlx_stack_home, _TWO_TIER_STACK)
        runner = CliRunner()

        result = runner.invoke(
            setup,
            [
                "--add", "mlx-community/Model1-4bit",
                "--add", "mlx-community/Model2-4bit",
            ],
        )

        assert result.exit_code == 0, f"Exit {result.exit_code}:\n{result.output}"
        stack = yaml.safe_load(
            (mlx_stack_home / "stacks" / "default.yaml").read_text()
        )
        assert len(stack["tiers"]) == 4


class TestSetupAddNoExistingStack:
    """--add on nonexistent stack produces error."""

    def test_add_without_stack_errors(self, mlx_stack_home: Path) -> None:
        """--add with no existing stack.yaml shows error."""
        runner = CliRunner()

        result = runner.invoke(
            setup,
            ["--add", "mlx-community/Model-4bit"],
        )

        assert result.exit_code != 0
        assert "setup" in result.output.lower()


# --------------------------------------------------------------------------- #
# Tests for --remove flag
# --------------------------------------------------------------------------- #


class TestSetupRemove:
    """--remove removes tier from existing stack."""

    def test_remove_tier(self, mlx_stack_home: Path) -> None:
        """--remove fast removes fast tier from stack.yaml."""
        _setup_existing_stack(mlx_stack_home, _TWO_TIER_STACK)
        runner = CliRunner()

        result = runner.invoke(setup, ["--remove", "fast"])

        assert result.exit_code == 0, f"Exit {result.exit_code}:\n{result.output}"
        stack = yaml.safe_load(
            (mlx_stack_home / "stacks" / "default.yaml").read_text()
        )
        tier_names = [t["name"] for t in stack["tiers"]]
        assert "fast" not in tier_names
        assert len(stack["tiers"]) == 1

    def test_remove_tier_output_mentions_up(self, mlx_stack_home: Path) -> None:
        """Output tells user to run 'mlx-stack up' after removal."""
        _setup_existing_stack(mlx_stack_home, _TWO_TIER_STACK)
        runner = CliRunner()

        result = runner.invoke(setup, ["--remove", "fast"])

        assert result.exit_code == 0
        assert "mlx-stack up" in result.output

    def test_remove_tier_describes_change(self, mlx_stack_home: Path) -> None:
        """Output describes what was removed."""
        _setup_existing_stack(mlx_stack_home, _TWO_TIER_STACK)
        runner = CliRunner()

        result = runner.invoke(setup, ["--remove", "fast"])

        assert result.exit_code == 0
        assert "Removed" in result.output or "removed" in result.output

    def test_remove_updates_litellm(self, mlx_stack_home: Path) -> None:
        """--remove also updates litellm.yaml."""
        _setup_existing_stack(mlx_stack_home, _TWO_TIER_STACK)
        runner = CliRunner()

        result = runner.invoke(setup, ["--remove", "fast"])

        assert result.exit_code == 0
        litellm = yaml.safe_load(
            (mlx_stack_home / "litellm.yaml").read_text()
        )
        model_names = [m["model_name"] for m in litellm["model_list"]]
        assert "fast" not in model_names

    def test_remove_nonexistent_tier_errors(self, mlx_stack_home: Path) -> None:
        """--remove with nonexistent tier shows error with valid tier names."""
        _setup_existing_stack(mlx_stack_home, _TWO_TIER_STACK)
        runner = CliRunner()

        result = runner.invoke(setup, ["--remove", "nonexistent"])

        assert result.exit_code != 0
        assert "nonexistent" in result.output
        # Should list valid tier names
        assert "standard" in result.output or "fast" in result.output

    def test_remove_all_tiers_errors(self, mlx_stack_home: Path) -> None:
        """--remove that would empty the stack shows error."""
        _setup_existing_stack(mlx_stack_home, _ONE_TIER_STACK)
        runner = CliRunner()

        result = runner.invoke(setup, ["--remove", "standard"])

        assert result.exit_code != 0
        assert "cannot" in result.output.lower() or "at least" in result.output.lower()


class TestSetupRemoveMultiple:
    """Multiple --remove flags in one invocation."""

    def test_remove_two_tiers(self, mlx_stack_home: Path) -> None:
        """Two --remove flags remove two tiers."""
        _setup_existing_stack(mlx_stack_home, _THREE_TIER_STACK)
        runner = CliRunner()

        result = runner.invoke(
            setup,
            ["--remove", "fast", "--remove", "reasoning"],
        )

        assert result.exit_code == 0, f"Exit {result.exit_code}:\n{result.output}"
        stack = yaml.safe_load(
            (mlx_stack_home / "stacks" / "default.yaml").read_text()
        )
        tier_names = [t["name"] for t in stack["tiers"]]
        assert "fast" not in tier_names
        assert "reasoning" not in tier_names
        assert len(stack["tiers"]) == 1

    def test_remove_all_via_multiple_flags_errors(self, mlx_stack_home: Path) -> None:
        """Multiple --remove that would empty stack errors."""
        _setup_existing_stack(mlx_stack_home, _TWO_TIER_STACK)
        runner = CliRunner()

        result = runner.invoke(
            setup,
            ["--remove", "standard", "--remove", "fast"],
        )

        assert result.exit_code != 0
        assert "cannot" in result.output.lower() or "at least" in result.output.lower()


class TestSetupRemoveNoExistingStack:
    """--remove on nonexistent stack produces error."""

    def test_remove_without_stack_errors(self, mlx_stack_home: Path) -> None:
        """--remove with no existing stack.yaml shows error."""
        runner = CliRunner()

        result = runner.invoke(setup, ["--remove", "fast"])

        assert result.exit_code != 0
        assert "setup" in result.output.lower()


# --------------------------------------------------------------------------- #
# Tests for --add + --remove combined
# --------------------------------------------------------------------------- #


class TestSetupAddAndRemoveCombined:
    """--add and --remove can be used together."""

    def test_add_and_remove_in_same_invocation(self, mlx_stack_home: Path) -> None:
        """--add + --remove atomically modifies the stack."""
        _setup_existing_stack(mlx_stack_home, _TWO_TIER_STACK)
        runner = CliRunner()

        result = runner.invoke(
            setup,
            [
                "--add", "mlx-community/NewModel-4bit",
                "--remove", "fast",
            ],
        )

        assert result.exit_code == 0, f"Exit {result.exit_code}:\n{result.output}"
        stack = yaml.safe_load(
            (mlx_stack_home / "stacks" / "default.yaml").read_text()
        )
        tier_names = [t["name"] for t in stack["tiers"]]
        assert "fast" not in tier_names
        # standard + new model
        assert len(stack["tiers"]) == 2
        assert "standard" in tier_names


# --------------------------------------------------------------------------- #
# Tests for --add with --no-pull
# --------------------------------------------------------------------------- #


class TestSetupAddNoPull:
    """--add with --no-pull modifies config without downloading."""

    def test_add_no_pull_does_not_download(self, mlx_stack_home: Path) -> None:
        """--add with --no-pull skips model download."""
        _setup_existing_stack(mlx_stack_home, _TWO_TIER_STACK)
        runner = CliRunner()

        with patch("mlx_stack.cli.setup.pull_setup_models") as mock_pull:
            result = runner.invoke(
                setup,
                ["--add", "mlx-community/Model-4bit", "--no-pull"],
            )

        assert result.exit_code == 0, f"Exit {result.exit_code}:\n{result.output}"
        mock_pull.assert_not_called()
        stack = yaml.safe_load(
            (mlx_stack_home / "stacks" / "default.yaml").read_text()
        )
        assert len(stack["tiers"]) == 3


# --------------------------------------------------------------------------- #
# Tests for wizard flow unchanged
# --------------------------------------------------------------------------- #


class TestSetupWizardUnchanged:
    """Plain setup (no modification flags) runs the original wizard flow."""

    def test_wizard_flow_runs_normally(self, mlx_stack_home: Path) -> None:
        """--accept-defaults with no --add/--remove runs full wizard."""
        result = _run_setup(["--accept-defaults"], mlx_stack_home)
        assert result.exit_code == 0
        assert "Hardware" in result.output
        assert "Model Selection" in result.output
        assert "Tier Assignment" in result.output
        assert "Starting Stack" in result.output

    def test_no_modification_flags_does_not_modify_existing(
        self, mlx_stack_home: Path
    ) -> None:
        """Wizard flow with --accept-defaults completes without modification logic."""
        result = _run_setup(["--accept-defaults"], mlx_stack_home)
        assert result.exit_code == 0
        # Should NOT contain modification-specific output
        assert "Added tier" not in result.output
        assert "Removed tier" not in result.output


# --------------------------------------------------------------------------- #
# Tests for setup --help
# --------------------------------------------------------------------------- #


class TestSetupHelp:
    """Help output shows modification flags."""

    def test_help_shows_add_flag(self) -> None:
        """setup --help shows --add flag."""
        runner = CliRunner()
        result = runner.invoke(setup, ["--help"])
        assert "--add" in result.output

    def test_help_shows_as_flag(self) -> None:
        """setup --help shows --as flag."""
        runner = CliRunner()
        result = runner.invoke(setup, ["--help"])
        assert "--as" in result.output

    def test_help_shows_remove_flag(self) -> None:
        """setup --help shows --remove flag."""
        runner = CliRunner()
        result = runner.invoke(setup, ["--help"])
        assert "--remove" in result.output

    def test_help_shows_model_flag(self) -> None:
        """setup --help shows --model flag."""
        runner = CliRunner()
        result = runner.invoke(setup, ["--help"])
        assert "--model" in result.output

    def test_help_shows_no_pull_flag(self) -> None:
        """setup --help shows --no-pull flag."""
        runner = CliRunner()
        result = runner.invoke(setup, ["--help"])
        assert "--no-pull" in result.output

    def test_help_shows_no_start_flag(self) -> None:
        """setup --help shows --no-start flag."""
        runner = CliRunner()
        result = runner.invoke(setup, ["--help"])
        assert "--no-start" in result.output


# --------------------------------------------------------------------------- #
# Tests for --model flag (single-model quick setup)
# --------------------------------------------------------------------------- #


class TestSetupModelHfRepo:
    """--model with HF repo creates single-tier stack, no wizard."""

    def test_model_hf_repo_creates_single_tier_stack(self, mlx_stack_home: Path) -> None:
        """--model mlx-community/Qwen3-8B-4bit creates stack with 1 'standard' tier."""
        runner = CliRunner()

        with (
            patch("mlx_stack.cli.setup.pull_setup_models"),
            patch("mlx_stack.cli.setup.start_stack", return_value=MOCK_UP_RESULT),
        ):
            result = runner.invoke(
                setup,
                ["--model", "mlx-community/Qwen3-8B-4bit"],
            )

        assert result.exit_code == 0, f"Exit {result.exit_code}:\n{result.output}"
        stack = yaml.safe_load(
            (mlx_stack_home / "stacks" / "default.yaml").read_text()
        )
        assert len(stack["tiers"]) == 1
        tier = stack["tiers"][0]
        assert tier["name"] == "standard"
        assert tier["source"] == "mlx-community/Qwen3-8B-4bit"

    def test_model_hf_repo_generates_litellm_yaml(self, mlx_stack_home: Path) -> None:
        """--model creates litellm.yaml with the new tier."""
        runner = CliRunner()

        with (
            patch("mlx_stack.cli.setup.pull_setup_models"),
            patch("mlx_stack.cli.setup.start_stack", return_value=MOCK_UP_RESULT),
        ):
            result = runner.invoke(
                setup,
                ["--model", "mlx-community/Qwen3-8B-4bit"],
            )

        assert result.exit_code == 0
        litellm_path = mlx_stack_home / "litellm.yaml"
        assert litellm_path.exists()
        litellm = yaml.safe_load(litellm_path.read_text())
        model_names = [m["model_name"] for m in litellm["model_list"]]
        assert "standard" in model_names

    def test_model_hf_repo_skips_wizard(self, mlx_stack_home: Path) -> None:
        """--model does NOT show wizard steps (Hardware, Model Selection, etc.)."""
        runner = CliRunner()

        with (
            patch("mlx_stack.cli.setup.pull_setup_models"),
            patch("mlx_stack.cli.setup.start_stack", return_value=MOCK_UP_RESULT),
        ):
            result = runner.invoke(
                setup,
                ["--model", "mlx-community/Qwen3-8B-4bit"],
            )

        assert result.exit_code == 0
        assert "Hardware" not in result.output
        assert "Model Selection" not in result.output
        assert "Tier Assignment" not in result.output

    def test_model_hf_repo_calls_pull_and_start(self, mlx_stack_home: Path) -> None:
        """--model without --no-pull/--no-start calls pull and start."""
        runner = CliRunner()

        with (
            patch("mlx_stack.cli.setup.pull_setup_models") as mock_pull,
            patch("mlx_stack.cli.setup.start_stack", return_value=MOCK_UP_RESULT) as mock_start,
        ):
            result = runner.invoke(
                setup,
                ["--model", "mlx-community/Qwen3-8B-4bit"],
            )

        assert result.exit_code == 0
        mock_pull.assert_called_once()
        mock_start.assert_called_once()

    def test_model_hf_repo_overwrites_existing_stack(self, mlx_stack_home: Path) -> None:
        """--model replaces existing multi-tier stack with single-tier."""
        _setup_existing_stack(mlx_stack_home, _TWO_TIER_STACK)
        runner = CliRunner()

        with (
            patch("mlx_stack.cli.setup.pull_setup_models"),
            patch("mlx_stack.cli.setup.start_stack", return_value=MOCK_UP_RESULT),
        ):
            result = runner.invoke(
                setup,
                ["--model", "mlx-community/Qwen3-8B-4bit"],
            )

        assert result.exit_code == 0
        stack = yaml.safe_load(
            (mlx_stack_home / "stacks" / "default.yaml").read_text()
        )
        assert len(stack["tiers"]) == 1
        assert stack["tiers"][0]["name"] == "standard"


class TestSetupModelCatalogId:
    """--model with catalog ID resolves and creates single-tier stack."""

    def test_model_catalog_id_resolves(self, mlx_stack_home: Path) -> None:
        """--model qwen3.5-8b resolves catalog ID to HF repo."""
        runner = CliRunner()

        with (
            patch(
                "mlx_stack.cli.setup.get_entry_by_id",
                return_value=_MOCK_CATALOG_ENTRY,
            ),
            patch("mlx_stack.cli.setup.load_catalog", return_value=[_MOCK_CATALOG_ENTRY]),
            patch("mlx_stack.cli.setup.pull_setup_models"),
            patch("mlx_stack.cli.setup.start_stack", return_value=MOCK_UP_RESULT),
        ):
            result = runner.invoke(setup, ["--model", "qwen3.5-8b"])

        assert result.exit_code == 0, f"Exit {result.exit_code}:\n{result.output}"
        stack = yaml.safe_load(
            (mlx_stack_home / "stacks" / "default.yaml").read_text()
        )
        assert len(stack["tiers"]) == 1
        assert stack["tiers"][0]["name"] == "standard"
        assert "mlx-community" in stack["tiers"][0]["source"]

    def test_model_invalid_catalog_id_shows_error(self, mlx_stack_home: Path) -> None:
        """--model with invalid catalog ID produces clear error, no traceback."""
        runner = CliRunner()

        with (
            patch("mlx_stack.cli.setup.get_entry_by_id", return_value=None),
            patch("mlx_stack.cli.setup.load_catalog", return_value=[]),
        ):
            result = runner.invoke(setup, ["--model", "nonexistent-xyz"])

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()
        assert "Traceback" not in result.output


# --------------------------------------------------------------------------- #
# Tests for --model with --no-pull and --no-start
# --------------------------------------------------------------------------- #


class TestSetupModelNoPull:
    """--model with --no-pull creates config without download or start."""

    def test_model_no_pull_skips_download_and_start(self, mlx_stack_home: Path) -> None:
        """--model --no-pull creates stack.yaml but does not download or start."""
        runner = CliRunner()

        with (
            patch("mlx_stack.cli.setup.pull_setup_models") as mock_pull,
            patch("mlx_stack.cli.setup.start_stack") as mock_start,
        ):
            result = runner.invoke(
                setup,
                ["--model", "mlx-community/Model-4bit", "--no-pull"],
            )

        assert result.exit_code == 0, f"Exit {result.exit_code}:\n{result.output}"
        mock_pull.assert_not_called()
        mock_start.assert_not_called()
        stack = yaml.safe_load(
            (mlx_stack_home / "stacks" / "default.yaml").read_text()
        )
        assert len(stack["tiers"]) == 1

    def test_model_no_pull_tells_user_to_run_up(self, mlx_stack_home: Path) -> None:
        """--model --no-pull output tells user to run 'mlx-stack up'."""
        runner = CliRunner()

        with (
            patch("mlx_stack.cli.setup.pull_setup_models"),
            patch("mlx_stack.cli.setup.start_stack"),
        ):
            result = runner.invoke(
                setup,
                ["--model", "mlx-community/Model-4bit", "--no-pull"],
            )

        assert result.exit_code == 0
        assert "mlx-stack up" in result.output


class TestSetupModelNoStart:
    """--model with --no-start creates config and pulls but doesn't start."""

    def test_model_no_start_pulls_but_does_not_start(self, mlx_stack_home: Path) -> None:
        """--model --no-start pulls model but does not start stack."""
        runner = CliRunner()

        with (
            patch("mlx_stack.cli.setup.pull_setup_models") as mock_pull,
            patch("mlx_stack.cli.setup.start_stack") as mock_start,
        ):
            result = runner.invoke(
                setup,
                ["--model", "mlx-community/Model-4bit", "--no-start"],
            )

        assert result.exit_code == 0, f"Exit {result.exit_code}:\n{result.output}"
        mock_pull.assert_called_once()
        mock_start.assert_not_called()

    def test_model_no_start_tells_user_to_run_up(self, mlx_stack_home: Path) -> None:
        """--model --no-start output tells user to run 'mlx-stack up'."""
        runner = CliRunner()

        with (
            patch("mlx_stack.cli.setup.pull_setup_models"),
            patch("mlx_stack.cli.setup.start_stack"),
        ):
            result = runner.invoke(
                setup,
                ["--model", "mlx-community/Model-4bit", "--no-start"],
            )

        assert result.exit_code == 0
        assert "mlx-stack up" in result.output


# --------------------------------------------------------------------------- #
# Tests for --no-pull and --no-start in wizard flow
# --------------------------------------------------------------------------- #


class TestSetupWizardNoPull:
    """--no-pull skips model download in wizard flow."""

    def test_wizard_no_pull_skips_download_and_start(self, mlx_stack_home: Path) -> None:
        """--accept-defaults --no-pull runs wizard but skips pull and start."""
        runner = CliRunner()

        with (
            patch("mlx_stack.core.onboarding.detect_hardware", return_value=MOCK_PROFILE),
            patch("mlx_stack.core.onboarding.save_profile"),
            patch("mlx_stack.core.discovery.query_hf_models", return_value=[]),
            patch(
                "mlx_stack.core.discovery.load_benchmark_data",
                return_value=MOCK_BENCHMARK_DATA,
            ),
            patch(
                "mlx_stack.cli.setup.generate_config",
                return_value=(
                    mlx_stack_home / "stacks" / "default.yaml",
                    mlx_stack_home / "litellm.yaml",
                ),
            ),
            patch("mlx_stack.cli.setup.pull_setup_models") as mock_pull,
            patch("mlx_stack.cli.setup.start_stack") as mock_start,
        ):
            result = runner.invoke(setup, ["--accept-defaults", "--no-pull"])

        assert result.exit_code == 0, f"Exit {result.exit_code}:\n{result.output}"
        mock_pull.assert_not_called()
        mock_start.assert_not_called()

    def test_wizard_no_pull_tells_user_to_run_up(self, mlx_stack_home: Path) -> None:
        """--accept-defaults --no-pull tells user to run 'mlx-stack up'."""
        runner = CliRunner()

        with (
            patch("mlx_stack.core.onboarding.detect_hardware", return_value=MOCK_PROFILE),
            patch("mlx_stack.core.onboarding.save_profile"),
            patch("mlx_stack.core.discovery.query_hf_models", return_value=[]),
            patch(
                "mlx_stack.core.discovery.load_benchmark_data",
                return_value=MOCK_BENCHMARK_DATA,
            ),
            patch(
                "mlx_stack.cli.setup.generate_config",
                return_value=(
                    mlx_stack_home / "stacks" / "default.yaml",
                    mlx_stack_home / "litellm.yaml",
                ),
            ),
            patch("mlx_stack.cli.setup.pull_setup_models"),
            patch("mlx_stack.cli.setup.start_stack"),
        ):
            result = runner.invoke(setup, ["--accept-defaults", "--no-pull"])

        assert result.exit_code == 0
        assert "mlx-stack up" in result.output


class TestSetupWizardNoStart:
    """--no-start skips stack startup in wizard flow."""

    def test_wizard_no_start_pulls_but_does_not_start(self, mlx_stack_home: Path) -> None:
        """--accept-defaults --no-start pulls models but skips start."""
        runner = CliRunner()

        with (
            patch("mlx_stack.core.onboarding.detect_hardware", return_value=MOCK_PROFILE),
            patch("mlx_stack.core.onboarding.save_profile"),
            patch("mlx_stack.core.discovery.query_hf_models", return_value=[]),
            patch(
                "mlx_stack.core.discovery.load_benchmark_data",
                return_value=MOCK_BENCHMARK_DATA,
            ),
            patch(
                "mlx_stack.cli.setup.generate_config",
                return_value=(
                    mlx_stack_home / "stacks" / "default.yaml",
                    mlx_stack_home / "litellm.yaml",
                ),
            ),
            patch("mlx_stack.cli.setup.pull_setup_models", return_value=[]) as mock_pull,
            patch("mlx_stack.cli.setup.start_stack") as mock_start,
        ):
            result = runner.invoke(setup, ["--accept-defaults", "--no-start"])

        assert result.exit_code == 0, f"Exit {result.exit_code}:\n{result.output}"
        mock_pull.assert_called_once()
        mock_start.assert_not_called()

    def test_wizard_no_start_tells_user_to_run_up(self, mlx_stack_home: Path) -> None:
        """--accept-defaults --no-start tells user to run 'mlx-stack up'."""
        runner = CliRunner()

        with (
            patch("mlx_stack.core.onboarding.detect_hardware", return_value=MOCK_PROFILE),
            patch("mlx_stack.core.onboarding.save_profile"),
            patch("mlx_stack.core.discovery.query_hf_models", return_value=[]),
            patch(
                "mlx_stack.core.discovery.load_benchmark_data",
                return_value=MOCK_BENCHMARK_DATA,
            ),
            patch(
                "mlx_stack.cli.setup.generate_config",
                return_value=(
                    mlx_stack_home / "stacks" / "default.yaml",
                    mlx_stack_home / "litellm.yaml",
                ),
            ),
            patch("mlx_stack.cli.setup.pull_setup_models", return_value=[]),
            patch("mlx_stack.cli.setup.start_stack"),
        ):
            result = runner.invoke(setup, ["--accept-defaults", "--no-start"])

        assert result.exit_code == 0
        assert "mlx-stack up" in result.output


# --------------------------------------------------------------------------- #
# Tests for --no-pull implies --no-start
# --------------------------------------------------------------------------- #


class TestSetupNoPullImpliesNoStart:
    """--no-pull without --no-start still skips both download and startup."""

    def test_no_pull_implies_no_start_wizard(self, mlx_stack_home: Path) -> None:
        """--no-pull alone skips both pull and start in wizard flow."""
        runner = CliRunner()

        with (
            patch("mlx_stack.core.onboarding.detect_hardware", return_value=MOCK_PROFILE),
            patch("mlx_stack.core.onboarding.save_profile"),
            patch("mlx_stack.core.discovery.query_hf_models", return_value=[]),
            patch(
                "mlx_stack.core.discovery.load_benchmark_data",
                return_value=MOCK_BENCHMARK_DATA,
            ),
            patch(
                "mlx_stack.cli.setup.generate_config",
                return_value=(
                    mlx_stack_home / "stacks" / "default.yaml",
                    mlx_stack_home / "litellm.yaml",
                ),
            ),
            patch("mlx_stack.cli.setup.pull_setup_models") as mock_pull,
            patch("mlx_stack.cli.setup.start_stack") as mock_start,
        ):
            result = runner.invoke(setup, ["--accept-defaults", "--no-pull"])

        assert result.exit_code == 0
        mock_pull.assert_not_called()
        mock_start.assert_not_called()

    def test_no_pull_implies_no_start_model(self, mlx_stack_home: Path) -> None:
        """--model --no-pull skips both pull and start."""
        runner = CliRunner()

        with (
            patch("mlx_stack.cli.setup.pull_setup_models") as mock_pull,
            patch("mlx_stack.cli.setup.start_stack") as mock_start,
        ):
            result = runner.invoke(
                setup,
                ["--model", "mlx-community/Model-4bit", "--no-pull"],
            )

        assert result.exit_code == 0
        mock_pull.assert_not_called()
        mock_start.assert_not_called()


# --------------------------------------------------------------------------- #
# Tests for mutual exclusivity (--model vs --add/--remove)
# --------------------------------------------------------------------------- #


class TestSetupModelMutualExclusivity:
    """--model conflicts with --add and --remove."""

    def test_model_with_add_errors(self, mlx_stack_home: Path) -> None:
        """--model combined with --add produces error about conflicting flags."""
        runner = CliRunner()

        result = runner.invoke(
            setup,
            ["--model", "mlx-community/Model-4bit", "--add", "mlx-community/Other-4bit"],
        )

        assert result.exit_code != 0
        assert "cannot" in result.output.lower() or "mutually exclusive" in result.output.lower() or "conflict" in result.output.lower()

    def test_model_with_remove_errors(self, mlx_stack_home: Path) -> None:
        """--model combined with --remove produces error about conflicting flags."""
        _setup_existing_stack(mlx_stack_home, _TWO_TIER_STACK)
        runner = CliRunner()

        result = runner.invoke(
            setup,
            ["--model", "mlx-community/Model-4bit", "--remove", "fast"],
        )

        assert result.exit_code != 0
        assert "cannot" in result.output.lower() or "mutually exclusive" in result.output.lower() or "conflict" in result.output.lower()

    def test_model_with_add_and_remove_errors(self, mlx_stack_home: Path) -> None:
        """--model combined with --add and --remove produces error."""
        runner = CliRunner()

        result = runner.invoke(
            setup,
            [
                "--model", "mlx-community/Model-4bit",
                "--add", "mlx-community/Other-4bit",
                "--remove", "fast",
            ],
        )

        assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# Tests for --as without --add still errors
# --------------------------------------------------------------------------- #


class TestSetupAsWithoutAddErrors:
    """--as without --add produces error (existing behavior preserved)."""

    def test_as_without_add_errors(self, mlx_stack_home: Path) -> None:
        """--as without --add produces an error."""
        runner = CliRunner()

        result = runner.invoke(setup, ["--as", "custom-name"])

        assert result.exit_code != 0
        assert "--as" in result.output and "--add" in result.output

    def test_as_with_model_errors(self, mlx_stack_home: Path) -> None:
        """--as with --model (but no --add) still produces error."""
        runner = CliRunner()

        result = runner.invoke(
            setup,
            ["--model", "mlx-community/Model-4bit", "--as", "custom-name"],
        )

        assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# Tests for backward compatibility
# --------------------------------------------------------------------------- #


class TestSetupBackwardCompat:
    """Existing --accept-defaults and wizard flow unchanged."""

    def test_accept_defaults_still_works(self, mlx_stack_home: Path) -> None:
        """--accept-defaults with no new flags runs full wizard."""
        result = _run_setup(["--accept-defaults"], mlx_stack_home)
        assert result.exit_code == 0
        assert "Hardware" in result.output
        assert "Model Selection" in result.output
        assert "Tier Assignment" in result.output
        assert "Starting Stack" in result.output
