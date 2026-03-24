"""Tests for the CLI config commands — set, get, list, reset.

Uses Click CliRunner for testing CLI invocations with isolated
MLX_STACK_HOME via conftest fixtures.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from click.testing import CliRunner

from mlx_stack.cli.main import cli
from mlx_stack.core.paths import get_config_path

# --------------------------------------------------------------------------- #
# config --help
# --------------------------------------------------------------------------- #


class TestConfigHelp:
    """Tests for config help output."""

    def test_config_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "--help"])
        assert result.exit_code == 0

    def test_config_help_lists_subcommands(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "--help"])
        assert "set" in result.output
        assert "get" in result.output
        assert "list" in result.output
        assert "reset" in result.output

    def test_config_set_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "set", "--help"])
        assert result.exit_code == 0
        assert "KEY" in result.output
        assert "VALUE" in result.output

    def test_config_get_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "get", "--help"])
        assert result.exit_code == 0
        assert "KEY" in result.output

    def test_config_list_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "list", "--help"])
        assert result.exit_code == 0

    def test_config_reset_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "reset", "--help"])
        assert result.exit_code == 0
        assert "--yes" in result.output
        assert "--force" in result.output


# --------------------------------------------------------------------------- #
# config set
# --------------------------------------------------------------------------- #


class TestConfigSet:
    """Tests for `mlx-stack config set`."""

    def test_set_string_value(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "set", "openrouter-key", "sk-test-123"])
        assert result.exit_code == 0
        assert "openrouter-key" in result.output

    def test_set_integer_value(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "set", "memory-budget-pct", "60"])
        assert result.exit_code == 0

    def test_set_boolean_value(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "set", "auto-health-check", "false"])
        assert result.exit_code == 0

    def test_set_path_value(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "set", "model-dir", "/tmp/my-models"])
        assert result.exit_code == 0

    def test_set_quant_value(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "set", "default-quant", "int8"])
        assert result.exit_code == 0

    def test_set_port_value(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "set", "litellm-port", "5000"])
        assert result.exit_code == 0

    def test_set_invalid_key_rejected(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "set", "bad-key", "value"])
        assert result.exit_code != 0
        assert "Unknown config key" in result.output

    def test_set_invalid_key_shows_valid_keys(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "set", "bad-key", "value"])
        assert "default-quant" in result.output
        assert "litellm-port" in result.output

    def test_set_invalid_quant_rejected(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "set", "default-quant", "int6"])
        assert result.exit_code != 0
        assert "Invalid quantization" in result.output

    def test_set_invalid_port_rejected(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "set", "litellm-port", "65536"])
        assert result.exit_code != 0

    def test_set_invalid_memory_pct_rejected(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "set", "memory-budget-pct", "101"])
        assert result.exit_code != 0

    def test_set_non_integer_for_port(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "set", "litellm-port", "abc"])
        assert result.exit_code != 0
        assert "Expected an integer" in result.output

    def test_set_shows_masked_api_key(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "set", "openrouter-key", "sk-secret-key-12345"])
        assert result.exit_code == 0
        # Should NOT show full key in output
        assert "sk-secret-key-12345" not in result.output
        assert "****" in result.output

    def test_set_no_traceback(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "set", "bad-key", "value"])
        assert "Traceback" not in result.output


# --------------------------------------------------------------------------- #
# config get
# --------------------------------------------------------------------------- #


class TestConfigGet:
    """Tests for `mlx-stack config get`."""

    def test_get_default_value(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "get", "default-quant"])
        assert result.exit_code == 0
        assert "int4" in result.output

    def test_get_user_set_value(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        runner.invoke(cli, ["config", "set", "litellm-port", "5000"])
        result = runner.invoke(cli, ["config", "get", "litellm-port"])
        assert result.exit_code == 0
        assert "5000" in result.output

    def test_get_invalid_key(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "get", "bad-key"])
        assert result.exit_code != 0
        assert "Unknown config key" in result.output

    def test_get_api_key_masked(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        runner.invoke(cli, ["config", "set", "openrouter-key", "sk-secret-key-12345"])
        result = runner.invoke(cli, ["config", "get", "openrouter-key"])
        assert result.exit_code == 0
        assert "sk-secret-key-12345" not in result.output
        assert "****" in result.output

    def test_get_default_memory_budget(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "get", "memory-budget-pct"])
        assert result.exit_code == 0
        assert "40" in result.output

    def test_get_default_port(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "get", "litellm-port"])
        assert result.exit_code == 0
        assert "4000" in result.output

    def test_get_default_auto_health_check(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "get", "auto-health-check"])
        assert result.exit_code == 0
        assert "True" in result.output

    def test_get_no_traceback_on_error(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "get", "bad-key"])
        assert "Traceback" not in result.output


# --------------------------------------------------------------------------- #
# config set + get round-trip
# --------------------------------------------------------------------------- #


class TestConfigRoundTrip:
    """Tests for set/get round-trips with persistence across invocations."""

    def test_string_round_trip(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        runner.invoke(cli, ["config", "set", "openrouter-key", "sk-abc-def-12345"])
        # Verify via file that full key is stored
        config_path = get_config_path()
        data = yaml.safe_load(config_path.read_text())
        assert data["openrouter-key"] == "sk-abc-def-12345"

    def test_integer_round_trip(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        runner.invoke(cli, ["config", "set", "memory-budget-pct", "75"])
        result = runner.invoke(cli, ["config", "get", "memory-budget-pct"])
        assert "75" in result.output

    def test_boolean_round_trip(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        runner.invoke(cli, ["config", "set", "auto-health-check", "false"])
        result = runner.invoke(cli, ["config", "get", "auto-health-check"])
        assert "False" in result.output

    def test_path_round_trip(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        runner.invoke(cli, ["config", "set", "model-dir", "/custom/path"])
        result = runner.invoke(cli, ["config", "get", "model-dir"])
        assert "/custom/path" in result.output


# --------------------------------------------------------------------------- #
# config list
# --------------------------------------------------------------------------- #


class TestConfigList:
    """Tests for `mlx-stack config list`."""

    def test_list_exits_zero(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "list"])
        assert result.exit_code == 0

    def test_list_shows_all_keys(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "list"])
        assert "openrouter-key" in result.output
        assert "default-quant" in result.output
        assert "memory-budget-pct" in result.output
        assert "litellm-port" in result.output
        assert "model-dir" in result.output
        assert "auto-health-check" in result.output

    def test_list_shows_default_values(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "list"])
        assert "int4" in result.output
        assert "40" in result.output
        assert "4000" in result.output

    def test_list_shows_user_set_indicator(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        runner.invoke(cli, ["config", "set", "litellm-port", "5000"])
        result = runner.invoke(cli, ["config", "list"])
        assert "user-set" in result.output

    def test_list_masks_api_key(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        runner.invoke(cli, ["config", "set", "openrouter-key", "sk-secret-key-12345"])
        result = runner.invoke(cli, ["config", "list"])
        assert "sk-secret-key-12345" not in result.output
        assert "****" in result.output

    def test_list_shows_default_source(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "list"])
        assert "default" in result.output

    def test_list_corrupt_file_error(self, mlx_stack_home: Path) -> None:
        config_path = get_config_path()
        config_path.write_text("{{{invalid")
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "list"])
        assert result.exit_code != 0
        assert "corrupt" in result.output


# --------------------------------------------------------------------------- #
# config reset
# --------------------------------------------------------------------------- #


class TestConfigReset:
    """Tests for `mlx-stack config reset`."""

    def test_reset_with_yes_flag(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        runner.invoke(cli, ["config", "set", "litellm-port", "5000"])
        result = runner.invoke(cli, ["config", "reset", "--yes"])
        assert result.exit_code == 0
        assert "reset" in result.output.lower()

    def test_reset_with_force_flag(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        runner.invoke(cli, ["config", "set", "litellm-port", "5000"])
        result = runner.invoke(cli, ["config", "reset", "--force"])
        assert result.exit_code == 0

    def test_reset_restores_defaults(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        runner.invoke(cli, ["config", "set", "litellm-port", "5000"])
        runner.invoke(cli, ["config", "set", "default-quant", "int8"])
        runner.invoke(cli, ["config", "reset", "--yes"])
        result = runner.invoke(cli, ["config", "get", "litellm-port"])
        assert "4000" in result.output
        result = runner.invoke(cli, ["config", "get", "default-quant"])
        assert "int4" in result.output

    def test_reset_without_flag_non_interactive(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "reset"])
        assert result.exit_code != 0
        assert "--yes" in result.output or "--force" in result.output

    def test_reset_no_traceback(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "reset"])
        assert "Traceback" not in result.output

    def test_reset_when_no_config_exists(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "reset", "--yes"])
        assert result.exit_code == 0

    def test_all_defaults_after_reset(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        # Set several values
        runner.invoke(cli, ["config", "set", "litellm-port", "5000"])
        runner.invoke(cli, ["config", "set", "memory-budget-pct", "80"])
        runner.invoke(cli, ["config", "set", "auto-health-check", "false"])
        runner.invoke(cli, ["config", "set", "default-quant", "bf16"])
        # Reset
        runner.invoke(cli, ["config", "reset", "--yes"])
        # Check all defaults
        for key, expected in [
            ("default-quant", "int4"),
            ("memory-budget-pct", "40"),
            ("litellm-port", "4000"),
            ("auto-health-check", "True"),
        ]:
            result = runner.invoke(cli, ["config", "get", key])
            assert result.exit_code == 0
            assert expected in result.output, f"Expected {expected} for {key}, got: {result.output}"


# --------------------------------------------------------------------------- #
# Corrupt config file via CLI
# --------------------------------------------------------------------------- #


class TestCorruptConfigCLI:
    """Tests for corrupt config file handling via CLI."""

    def test_get_with_corrupt_file(self, mlx_stack_home: Path) -> None:
        config_path = get_config_path()
        config_path.write_text("{{{bad yaml")
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "get", "default-quant"])
        assert result.exit_code != 0
        assert "config reset --yes" in result.output

    def test_set_with_corrupt_file(self, mlx_stack_home: Path) -> None:
        config_path = get_config_path()
        config_path.write_text("{{{bad yaml")
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "set", "default-quant", "int4"])
        assert result.exit_code != 0

    def test_empty_file_works(self, mlx_stack_home: Path) -> None:
        config_path = get_config_path()
        config_path.write_text("")
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "get", "default-quant"])
        assert result.exit_code == 0
        assert "int4" in result.output

    def test_missing_file_works(self, mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "get", "default-quant"])
        assert result.exit_code == 0
        assert "int4" in result.output

    def test_no_traceback_on_corrupt(self, mlx_stack_home: Path) -> None:
        config_path = get_config_path()
        config_path.write_text("{{{")
        runner = CliRunner(env={"MLX_STACK_HOME": str(mlx_stack_home)})
        result = runner.invoke(cli, ["config", "get", "default-quant"])
        assert "Traceback" not in result.output


# --------------------------------------------------------------------------- #
# Auto-creation of data directory
# --------------------------------------------------------------------------- #


class TestAutoCreateDataDir:
    """Tests for auto-creation of data directory on config operations."""

    def test_set_creates_data_dir(self, clean_mlx_stack_home: Path) -> None:
        runner = CliRunner(env={"MLX_STACK_HOME": str(clean_mlx_stack_home)})
        result = runner.invoke(cli, ["config", "set", "default-quant", "int4"])
        assert result.exit_code == 0
        assert clean_mlx_stack_home.exists()
