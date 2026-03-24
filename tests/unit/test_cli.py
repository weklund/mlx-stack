"""Tests for the CLI entry point — --help, --version, unknown commands, typo suggestions."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from mlx_stack import __version__
from mlx_stack.cli.main import cli


class TestCLIHelp:
    """Tests for --help output."""

    def test_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

    def test_help_shows_command_names(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        # All planned commands should appear in help output
        for cmd in [
            "profile",
            "recommend",
            "init",
            "pull",
            "models",
            "up",
            "down",
            "status",
            "bench",
            "config",
        ]:
            assert cmd in result.output, f"Command '{cmd}' not found in --help output"

    def test_help_shows_categories(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "Setup & Configuration" in result.output
        assert "Model Management" in result.output
        assert "Stack Lifecycle" in result.output
        assert "Diagnostics" in result.output

    def test_bare_command_shows_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, [])
        assert result.exit_code == 0
        assert "mlx-stack" in result.output


class TestCLIVersion:
    """Tests for --version output."""

    def test_version_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0

    def test_version_matches_package(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert __version__ in result.output

    def test_version_format(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.output.strip() == f"mlx-stack, version {__version__}"


class TestUnknownCommand:
    """Tests for unknown subcommands and typo suggestions."""

    def test_unknown_command_nonzero_exit(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["foobar"])
        assert result.exit_code != 0

    def test_unknown_command_shows_error(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["foobar"])
        assert "foobar" in result.output

    def test_unknown_command_no_traceback(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["foobar"])
        assert "Traceback" not in result.output

    def test_typo_suggests_close_match(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["proflie"])
        assert "profile" in result.output
        assert "Did you mean" in result.output

    def test_typo_suggest_init(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["inti"])
        assert "init" in result.output

    def test_typo_suggest_status(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["statu"])
        assert "status" in result.output

    def test_unknown_command_suggests_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["foobar"])
        assert "--help" in result.output


class TestDataHomeAutoCreation:
    """Tests for data-home gating: only state-writing commands create it."""

    def test_bare_command_does_not_create_data_home(
        self, clean_mlx_stack_home: Path,
    ) -> None:
        """Running bare mlx-stack (help) does NOT create the data directory."""
        assert not clean_mlx_stack_home.exists()
        runner = CliRunner()
        result = runner.invoke(cli, [])
        assert result.exit_code == 0
        assert not clean_mlx_stack_home.exists()

    def test_readonly_subcommand_does_not_create_data_home(
        self, clean_mlx_stack_home: Path,
    ) -> None:
        """Read-only subcommands do NOT create the data directory."""
        assert not clean_mlx_stack_home.exists()
        runner = CliRunner()
        runner.invoke(cli, ["status"])
        assert not clean_mlx_stack_home.exists()

    def test_state_writing_subcommand_creates_data_home(
        self, clean_mlx_stack_home: Path,
    ) -> None:
        """State-writing subcommands (config set) auto-create the data directory."""
        assert not clean_mlx_stack_home.exists()
        runner = CliRunner()
        runner.invoke(cli, ["config", "set", "default-quant", "int4"])
        assert clean_mlx_stack_home.exists()
        assert clean_mlx_stack_home.is_dir()

    def test_existing_data_home_not_affected(
        self, mlx_stack_home: Path,
    ) -> None:
        """Running CLI when data home already exists doesn't cause errors."""
        assert mlx_stack_home.exists()
        marker = mlx_stack_home / "test-marker.txt"
        marker.write_text("keep me")
        runner = CliRunner()
        result = runner.invoke(cli, [])
        assert result.exit_code == 0
        assert mlx_stack_home.exists()
        assert marker.read_text() == "keep me"


class TestPlaceholderCommands:
    """Tests for placeholder commands that are not yet implemented."""

    def test_config_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "--help"])
        assert result.exit_code == 0
        assert "set" in result.output
        assert "get" in result.output
        assert "list" in result.output
        assert "reset" in result.output
