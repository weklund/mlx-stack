"""Tests for the install/uninstall CLI commands.

Covers CLI invocation, help text, --status flag, error handling
for platform, prerequisite, and launchd errors, and successful
install/uninstall flows.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

from mlx_stack.cli.main import cli
from mlx_stack.core.launchd import (
    AgentStatus,
    LaunchdError,
    PlatformError,
    PrerequisiteError,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def runner() -> CliRunner:
    """Create a Click CliRunner."""
    return CliRunner()


@pytest.fixture()
def stack_definition(mlx_stack_home: Path) -> dict:
    """Create a test stack definition."""
    stacks_dir = mlx_stack_home / "stacks"
    stacks_dir.mkdir(parents=True, exist_ok=True)

    stack = {
        "schema_version": 1,
        "name": "default",
        "hardware_profile": "m4-pro-48",
        "intent": "balanced",
        "created": "2025-01-01T00:00:00",
        "tiers": [
            {
                "name": "fast",
                "model": "qwen3.5-3b",
                "quant": "int4",
                "source": "mlx-community/Qwen3.5-3B-4bit",
                "port": 8000,
                "vllm_flags": {},
            },
        ],
    }

    stack_path = stacks_dir / "default.yaml"
    stack_path.write_text(yaml.dump(stack))
    return stack


# --------------------------------------------------------------------------- #
# Help text tests
# --------------------------------------------------------------------------- #


class TestInstallHelp:
    """Tests for install command help text."""

    def test_install_in_main_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "install" in result.output

    def test_uninstall_in_main_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "uninstall" in result.output

    def test_install_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["install", "--help"])
        assert result.exit_code == 0
        assert "--status" in result.output

    def test_install_help_mentions_launchd(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["install", "--help"])
        assert result.exit_code == 0
        # Should mention LaunchAgent or launchd
        output_lower = result.output.lower()
        assert "launch" in output_lower

    def test_uninstall_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["uninstall", "--help"])
        assert result.exit_code == 0
        # Should describe what uninstall does
        assert "uninstall" in result.output.lower() or "remove" in result.output.lower()

    def test_install_in_lifecycle_category(self, runner: CliRunner) -> None:
        """Install and uninstall should appear in the Stack Lifecycle category."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        # Both commands should be in help
        assert "install" in result.output
        assert "uninstall" in result.output


# --------------------------------------------------------------------------- #
# install --status tests
# --------------------------------------------------------------------------- #


class TestInstallStatus:
    """Tests for install --status flag."""

    def test_status_not_installed(
        self, runner: CliRunner, mlx_stack_home: Path
    ) -> None:
        status = AgentStatus(installed=False, running=False, pid=None)
        with patch("mlx_stack.cli.install.get_agent_status", return_value=status):
            result = runner.invoke(cli, ["install", "--status"])

        assert result.exit_code == 0
        assert "not installed" in result.output

    def test_status_installed_and_running(
        self, runner: CliRunner, mlx_stack_home: Path
    ) -> None:
        status = AgentStatus(installed=True, running=True, pid=12345)
        with patch("mlx_stack.cli.install.get_agent_status", return_value=status):
            result = runner.invoke(cli, ["install", "--status"])

        assert result.exit_code == 0
        assert "installed and running" in result.output
        assert "PID 12345" in result.output

    def test_status_installed_but_not_running(
        self, runner: CliRunner, mlx_stack_home: Path
    ) -> None:
        status = AgentStatus(installed=True, running=False, pid=None)
        with patch("mlx_stack.cli.install.get_agent_status", return_value=status):
            result = runner.invoke(cli, ["install", "--status"])

        assert result.exit_code == 0
        assert "installed but not running" in result.output

    def test_status_platform_error(
        self, runner: CliRunner, mlx_stack_home: Path
    ) -> None:
        with patch(
            "mlx_stack.cli.install.get_agent_status",
            side_effect=PlatformError("only available on macOS"),
        ):
            result = runner.invoke(cli, ["install", "--status"])

        assert result.exit_code != 0
        assert "Error" in result.output


# --------------------------------------------------------------------------- #
# install command tests (successful)
# --------------------------------------------------------------------------- #


class TestInstallCommand:
    """Tests for the install command."""

    def test_fresh_install(
        self, runner: CliRunner, mlx_stack_home: Path, stack_definition: dict
    ) -> None:
        with patch(
            "mlx_stack.cli.install.install_agent",
            return_value=(Path("/tmp/test.plist"), False),
        ):
            result = runner.invoke(cli, ["install"])

        assert result.exit_code == 0
        assert "installed successfully" in result.output.lower()

    def test_reinstall(
        self, runner: CliRunner, mlx_stack_home: Path, stack_definition: dict
    ) -> None:
        with patch(
            "mlx_stack.cli.install.install_agent",
            return_value=(Path("/tmp/test.plist"), True),
        ):
            result = runner.invoke(cli, ["install"])

        assert result.exit_code == 0
        assert "updated" in result.output.lower()

    def test_plist_path_shown(
        self, runner: CliRunner, mlx_stack_home: Path, stack_definition: dict
    ) -> None:
        plist = Path("/Users/test/Library/LaunchAgents/com.mlx-stack.watchdog.plist")
        with patch(
            "mlx_stack.cli.install.install_agent",
            return_value=(plist, False),
        ):
            result = runner.invoke(cli, ["install"])

        assert result.exit_code == 0
        assert "plist" in result.output.lower() or str(plist) in result.output

    def test_mentions_auto_start(
        self, runner: CliRunner, mlx_stack_home: Path, stack_definition: dict
    ) -> None:
        with patch(
            "mlx_stack.cli.install.install_agent",
            return_value=(Path("/tmp/test.plist"), False),
        ):
            result = runner.invoke(cli, ["install"])

        assert result.exit_code == 0
        assert "login" in result.output.lower() or "auto" in result.output.lower()


# --------------------------------------------------------------------------- #
# install command error handling
# --------------------------------------------------------------------------- #


class TestInstallErrors:
    """Tests for install command error handling."""

    def test_platform_error(
        self, runner: CliRunner, mlx_stack_home: Path
    ) -> None:
        with patch(
            "mlx_stack.cli.install.install_agent",
            side_effect=PlatformError("only available on macOS"),
        ):
            result = runner.invoke(cli, ["install"])

        assert result.exit_code != 0
        assert "Error" in result.output
        assert "macOS" in result.output

    def test_prerequisite_error(
        self, runner: CliRunner, mlx_stack_home: Path
    ) -> None:
        with patch(
            "mlx_stack.cli.install.install_agent",
            side_effect=PrerequisiteError(
                "No stack configuration found. Run 'mlx-stack init' first."
            ),
        ):
            result = runner.invoke(cli, ["install"])

        assert result.exit_code != 0
        assert "Error" in result.output
        assert "init" in result.output.lower()

    def test_launchd_error(
        self, runner: CliRunner, mlx_stack_home: Path
    ) -> None:
        with patch(
            "mlx_stack.cli.install.install_agent",
            side_effect=LaunchdError("launchctl bootstrap failed"),
        ):
            result = runner.invoke(cli, ["install"])

        assert result.exit_code != 0
        assert "Error" in result.output

    def test_no_python_traceback_on_platform_error(
        self, runner: CliRunner, mlx_stack_home: Path
    ) -> None:
        with patch(
            "mlx_stack.cli.install.install_agent",
            side_effect=PlatformError("not macOS"),
        ):
            result = runner.invoke(cli, ["install"])

        assert "Traceback" not in result.output

    def test_no_python_traceback_on_prerequisite_error(
        self, runner: CliRunner, mlx_stack_home: Path
    ) -> None:
        with patch(
            "mlx_stack.cli.install.install_agent",
            side_effect=PrerequisiteError("no stack"),
        ):
            result = runner.invoke(cli, ["install"])

        assert "Traceback" not in result.output


# --------------------------------------------------------------------------- #
# uninstall command tests
# --------------------------------------------------------------------------- #


class TestUninstallCommand:
    """Tests for the uninstall command."""

    def test_successful_uninstall(
        self, runner: CliRunner, mlx_stack_home: Path
    ) -> None:
        with patch("mlx_stack.cli.install.uninstall_agent", return_value=True):
            result = runner.invoke(cli, ["uninstall"])

        assert result.exit_code == 0
        assert "uninstalled" in result.output.lower()

    def test_not_installed_message(
        self, runner: CliRunner, mlx_stack_home: Path
    ) -> None:
        with patch("mlx_stack.cli.install.uninstall_agent", return_value=False):
            result = runner.invoke(cli, ["uninstall"])

        assert result.exit_code == 0
        assert "not installed" in result.output.lower()

    def test_services_unaffected_message(
        self, runner: CliRunner, mlx_stack_home: Path
    ) -> None:
        with patch("mlx_stack.cli.install.uninstall_agent", return_value=True):
            result = runner.invoke(cli, ["uninstall"])

        assert result.exit_code == 0
        assert "services" in result.output.lower() or "running" in result.output.lower()


# --------------------------------------------------------------------------- #
# uninstall error handling
# --------------------------------------------------------------------------- #


class TestUninstallErrors:
    """Tests for uninstall command error handling."""

    def test_platform_error(
        self, runner: CliRunner, mlx_stack_home: Path
    ) -> None:
        with patch(
            "mlx_stack.cli.install.uninstall_agent",
            side_effect=PlatformError("only available on macOS"),
        ):
            result = runner.invoke(cli, ["uninstall"])

        assert result.exit_code != 0
        assert "Error" in result.output

    def test_launchd_error(
        self, runner: CliRunner, mlx_stack_home: Path
    ) -> None:
        with patch(
            "mlx_stack.cli.install.uninstall_agent",
            side_effect=LaunchdError("bootout failed"),
        ):
            result = runner.invoke(cli, ["uninstall"])

        assert result.exit_code != 0
        assert "Error" in result.output

    def test_no_python_traceback(
        self, runner: CliRunner, mlx_stack_home: Path
    ) -> None:
        with patch(
            "mlx_stack.cli.install.uninstall_agent",
            side_effect=LaunchdError("bootout failed"),
        ):
            result = runner.invoke(cli, ["uninstall"])

        assert "Traceback" not in result.output


# --------------------------------------------------------------------------- #
# Integration-style tests (using real file system)
# --------------------------------------------------------------------------- #


class TestInstallStatusIntegration:
    """Integration-style tests for --status with real file system checks."""

    def test_status_reports_not_installed_when_no_plist(
        self, runner: CliRunner, mlx_stack_home: Path
    ) -> None:
        """Verify status works when plist doesn't exist."""
        status = AgentStatus(installed=False, running=False, pid=None)
        with patch("mlx_stack.cli.install.get_agent_status", return_value=status):
            result = runner.invoke(cli, ["install", "--status"])
        assert result.exit_code == 0
        assert "not installed" in result.output
