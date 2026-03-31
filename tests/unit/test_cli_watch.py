"""Tests for the watch CLI command.

Covers CLI invocation, help text, parameter validation, error
handling, and integration with the watchdog core module.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

from mlx_stack.cli.main import cli
from mlx_stack.core.watchdog import WatchdogError, WatchdogState

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


class TestWatchHelp:
    """Tests for watch command help text."""

    def test_watch_in_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "watch" in result.output

    def test_watch_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["watch", "--help"])
        assert result.exit_code == 0
        assert "--interval" in result.output
        assert "--max-restarts" in result.output
        assert "--restart-delay" in result.output
        assert "--daemon" in result.output

    def test_watch_help_shows_defaults(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["watch", "--help"])
        assert result.exit_code == 0
        assert "30" in result.output  # default interval
        assert "5" in result.output  # default max-restarts
        assert "10" in result.output  # default restart-delay


# --------------------------------------------------------------------------- #
# Parameter validation tests
# --------------------------------------------------------------------------- #


class TestWatchParameterValidation:
    """Tests for watch command parameter validation."""

    def test_invalid_interval_zero(
        self, runner: CliRunner, mlx_stack_home: Path
    ) -> None:
        result = runner.invoke(cli, ["watch", "--interval", "0"])
        assert result.exit_code != 0
        assert "positive integer" in result.output.lower() or "Invalid" in result.output

    def test_invalid_interval_negative(
        self, runner: CliRunner, mlx_stack_home: Path
    ) -> None:
        result = runner.invoke(cli, ["watch", "--interval", "-5"])
        assert result.exit_code != 0

    def test_invalid_max_restarts_zero(
        self, runner: CliRunner, mlx_stack_home: Path
    ) -> None:
        result = runner.invoke(cli, ["watch", "--max-restarts", "0"])
        assert result.exit_code != 0

    def test_invalid_restart_delay_negative(
        self, runner: CliRunner, mlx_stack_home: Path
    ) -> None:
        result = runner.invoke(cli, ["watch", "--restart-delay", "-1"])
        assert result.exit_code != 0

    def test_invalid_interval_non_integer(
        self, runner: CliRunner, mlx_stack_home: Path
    ) -> None:
        result = runner.invoke(cli, ["watch", "--interval", "abc"])
        assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# No stack error tests
# --------------------------------------------------------------------------- #


class TestWatchNoStack:
    """Tests for watch command when no stack is configured."""

    def test_no_stack_exits_with_error(
        self, runner: CliRunner, mlx_stack_home: Path
    ) -> None:
        result = runner.invoke(cli, ["watch"])
        assert result.exit_code != 0
        assert "init" in result.output.lower() or "stack" in result.output.lower()


# --------------------------------------------------------------------------- #
# Already running error tests
# --------------------------------------------------------------------------- #


class TestWatchAlreadyRunning:
    """Tests for watch when another watchdog is running."""

    def test_already_running_error(
        self,
        runner: CliRunner,
        mlx_stack_home: Path,
        stack_definition: dict,
    ) -> None:
        with patch(
            "mlx_stack.cli.watch.run_watchdog",
            side_effect=WatchdogError("A watchdog is already running (PID 12345)."),
        ):
            result = runner.invoke(cli, ["watch"])

        assert result.exit_code != 0
        assert "already running" in result.output.lower() or "Error" in result.output


# --------------------------------------------------------------------------- #
# Successful invocation tests
# --------------------------------------------------------------------------- #


class TestWatchInvocation:
    """Tests for successful watch command invocation."""

    def test_basic_invocation(
        self,
        runner: CliRunner,
        mlx_stack_home: Path,
        stack_definition: dict,
    ) -> None:
        mock_state = WatchdogState(cycle_count=1)

        with patch(
            "mlx_stack.cli.watch.run_watchdog",
            return_value=mock_state,
        ) as mock_run:
            result = runner.invoke(cli, ["watch"])

        assert result.exit_code == 0
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["interval"] == 30
        assert call_kwargs.kwargs["max_restarts"] == 5
        assert call_kwargs.kwargs["restart_delay"] == 10
        assert call_kwargs.kwargs["daemon"] is False

    def test_custom_interval(
        self,
        runner: CliRunner,
        mlx_stack_home: Path,
        stack_definition: dict,
    ) -> None:
        with patch(
            "mlx_stack.cli.watch.run_watchdog",
            return_value=WatchdogState(),
        ) as mock_run:
            result = runner.invoke(cli, ["watch", "--interval", "60"])

        assert result.exit_code == 0
        assert mock_run.call_args.kwargs["interval"] == 60

    def test_custom_max_restarts(
        self,
        runner: CliRunner,
        mlx_stack_home: Path,
        stack_definition: dict,
    ) -> None:
        with patch(
            "mlx_stack.cli.watch.run_watchdog",
            return_value=WatchdogState(),
        ) as mock_run:
            result = runner.invoke(cli, ["watch", "--max-restarts", "3"])

        assert result.exit_code == 0
        assert mock_run.call_args.kwargs["max_restarts"] == 3

    def test_custom_restart_delay(
        self,
        runner: CliRunner,
        mlx_stack_home: Path,
        stack_definition: dict,
    ) -> None:
        with patch(
            "mlx_stack.cli.watch.run_watchdog",
            return_value=WatchdogState(),
        ) as mock_run:
            result = runner.invoke(cli, ["watch", "--restart-delay", "20"])

        assert result.exit_code == 0
        assert mock_run.call_args.kwargs["restart_delay"] == 20

    def test_daemon_flag(
        self,
        runner: CliRunner,
        mlx_stack_home: Path,
        stack_definition: dict,
    ) -> None:
        with patch(
            "mlx_stack.cli.watch.run_watchdog",
            return_value=WatchdogState(),
        ) as mock_run:
            result = runner.invoke(cli, ["watch", "--daemon"])

        assert result.exit_code == 0
        assert mock_run.call_args.kwargs["daemon"] is True

    def test_all_options_combined(
        self,
        runner: CliRunner,
        mlx_stack_home: Path,
        stack_definition: dict,
    ) -> None:
        with patch(
            "mlx_stack.cli.watch.run_watchdog",
            return_value=WatchdogState(),
        ) as mock_run:
            result = runner.invoke(
                cli,
                [
                    "watch",
                    "--interval", "45",
                    "--max-restarts", "10",
                    "--restart-delay", "15",
                    "--daemon",
                ],
            )

        assert result.exit_code == 0
        kwargs = mock_run.call_args.kwargs
        assert kwargs["interval"] == 45
        assert kwargs["max_restarts"] == 10
        assert kwargs["restart_delay"] == 15
        assert kwargs["daemon"] is True

    def test_callbacks_are_passed(
        self,
        runner: CliRunner,
        mlx_stack_home: Path,
        stack_definition: dict,
    ) -> None:
        with patch(
            "mlx_stack.cli.watch.run_watchdog",
            return_value=WatchdogState(),
        ) as mock_run:
            runner.invoke(cli, ["watch"])

        kwargs = mock_run.call_args.kwargs
        assert kwargs["status_callback"] is not None
        assert kwargs["restart_callback"] is not None
