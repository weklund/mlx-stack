"""Tests for the logs CLI command (cli/logs.py).

Covers all CLI flags and behaviors: listing, tail viewing, following,
service filtering, rotation, and archived log viewing.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from mlx_stack.cli.logs import logs

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _create_log(logs_dir: Path, service: str, content: str = "") -> Path:
    """Create a log file for a service."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{service}.log"
    log_path.write_text(content)
    return log_path


def _create_archive(
    logs_dir: Path, service: str, number: int, content: str
) -> Path:
    """Create a gzip archive for a service."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    archive_path = logs_dir / f"{service}.log.{number}.gz"
    with gzip.open(str(archive_path), "wb") as f:
        f.write(content.encode("utf-8"))
    return archive_path


# --------------------------------------------------------------------------- #
# Listing (no args)
# --------------------------------------------------------------------------- #


class TestListLogs:
    """Tests for listing log files (no args)."""

    def test_lists_log_files(self, mlx_stack_home: Path) -> None:
        """Lists available log files with sizes and times."""
        logs_dir = mlx_stack_home / "logs"
        _create_log(logs_dir, "fast", "some log content here\n")
        _create_log(logs_dir, "litellm", "litellm output\n")

        runner = CliRunner()
        result = runner.invoke(logs, [])

        assert result.exit_code == 0
        assert "fast.log" in result.output
        assert "litellm.log" in result.output

    def test_empty_logs_dir(self, mlx_stack_home: Path) -> None:
        """Shows message when no log files exist."""
        logs_dir = mlx_stack_home / "logs"
        logs_dir.mkdir(parents=True)

        runner = CliRunner()
        result = runner.invoke(logs, [])

        assert result.exit_code == 0
        assert "No log files found" in result.output

    def test_no_logs_dir(self, mlx_stack_home: Path) -> None:
        """Shows message when logs directory doesn't exist."""
        runner = CliRunner()
        result = runner.invoke(logs, [])

        assert result.exit_code == 0
        assert "No log files found" in result.output


# --------------------------------------------------------------------------- #
# Basic viewing
# --------------------------------------------------------------------------- #


class TestViewLogs:
    """Tests for viewing log content."""

    def test_shows_tail_default(self, mlx_stack_home: Path) -> None:
        """Shows last 50 lines by default."""
        logs_dir = mlx_stack_home / "logs"
        lines = [f"line {i}" for i in range(100)]
        _create_log(logs_dir, "fast", "\n".join(lines))

        runner = CliRunner()
        result = runner.invoke(logs, ["fast"])

        assert result.exit_code == 0
        output_lines = result.output.strip().splitlines()
        assert len(output_lines) == 50
        assert "line 50" in output_lines[0]
        assert "line 99" in output_lines[-1]

    def test_tail_n_lines(self, mlx_stack_home: Path) -> None:
        """--tail N shows exactly N lines."""
        logs_dir = mlx_stack_home / "logs"
        lines = [f"line {i}" for i in range(50)]
        _create_log(logs_dir, "fast", "\n".join(lines))

        runner = CliRunner()
        result = runner.invoke(logs, ["fast", "--tail", "10"])

        assert result.exit_code == 0
        output_lines = result.output.strip().splitlines()
        assert len(output_lines) == 10

    def test_empty_log(self, mlx_stack_home: Path) -> None:
        """Empty log shows informational message."""
        logs_dir = mlx_stack_home / "logs"
        _create_log(logs_dir, "fast", "")

        runner = CliRunner()
        result = runner.invoke(logs, ["fast"])

        assert result.exit_code == 0
        assert "empty" in result.output.lower()


# --------------------------------------------------------------------------- #
# Service filtering
# --------------------------------------------------------------------------- #


class TestServiceFiltering:
    """Tests for --service filtering and service argument."""

    def test_service_argument(self, mlx_stack_home: Path) -> None:
        """Positional service argument shows correct log."""
        logs_dir = mlx_stack_home / "logs"
        _create_log(logs_dir, "fast", "fast content\n")
        _create_log(logs_dir, "standard", "standard content\n")

        runner = CliRunner()
        result = runner.invoke(logs, ["fast"])

        assert result.exit_code == 0
        assert "fast content" in result.output

    def test_service_flag(self, mlx_stack_home: Path) -> None:
        """--service flag shows correct log."""
        logs_dir = mlx_stack_home / "logs"
        _create_log(logs_dir, "fast", "fast content\n")

        runner = CliRunner()
        result = runner.invoke(logs, ["--service", "fast"])

        assert result.exit_code == 0
        assert "fast content" in result.output

    def test_invalid_service_error(self, mlx_stack_home: Path) -> None:
        """Invalid service name produces clear error."""
        logs_dir = mlx_stack_home / "logs"
        _create_log(logs_dir, "fast", "content")

        runner = CliRunner()
        result = runner.invoke(logs, ["nonexistent"])

        assert result.exit_code != 0
        assert "No log file found" in result.output
        assert "nonexistent" in result.output
        assert "fast" in result.output  # Shows available services

    def test_invalid_service_no_logs_exist(self, mlx_stack_home: Path) -> None:
        """Invalid service with no logs shows appropriate message."""
        logs_dir = mlx_stack_home / "logs"
        logs_dir.mkdir(parents=True)

        runner = CliRunner()
        result = runner.invoke(logs, ["nonexistent"])

        assert result.exit_code != 0
        assert "No log file found" in result.output
        assert "No log files exist" in result.output


# --------------------------------------------------------------------------- #
# Rotation
# --------------------------------------------------------------------------- #


class TestRotation:
    """Tests for --rotate flag."""

    def test_rotate_all(self, mlx_stack_home: Path) -> None:
        """--rotate rotates all eligible logs."""
        logs_dir = mlx_stack_home / "logs"
        logs_dir.mkdir(parents=True)
        # Create a file above threshold
        (logs_dir / "fast.log").write_bytes(b"x" * (2 * 1024 * 1024))
        (logs_dir / "standard.log").write_text("small")

        with patch(
            "mlx_stack.core.log_viewer.get_value",
            side_effect=lambda k: 1 if k == "log-max-size-mb" else 5,
        ):
            runner = CliRunner()
            result = runner.invoke(logs, ["--rotate"])

        assert result.exit_code == 0
        assert "fast" in result.output
        assert "rotated" in result.output

    def test_rotate_specific_service(self, mlx_stack_home: Path) -> None:
        """--rotate --service <name> rotates only that service."""
        logs_dir = mlx_stack_home / "logs"
        logs_dir.mkdir(parents=True)
        (logs_dir / "fast.log").write_bytes(b"x" * (2 * 1024 * 1024))

        with patch(
            "mlx_stack.core.log_viewer.get_value",
            side_effect=lambda k: 1 if k == "log-max-size-mb" else 5,
        ):
            runner = CliRunner()
            result = runner.invoke(logs, ["--rotate", "--service", "fast"])

        assert result.exit_code == 0
        assert "fast" in result.output

    def test_rotate_no_rotation_needed(self, mlx_stack_home: Path) -> None:
        """Reports 'no rotation needed' when files are small."""
        logs_dir = mlx_stack_home / "logs"
        _create_log(logs_dir, "fast", "small content")

        with patch(
            "mlx_stack.core.log_viewer.get_value",
            side_effect=lambda k: 50 if k == "log-max-size-mb" else 5,
        ):
            runner = CliRunner()
            result = runner.invoke(logs, ["--rotate"])

        assert result.exit_code == 0
        assert "no rotation needed" in result.output.lower()

    def test_rotate_no_logs(self, mlx_stack_home: Path) -> None:
        """Reports appropriately when no logs exist to rotate."""
        runner = CliRunner()
        result = runner.invoke(logs, ["--rotate"])

        assert result.exit_code == 0
        assert "No log files found" in result.output

    def test_rotate_invalid_service(self, mlx_stack_home: Path) -> None:
        """--rotate with invalid --service shows error."""
        logs_dir = mlx_stack_home / "logs"
        _create_log(logs_dir, "fast", "content")

        runner = CliRunner()
        result = runner.invoke(logs, ["--rotate", "--service", "invalid"])

        assert result.exit_code != 0
        assert "No log file found" in result.output
        assert "fast" in result.output


# --------------------------------------------------------------------------- #
# Archived logs
# --------------------------------------------------------------------------- #


class TestAllLogs:
    """Tests for --all flag."""

    def test_shows_archives_and_current(self, mlx_stack_home: Path) -> None:
        """--all shows archived and current logs chronologically."""
        logs_dir = mlx_stack_home / "logs"
        _create_archive(logs_dir, "fast", 2, "oldest archived")
        _create_archive(logs_dir, "fast", 1, "newest archived")
        _create_log(logs_dir, "fast", "current content")

        runner = CliRunner()
        result = runner.invoke(logs, ["fast", "--all"])

        assert result.exit_code == 0
        assert "oldest archived" in result.output
        assert "newest archived" in result.output
        assert "current content" in result.output
        # Check chronological order
        oldest_pos = result.output.index("oldest archived")
        newest_pos = result.output.index("newest archived")
        current_pos = result.output.index("current content")
        assert oldest_pos < newest_pos < current_pos

    def test_all_no_logs(self, mlx_stack_home: Path) -> None:
        """--all with no logs (no archives, no current) shows informative message."""
        logs_dir = mlx_stack_home / "logs"
        logs_dir.mkdir(parents=True)

        runner = CliRunner()
        result = runner.invoke(logs, ["nonexistent", "--all"])

        # No archives or current log → informative "no content" message, exit 0
        assert result.exit_code == 0
        assert "No log content found" in result.output

    def test_all_archives_only_no_current(self, mlx_stack_home: Path) -> None:
        """--all shows archives even when current log file is missing."""
        logs_dir = mlx_stack_home / "logs"
        _create_archive(logs_dir, "fast", 2, "old archived content")
        _create_archive(logs_dir, "fast", 1, "recent archived content")
        # No fast.log current file

        runner = CliRunner()
        result = runner.invoke(logs, ["fast", "--all"])

        assert result.exit_code == 0
        assert "old archived content" in result.output
        assert "recent archived content" in result.output


# --------------------------------------------------------------------------- #
# Follow mode
# --------------------------------------------------------------------------- #


class TestFollowMode:
    """Tests for --follow flag."""

    def test_follow_invalid_service(self, mlx_stack_home: Path) -> None:
        """--follow with invalid service shows error."""
        logs_dir = mlx_stack_home / "logs"
        logs_dir.mkdir(parents=True)

        runner = CliRunner()
        result = runner.invoke(logs, ["nonexistent", "--follow"])

        assert result.exit_code != 0
        assert "No log file found" in result.output


# --------------------------------------------------------------------------- #
# Help
# --------------------------------------------------------------------------- #


class TestHelp:
    """Tests for help text."""

    def test_help_shows_all_options(self) -> None:
        """--help documents all flags."""
        runner = CliRunner()
        result = runner.invoke(logs, ["--help"])

        assert result.exit_code == 0
        assert "--follow" in result.output
        assert "--tail" in result.output
        assert "--service" in result.output
        assert "--rotate" in result.output
        assert "--all" in result.output

    def test_logs_in_main_help(self) -> None:
        """logs appears in mlx-stack --help."""
        from mlx_stack.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "logs" in result.output


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #


class TestEdgeCases:
    """Edge case tests."""

    def test_service_argument_takes_precedence_over_flag(
        self, mlx_stack_home: Path
    ) -> None:
        """Positional argument takes precedence over --service flag."""
        logs_dir = mlx_stack_home / "logs"
        _create_log(logs_dir, "fast", "fast content\n")
        _create_log(logs_dir, "standard", "standard content\n")

        runner = CliRunner()
        result = runner.invoke(logs, ["fast", "--service", "standard"])

        assert result.exit_code == 0
        assert "fast content" in result.output

    def test_rotate_with_service_arg(self, mlx_stack_home: Path) -> None:
        """--rotate with positional service arg rotates just that service."""
        logs_dir = mlx_stack_home / "logs"
        logs_dir.mkdir(parents=True)
        (logs_dir / "fast.log").write_bytes(b"x" * (2 * 1024 * 1024))

        with patch(
            "mlx_stack.core.log_viewer.get_value",
            side_effect=lambda k: 1 if k == "log-max-size-mb" else 5,
        ):
            runner = CliRunner()
            result = runner.invoke(logs, ["fast", "--rotate"])

        assert result.exit_code == 0
        assert "fast" in result.output
