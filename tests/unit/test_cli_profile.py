"""Tests for the `mlx-stack profile` CLI command.

Validates VAL-PROFILE-001 through VAL-PROFILE-007: chip detection,
unknown chip handling, non-Apple-Silicon rejection, profile JSON format,
Rich table output, overwrite behavior, and error handling.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from mlx_stack.cli.main import cli
from mlx_stack.core.hardware import HardwareError, HardwareProfile


def _mock_known_hardware() -> HardwareProfile:
    """Return a mock profile for Apple M4 Pro."""
    return HardwareProfile(
        chip="Apple M4 Pro",
        gpu_cores=20,
        memory_gb=64,
        bandwidth_gbps=273.0,
        is_estimate=False,
    )


def _mock_unknown_hardware() -> HardwareProfile:
    """Return a mock profile for an unknown future chip."""
    return HardwareProfile(
        chip="Apple M6",
        gpu_cores=32,
        memory_gb=64,
        bandwidth_gbps=400.0,
        is_estimate=True,
    )


class TestProfileKnownChip:
    """VAL-PROFILE-001: Known Apple Silicon chip detection and display."""

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_exits_zero(
        self, mock_detect: object, mlx_stack_home: Path
    ) -> None:
        mock_detect.return_value = _mock_known_hardware()  # type: ignore[attr-defined]
        runner = CliRunner()
        result = runner.invoke(cli, ["profile"])
        assert result.exit_code == 0

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_shows_chip_name(
        self, mock_detect: object, mlx_stack_home: Path
    ) -> None:
        mock_detect.return_value = _mock_known_hardware()  # type: ignore[attr-defined]
        runner = CliRunner()
        result = runner.invoke(cli, ["profile"])
        assert "Apple M4 Pro" in result.output

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_shows_gpu_cores(
        self, mock_detect: object, mlx_stack_home: Path
    ) -> None:
        mock_detect.return_value = _mock_known_hardware()  # type: ignore[attr-defined]
        runner = CliRunner()
        result = runner.invoke(cli, ["profile"])
        assert "20" in result.output

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_shows_memory(
        self, mock_detect: object, mlx_stack_home: Path
    ) -> None:
        mock_detect.return_value = _mock_known_hardware()  # type: ignore[attr-defined]
        runner = CliRunner()
        result = runner.invoke(cli, ["profile"])
        assert "64 GB" in result.output

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_shows_bandwidth(
        self, mock_detect: object, mlx_stack_home: Path
    ) -> None:
        mock_detect.return_value = _mock_known_hardware()  # type: ignore[attr-defined]
        runner = CliRunner()
        result = runner.invoke(cli, ["profile"])
        assert "273.0 GB/s" in result.output

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_shows_profile_id(
        self, mock_detect: object, mlx_stack_home: Path
    ) -> None:
        mock_detect.return_value = _mock_known_hardware()  # type: ignore[attr-defined]
        runner = CliRunner()
        result = runner.invoke(cli, ["profile"])
        assert "m4-pro-64" in result.output

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_no_warning_for_known_chip(
        self, mock_detect: object, mlx_stack_home: Path
    ) -> None:
        mock_detect.return_value = _mock_known_hardware()  # type: ignore[attr-defined]
        runner = CliRunner()
        result = runner.invoke(cli, ["profile"])
        assert "(estimate)" not in result.output
        assert "unknown chip" not in result.output.lower()
        assert "bench --save" not in result.output


class TestProfileUnknownChip:
    """VAL-PROFILE-002: Unknown chip estimation with bench suggestion."""

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_exits_zero(
        self, mock_detect: object, mlx_stack_home: Path
    ) -> None:
        mock_detect.return_value = _mock_unknown_hardware()  # type: ignore[attr-defined]
        runner = CliRunner()
        result = runner.invoke(cli, ["profile"])
        assert result.exit_code == 0

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_shows_estimate_label(
        self, mock_detect: object, mlx_stack_home: Path
    ) -> None:
        mock_detect.return_value = _mock_unknown_hardware()  # type: ignore[attr-defined]
        runner = CliRunner()
        result = runner.invoke(cli, ["profile"])
        assert "estimate" in result.output.lower()

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_shows_bench_suggestion(
        self, mock_detect: object, mlx_stack_home: Path
    ) -> None:
        mock_detect.return_value = _mock_unknown_hardware()  # type: ignore[attr-defined]
        runner = CliRunner()
        result = runner.invoke(cli, ["profile"])
        assert "bench --save" in result.output

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_profile_still_written(
        self, mock_detect: object, mlx_stack_home: Path
    ) -> None:
        mock_detect.return_value = _mock_unknown_hardware()  # type: ignore[attr-defined]
        runner = CliRunner()
        result = runner.invoke(cli, ["profile"])
        assert result.exit_code == 0

        profile_path = mlx_stack_home / "profile.json"
        assert profile_path.exists()
        data = json.loads(profile_path.read_text())
        assert data["chip"] == "Apple M6"


class TestProfileNonAppleSilicon:
    """VAL-PROFILE-003: Non-Apple-Silicon rejection."""

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_nonzero_exit(
        self, mock_detect: object, mlx_stack_home: Path
    ) -> None:
        mock_detect.side_effect = HardwareError(  # type: ignore[attr-defined]
            "mlx-stack requires Apple Silicon (M1 or later)"
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["profile"])
        assert result.exit_code != 0

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_error_message(
        self, mock_detect: object, mlx_stack_home: Path
    ) -> None:
        mock_detect.side_effect = HardwareError(  # type: ignore[attr-defined]
            "mlx-stack requires Apple Silicon (M1 or later)"
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["profile"])
        assert "requires Apple Silicon" in result.output

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_no_traceback(
        self, mock_detect: object, mlx_stack_home: Path
    ) -> None:
        mock_detect.side_effect = HardwareError(  # type: ignore[attr-defined]
            "mlx-stack requires Apple Silicon (M1 or later)"
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["profile"])
        assert "Traceback" not in result.output

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_no_profile_written(
        self, mock_detect: object, mlx_stack_home: Path
    ) -> None:
        mock_detect.side_effect = HardwareError(  # type: ignore[attr-defined]
            "mlx-stack requires Apple Silicon (M1 or later)"
        )
        runner = CliRunner()
        runner.invoke(cli, ["profile"])
        profile_path = mlx_stack_home / "profile.json"
        assert not profile_path.exists()


class TestProfileJsonFormat:
    """VAL-PROFILE-004: Profile JSON is valid, complete, and correctly located."""

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_valid_json(
        self, mock_detect: object, mlx_stack_home: Path
    ) -> None:
        mock_detect.return_value = _mock_known_hardware()  # type: ignore[attr-defined]
        runner = CliRunner()
        runner.invoke(cli, ["profile"])

        profile_path = mlx_stack_home / "profile.json"
        data = json.loads(profile_path.read_text())
        assert isinstance(data, dict)

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_all_required_fields(
        self, mock_detect: object, mlx_stack_home: Path
    ) -> None:
        mock_detect.return_value = _mock_known_hardware()  # type: ignore[attr-defined]
        runner = CliRunner()
        runner.invoke(cli, ["profile"])

        profile_path = mlx_stack_home / "profile.json"
        data = json.loads(profile_path.read_text())

        assert "chip" in data
        assert "gpu_cores" in data
        assert "memory_gb" in data
        assert "bandwidth_gbps" in data
        assert "profile_id" in data

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_field_types(
        self, mock_detect: object, mlx_stack_home: Path
    ) -> None:
        mock_detect.return_value = _mock_known_hardware()  # type: ignore[attr-defined]
        runner = CliRunner()
        runner.invoke(cli, ["profile"])

        profile_path = mlx_stack_home / "profile.json"
        data = json.loads(profile_path.read_text())

        assert isinstance(data["chip"], str)
        assert isinstance(data["gpu_cores"], int)
        assert isinstance(data["memory_gb"], int)
        assert isinstance(data["bandwidth_gbps"], (int, float))
        assert isinstance(data["profile_id"], str)

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_profile_id_pattern(
        self, mock_detect: object, mlx_stack_home: Path
    ) -> None:
        mock_detect.return_value = _mock_known_hardware()  # type: ignore[attr-defined]
        runner = CliRunner()
        runner.invoke(cli, ["profile"])

        profile_path = mlx_stack_home / "profile.json"
        data = json.loads(profile_path.read_text())
        # profile_id should follow <chip-variant>-<memory_gb> pattern
        assert data["profile_id"] == "m4-pro-64"

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_all_values_non_null(
        self, mock_detect: object, mlx_stack_home: Path
    ) -> None:
        mock_detect.return_value = _mock_known_hardware()  # type: ignore[attr-defined]
        runner = CliRunner()
        runner.invoke(cli, ["profile"])

        profile_path = mlx_stack_home / "profile.json"
        data = json.loads(profile_path.read_text())

        for key in ("chip", "gpu_cores", "memory_gb", "bandwidth_gbps", "profile_id"):
            assert data[key] is not None, f"Field '{key}' should not be null"


class TestProfileRichTable:
    """VAL-PROFILE-005: Output is a Rich-formatted table."""

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_table_header_present(
        self, mock_detect: object, mlx_stack_home: Path
    ) -> None:
        mock_detect.return_value = _mock_known_hardware()  # type: ignore[attr-defined]
        runner = CliRunner()
        result = runner.invoke(cli, ["profile"])
        assert "Hardware Profile" in result.output

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_table_has_property_labels(
        self, mock_detect: object, mlx_stack_home: Path
    ) -> None:
        mock_detect.return_value = _mock_known_hardware()  # type: ignore[attr-defined]
        runner = CliRunner()
        result = runner.invoke(cli, ["profile"])
        assert "Chip" in result.output
        assert "GPU Cores" in result.output
        assert "Memory" in result.output or "Unified Memory" in result.output
        assert "Bandwidth" in result.output or "Memory Bandwidth" in result.output
        assert "Profile ID" in result.output

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_table_has_borders(
        self, mock_detect: object, mlx_stack_home: Path
    ) -> None:
        """Rich tables include box-drawing characters or similar formatting."""
        mock_detect.return_value = _mock_known_hardware()  # type: ignore[attr-defined]
        runner = CliRunner()
        result = runner.invoke(cli, ["profile"])
        # Rich tables typically use ─, │, ┌, ┐, etc.
        assert any(
            c in result.output for c in ("─", "│", "┌", "┐", "└", "┘", "┬", "┴", "├", "┤")
        ), "Expected Rich table border characters in output"


class TestProfileOverwrite:
    """VAL-PROFILE-006: Re-running profile overwrites existing data."""

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_overwrite(
        self, mock_detect: object, mlx_stack_home: Path
    ) -> None:
        # First run with M1
        hw1 = HardwareProfile("Apple M1", 8, 16, 68.25, False)
        mock_detect.return_value = hw1  # type: ignore[attr-defined]
        runner = CliRunner()
        runner.invoke(cli, ["profile"])

        profile_path = mlx_stack_home / "profile.json"
        data1 = json.loads(profile_path.read_text())
        assert data1["chip"] == "Apple M1"

        # Second run with M4 Pro
        hw2 = _mock_known_hardware()
        mock_detect.return_value = hw2  # type: ignore[attr-defined]
        runner.invoke(cli, ["profile"])

        data2 = json.loads(profile_path.read_text())
        assert data2["chip"] == "Apple M4 Pro"


class TestProfileErrorHandling:
    """VAL-PROFILE-007: System command failures handled gracefully."""

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_sysctl_error_no_traceback(
        self, mock_detect: object, mlx_stack_home: Path
    ) -> None:
        mock_detect.side_effect = HardwareError(  # type: ignore[attr-defined]
            "sysctl failed for key 'machdep.cpu.brand_string': Operation not permitted"
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["profile"])
        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert "Error" in result.output

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_profiler_error_no_traceback(
        self, mock_detect: object, mlx_stack_home: Path
    ) -> None:
        mock_detect.side_effect = HardwareError(  # type: ignore[attr-defined]
            "system_profiler command not found — are you running on macOS?"
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["profile"])
        assert result.exit_code != 0
        assert "Traceback" not in result.output

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_descriptive_error_message(
        self, mock_detect: object, mlx_stack_home: Path
    ) -> None:
        mock_detect.side_effect = HardwareError(  # type: ignore[attr-defined]
            "sysctl timed out reading key 'hw.memsize'"
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["profile"])
        assert "sysctl timed out" in result.output


class TestProfileAutoCreatesDirectory:
    """VAL-SETUP-004: Profile auto-creates ~/.mlx-stack/ on first use."""

    @patch("mlx_stack.cli.profile.detect_hardware")
    def test_creates_data_dir(
        self, mock_detect: object, clean_mlx_stack_home: Path
    ) -> None:
        assert not clean_mlx_stack_home.exists()
        mock_detect.return_value = _mock_known_hardware()  # type: ignore[attr-defined]
        runner = CliRunner()
        result = runner.invoke(cli, ["profile"])
        assert result.exit_code == 0
        assert clean_mlx_stack_home.exists()
        assert (clean_mlx_stack_home / "profile.json").exists()
