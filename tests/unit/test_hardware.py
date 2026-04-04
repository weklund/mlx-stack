"""Tests for the hardware detection module — core/hardware.py.

Covers chip detection, GPU core count, memory detection, bandwidth lookup,
unknown chip estimation, non-Apple-Silicon rejection, profile save/load,
and system command failure handling.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from mlx_stack.core.hardware import (
    CHIP_SPECS,
    HardwareError,
    HardwareProfile,
    detect_chip,
    detect_gpu_cores,
    detect_hardware,
    detect_memory_gb,
    estimate_bandwidth,
    load_profile,
    lookup_bandwidth,
    save_profile,
)

# --------------------------------------------------------------------------- #
# Helper: mock subprocess.run
# --------------------------------------------------------------------------- #


def _make_completed(
    stdout: str = "", returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["mock"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


# --------------------------------------------------------------------------- #
# HardwareProfile dataclass
# --------------------------------------------------------------------------- #


class TestHardwareProfile:
    """Tests for the HardwareProfile dataclass."""

    def test_profile_id_standard(self) -> None:
        p = HardwareProfile("Apple M4 Pro", 24, 64, 273.0, False)
        assert p.profile_id == "m4-pro-64"

    def test_profile_id_base_chip(self) -> None:
        p = HardwareProfile("Apple M1", 8, 16, 68.25, False)
        assert p.profile_id == "m1-16"

    def test_profile_id_ultra(self) -> None:
        p = HardwareProfile("Apple M2 Ultra", 76, 192, 800.0, False)
        assert p.profile_id == "m2-ultra-192"

    def test_profile_id_max(self) -> None:
        p = HardwareProfile("Apple M5 Max", 40, 128, 546.0, False)
        assert p.profile_id == "m5-max-128"

    def test_to_dict_has_required_fields(self) -> None:
        p = HardwareProfile("Apple M4", 10, 32, 120.0, False)
        d = p.to_dict()
        assert d["chip"] == "Apple M4"
        assert d["gpu_cores"] == 10
        assert d["memory_gb"] == 32
        assert d["bandwidth_gbps"] == 120.0
        assert d["profile_id"] == "m4-32"
        assert d["is_estimate"] is False

    def test_to_dict_includes_is_estimate_true(self) -> None:
        p = HardwareProfile("Apple M6", 32, 64, 400.0, True)
        d = p.to_dict()
        assert d["is_estimate"] is True

    def test_to_dict_values_non_null(self) -> None:
        p = HardwareProfile("Apple M3 Pro", 18, 36, 150.0, False)
        d = p.to_dict()
        for key in ("chip", "gpu_cores", "memory_gb", "bandwidth_gbps", "profile_id"):
            assert d[key] is not None

    def test_frozen(self) -> None:
        p = HardwareProfile("Apple M1", 8, 16, 68.25, False)
        with pytest.raises(AttributeError):
            p.chip = "Apple M2"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# detect_chip()
# --------------------------------------------------------------------------- #


class TestDetectChip:
    """Tests for detect_chip() with mocked sysctl."""

    @patch("mlx_stack.core.hardware._run_sysctl")
    def test_known_chip(self, mock_sysctl: object) -> None:
        mock_sysctl.return_value = "Apple M4 Pro"  # type: ignore[attr-defined]
        assert detect_chip() == "Apple M4 Pro"

    @patch("mlx_stack.core.hardware._run_sysctl")
    def test_m1_base(self, mock_sysctl: object) -> None:
        mock_sysctl.return_value = "Apple M1"  # type: ignore[attr-defined]
        assert detect_chip() == "Apple M1"

    @patch("mlx_stack.core.hardware._run_sysctl")
    def test_m5_max(self, mock_sysctl: object) -> None:
        mock_sysctl.return_value = "Apple M5 Max"  # type: ignore[attr-defined]
        assert detect_chip() == "Apple M5 Max"

    @patch("mlx_stack.core.hardware._run_sysctl")
    def test_future_chip(self, mock_sysctl: object) -> None:
        mock_sysctl.return_value = "Apple M6"  # type: ignore[attr-defined]
        # Unknown but still M-series — should succeed
        assert detect_chip() == "Apple M6"

    @patch("mlx_stack.core.hardware._run_sysctl")
    def test_intel_rejected(self, mock_sysctl: object) -> None:
        mock_sysctl.return_value = "Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz"  # type: ignore[attr-defined]
        with pytest.raises(HardwareError, match="requires Apple Silicon"):
            detect_chip()

    @patch("mlx_stack.core.hardware._run_sysctl")
    def test_empty_brand_string(self, mock_sysctl: object) -> None:
        mock_sysctl.return_value = ""  # type: ignore[attr-defined]
        with pytest.raises(HardwareError, match="Could not read CPU brand"):
            detect_chip()

    @patch("mlx_stack.core.hardware._run_sysctl")
    def test_sysctl_error_propagated(self, mock_sysctl: object) -> None:
        mock_sysctl.side_effect = HardwareError("sysctl failed")  # type: ignore[attr-defined]
        with pytest.raises(HardwareError, match="sysctl failed"):
            detect_chip()


# --------------------------------------------------------------------------- #
# detect_memory_gb()
# --------------------------------------------------------------------------- #


class TestDetectMemoryGb:
    """Tests for detect_memory_gb() with mocked sysctl."""

    @patch("mlx_stack.core.hardware._run_sysctl")
    def test_16gb(self, mock_sysctl: object) -> None:
        # 16 GB = 17179869184 bytes
        mock_sysctl.return_value = "17179869184"  # type: ignore[attr-defined]
        assert detect_memory_gb() == 16

    @patch("mlx_stack.core.hardware._run_sysctl")
    def test_32gb(self, mock_sysctl: object) -> None:
        mock_sysctl.return_value = "34359738368"  # type: ignore[attr-defined]
        assert detect_memory_gb() == 32

    @patch("mlx_stack.core.hardware._run_sysctl")
    def test_64gb(self, mock_sysctl: object) -> None:
        mock_sysctl.return_value = "68719476736"  # type: ignore[attr-defined]
        assert detect_memory_gb() == 64

    @patch("mlx_stack.core.hardware._run_sysctl")
    def test_128gb(self, mock_sysctl: object) -> None:
        mock_sysctl.return_value = "137438953472"  # type: ignore[attr-defined]
        assert detect_memory_gb() == 128

    @patch("mlx_stack.core.hardware._run_sysctl")
    def test_8gb(self, mock_sysctl: object) -> None:
        mock_sysctl.return_value = "8589934592"  # type: ignore[attr-defined]
        assert detect_memory_gb() == 8

    @patch("mlx_stack.core.hardware._run_sysctl")
    def test_invalid_value(self, mock_sysctl: object) -> None:
        mock_sysctl.return_value = "not-a-number"  # type: ignore[attr-defined]
        with pytest.raises(HardwareError, match=r"Unexpected hw\.memsize"):
            detect_memory_gb()


# --------------------------------------------------------------------------- #
# detect_gpu_cores()
# --------------------------------------------------------------------------- #


class TestDetectGpuCores:
    """Tests for detect_gpu_cores() with mocked system_profiler."""

    @patch("mlx_stack.core.hardware._run_system_profiler")
    def test_40_cores(self, mock_profiler: object) -> None:
        mock_profiler.return_value = (  # type: ignore[attr-defined]
            "Graphics/Displays:\n\n"
            "    Apple M5 Max:\n\n"
            "      Chipset Model: Apple M5 Max\n"
            "      Type: GPU\n"
            "      Bus: Built-In\n"
            "      Total Number of Cores: 40\n"
        )
        assert detect_gpu_cores() == 40

    @patch("mlx_stack.core.hardware._run_system_profiler")
    def test_10_cores(self, mock_profiler: object) -> None:
        mock_profiler.return_value = "Total Number of Cores: 10\n"  # type: ignore[attr-defined]
        assert detect_gpu_cores() == 10

    @patch("mlx_stack.core.hardware._run_system_profiler")
    def test_8_cores(self, mock_profiler: object) -> None:
        mock_profiler.return_value = "Total Number of Cores: 8\n"  # type: ignore[attr-defined]
        assert detect_gpu_cores() == 8

    @patch("mlx_stack.core.hardware._run_system_profiler")
    def test_no_cores_info(self, mock_profiler: object) -> None:
        mock_profiler.return_value = "Some unexpected output"  # type: ignore[attr-defined]
        with pytest.raises(HardwareError, match="Could not determine GPU core count"):
            detect_gpu_cores()


# --------------------------------------------------------------------------- #
# estimate_bandwidth() & lookup_bandwidth()
# --------------------------------------------------------------------------- #


class TestBandwidthLookup:
    """Tests for bandwidth lookup and estimation."""

    def test_all_17_known_chips_in_table(self) -> None:
        """Verify the lookup table has exactly 17 known variants."""
        assert len(CHIP_SPECS) == 17

    @pytest.mark.parametrize(
        ("chip", "expected_bw"),
        [
            ("Apple M1", 68.25),
            ("Apple M1 Pro", 200.0),
            ("Apple M1 Max", 400.0),
            ("Apple M1 Ultra", 800.0),
            ("Apple M2", 100.0),
            ("Apple M2 Pro", 200.0),
            ("Apple M2 Max", 400.0),
            ("Apple M2 Ultra", 800.0),
            ("Apple M3", 100.0),
            ("Apple M3 Pro", 150.0),
            ("Apple M3 Max", 400.0),
            ("Apple M3 Ultra", 800.0),
            ("Apple M4", 120.0),
            ("Apple M4 Pro", 273.0),
            ("Apple M4 Max", 546.0),
            ("Apple M4 Ultra", 819.2),
            ("Apple M5 Max", 546.0),
        ],
    )
    def test_known_chip_bandwidth(self, chip: str, expected_bw: float) -> None:
        bw, is_estimate = lookup_bandwidth(chip, 32)
        assert bw == expected_bw
        assert is_estimate is False

    def test_unknown_chip_returns_estimate(self) -> None:
        bw, is_estimate = lookup_bandwidth("Apple M6", 64)
        assert is_estimate is True
        assert bw > 0

    def test_estimate_8gb(self) -> None:
        assert estimate_bandwidth(8) == 68.0

    def test_estimate_16gb(self) -> None:
        assert estimate_bandwidth(16) == 100.0

    def test_estimate_32gb(self) -> None:
        assert estimate_bandwidth(32) == 200.0

    def test_estimate_64gb(self) -> None:
        assert estimate_bandwidth(64) == 400.0

    def test_estimate_128gb(self) -> None:
        assert estimate_bandwidth(128) == 546.0

    def test_estimate_192gb(self) -> None:
        assert estimate_bandwidth(192) == 800.0

    def test_estimate_4gb(self) -> None:
        assert estimate_bandwidth(4) == 68.0


# --------------------------------------------------------------------------- #
# detect_hardware() — full flow
# --------------------------------------------------------------------------- #


class TestDetectHardware:
    """Tests for the full detect_hardware() flow."""

    @patch("mlx_stack.core.hardware._run_system_profiler")
    @patch("mlx_stack.core.hardware._run_sysctl")
    def test_known_chip_full_detection(self, mock_sysctl: object, mock_profiler: object) -> None:
        def sysctl_side_effect(key: str) -> str:
            if key == "machdep.cpu.brand_string":
                return "Apple M4 Pro"
            if key == "hw.memsize":
                return "68719476736"  # 64 GB
            raise HardwareError(f"Unknown key {key}")

        mock_sysctl.side_effect = sysctl_side_effect  # type: ignore[attr-defined]
        mock_profiler.return_value = "Total Number of Cores: 20\n"  # type: ignore[attr-defined]

        profile = detect_hardware()

        assert profile.chip == "Apple M4 Pro"
        assert profile.gpu_cores == 20
        assert profile.memory_gb == 64
        assert profile.bandwidth_gbps == 273.0
        assert profile.is_estimate is False
        assert profile.profile_id == "m4-pro-64"

    @patch("mlx_stack.core.hardware._run_system_profiler")
    @patch("mlx_stack.core.hardware._run_sysctl")
    def test_unknown_chip_uses_estimate(self, mock_sysctl: object, mock_profiler: object) -> None:
        def sysctl_side_effect(key: str) -> str:
            if key == "machdep.cpu.brand_string":
                return "Apple M6"
            if key == "hw.memsize":
                return "68719476736"  # 64 GB
            raise HardwareError(f"Unknown key {key}")

        mock_sysctl.side_effect = sysctl_side_effect  # type: ignore[attr-defined]
        mock_profiler.return_value = "Total Number of Cores: 32\n"  # type: ignore[attr-defined]

        profile = detect_hardware()

        assert profile.chip == "Apple M6"
        assert profile.gpu_cores == 32
        assert profile.memory_gb == 64
        assert profile.bandwidth_gbps == 400.0
        assert profile.is_estimate is True

    @patch("mlx_stack.core.hardware._run_sysctl")
    def test_non_apple_silicon_rejected(self, mock_sysctl: object) -> None:
        mock_sysctl.return_value = "Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz"  # type: ignore[attr-defined]
        with pytest.raises(HardwareError, match="requires Apple Silicon"):
            detect_hardware()


# --------------------------------------------------------------------------- #
# _run_sysctl() — system command wrapper
# --------------------------------------------------------------------------- #


class TestRunSysctl:
    """Tests for _run_sysctl() — system command failures."""

    @patch("subprocess.run")
    def test_sysctl_not_found(self, mock_run: object) -> None:
        mock_run.side_effect = FileNotFoundError  # type: ignore[attr-defined]
        from mlx_stack.core.hardware import _run_sysctl

        with pytest.raises(HardwareError, match="sysctl command not found"):
            _run_sysctl("hw.memsize")

    @patch("subprocess.run")
    def test_sysctl_timeout(self, mock_run: object) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="sysctl", timeout=10)  # type: ignore[attr-defined]
        from mlx_stack.core.hardware import _run_sysctl

        with pytest.raises(HardwareError, match="sysctl timed out"):
            _run_sysctl("hw.memsize")

    @patch("subprocess.run")
    def test_sysctl_nonzero_return(self, mock_run: object) -> None:
        mock_run.return_value = _make_completed(returncode=1, stderr="Unknown key")  # type: ignore[attr-defined]
        from mlx_stack.core.hardware import _run_sysctl

        with pytest.raises(HardwareError, match="sysctl failed"):
            _run_sysctl("invalid.key")


# --------------------------------------------------------------------------- #
# _run_system_profiler() — system command wrapper
# --------------------------------------------------------------------------- #


class TestRunSystemProfiler:
    """Tests for _run_system_profiler() — system command failures."""

    @patch("subprocess.run")
    def test_profiler_not_found(self, mock_run: object) -> None:
        mock_run.side_effect = FileNotFoundError  # type: ignore[attr-defined]
        from mlx_stack.core.hardware import _run_system_profiler

        with pytest.raises(HardwareError, match="system_profiler command not found"):
            _run_system_profiler()

    @patch("subprocess.run")
    def test_profiler_timeout(self, mock_run: object) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="system_profiler", timeout=30)  # type: ignore[attr-defined]
        from mlx_stack.core.hardware import _run_system_profiler

        with pytest.raises(HardwareError, match="system_profiler timed out"):
            _run_system_profiler()

    @patch("subprocess.run")
    def test_profiler_nonzero_return(self, mock_run: object) -> None:
        mock_run.return_value = _make_completed(returncode=1, stderr="Error")  # type: ignore[attr-defined]
        from mlx_stack.core.hardware import _run_system_profiler

        with pytest.raises(HardwareError, match="system_profiler failed"):
            _run_system_profiler()


# --------------------------------------------------------------------------- #
# save_profile() & load_profile()
# --------------------------------------------------------------------------- #


class TestProfilePersistence:
    """Tests for saving and loading hardware profiles."""

    def test_save_creates_file(self, mlx_stack_home: Path) -> None:
        profile = HardwareProfile("Apple M4 Pro", 20, 64, 273.0, False)
        save_profile(profile)
        profile_path = mlx_stack_home / "profile.json"
        assert profile_path.exists()

    def test_save_valid_json(self, mlx_stack_home: Path) -> None:
        profile = HardwareProfile("Apple M4 Pro", 20, 64, 273.0, False)
        save_profile(profile)
        profile_path = mlx_stack_home / "profile.json"
        data = json.loads(profile_path.read_text())
        assert data["chip"] == "Apple M4 Pro"
        assert data["gpu_cores"] == 20
        assert data["memory_gb"] == 64
        assert data["bandwidth_gbps"] == 273.0
        assert data["profile_id"] == "m4-pro-64"

    def test_save_overwrites_existing(self, mlx_stack_home: Path) -> None:
        profile1 = HardwareProfile("Apple M1", 8, 16, 68.25, False)
        save_profile(profile1)

        profile2 = HardwareProfile("Apple M4 Pro", 20, 64, 273.0, False)
        save_profile(profile2)

        profile_path = mlx_stack_home / "profile.json"
        data = json.loads(profile_path.read_text())
        assert data["chip"] == "Apple M4 Pro"

    def test_save_creates_directory_if_missing(self, clean_mlx_stack_home: Path) -> None:
        assert not clean_mlx_stack_home.exists()
        profile = HardwareProfile("Apple M4", 10, 32, 120.0, False)
        save_profile(profile)
        assert (clean_mlx_stack_home / "profile.json").exists()

    def test_load_returns_profile(self, mlx_stack_home: Path) -> None:
        profile = HardwareProfile("Apple M3 Max", 40, 96, 400.0, False)
        save_profile(profile)

        loaded = load_profile()
        assert loaded is not None
        assert loaded.chip == "Apple M3 Max"
        assert loaded.gpu_cores == 40
        assert loaded.memory_gb == 96
        assert loaded.bandwidth_gbps == 400.0
        assert loaded.profile_id == "m3-max-96"

    def test_load_returns_none_when_missing(self, mlx_stack_home: Path) -> None:
        loaded = load_profile()
        assert loaded is None

    def test_load_returns_none_on_invalid_json(self, mlx_stack_home: Path) -> None:
        profile_path = mlx_stack_home / "profile.json"
        profile_path.write_text("not valid json {{{")
        loaded = load_profile()
        assert loaded is None

    def test_load_returns_none_on_missing_fields(self, mlx_stack_home: Path) -> None:
        profile_path = mlx_stack_home / "profile.json"
        profile_path.write_text(json.dumps({"chip": "Apple M1"}))
        loaded = load_profile()
        assert loaded is None

    def test_roundtrip_all_fields(self, mlx_stack_home: Path) -> None:
        original = HardwareProfile("Apple M4 Max", 40, 128, 546.0, False)
        save_profile(original)
        loaded = load_profile()
        assert loaded is not None
        assert loaded.chip == original.chip
        assert loaded.gpu_cores == original.gpu_cores
        assert loaded.memory_gb == original.memory_gb
        assert loaded.bandwidth_gbps == original.bandwidth_gbps
        assert loaded.profile_id == original.profile_id
