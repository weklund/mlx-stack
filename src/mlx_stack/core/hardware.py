"""Hardware detection module for Apple Silicon Macs.

Detects chip model, GPU core count, unified memory, and memory bandwidth.
Uses a lookup table of 17 known M-series variants (M1 through M5) and
estimates bandwidth for unknown chips.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from typing import Any

from mlx_stack.core.paths import ensure_data_home, get_profile_path

# --------------------------------------------------------------------------- #
# Lookup table — 17 known Apple Silicon M-series variants
# bandwidth_gbps values sourced from Apple published specs
# --------------------------------------------------------------------------- #
CHIP_SPECS: dict[str, dict[str, float | int]] = {
    "Apple M1": {"bandwidth_gbps": 68.25},
    "Apple M1 Pro": {"bandwidth_gbps": 200.0},
    "Apple M1 Max": {"bandwidth_gbps": 400.0},
    "Apple M1 Ultra": {"bandwidth_gbps": 800.0},
    "Apple M2": {"bandwidth_gbps": 100.0},
    "Apple M2 Pro": {"bandwidth_gbps": 200.0},
    "Apple M2 Max": {"bandwidth_gbps": 400.0},
    "Apple M2 Ultra": {"bandwidth_gbps": 800.0},
    "Apple M3": {"bandwidth_gbps": 100.0},
    "Apple M3 Pro": {"bandwidth_gbps": 150.0},
    "Apple M3 Max": {"bandwidth_gbps": 400.0},
    "Apple M3 Ultra": {"bandwidth_gbps": 800.0},
    "Apple M4": {"bandwidth_gbps": 120.0},
    "Apple M4 Pro": {"bandwidth_gbps": 273.0},
    "Apple M4 Max": {"bandwidth_gbps": 546.0},
    "Apple M4 Ultra": {"bandwidth_gbps": 819.2},
    "Apple M5 Max": {"bandwidth_gbps": 546.0},
}


class HardwareError(Exception):
    """Raised when hardware detection fails or hardware is unsupported."""


@dataclass(frozen=True)
class HardwareProfile:
    """Detected hardware profile for the current machine."""

    chip: str
    gpu_cores: int
    memory_gb: int
    bandwidth_gbps: float
    is_estimate: bool

    @property
    def profile_id(self) -> str:
        """Generate a profile ID like 'm4-pro-64'."""
        # Strip 'Apple ' prefix, lowercase, replace spaces with hyphens
        chip_part = self.chip.removeprefix("Apple ").lower().replace(" ", "-")
        return f"{chip_part}-{self.memory_gb}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary suitable for JSON output."""
        return {
            "chip": self.chip,
            "gpu_cores": self.gpu_cores,
            "memory_gb": self.memory_gb,
            "bandwidth_gbps": self.bandwidth_gbps,
            "profile_id": self.profile_id,
        }


# --------------------------------------------------------------------------- #
# System command wrappers (mockable for tests)
# --------------------------------------------------------------------------- #


def _run_sysctl(key: str) -> str:
    """Run sysctl and return the output for a given key.

    Raises:
        HardwareError: If the sysctl command fails.
    """
    try:
        result = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            msg = f"sysctl failed for key '{key}': {result.stderr.strip()}"
            raise HardwareError(msg)
        return result.stdout.strip()
    except FileNotFoundError:
        msg = "sysctl command not found — are you running on macOS?"
        raise HardwareError(msg) from None
    except subprocess.TimeoutExpired:
        msg = f"sysctl timed out reading key '{key}'"
        raise HardwareError(msg) from None


def _run_system_profiler() -> str:
    """Run system_profiler SPDisplaysDataType and return output.

    Raises:
        HardwareError: If the system_profiler command fails.
    """
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            msg = f"system_profiler failed: {result.stderr.strip()}"
            raise HardwareError(msg)
        return result.stdout
    except FileNotFoundError:
        msg = "system_profiler command not found — are you running on macOS?"
        raise HardwareError(msg) from None
    except subprocess.TimeoutExpired:
        msg = "system_profiler timed out"
        raise HardwareError(msg) from None


# --------------------------------------------------------------------------- #
# Detection functions
# --------------------------------------------------------------------------- #


def detect_chip() -> str:
    """Detect the chip model name from sysctl.

    Returns:
        Chip name string, e.g. 'Apple M4 Pro'.

    Raises:
        HardwareError: If the chip cannot be detected or is not Apple Silicon.
    """
    brand = _run_sysctl("machdep.cpu.brand_string")
    if not brand:
        msg = "Could not read CPU brand string from sysctl"
        raise HardwareError(msg)

    # Validate it's Apple Silicon (M-series)
    if not re.match(r"Apple M\d", brand):
        msg = "mlx-stack requires Apple Silicon (M1 or later)"
        raise HardwareError(msg)

    return brand


def detect_memory_gb() -> int:
    """Detect unified memory in GB from sysctl hw.memsize.

    Returns:
        Memory in GB as an integer.

    Raises:
        HardwareError: If memory size cannot be read.
    """
    raw = _run_sysctl("hw.memsize")
    try:
        memsize_bytes = int(raw)
    except ValueError:
        msg = f"Unexpected hw.memsize value: {raw!r}"
        raise HardwareError(msg) from None

    return memsize_bytes // (1024**3)


def detect_gpu_cores() -> int:
    """Detect GPU core count from system_profiler.

    Returns:
        Number of GPU cores.

    Raises:
        HardwareError: If GPU core count cannot be determined.
    """
    output = _run_system_profiler()

    # Parse "Total Number of Cores: 40" from system_profiler output
    match = re.search(r"Total Number of Cores:\s*(\d+)", output)
    if not match:
        msg = "Could not determine GPU core count from system_profiler output"
        raise HardwareError(msg)

    return int(match.group(1))


def estimate_bandwidth(memory_gb: int) -> float:
    """Estimate memory bandwidth for unknown chips based on memory size.

    Uses a simple heuristic: larger memory configurations tend to have
    higher bandwidth. This is a rough estimate — users should run
    `mlx-stack bench --save` for accurate numbers.

    Args:
        memory_gb: Total unified memory in GB.

    Returns:
        Estimated bandwidth in GB/s.
    """
    if memory_gb <= 8:
        return 68.0
    if memory_gb <= 16:
        return 100.0
    if memory_gb <= 32:
        return 200.0
    if memory_gb <= 64:
        return 400.0
    if memory_gb <= 128:
        return 546.0
    return 800.0


def lookup_bandwidth(chip: str, memory_gb: int) -> tuple[float, bool]:
    """Look up bandwidth from the chip specs table.

    Args:
        chip: Full chip name (e.g. 'Apple M4 Pro').
        memory_gb: Total unified memory in GB.

    Returns:
        Tuple of (bandwidth_gbps, is_estimate). is_estimate is True when
        the chip is not in the lookup table and bandwidth was estimated.
    """
    specs = CHIP_SPECS.get(chip)
    if specs is not None:
        return float(specs["bandwidth_gbps"]), False
    return estimate_bandwidth(memory_gb), True


# --------------------------------------------------------------------------- #
# Main detection entry point
# --------------------------------------------------------------------------- #


def detect_hardware() -> HardwareProfile:
    """Detect hardware and build a HardwareProfile.

    Detects chip model, GPU core count, unified memory, and memory
    bandwidth. Known chips use the lookup table; unknown chips use
    a bandwidth estimation.

    Returns:
        A HardwareProfile with all detected values.

    Raises:
        HardwareError: If hardware cannot be detected or is unsupported.
    """
    chip = detect_chip()
    memory_gb = detect_memory_gb()
    gpu_cores = detect_gpu_cores()
    bandwidth_gbps, is_estimate = lookup_bandwidth(chip, memory_gb)

    return HardwareProfile(
        chip=chip,
        gpu_cores=gpu_cores,
        memory_gb=memory_gb,
        bandwidth_gbps=bandwidth_gbps,
        is_estimate=is_estimate,
    )


def save_profile(profile: HardwareProfile) -> None:
    """Write the hardware profile to ~/.mlx-stack/profile.json.

    Overwrites any existing profile file.

    Args:
        profile: The hardware profile to save.
    """
    ensure_data_home()
    profile_path = get_profile_path()
    profile_path.write_text(json.dumps(profile.to_dict(), indent=2) + "\n")


def load_profile() -> HardwareProfile | None:
    """Load the hardware profile from disk, if it exists.

    Returns:
        A HardwareProfile if the file exists and is valid, None otherwise.
    """
    profile_path = get_profile_path()
    if not profile_path.exists():
        return None

    try:
        data = json.loads(profile_path.read_text())
        return HardwareProfile(
            chip=data["chip"],
            gpu_cores=data["gpu_cores"],
            memory_gb=data["memory_gb"],
            bandwidth_gbps=data["bandwidth_gbps"],
            is_estimate=False,  # saved profiles are considered authoritative
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
