"""Compatibility matrix report generator.

Pytest plugin that collects per-model test results from smoke tests
and writes a JSON compatibility matrix to disk.

The matrix records pass/fail/skip status for inference, tool calling,
and thinking capabilities, along with timing and memory data.

**Output location:** ``~/.mlx-stack-test-cache/reports/compatibility-matrix-{timestamp}.json``

Usage: This plugin is auto-loaded by pytest when present in the test
directory. No registration needed.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# Report data classes
# --------------------------------------------------------------------------- #

REPORT_DIR = Path.home() / ".mlx-stack-test-cache" / "reports"


@dataclass
class ModelTestResult:
    """Result of smoke-testing a single model."""

    model_id: str
    quant: str
    inference: str = "skip"  # "pass" | "fail" | "skip"
    tool_calling: str | None = None  # "pass" | "fail" | "skip" | None (N/A)
    thinking: str | None = None  # "pass" | "fail" | "skip" | None (N/A)
    memory_actual_gb: float | None = None
    memory_catalog_gb: float | None = None
    startup_time_s: float | None = None
    inference_time_s: float | None = None
    error: str | None = None


@dataclass
class CompatibilityMatrix:
    """Full compatibility matrix for a test run."""

    timestamp: str = ""
    platform: str = ""
    chip: str = ""
    memory_gb: float = 0.0
    vllm_mlx_version: str = ""
    results: list[ModelTestResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.platform:
            self.platform = f"{platform.system()} {platform.release()}"

    def add_result(self, result: ModelTestResult) -> None:
        """Add a model test result to the matrix."""
        self.results.append(result)

    def write(self, output_dir: Path | None = None) -> Path:
        """Write the matrix to a JSON file.

        Returns:
            Path to the written JSON file.
        """
        out_dir = output_dir or REPORT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = out_dir / f"compatibility-matrix-{ts}.json"

        data = {
            "timestamp": self.timestamp,
            "platform": self.platform,
            "chip": self.chip,
            "memory_gb": self.memory_gb,
            "vllm_mlx_version": self.vllm_mlx_version,
            "results": [asdict(r) for r in self.results],
        }

        path.write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8",
        )
        return path

    def summary(self) -> dict[str, int]:
        """Return pass/fail/skip counts for inference tests."""
        counts: dict[str, int] = {"pass": 0, "fail": 0, "skip": 0}
        for r in self.results:
            counts[r.inference] = counts.get(r.inference, 0) + 1
        return counts
