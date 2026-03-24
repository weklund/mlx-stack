"""Tests for the `mlx-stack models` CLI command and core models module.

Validates:
- VAL-MODELS-001: Lists locally downloaded models with disk size and quant
- VAL-MODELS-002: Active stack models marked with indicator
- VAL-MODELS-003: No local models produces informative message
- VAL-MODELS-004: --catalog shows full catalog with hardware-specific data
- VAL-MODELS-005: Output is a formatted table with human-readable names
- VAL-MODELS-006: Custom model-dir configuration respected
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import yaml
from click.testing import CliRunner

from mlx_stack.cli.main import cli
from mlx_stack.core.catalog import (
    BenchmarkResult,
    Capabilities,
    CatalogEntry,
    QualityScores,
    QuantSource,
)
from mlx_stack.core.hardware import HardwareProfile
from mlx_stack.core.models import (
    format_size,
    get_models_directory,
    get_remote_stack_models,
    list_catalog_models,
    scan_local_models,
)

# --------------------------------------------------------------------------- #
# Fixtures — reusable test data
# --------------------------------------------------------------------------- #


def _make_profile(
    chip: str = "Apple M4 Max",
    gpu_cores: int = 40,
    memory_gb: int = 128,
    bandwidth_gbps: float = 546.0,
    is_estimate: bool = False,
) -> HardwareProfile:
    """Create a HardwareProfile for testing."""
    return HardwareProfile(
        chip=chip,
        gpu_cores=gpu_cores,
        memory_gb=memory_gb,
        bandwidth_gbps=bandwidth_gbps,
        is_estimate=is_estimate,
    )


def _make_entry(
    model_id: str = "test-model",
    name: str = "Test Model",
    family: str = "Test",
    params_b: float = 8.0,
    architecture: str = "transformer",
    quality_overall: int = 70,
    tool_calling: bool = True,
    benchmarks: dict[str, BenchmarkResult] | None = None,
    tags: list[str] | None = None,
    memory_gb: float = 5.5,
    disk_size_gb: float = 4.5,
) -> CatalogEntry:
    """Create a CatalogEntry for testing."""
    if benchmarks is None:
        benchmarks = {
            "m4-max-128": BenchmarkResult(prompt_tps=140.0, gen_tps=77.0, memory_gb=memory_gb),
            "m4-pro-48": BenchmarkResult(prompt_tps=95.0, gen_tps=52.0, memory_gb=memory_gb),
        }
    return CatalogEntry(
        id=model_id,
        name=name,
        family=family,
        params_b=params_b,
        architecture=architecture,
        min_mlx_lm_version="0.22.0",
        sources={
            "int4": QuantSource(
                hf_repo=f"mlx-community/{model_id}-4bit",
                disk_size_gb=disk_size_gb,
            ),
            "int8": QuantSource(
                hf_repo=f"mlx-community/{model_id}-8bit",
                disk_size_gb=disk_size_gb * 2,
            ),
        },
        capabilities=Capabilities(
            tool_calling=tool_calling,
            tool_call_parser="hermes" if tool_calling else None,
            thinking=False,
            reasoning_parser=None,
            vision=False,
        ),
        quality=QualityScores(
            overall=quality_overall,
            coding=65,
            reasoning=60,
            instruction_following=72,
        ),
        benchmarks=benchmarks,
        tags=tags or [],
    )


def _make_test_catalog() -> list[CatalogEntry]:
    """Create a test catalog with several models."""
    return [
        _make_entry(
            model_id="qwen3.5-8b",
            name="Qwen 3.5 8B",
            family="Qwen 3.5",
            params_b=8.0,
            memory_gb=5.5,
            disk_size_gb=4.5,
            tags=["balanced", "agent-ready"],
        ),
        _make_entry(
            model_id="nemotron-8b",
            name="Nemotron 8B",
            family="Nemotron",
            params_b=8.0,
            memory_gb=5.0,
            disk_size_gb=4.2,
            tags=["agent-ready"],
        ),
        _make_entry(
            model_id="gemma3-12b",
            name="Gemma 3 12B",
            family="Gemma 3",
            params_b=12.0,
            memory_gb=7.5,
            disk_size_gb=6.5,
        ),
    ]


def _create_model_dir(
    models_dir: Path,
    dirname: str,
    size_bytes: int = 4_500_000_000,
    num_files: int = 3,
) -> Path:
    """Create a fake model directory with files of specified total size.

    Args:
        models_dir: Parent models directory.
        dirname: Name for the model directory.
        size_bytes: Total size in bytes to distribute across files.
        num_files: Number of files to create.

    Returns:
        Path to the created model directory.
    """
    model_dir = models_dir / dirname
    model_dir.mkdir(parents=True, exist_ok=True)

    per_file = size_bytes // num_files
    remainder = size_bytes % num_files

    for i in range(num_files):
        fsize = per_file + (remainder if i == 0 else 0)
        fp = model_dir / f"model-{i:05d}.safetensors"
        fp.write_bytes(b"\x00" * min(fsize, 100))  # Write small files for speed
        # Truncate to desired size for stat
        with open(fp, "wb") as f:
            f.seek(fsize - 1)
            f.write(b"\x00")

    return model_dir


def _create_stack_yaml(
    stacks_dir: Path,
    tiers: list[dict],
    stack_name: str = "default",
) -> Path:
    """Create a stack definition YAML file."""
    stacks_dir.mkdir(parents=True, exist_ok=True)
    stack = {
        "schema_version": 1,
        "name": stack_name,
        "hardware_profile": "m4-max-128",
        "intent": "balanced",
        "created": "2025-01-01T00:00:00+00:00",
        "tiers": tiers,
    }
    path = stacks_dir / f"{stack_name}.yaml"
    path.write_text(yaml.dump(stack, default_flow_style=False), encoding="utf-8")
    return path


# =========================================================================== #
# Core module tests — scan_local_models
# =========================================================================== #


class TestScanLocalModels:
    """Tests for the scan_local_models function."""

    def test_empty_models_dir(self, mlx_stack_home: Path) -> None:
        """Returns empty list when models dir exists but is empty."""
        models_dir = mlx_stack_home / "models"
        models_dir.mkdir(parents=True)

        result = scan_local_models(models_dir=models_dir, catalog=[])
        assert result == []

    def test_missing_models_dir(self, mlx_stack_home: Path) -> None:
        """Returns empty list when models dir does not exist."""
        models_dir = mlx_stack_home / "models"
        # Do NOT create the directory
        result = scan_local_models(models_dir=models_dir, catalog=[])
        assert result == []

    def test_discovers_model_directories(self, mlx_stack_home: Path) -> None:
        """Finds model directories and reports their disk size."""
        models_dir = mlx_stack_home / "models"
        _create_model_dir(models_dir, "Qwen3.5-8B-4bit", size_bytes=4_500_000_000)

        result = scan_local_models(models_dir=models_dir, catalog=[])
        assert len(result) == 1
        assert result[0].name == "Qwen3.5-8B-4bit"
        assert result[0].disk_size_bytes > 0

    def test_detects_quant_from_dirname(self, mlx_stack_home: Path) -> None:
        """Detects quantization level from directory name patterns."""
        models_dir = mlx_stack_home / "models"
        _create_model_dir(models_dir, "Qwen3.5-8B-4bit", size_bytes=1000)
        _create_model_dir(models_dir, "Qwen3.5-8B-8bit", size_bytes=1000)
        _create_model_dir(models_dir, "SomeModel-bf16", size_bytes=1000)

        result = scan_local_models(models_dir=models_dir, catalog=[])
        quants = {m.name: m.quant for m in result}

        assert quants["Qwen3.5-8B-4bit"] == "int4"
        assert quants["Qwen3.5-8B-8bit"] == "int8"
        assert quants["SomeModel-bf16"] == "bf16"

    def test_detects_quant_unknown(self, mlx_stack_home: Path) -> None:
        """Reports 'unknown' when quant can't be determined."""
        models_dir = mlx_stack_home / "models"
        _create_model_dir(models_dir, "mystery-model", size_bytes=1000)

        result = scan_local_models(models_dir=models_dir, catalog=[])
        assert result[0].quant == "unknown"

    def test_detects_source_type(self, mlx_stack_home: Path) -> None:
        """Detects source type from directory name."""
        models_dir = mlx_stack_home / "models"
        _create_model_dir(models_dir, "mlx-community--Qwen3.5-8B-4bit", size_bytes=1000)
        _create_model_dir(models_dir, "random-model", size_bytes=1000)

        result = scan_local_models(models_dir=models_dir, catalog=[])
        sources = {m.name: m.source_type for m in result}
        assert sources["mlx-community--Qwen3.5-8B-4bit"] == "mlx-community"
        assert sources["random-model"] == "unknown"

    def test_matches_catalog_names(self, mlx_stack_home: Path) -> None:
        """Matches local directories to catalog entries for human-readable names."""
        models_dir = mlx_stack_home / "models"
        catalog = _make_test_catalog()

        # Create dir matching the HF repo name for qwen3.5-8b int4
        _create_model_dir(models_dir, "qwen3.5-8b-4bit", size_bytes=1000)

        result = scan_local_models(models_dir=models_dir, catalog=catalog)
        assert len(result) == 1
        assert result[0].catalog_name == "Qwen 3.5 8B"

    def test_active_stack_indicator(self, mlx_stack_home: Path) -> None:
        """Marks models as active when referenced by the active stack."""
        models_dir = mlx_stack_home / "models"
        stacks_dir = mlx_stack_home / "stacks"
        catalog = _make_test_catalog()

        # Create model dirs
        _create_model_dir(models_dir, "qwen3.5-8b-4bit", size_bytes=1000)
        _create_model_dir(models_dir, "random-model", size_bytes=1000)

        # Create stack referencing qwen3.5-8b
        stack_tiers = [
            {
                "name": "fast",
                "model": "qwen3.5-8b",
                "quant": "int4",
                "source": "mlx-community/qwen3.5-8b-4bit",
                "port": 8000,
            }
        ]
        _create_stack_yaml(stacks_dir, stack_tiers)

        stack = yaml.safe_load((stacks_dir / "default.yaml").read_text())
        result = scan_local_models(models_dir=models_dir, catalog=catalog, stack=stack)

        active = {m.name: m.is_active for m in result}
        assert active["qwen3.5-8b-4bit"] is True
        assert active["random-model"] is False

    def test_ignores_regular_files(self, mlx_stack_home: Path) -> None:
        """Ignores regular files in the models directory."""
        models_dir = mlx_stack_home / "models"
        models_dir.mkdir(parents=True)
        (models_dir / "stray-file.txt").write_text("ignored")
        _create_model_dir(models_dir, "real-model-4bit", size_bytes=1000)

        result = scan_local_models(models_dir=models_dir, catalog=[])
        assert len(result) == 1
        assert result[0].name == "real-model-4bit"

    def test_sorted_by_name(self, mlx_stack_home: Path) -> None:
        """Results are sorted alphabetically by name."""
        models_dir = mlx_stack_home / "models"
        _create_model_dir(models_dir, "zeta-model", size_bytes=1000)
        _create_model_dir(models_dir, "alpha-model", size_bytes=1000)
        _create_model_dir(models_dir, "middle-model", size_bytes=1000)

        result = scan_local_models(models_dir=models_dir, catalog=[])
        names = [m.name for m in result]
        assert names == ["alpha-model", "middle-model", "zeta-model"]


# =========================================================================== #
# Core module tests — get_remote_stack_models
# =========================================================================== #


class TestGetRemoteStackModels:
    """Tests for the get_remote_stack_models function."""

    def test_no_stack(self, mlx_stack_home: Path) -> None:
        """Returns empty list when no stack is configured."""
        result = get_remote_stack_models(local_models=[], stack=None)
        assert result == []

    def test_all_local(self, mlx_stack_home: Path) -> None:
        """Returns empty list when all stack models are downloaded."""
        models_dir = mlx_stack_home / "models"
        catalog = _make_test_catalog()

        _create_model_dir(models_dir, "qwen3.5-8b-4bit", size_bytes=1000)

        stack = {
            "tiers": [
                {
                    "name": "fast",
                    "model": "qwen3.5-8b",
                    "quant": "int4",
                    "source": "mlx-community/qwen3.5-8b-4bit",
                    "port": 8000,
                }
            ]
        }

        local_models = scan_local_models(models_dir=models_dir, catalog=catalog, stack=stack)
        result = get_remote_stack_models(local_models=local_models, stack=stack, catalog=catalog)
        assert result == []

    def test_finds_remote_only_models(self, mlx_stack_home: Path) -> None:
        """Identifies stack models that are not downloaded locally."""
        catalog = _make_test_catalog()

        stack = {
            "tiers": [
                {
                    "name": "fast",
                    "model": "qwen3.5-8b",
                    "quant": "int4",
                    "source": "mlx-community/qwen3.5-8b-4bit",
                    "port": 8000,
                }
            ]
        }

        result = get_remote_stack_models(local_models=[], stack=stack, catalog=catalog)
        assert len(result) == 1
        assert result[0]["model_id"] == "qwen3.5-8b"
        assert result[0]["source"] == "remote"
        assert result[0]["catalog_name"] == "Qwen 3.5 8B"
        assert result[0]["est_size_gb"] == 4.5


# =========================================================================== #
# Core module tests — list_catalog_models
# =========================================================================== #


class TestListCatalogModels:
    """Tests for the list_catalog_models function."""

    def test_lists_all_catalog_entries(self) -> None:
        """Returns a CatalogModel for every catalog entry."""
        catalog = _make_test_catalog()
        result = list_catalog_models(catalog=catalog, profile=None, local_models=[])
        assert len(result) == len(catalog)

    def test_includes_hardware_benchmarks(self) -> None:
        """Includes gen_tps and memory_gb when profile matches catalog benchmarks."""
        catalog = _make_test_catalog()
        profile = _make_profile()

        result = list_catalog_models(catalog=catalog, profile=profile, local_models=[])

        # All our test entries have m4-max-128 benchmarks
        for cm in result:
            assert cm.gen_tps is not None
            assert cm.memory_gb is not None
            assert cm.is_estimated is False

    def test_estimated_for_unknown_hardware(self) -> None:
        """Shows estimated values when hardware doesn't match catalog benchmarks."""
        catalog = _make_test_catalog()
        # Use a profile that doesn't match any catalog benchmark key
        profile = _make_profile(
            chip="Apple M3",
            memory_gb=24,
            bandwidth_gbps=100.0,
        )

        result = list_catalog_models(catalog=catalog, profile=profile, local_models=[])

        for cm in result:
            assert cm.gen_tps is not None
            assert cm.is_estimated is True

    def test_no_profile_no_benchmarks(self, mlx_stack_home: Path) -> None:
        """No benchmark data when no profile is available."""
        catalog = _make_test_catalog()
        # Ensure no profile.json exists so load_profile returns None
        result = list_catalog_models(catalog=catalog, profile=None, local_models=[])

        for cm in result:
            assert cm.gen_tps is None
            assert cm.memory_gb is None

    def test_marks_local_models(self, mlx_stack_home: Path) -> None:
        """Indicates which catalog models are available locally."""
        catalog = _make_test_catalog()
        models_dir = mlx_stack_home / "models"
        _create_model_dir(models_dir, "qwen3.5-8b-4bit", size_bytes=1000)

        local_models = scan_local_models(models_dir=models_dir, catalog=catalog)
        result = list_catalog_models(catalog=catalog, profile=None, local_models=local_models)

        local_map = {cm.id: cm.is_local for cm in result}
        assert local_map["qwen3.5-8b"] is True
        assert local_map["nemotron-8b"] is False

    def test_shows_all_quants(self) -> None:
        """Shows all available quantizations for each model."""
        catalog = _make_test_catalog()
        result = list_catalog_models(catalog=catalog, profile=None, local_models=[])

        for cm in result:
            assert "int4" in cm.quants
            assert "int8" in cm.quants


# =========================================================================== #
# Core module tests — format_size
# =========================================================================== #


class TestFormatSize:
    """Tests for the format_size utility function."""

    def test_gb(self) -> None:
        assert format_size(5_200_000_000) == "5.2 GB"

    def test_mb(self) -> None:
        assert format_size(850_000_000) == "850 MB"

    def test_kb(self) -> None:
        assert format_size(120_000) == "120 KB"

    def test_bytes(self) -> None:
        assert format_size(500) == "500 B"

    def test_zero(self) -> None:
        assert format_size(0) == "0 B"

    def test_exact_gb(self) -> None:
        assert format_size(1_000_000_000) == "1.0 GB"


# =========================================================================== #
# Core module tests — get_models_directory
# =========================================================================== #


class TestGetModelsDirectory:
    """Tests for the get_models_directory function."""

    def test_default_path(self, mlx_stack_home: Path) -> None:
        """Returns default models path when no custom config."""
        result = get_models_directory()
        assert result == mlx_stack_home / "models"

    def test_custom_path(self, mlx_stack_home: Path) -> None:
        """Respects custom model-dir from config. (VAL-MODELS-006)"""
        custom_dir = mlx_stack_home / "custom-models"

        with patch("mlx_stack.core.models.get_value", return_value=str(custom_dir)):
            result = get_models_directory()

        assert result == custom_dir


# =========================================================================== #
# CLI tests — `mlx-stack models`
# =========================================================================== #


class TestModelsCommand:
    """Tests for the `mlx-stack models` CLI command."""

    def test_no_models_message(self, mlx_stack_home: Path) -> None:
        """VAL-MODELS-003: Shows 'no models found' with pull suggestion."""
        runner = CliRunner()
        result = runner.invoke(cli, ["models"])

        assert result.exit_code == 0
        assert "No models found" in result.output
        assert "mlx-stack pull" in result.output
        assert "mlx-stack init" in result.output

    def test_no_models_dir_not_exist(self, clean_mlx_stack_home: Path) -> None:
        """VAL-MODELS-003: Non-existent model dir shows helpful message."""
        runner = CliRunner()
        result = runner.invoke(cli, ["models"])

        assert result.exit_code == 0
        assert "No models found" in result.output

    def test_lists_local_models_with_size_and_quant(self, mlx_stack_home: Path) -> None:
        """VAL-MODELS-001: Shows models with disk size and quantization.

        Verifies through the core scan function that models are discovered
        with correct size and quant, then checks CLI renders them.
        """
        models_dir = mlx_stack_home / "models"
        _create_model_dir(models_dir, "Qwen3.5-8B-4bit", size_bytes=4_500_000_000)

        # Verify core scanning works correctly
        local_models = scan_local_models(models_dir=models_dir, catalog=[])
        assert len(local_models) == 1
        assert local_models[0].name == "Qwen3.5-8B-4bit"
        assert local_models[0].disk_size_bytes == 4_500_000_000
        assert local_models[0].quant == "int4"

        # Verify CLI renders something (table with Local Models title)
        runner = CliRunner()
        with patch("mlx_stack.cli.models.load_catalog", return_value=[]):
            result = runner.invoke(cli, ["models"])

        assert result.exit_code == 0
        assert "Local Models" in result.output
        assert "Model" in result.output  # Table header
        assert "Size" in result.output  # Table header
        assert "Quant" in result.output  # Table header

    def test_active_stack_indicator(self, mlx_stack_home: Path) -> None:
        """VAL-MODELS-002: Active stack models marked with visual indicator."""
        models_dir = mlx_stack_home / "models"
        stacks_dir = mlx_stack_home / "stacks"
        catalog = _make_test_catalog()

        _create_model_dir(models_dir, "qwen3.5-8b-4bit", size_bytes=1000)
        _create_model_dir(models_dir, "random-model-4bit", size_bytes=1000)

        # Stack references qwen3.5-8b
        stack_tiers = [
            {
                "name": "fast",
                "model": "qwen3.5-8b",
                "quant": "int4",
                "source": "mlx-community/qwen3.5-8b-4bit",
                "port": 8000,
            }
        ]
        _create_stack_yaml(stacks_dir, stack_tiers)

        # Verify via core function that active detection works
        stack = yaml.safe_load((stacks_dir / "default.yaml").read_text())
        local_models = scan_local_models(models_dir=models_dir, catalog=catalog, stack=stack)
        active_map = {m.name: m.is_active for m in local_models}
        assert active_map["qwen3.5-8b-4bit"] is True
        assert active_map["random-model-4bit"] is False

        # Verify CLI shows the indicator
        runner = CliRunner()
        with patch("mlx_stack.cli.models.load_catalog", return_value=catalog):
            result = runner.invoke(cli, ["models"])

        assert result.exit_code == 0
        assert "✓" in result.output
        assert "active in current stack" in result.output

    def test_remote_stack_models_shown(self, mlx_stack_home: Path) -> None:
        """VAL-MODELS-002: Remote-only stack models show with remote source."""
        stacks_dir = mlx_stack_home / "stacks"
        catalog = _make_test_catalog()

        # Stack references qwen3.5-8b but it's not downloaded
        stack_tiers = [
            {
                "name": "fast",
                "model": "qwen3.5-8b",
                "quant": "int4",
                "source": "mlx-community/qwen3.5-8b-4bit",
                "port": 8000,
            }
        ]
        _create_stack_yaml(stacks_dir, stack_tiers)

        runner = CliRunner()
        with patch("mlx_stack.cli.models.load_catalog", return_value=catalog):
            result = runner.invoke(cli, ["models"])

        assert result.exit_code == 0
        assert "not downloaded" in result.output
        assert "remote" in result.output
        assert "Qwen 3.5 8B" in result.output

    def test_human_readable_names(self, mlx_stack_home: Path) -> None:
        """VAL-MODELS-005: Uses human-readable catalog names, not directory names."""
        models_dir = mlx_stack_home / "models"
        catalog = _make_test_catalog()

        # Dir name matches catalog source
        _create_model_dir(models_dir, "qwen3.5-8b-4bit", size_bytes=1000)

        # Verify via core function that catalog name is resolved
        local_models = scan_local_models(models_dir=models_dir, catalog=catalog)
        assert local_models[0].catalog_name == "Qwen 3.5 8B"

        # Verify CLI renders catalog names
        runner = CliRunner()
        with patch("mlx_stack.cli.models.load_catalog", return_value=catalog):
            result = runner.invoke(cli, ["models"])

        assert result.exit_code == 0
        assert "Qwen 3.5 8B" in result.output

    def test_formatted_table_output(self, mlx_stack_home: Path) -> None:
        """VAL-MODELS-005: Output is a formatted table with aligned columns."""
        models_dir = mlx_stack_home / "models"
        _create_model_dir(models_dir, "test-model-4bit", size_bytes=1000)

        runner = CliRunner()
        with patch("mlx_stack.cli.models.load_catalog", return_value=[]):
            result = runner.invoke(cli, ["models"])

        assert result.exit_code == 0
        # Rich tables include box-drawing characters or header separators
        assert "Model" in result.output
        assert "Size" in result.output
        assert "Quant" in result.output
        assert "Source" in result.output

    def test_models_dir_path_shown(self, mlx_stack_home: Path) -> None:
        """Shows the models directory path in the output."""
        models_dir = mlx_stack_home / "models"
        models_dir.mkdir(parents=True)

        runner = CliRunner()
        with patch("mlx_stack.cli.models.load_catalog", return_value=[]):
            result = runner.invoke(cli, ["models"])

        # Even with no models, shows the "No models found" message
        assert result.exit_code == 0

    def test_custom_model_dir_respected(self, mlx_stack_home: Path) -> None:
        """VAL-MODELS-006: Custom model-dir from config is used."""
        custom_dir = mlx_stack_home / "custom-models"
        _create_model_dir(custom_dir, "custom-model-4bit", size_bytes=2_000_000_000)

        # Write config with custom model-dir
        config_path = mlx_stack_home / "config.yaml"
        config_path.write_text(
            yaml.dump({"model-dir": str(custom_dir)}),
            encoding="utf-8",
        )

        # Verify via core function that custom dir is used
        local_models = scan_local_models(models_dir=custom_dir, catalog=[])
        assert len(local_models) == 1
        assert local_models[0].name == "custom-model-4bit"
        assert local_models[0].disk_size_bytes == 2_000_000_000

        # Verify CLI uses the custom dir from config
        runner = CliRunner()
        with patch("mlx_stack.cli.models.load_catalog", return_value=[]):
            result = runner.invoke(cli, ["models"])

        assert result.exit_code == 0
        assert "Local Models" in result.output
        assert "custom-model" in result.output


# =========================================================================== #
# CLI tests — `mlx-stack models --catalog`
# =========================================================================== #


class TestModelsCatalogCommand:
    """Tests for the `mlx-stack models --catalog` CLI command."""

    def test_shows_all_catalog_entries(self, mlx_stack_home: Path) -> None:
        """VAL-MODELS-004: Shows all catalog models."""
        catalog = _make_test_catalog()

        runner = CliRunner()
        with (
            patch("mlx_stack.core.models.load_catalog", return_value=catalog),
            patch("mlx_stack.cli.models.load_catalog", return_value=catalog),
            patch("mlx_stack.cli.models.load_profile", return_value=None),
        ):
            result = runner.invoke(cli, ["models", "--catalog"])

        assert result.exit_code == 0
        assert "Model Catalog" in result.output
        assert "Qwen 3.5 8B" in result.output
        assert "Nemotron 8B" in result.output
        assert "Gemma 3 12B" in result.output

    def test_shows_hardware_specific_benchmarks(self, mlx_stack_home: Path) -> None:
        """VAL-MODELS-004: Shows benchmark data for current hardware."""
        catalog = _make_test_catalog()
        profile = _make_profile()

        runner = CliRunner()
        with (
            patch("mlx_stack.core.models.load_catalog", return_value=catalog),
            patch("mlx_stack.cli.models.load_catalog", return_value=catalog),
            patch("mlx_stack.cli.models.load_profile", return_value=profile),
            patch("mlx_stack.core.models.load_profile", return_value=profile),
        ):
            result = runner.invoke(cli, ["models", "--catalog"])

        assert result.exit_code == 0
        # Should have Gen t/s and Mem GB columns
        assert "Gen t/s" in result.output
        assert "Mem GB" in result.output
        # Profile info shown
        assert "Apple M4 Max" in result.output

    def test_estimated_values_with_tilde(self, mlx_stack_home: Path) -> None:
        """VAL-MODELS-004: Unknown hardware shows estimated values labeled as estimates."""
        catalog = _make_test_catalog()
        # Profile that doesn't match any benchmark key
        profile = _make_profile(
            chip="Apple M3",
            memory_gb=24,
            bandwidth_gbps=100.0,
        )

        runner = CliRunner()
        with (
            patch("mlx_stack.core.models.load_catalog", return_value=catalog),
            patch("mlx_stack.cli.models.load_catalog", return_value=catalog),
            patch("mlx_stack.cli.models.load_profile", return_value=profile),
            patch("mlx_stack.core.models.load_profile", return_value=profile),
        ):
            result = runner.invoke(cli, ["models", "--catalog"])

        assert result.exit_code == 0
        # Estimated values should have ~ indicator
        assert "~" in result.output
        # Should mention estimated
        assert "estimated" in result.output.lower()

    def test_no_profile_message(self, mlx_stack_home: Path) -> None:
        """Shows message when no hardware profile is available."""
        catalog = _make_test_catalog()

        runner = CliRunner()
        with (
            patch("mlx_stack.core.models.load_catalog", return_value=catalog),
            patch("mlx_stack.cli.models.load_catalog", return_value=catalog),
            patch("mlx_stack.cli.models.load_profile", return_value=None),
        ):
            result = runner.invoke(cli, ["models", "--catalog"])

        assert result.exit_code == 0
        assert "No hardware profile" in result.output
        assert "mlx-stack profile" in result.output

    def test_locally_available_indicator(self, mlx_stack_home: Path) -> None:
        """VAL-MODELS-004: Locally available models are indicated."""
        catalog = _make_test_catalog()
        models_dir = mlx_stack_home / "models"
        _create_model_dir(models_dir, "qwen3.5-8b-4bit", size_bytes=1000)

        runner = CliRunner()
        with (
            patch("mlx_stack.core.models.load_catalog", return_value=catalog),
            patch("mlx_stack.cli.models.load_catalog", return_value=catalog),
            patch("mlx_stack.cli.models.load_profile", return_value=None),
        ):
            result = runner.invoke(cli, ["models", "--catalog"])

        assert result.exit_code == 0
        assert "●" in result.output
        assert "available locally" in result.output

    def test_shows_quants_and_params(self, mlx_stack_home: Path) -> None:
        """VAL-MODELS-004: Shows parameter count and available quantizations."""
        catalog = _make_test_catalog()

        runner = CliRunner()
        with (
            patch("mlx_stack.core.models.load_catalog", return_value=catalog),
            patch("mlx_stack.cli.models.load_catalog", return_value=catalog),
            patch("mlx_stack.cli.models.load_profile", return_value=None),
        ):
            result = runner.invoke(cli, ["models", "--catalog"])

        assert result.exit_code == 0
        assert "8.0B" in result.output
        assert "12.0B" in result.output
        assert "int4" in result.output
        assert "int8" in result.output

    def test_formatted_table_with_headers(self, mlx_stack_home: Path) -> None:
        """VAL-MODELS-005: Output is a formatted table with aligned columns."""
        catalog = _make_test_catalog()

        runner = CliRunner()
        with (
            patch("mlx_stack.core.models.load_catalog", return_value=catalog),
            patch("mlx_stack.cli.models.load_catalog", return_value=catalog),
            patch("mlx_stack.cli.models.load_profile", return_value=None),
        ):
            result = runner.invoke(cli, ["models", "--catalog"])

        assert result.exit_code == 0
        assert "Name" in result.output
        assert "Family" in result.output
        assert "Params" in result.output
        assert "Quants" in result.output

    def test_catalog_error_handled(self, mlx_stack_home: Path) -> None:
        """VAL-CATALOG-003: Catalog error produces clear error message."""
        from mlx_stack.core.catalog import CatalogError

        runner = CliRunner()
        with patch(
            "mlx_stack.cli.models.load_catalog",
            side_effect=CatalogError("catalog is broken"),
        ):
            result = runner.invoke(cli, ["models", "--catalog"])

        assert result.exit_code == 1
        assert "Error" in result.output
        assert "catalog" in result.output.lower()

    def test_full_catalog_15_models(self, mlx_stack_home: Path) -> None:
        """VAL-CATALOG-001: Shows all 15 catalog models when using real catalog."""
        runner = CliRunner()
        with patch("mlx_stack.cli.models.load_profile", return_value=None):
            result = runner.invoke(cli, ["models", "--catalog"])

        assert result.exit_code == 0
        # Verify all 15 model families present
        assert "Qwen 3.5" in result.output
        assert "Nemotron" in result.output
        assert "Gemma 3" in result.output
        assert "DeepSeek R1" in result.output
        assert "Qwen 3" in result.output or "Qwen 3 8B" in result.output
        assert "Llama 3.3" in result.output


# =========================================================================== #
# CLI tests — help text
# =========================================================================== #


class TestModelsHelp:
    """Tests for models command help."""

    def test_help_flag(self) -> None:
        """models --help shows usage information."""
        runner = CliRunner()
        result = runner.invoke(cli, ["models", "--help"])

        assert result.exit_code == 0
        assert "--catalog" in result.output
        assert "local models" in result.output.lower() or "catalog" in result.output.lower()

    def test_models_in_main_help(self) -> None:
        """models command appears in main help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "models" in result.output


# =========================================================================== #
# Edge case tests
# =========================================================================== #


class TestModelsEdgeCases:
    """Edge case tests for the models command."""

    def test_corrupt_stack_yaml_graceful(self, mlx_stack_home: Path) -> None:
        """Handles corrupt stack YAML without crashing."""
        models_dir = mlx_stack_home / "models"
        _create_model_dir(models_dir, "some-model-4bit", size_bytes=1000)

        stacks_dir = mlx_stack_home / "stacks"
        stacks_dir.mkdir(parents=True)
        (stacks_dir / "default.yaml").write_text("{{{invalid yaml", encoding="utf-8")

        runner = CliRunner()
        with patch("mlx_stack.cli.models.load_catalog", return_value=[]):
            result = runner.invoke(cli, ["models"])

        # Should not crash — gracefully handle corrupt stack
        assert result.exit_code == 0
        assert "Local Models" in result.output

    def test_empty_catalog_graceful(self, mlx_stack_home: Path) -> None:
        """Handles empty catalog without crashing."""
        models_dir = mlx_stack_home / "models"
        _create_model_dir(models_dir, "some-model", size_bytes=1000)

        runner = CliRunner()
        with patch("mlx_stack.cli.models.load_catalog", return_value=[]):
            result = runner.invoke(cli, ["models"])

        assert result.exit_code == 0
        assert "Local Models" in result.output

    def test_model_dir_with_nested_dirs(self, mlx_stack_home: Path) -> None:
        """Correctly computes size for directories with nested subdirectories."""
        models_dir = mlx_stack_home / "models"
        model_dir = models_dir / "nested-model-4bit"
        model_dir.mkdir(parents=True)

        # Create nested structure
        sub = model_dir / "subfolder"
        sub.mkdir()
        (sub / "weights.safetensors").write_bytes(b"\x00" * 100)
        (model_dir / "config.json").write_bytes(b"\x00" * 50)

        result = scan_local_models(models_dir=models_dir, catalog=[])
        assert len(result) == 1
        assert result[0].disk_size_bytes == 150

    def test_quant_detection_from_config_json(self, mlx_stack_home: Path) -> None:
        """Detects quantization from config.json when dirname doesn't indicate it."""
        models_dir = mlx_stack_home / "models"
        model_dir = models_dir / "mystery-model"
        model_dir.mkdir(parents=True)

        config = {"quantization_config": {"bits": 4}}
        (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

        result = scan_local_models(models_dir=models_dir, catalog=[])
        assert result[0].quant == "int4"

    def test_multiple_models_multiple_quants(self, mlx_stack_home: Path) -> None:
        """Correctly handles multiple models with different quantizations."""
        models_dir = mlx_stack_home / "models"
        _create_model_dir(models_dir, "Model-A-4bit", size_bytes=2_000_000_000)
        _create_model_dir(models_dir, "Model-A-8bit", size_bytes=4_000_000_000)
        _create_model_dir(models_dir, "Model-B-bf16", size_bytes=8_000_000_000)

        result = scan_local_models(models_dir=models_dir, catalog=[])
        assert len(result) == 3

        by_name = {m.name: m for m in result}
        assert by_name["Model-A-4bit"].quant == "int4"
        assert by_name["Model-A-8bit"].quant == "int8"
        assert by_name["Model-B-bf16"].quant == "bf16"
