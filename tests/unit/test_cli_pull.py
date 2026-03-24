"""Tests for the `mlx-stack pull` CLI command and core pull module.

Validates:
- VAL-PULL-001: Downloads model to correct location
- VAL-PULL-002: Source resolution — mlx-community preferred, convert fallback
- VAL-PULL-003: Quant selection from config or --quant flag
- VAL-PULL-004: Disk space check before download
- VAL-PULL-005: Progress indicator during download
- VAL-PULL-006: Already-downloaded model detected
- VAL-PULL-007: Updates models inventory
- VAL-PULL-008: Network error handling
- VAL-PULL-009: Invalid model ID error
- VAL-PULL-010: Conversion failure produces clear error and cleans up
- VAL-PULL-011: Automatic retry on transient download failure
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from mlx_stack.cli.main import cli
from mlx_stack.core.catalog import (
    BenchmarkResult,
    Capabilities,
    CatalogEntry,
    CatalogError,
    QualityScores,
    QuantSource,
)
from mlx_stack.core.pull import (
    ConversionError,
    DiskSpaceError,
    DownloadError,
    InvalidModelError,
    ModelInventoryEntry,
    PullError,
    add_to_inventory,
    check_disk_space,
    find_in_inventory,
    get_model_local_path,
    is_model_downloaded,
    load_inventory,
    pull_model,
    resolve_source,
    save_inventory,
    validate_quant,
)

# --------------------------------------------------------------------------- #
# Fixtures — reusable test data
# --------------------------------------------------------------------------- #


def _make_entry(
    model_id: str = "qwen3.5-8b",
    name: str = "Qwen 3.5 8B",
    family: str = "Qwen 3.5",
    params_b: float = 8.0,
    architecture: str = "transformer",
    tool_calling: bool = True,
    disk_size_gb: float = 4.5,
    disk_size_gb_int8: float = 8.5,
    disk_size_gb_bf16: float = 16.0,
) -> CatalogEntry:
    """Create a CatalogEntry for testing."""
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
                disk_size_gb=disk_size_gb_int8,
            ),
            "bf16": QuantSource(
                hf_repo=f"Qwen/{name.replace(' ', '-')}",
                disk_size_gb=disk_size_gb_bf16,
                convert_from=True,
            ),
        },
        capabilities=Capabilities(
            tool_calling=tool_calling,
            tool_call_parser="hermes" if tool_calling else None,
            thinking=False,
            reasoning_parser=None,
            vision=False,
        ),
        quality=QualityScores(overall=68, coding=65, reasoning=62, instruction_following=72),
        benchmarks={
            "m4-max-128": BenchmarkResult(prompt_tps=140.0, gen_tps=77.0, memory_gb=5.5),
        },
        tags=["balanced", "agent-ready"],
    )


# =========================================================================== #
# Core module tests — pull.py
# =========================================================================== #


class TestValidateQuant:
    """Tests for quant validation."""

    def test_valid_int4(self) -> None:
        assert validate_quant("int4") == "int4"

    def test_valid_int8(self) -> None:
        assert validate_quant("int8") == "int8"

    def test_valid_bf16(self) -> None:
        assert validate_quant("bf16") == "bf16"

    def test_invalid_quant_raises(self) -> None:
        import pytest

        with pytest.raises(PullError, match="Invalid quantization 'int6'"):
            validate_quant("int6")

    def test_invalid_quant_fp32(self) -> None:
        import pytest

        with pytest.raises(PullError, match="Invalid quantization 'fp32'"):
            validate_quant("fp32")


class TestResolveSource:
    """Tests for source resolution — VAL-PULL-002."""

    def test_mlx_community_preferred_int4(self) -> None:
        """mlx-community source is used for int4 (convert_from=False)."""
        entry = _make_entry()
        source, source_type = resolve_source(entry, "int4")
        assert source_type == "mlx-community"
        assert "mlx-community" in source.hf_repo
        assert source.convert_from is False

    def test_mlx_community_preferred_int8(self) -> None:
        """mlx-community source is used for int8."""
        entry = _make_entry()
        source, source_type = resolve_source(entry, "int8")
        assert source_type == "mlx-community"
        assert "mlx-community" in source.hf_repo

    def test_conversion_fallback_bf16(self) -> None:
        """bf16 with convert_from=True uses conversion source type."""
        entry = _make_entry()
        source, source_type = resolve_source(entry, "bf16")
        assert source_type == "converted"
        assert source.convert_from is True
        assert "Qwen/" in source.hf_repo

    def test_unavailable_quant_raises(self) -> None:
        """Requesting unavailable quant raises PullError."""
        import pytest

        # Create entry with only int4 source
        entry = CatalogEntry(
            id="test-model",
            name="Test Model",
            family="Test",
            params_b=1.0,
            architecture="transformer",
            min_mlx_lm_version="0.22.0",
            sources={
                "int4": QuantSource(hf_repo="mlx-community/test-4bit", disk_size_gb=1.0),
            },
            capabilities=Capabilities(
                tool_calling=False,
                tool_call_parser=None,
                thinking=False,
                reasoning_parser=None,
                vision=False,
            ),
            quality=QualityScores(overall=50, coding=50, reasoning=50, instruction_following=50),
            benchmarks={},
            tags=[],
        )
        with pytest.raises(PullError, match="Quantization 'int8' is not available"):
            resolve_source(entry, "int8")


class TestDiskSpaceCheck:
    """Tests for disk space checking — VAL-PULL-004."""

    def test_sufficient_space(self, tmp_path: Path) -> None:
        """Returns True when there is enough space."""
        has_space, available = check_disk_space(tmp_path, 0.001)
        assert has_space is True
        assert available > 0

    def test_insufficient_space(self, tmp_path: Path) -> None:
        """Returns False when required exceeds available (simulated)."""
        # Request absurdly large amount
        has_space, available = check_disk_space(tmp_path, 999999999)
        assert has_space is False
        assert available > 0

    def test_creates_directory(self, tmp_path: Path) -> None:
        """Creates the models dir if it doesn't exist."""
        new_dir = tmp_path / "nonexistent" / "models"
        check_disk_space(new_dir, 0.001)
        assert new_dir.exists()


class TestModelLocalPath:
    """Tests for local path determination — VAL-PULL-001."""

    def test_mlx_community_repo(self, tmp_path: Path) -> None:
        """Extracts repo name from mlx-community/Model-4bit."""
        path = get_model_local_path(tmp_path, "mlx-community/Qwen3.5-8B-4bit")
        assert path == tmp_path / "Qwen3.5-8B-4bit"

    def test_base_repo(self, tmp_path: Path) -> None:
        """Extracts repo name from Qwen/Qwen3.5-8B."""
        path = get_model_local_path(tmp_path, "Qwen/Qwen3.5-8B")
        assert path == tmp_path / "Qwen3.5-8B"

    def test_no_org_prefix(self, tmp_path: Path) -> None:
        """Handles repo name without org prefix."""
        path = get_model_local_path(tmp_path, "SomeModel")
        assert path == tmp_path / "SomeModel"


class TestIsModelDownloaded:
    """Tests for download detection — VAL-PULL-006."""

    def test_empty_dir_not_downloaded(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        assert is_model_downloaded(model_dir) is False

    def test_nonexistent_dir_not_downloaded(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "nonexistent"
        assert is_model_downloaded(model_dir) is False

    def test_dir_with_files_is_downloaded(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        assert is_model_downloaded(model_dir) is True


class TestInventory:
    """Tests for inventory management — VAL-PULL-007."""

    def test_load_empty_inventory(self, mlx_stack_home: Path) -> None:
        """Returns empty list when no models.json exists."""
        entries = load_inventory()
        assert entries == []

    def test_save_and_load_inventory(self, mlx_stack_home: Path) -> None:
        """Round-trip save + load preserves entries."""
        entries = [
            {"model_id": "test-model", "quant": "int4", "name": "Test"},
        ]
        save_inventory(entries)

        loaded = load_inventory()
        assert len(loaded) == 1
        assert loaded[0]["model_id"] == "test-model"

    def test_add_to_inventory(self, mlx_stack_home: Path) -> None:
        """add_to_inventory appends a new entry."""
        entry = ModelInventoryEntry(
            model_id="qwen3.5-8b",
            name="Qwen 3.5 8B",
            quant="int4",
            source_type="mlx-community",
            hf_repo="mlx-community/qwen3.5-8b-4bit",
            local_path="/tmp/models/qwen3.5-8b-4bit",
            disk_size_gb=4.5,
            downloaded_at="2025-01-01T00:00:00+00:00",
        )
        add_to_inventory(entry)

        loaded = load_inventory()
        assert len(loaded) == 1
        assert loaded[0]["model_id"] == "qwen3.5-8b"
        assert loaded[0]["quant"] == "int4"

    def test_add_replaces_same_model_quant(self, mlx_stack_home: Path) -> None:
        """Adding same model_id+quant replaces the existing entry."""
        entry1 = ModelInventoryEntry(
            model_id="qwen3.5-8b",
            name="Qwen 3.5 8B",
            quant="int4",
            source_type="mlx-community",
            hf_repo="mlx-community/qwen3.5-8b-4bit",
            local_path="/tmp/models/old",
            disk_size_gb=4.5,
            downloaded_at="2025-01-01T00:00:00+00:00",
        )
        add_to_inventory(entry1)

        entry2 = ModelInventoryEntry(
            model_id="qwen3.5-8b",
            name="Qwen 3.5 8B",
            quant="int4",
            source_type="mlx-community",
            hf_repo="mlx-community/qwen3.5-8b-4bit",
            local_path="/tmp/models/new",
            disk_size_gb=4.5,
            downloaded_at="2025-01-02T00:00:00+00:00",
        )
        add_to_inventory(entry2)

        loaded = load_inventory()
        assert len(loaded) == 1
        assert loaded[0]["local_path"] == "/tmp/models/new"

    def test_add_different_quant_kept(self, mlx_stack_home: Path) -> None:
        """Adding same model with different quant keeps both entries."""
        entry1 = ModelInventoryEntry(
            model_id="qwen3.5-8b",
            name="Qwen 3.5 8B",
            quant="int4",
            source_type="mlx-community",
            hf_repo="mlx-community/qwen3.5-8b-4bit",
            local_path="/tmp/models/int4",
            disk_size_gb=4.5,
            downloaded_at="2025-01-01T00:00:00+00:00",
        )
        add_to_inventory(entry1)

        entry2 = ModelInventoryEntry(
            model_id="qwen3.5-8b",
            name="Qwen 3.5 8B",
            quant="int8",
            source_type="mlx-community",
            hf_repo="mlx-community/qwen3.5-8b-8bit",
            local_path="/tmp/models/int8",
            disk_size_gb=8.5,
            downloaded_at="2025-01-01T00:00:00+00:00",
        )
        add_to_inventory(entry2)

        loaded = load_inventory()
        assert len(loaded) == 2

    def test_find_in_inventory_found(self, mlx_stack_home: Path) -> None:
        """find_in_inventory returns the entry when present."""
        entry = ModelInventoryEntry(
            model_id="qwen3.5-8b",
            name="Qwen 3.5 8B",
            quant="int4",
            source_type="mlx-community",
            hf_repo="mlx-community/qwen3.5-8b-4bit",
            local_path="/tmp/models/qwen3.5-8b-4bit",
            disk_size_gb=4.5,
            downloaded_at="2025-01-01T00:00:00+00:00",
        )
        add_to_inventory(entry)

        found = find_in_inventory("qwen3.5-8b", "int4")
        assert found is not None
        assert found["model_id"] == "qwen3.5-8b"

    def test_find_in_inventory_not_found(self, mlx_stack_home: Path) -> None:
        """find_in_inventory returns None when not present."""
        found = find_in_inventory("nonexistent", "int4")
        assert found is None

    def test_corrupt_inventory_returns_empty(self, mlx_stack_home: Path) -> None:
        """Corrupt models.json returns empty list."""
        inv_path = mlx_stack_home / "models.json"
        inv_path.write_text("{{{invalid json", encoding="utf-8")
        entries = load_inventory()
        assert entries == []

    def test_inventory_file_is_valid_json(self, mlx_stack_home: Path) -> None:
        """Saved inventory is valid JSON."""
        entries = [{"model_id": "test", "quant": "int4"}]
        save_inventory(entries)

        inv_path = mlx_stack_home / "models.json"
        content = inv_path.read_text(encoding="utf-8")
        parsed = json.loads(content)
        assert isinstance(parsed, list)


# =========================================================================== #
# Core pull_model orchestrator tests
# =========================================================================== #


class TestPullModel:
    """Tests for the main pull_model function."""

    @patch("mlx_stack.core.pull.download_model")
    @patch("mlx_stack.core.pull.check_disk_space", return_value=(True, 100.0))
    def test_successful_download(
        self,
        mock_space: MagicMock,
        mock_download: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-PULL-001: Successful pull downloads to correct location."""
        catalog = [_make_entry()]
        result = pull_model("qwen3.5-8b", quant="int4", catalog=catalog)

        assert result.model_id == "qwen3.5-8b"
        assert result.name == "Qwen 3.5 8B"
        assert result.quant == "int4"
        assert result.source_type == "mlx-community"
        assert result.already_existed is False
        mock_download.assert_called_once()

    @patch("mlx_stack.core.pull.download_model")
    @patch("mlx_stack.core.pull.check_disk_space", return_value=(True, 100.0))
    def test_updates_inventory_after_download(
        self,
        mock_space: MagicMock,
        mock_download: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-PULL-007: Model appears in inventory after pull."""
        catalog = [_make_entry()]
        pull_model("qwen3.5-8b", quant="int4", catalog=catalog)

        inv = load_inventory()
        assert len(inv) == 1
        assert inv[0]["model_id"] == "qwen3.5-8b"
        assert inv[0]["quant"] == "int4"
        assert inv[0]["source_type"] == "mlx-community"
        assert "downloaded_at" in inv[0]

    def test_invalid_model_id_raises(self, mlx_stack_home: Path) -> None:
        """VAL-PULL-009: Invalid model ID raises InvalidModelError."""
        import pytest

        catalog = [_make_entry()]
        with pytest.raises(InvalidModelError, match="not found in catalog"):
            pull_model("nonexistent-model", catalog=catalog)

    def test_invalid_model_suggests_catalog(self, mlx_stack_home: Path) -> None:
        """VAL-PULL-009: Error message suggests models --catalog."""
        import pytest

        catalog = [_make_entry()]
        with pytest.raises(InvalidModelError, match="models --catalog"):
            pull_model("bad-model", catalog=catalog)

    def test_invalid_quant_rejected(self, mlx_stack_home: Path) -> None:
        """VAL-PULL-003: Invalid quant is rejected."""
        import pytest

        catalog = [_make_entry()]
        with pytest.raises(PullError, match="Invalid quantization"):
            pull_model("qwen3.5-8b", quant="int6", catalog=catalog)

    @patch("mlx_stack.core.pull.check_disk_space", return_value=(False, 2.0))
    def test_insufficient_disk_space(
        self,
        mock_space: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-PULL-004: Insufficient space raises DiskSpaceError."""
        import pytest

        catalog = [_make_entry()]
        with pytest.raises(DiskSpaceError, match="Insufficient disk space"):
            pull_model("qwen3.5-8b", quant="int4", catalog=catalog)

    @patch("mlx_stack.core.pull.check_disk_space", return_value=(False, 2.0))
    def test_disk_space_error_shows_requirements(
        self,
        mock_space: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-PULL-004: Disk space error shows required vs available."""
        import pytest

        catalog = [_make_entry()]
        with pytest.raises(DiskSpaceError, match=r"(?s)Required: 4\.5 GB.*Available: 2\.0 GB"):
            pull_model("qwen3.5-8b", quant="int4", catalog=catalog)

    @patch("mlx_stack.core.pull.download_model")
    @patch("mlx_stack.core.pull.check_disk_space", return_value=(True, 100.0))
    def test_already_downloaded_detected(
        self,
        mock_space: MagicMock,
        mock_download: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-PULL-006: Already-downloaded model skips re-download."""
        catalog = [_make_entry()]

        # Create the model directory with a file to simulate existing download
        models_dir = mlx_stack_home / "models"
        model_path = models_dir / "qwen3.5-8b-4bit"
        model_path.mkdir(parents=True)
        (model_path / "config.json").write_text("{}")

        result = pull_model("qwen3.5-8b", quant="int4", catalog=catalog)

        assert result.already_existed is True
        mock_download.assert_not_called()

    @patch("mlx_stack.core.pull.download_model")
    @patch("mlx_stack.core.pull.check_disk_space", return_value=(True, 100.0))
    def test_force_redownloads(
        self,
        mock_space: MagicMock,
        mock_download: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-PULL-006: --force re-downloads existing model."""
        catalog = [_make_entry()]

        # Create the model directory with a file
        models_dir = mlx_stack_home / "models"
        model_path = models_dir / "qwen3.5-8b-4bit"
        model_path.mkdir(parents=True)
        (model_path / "config.json").write_text("{}")

        result = pull_model("qwen3.5-8b", quant="int4", force=True, catalog=catalog)

        assert result.already_existed is False
        mock_download.assert_called_once()

    @patch("mlx_stack.core.pull.download_model")
    @patch("mlx_stack.core.pull.check_disk_space", return_value=(True, 100.0))
    def test_uses_config_default_quant(
        self,
        mock_space: MagicMock,
        mock_download: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-PULL-003: Without --quant, uses config default (int4)."""
        catalog = [_make_entry()]
        result = pull_model("qwen3.5-8b", quant=None, catalog=catalog)
        assert result.quant == "int4"

    @patch("mlx_stack.core.pull.get_value", return_value="int8")
    @patch("mlx_stack.core.pull.download_model")
    @patch("mlx_stack.core.pull.check_disk_space", return_value=(True, 100.0))
    def test_uses_config_custom_quant(
        self,
        mock_space: MagicMock,
        mock_download: MagicMock,
        mock_get_value: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-PULL-003: Uses config default-quant when set to int8."""
        catalog = [_make_entry()]
        result = pull_model("qwen3.5-8b", quant=None, catalog=catalog)
        assert result.quant == "int8"

    @patch("mlx_stack.core.pull.convert_model")
    @patch("mlx_stack.core.pull.check_disk_space", return_value=(True, 100.0))
    def test_bf16_uses_conversion(
        self,
        mock_space: MagicMock,
        mock_convert: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-PULL-002: bf16 with convert_from uses mlx_lm conversion."""
        catalog = [_make_entry()]
        result = pull_model("qwen3.5-8b", quant="bf16", catalog=catalog)

        assert result.source_type == "converted"
        mock_convert.assert_called_once()

    @patch("mlx_stack.core.pull.download_model", side_effect=DownloadError("Network error"))
    @patch("mlx_stack.core.pull.check_disk_space", return_value=(True, 100.0))
    def test_download_error_propagated(
        self,
        mock_space: MagicMock,
        mock_download: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-PULL-008: Download error is propagated."""
        import pytest

        catalog = [_make_entry()]
        with pytest.raises(DownloadError, match="Network error"):
            pull_model("qwen3.5-8b", quant="int4", catalog=catalog)

    @patch("mlx_stack.core.pull.convert_model", side_effect=ConversionError("Convert failed"))
    @patch("mlx_stack.core.pull.check_disk_space", return_value=(True, 100.0))
    def test_conversion_error_propagated(
        self,
        mock_space: MagicMock,
        mock_convert: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-PULL-010: Conversion error is propagated."""
        import pytest

        catalog = [_make_entry()]
        with pytest.raises(ConversionError, match="Convert failed"):
            pull_model("qwen3.5-8b", quant="bf16", catalog=catalog)


# =========================================================================== #
# Download and retry tests
# =========================================================================== #


class TestDownloadModel:
    """Tests for download_model with retry — VAL-PULL-008, VAL-PULL-011."""

    @patch("mlx_stack.core.pull._run_download")
    def test_successful_download(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Successful download on first attempt."""
        from rich.console import Console

        from mlx_stack.core.pull import download_model

        local_dir = tmp_path / "model"
        console = Console()
        download_model("mlx-community/test-4bit", local_dir, console)
        mock_run.assert_called_once()

    @patch("mlx_stack.core.pull._run_download")
    def test_retry_on_first_failure(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """VAL-PULL-011: Automatic retry on first transient failure."""
        from rich.console import Console

        from mlx_stack.core.pull import download_model

        # First call fails, second succeeds
        mock_run.side_effect = [DownloadError("Connection reset"), None]

        local_dir = tmp_path / "model"
        console = Console()
        download_model("mlx-community/test-4bit", local_dir, console)
        assert mock_run.call_count == 2

    @patch("mlx_stack.core.pull._run_download")
    def test_both_retries_fail(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """VAL-PULL-011: Both attempts fail shows network/auth suggestion."""
        import pytest
        from rich.console import Console

        from mlx_stack.core.pull import download_model

        mock_run.side_effect = DownloadError("Connection failed")

        local_dir = tmp_path / "model"
        console = Console()
        with pytest.raises(DownloadError, match="network connection"):
            download_model("mlx-community/test-4bit", local_dir, console)
        assert mock_run.call_count == 2

    @patch("mlx_stack.core.pull._run_download")
    def test_both_retries_fail_suggests_hf_token(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """VAL-PULL-011: Failure suggests HF_TOKEN."""
        import pytest
        from rich.console import Console

        from mlx_stack.core.pull import download_model

        mock_run.side_effect = DownloadError("Connection error")

        local_dir = tmp_path / "model"
        console = Console()
        with pytest.raises(DownloadError, match="HF_TOKEN"):
            download_model("test/model", local_dir, console)

    @patch("mlx_stack.core.pull._run_download")
    def test_cleanup_on_total_failure(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """VAL-PULL-008: Partial downloads cleaned up on total failure."""
        import pytest
        from rich.console import Console

        from mlx_stack.core.pull import download_model

        local_dir = tmp_path / "model"

        # Simulate partial content created during download
        def create_partial(*args: object, **kwargs: object) -> None:
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "partial.bin").write_text("partial data")
            raise DownloadError("Network error")

        mock_run.side_effect = create_partial

        console = Console()
        with pytest.raises(DownloadError):
            download_model("test/model", local_dir, console)

        # Partial download should be cleaned up
        assert not local_dir.exists()


# =========================================================================== #
# Traceback filtering tests
# =========================================================================== #


class TestFilterTraceback:
    """Tests for _filter_traceback — VAL-PULL-008."""

    def test_no_traceback_passes_through(self) -> None:
        """Output without traceback is returned as-is."""
        from mlx_stack.core.pull import _filter_traceback

        output = "Error: Repository not found"
        assert _filter_traceback(output) == "Error: Repository not found"

    def test_filters_full_traceback(self) -> None:
        """Full Python traceback is filtered to just the error message."""
        from mlx_stack.core.pull import _filter_traceback

        output = (
            "Traceback (most recent call last):\n"
            '  File "/usr/lib/python3.13/site-packages/requests/adapters.py", line 667\n'
            "    resp = conn.urlopen(...)\n"
            '  File "/usr/lib/python3.13/urllib3/connectionpool.py", line 843\n'
            "    retries = retries.increment(...)\n"
            "requests.exceptions.ConnectionError: Connection refused\n"
        )
        result = _filter_traceback(output)
        assert "Traceback" not in result
        assert "File " not in result
        assert "Connection refused" in result

    def test_empty_string(self) -> None:
        """Empty string returns empty string."""
        from mlx_stack.core.pull import _filter_traceback

        assert _filter_traceback("") == ""

    def test_multiline_without_traceback(self) -> None:
        """Multiple lines without traceback are preserved."""
        from mlx_stack.core.pull import _filter_traceback

        output = "Downloading file 1...\nDownloading file 2...\nError: timeout"
        result = _filter_traceback(output)
        assert "Downloading file 1..." in result
        assert "Error: timeout" in result

    def test_traceback_with_preamble(self) -> None:
        """Output with text before traceback preserves pre-traceback content."""
        from mlx_stack.core.pull import _filter_traceback

        output = (
            "Downloading model.safetensors\n"
            "Traceback (most recent call last):\n"
            '  File "/usr/lib/python3.13/site-packages/something.py", line 10\n'
            "    do_stuff()\n"
            "OSError: Network error\n"
        )
        result = _filter_traceback(output)
        assert "Downloading model.safetensors" in result
        assert "Network error" in result
        assert "Traceback" not in result


# =========================================================================== #
# Download progress visibility tests
# =========================================================================== #


class TestDownloadProgress:
    """Tests for real-time download progress display."""

    @patch("mlx_stack.core.pull.subprocess.Popen")
    def test_run_download_streams_output(
        self,
        mock_popen: MagicMock,
        tmp_path: Path,
    ) -> None:
        """_run_download streams stdout (merged with stderr) to console."""
        from io import StringIO

        from rich.console import Console

        from mlx_stack.core.pull import _run_download

        # Mock Popen to simulate line-by-line output (stderr merged via STDOUT)
        mock_proc = MagicMock()
        mock_proc.stdout = StringIO("Downloading model.safetensors: 50%\nDownloading: 100%\n")
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        local_dir = tmp_path / "model"
        local_dir.mkdir()
        output = StringIO()
        console = Console(file=output, no_color=True)
        _run_download("test/repo", local_dir, console)

        printed = output.getvalue()
        assert "Downloading" in printed

    @patch("mlx_stack.core.pull.subprocess.Popen")
    def test_run_download_uses_stderr_stdout_for_progress(
        self,
        mock_popen: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Verifies Popen uses stderr=STDOUT to merge tqdm progress bars."""
        import subprocess as sp
        from io import StringIO

        from rich.console import Console

        from mlx_stack.core.pull import _run_download

        mock_proc = MagicMock()
        mock_proc.stdout = StringIO("")
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        local_dir = tmp_path / "model"
        local_dir.mkdir()
        console = Console(file=StringIO())
        _run_download("test/repo", local_dir, console)

        mock_popen.assert_called_once()
        call_kwargs = mock_popen.call_args
        # Verify stderr=subprocess.STDOUT to merge progress bars into stdout
        assert call_kwargs[1].get("stderr") == sp.STDOUT
        # Verify stdout is PIPE for streaming
        assert call_kwargs[1].get("stdout") == sp.PIPE

    @patch("mlx_stack.core.pull.subprocess.Popen")
    def test_run_download_failure_reports_captured_output(
        self,
        mock_popen: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Failed download reports captured output in error."""
        from io import StringIO

        import pytest
        from rich.console import Console

        from mlx_stack.core.pull import _run_download

        # With stderr=STDOUT, error messages come through stdout
        mock_proc = MagicMock()
        mock_proc.stdout = StringIO("Error: Repository not found\n")
        mock_proc.returncode = 1
        mock_proc.wait.return_value = 1
        mock_popen.return_value = mock_proc

        local_dir = tmp_path / "model"
        local_dir.mkdir()
        console = Console(file=StringIO())

        with pytest.raises(DownloadError, match="Repository not found"):
            _run_download("test/repo", local_dir, console)

    @patch("mlx_stack.core.pull.subprocess.Popen")
    def test_run_download_filters_traceback_on_failure(
        self,
        mock_popen: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Failed download filters Python traceback, shows clean error."""
        from io import StringIO

        import pytest
        from rich.console import Console

        from mlx_stack.core.pull import _run_download

        # Simulate output with full Python traceback
        traceback_output = (
            "Traceback (most recent call last):\n"
            '  File "/usr/lib/python3.13/site-packages/huggingface_hub/utils.py", line 100\n'
            "    response.raise_for_status()\n"
            "requests.exceptions.ConnectionError: Connection refused\n"
        )
        mock_proc = MagicMock()
        mock_proc.stdout = StringIO(traceback_output)
        mock_proc.returncode = 1
        mock_proc.wait.return_value = 1
        mock_popen.return_value = mock_proc

        local_dir = tmp_path / "model"
        local_dir.mkdir()
        console = Console(file=StringIO())

        with pytest.raises(DownloadError) as exc_info:
            _run_download("test/repo", local_dir, console)

        error_msg = str(exc_info.value)
        # Should contain the clean error, not the traceback
        assert "Connection refused" in error_msg
        assert "Traceback (most recent call last)" not in error_msg
        assert '  File "' not in error_msg


# =========================================================================== #
# Conversion tests
# =========================================================================== #


class TestConvertModel:
    """Tests for convert_model — VAL-PULL-010."""

    @patch("mlx_stack.core.pull.subprocess.run")
    def test_successful_conversion(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Successful conversion creates model directory."""
        from rich.console import Console

        from mlx_stack.core.pull import convert_model

        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="Done")

        local_dir = tmp_path / "model"
        console = Console()
        convert_model("Qwen/Qwen3.5-8B", local_dir, "int4", console)
        mock_run.assert_called_once()

        # Check the command used correct quant args
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "-q" in cmd
        assert "--q-bits" in cmd
        assert "4" in cmd

    @patch("mlx_stack.core.pull.subprocess.run")
    def test_conversion_failure_cleans_up(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """VAL-PULL-010: Conversion failure cleans up partial artifacts."""
        import pytest
        from rich.console import Console

        from mlx_stack.core.pull import convert_model

        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="Error: failed to load model",
            stdout="",
        )

        local_dir = tmp_path / "model"
        local_dir.mkdir()
        (local_dir / "partial.bin").write_text("partial")

        console = Console()
        with pytest.raises(ConversionError, match="Conversion failed"):
            convert_model("Qwen/Qwen3.5-8B", local_dir, "int4", console)

        # Partial artifacts should be cleaned up
        assert not local_dir.exists()

    @patch("mlx_stack.core.pull.subprocess.run")
    def test_conversion_failure_includes_mlx_lm_output(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """VAL-PULL-010: Conversion error includes mlx_lm error output."""
        import pytest
        from rich.console import Console

        from mlx_stack.core.pull import convert_model

        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="mlx_lm error: Unsupported architecture",
            stdout="",
        )

        local_dir = tmp_path / "model"
        console = Console()
        with pytest.raises(ConversionError, match="Unsupported architecture"):
            convert_model("Qwen/Qwen3.5-8B", local_dir, "int4", console)

    @patch("mlx_stack.core.pull.subprocess.run")
    def test_conversion_int8(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Conversion with int8 uses correct q-bits."""
        from rich.console import Console

        from mlx_stack.core.pull import convert_model

        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="Done")

        local_dir = tmp_path / "model"
        console = Console()
        convert_model("Qwen/Qwen3.5-8B", local_dir, "int8", console)

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "8" in cmd


# =========================================================================== #
# CLI command tests
# =========================================================================== #


class TestPullCLI:
    """Tests for the `mlx-stack pull` CLI command."""

    def test_pull_help(self) -> None:
        """pull --help displays usage information."""
        runner = CliRunner()
        result = runner.invoke(cli, ["pull", "--help"])
        assert result.exit_code == 0
        assert "Download a model" in result.output
        assert "--quant" in result.output
        assert "--bench" in result.output
        assert "--force" in result.output

    @patch("mlx_stack.core.pull.download_model")
    @patch("mlx_stack.core.pull.check_disk_space", return_value=(True, 100.0))
    @patch("mlx_stack.core.pull.load_catalog")
    def test_pull_successful(
        self,
        mock_catalog: MagicMock,
        mock_space: MagicMock,
        mock_download: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-PULL-001: Successful pull via CLI."""
        mock_catalog.return_value = [_make_entry()]

        runner = CliRunner()
        result = runner.invoke(cli, ["pull", "qwen3.5-8b", "--quant", "int4"])
        assert result.exit_code == 0
        assert "Qwen 3.5 8B" in result.output
        assert "is ready" in result.output

    @patch("mlx_stack.core.pull.load_catalog")
    def test_pull_invalid_model(
        self,
        mock_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-PULL-009: Invalid model exits with error."""
        mock_catalog.return_value = [_make_entry()]

        runner = CliRunner()
        result = runner.invoke(cli, ["pull", "nonexistent-model"])
        assert result.exit_code == 1
        assert "not found in catalog" in result.output
        assert "models --catalog" in result.output

    @patch("mlx_stack.core.pull.load_catalog")
    def test_pull_invalid_quant(
        self,
        mock_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-PULL-003: Invalid quant exits with error."""
        mock_catalog.return_value = [_make_entry()]

        runner = CliRunner()
        result = runner.invoke(cli, ["pull", "qwen3.5-8b", "--quant", "int6"])
        assert result.exit_code == 1
        assert "Invalid quantization" in result.output

    @patch("mlx_stack.core.pull.check_disk_space", return_value=(False, 2.0))
    @patch("mlx_stack.core.pull.load_catalog")
    def test_pull_insufficient_space(
        self,
        mock_catalog: MagicMock,
        mock_space: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-PULL-004: Insufficient space exits with clear error."""
        mock_catalog.return_value = [_make_entry()]

        runner = CliRunner()
        result = runner.invoke(cli, ["pull", "qwen3.5-8b"])
        assert result.exit_code == 1
        assert "Insufficient disk space" in result.output

    @patch("mlx_stack.core.pull.download_model")
    @patch("mlx_stack.core.pull.check_disk_space", return_value=(True, 100.0))
    @patch("mlx_stack.core.pull.load_catalog")
    def test_pull_already_exists(
        self,
        mock_catalog: MagicMock,
        mock_space: MagicMock,
        mock_download: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-PULL-006: Already-downloaded model reports exists."""
        mock_catalog.return_value = [_make_entry()]

        # Create the model directory
        models_dir = mlx_stack_home / "models"
        model_path = models_dir / "qwen3.5-8b-4bit"
        model_path.mkdir(parents=True)
        (model_path / "config.json").write_text("{}")

        runner = CliRunner()
        result = runner.invoke(cli, ["pull", "qwen3.5-8b"])
        assert result.exit_code == 0
        assert "already exists" in result.output
        mock_download.assert_not_called()

    @patch("mlx_stack.core.pull.download_model")
    @patch("mlx_stack.core.pull.check_disk_space", return_value=(True, 100.0))
    @patch("mlx_stack.core.pull.load_catalog")
    def test_pull_force_redownloads(
        self,
        mock_catalog: MagicMock,
        mock_space: MagicMock,
        mock_download: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-PULL-006: --force re-downloads."""
        mock_catalog.return_value = [_make_entry()]

        # Create existing model
        models_dir = mlx_stack_home / "models"
        model_path = models_dir / "qwen3.5-8b-4bit"
        model_path.mkdir(parents=True)
        (model_path / "config.json").write_text("{}")

        runner = CliRunner()
        result = runner.invoke(cli, ["pull", "qwen3.5-8b", "--force"])
        assert result.exit_code == 0
        mock_download.assert_called_once()

    @patch(
        "mlx_stack.core.pull.download_model",
        side_effect=DownloadError("Connection failed"),
    )
    @patch("mlx_stack.core.pull.check_disk_space", return_value=(True, 100.0))
    @patch("mlx_stack.core.pull.load_catalog")
    def test_pull_network_error(
        self,
        mock_catalog: MagicMock,
        mock_space: MagicMock,
        mock_download: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-PULL-008: Network error shows clear message."""
        mock_catalog.return_value = [_make_entry()]

        runner = CliRunner()
        result = runner.invoke(cli, ["pull", "qwen3.5-8b"])
        assert result.exit_code == 1
        assert "Download error" in result.output

    @patch(
        "mlx_stack.core.pull.convert_model",
        side_effect=ConversionError("Convert failed\nmlx_lm: Error details"),
    )
    @patch("mlx_stack.core.pull.check_disk_space", return_value=(True, 100.0))
    @patch("mlx_stack.core.pull.load_catalog")
    def test_pull_conversion_error(
        self,
        mock_catalog: MagicMock,
        mock_space: MagicMock,
        mock_convert: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-PULL-010: Conversion error shows clear message."""
        mock_catalog.return_value = [_make_entry()]

        runner = CliRunner()
        result = runner.invoke(cli, ["pull", "qwen3.5-8b", "--quant", "bf16"])
        assert result.exit_code == 1
        assert "Conversion error" in result.output

    @patch("mlx_stack.core.pull.download_model")
    @patch("mlx_stack.core.pull.check_disk_space", return_value=(True, 100.0))
    @patch("mlx_stack.core.pull.load_catalog")
    def test_pull_updates_inventory(
        self,
        mock_catalog: MagicMock,
        mock_space: MagicMock,
        mock_download: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-PULL-007: After pull, model is in inventory."""
        mock_catalog.return_value = [_make_entry()]

        runner = CliRunner()
        result = runner.invoke(cli, ["pull", "qwen3.5-8b"])
        assert result.exit_code == 0

        # Check inventory
        inv = load_inventory()
        assert len(inv) == 1
        assert inv[0]["model_id"] == "qwen3.5-8b"

    @patch("mlx_stack.core.pull.download_model")
    @patch("mlx_stack.core.pull.check_disk_space", return_value=(True, 100.0))
    @patch("mlx_stack.core.pull.load_catalog")
    def test_pull_shows_progress_info(
        self,
        mock_catalog: MagicMock,
        mock_space: MagicMock,
        mock_download: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-PULL-005: Progress-related info shown during download."""
        mock_catalog.return_value = [_make_entry()]

        runner = CliRunner()
        result = runner.invoke(cli, ["pull", "qwen3.5-8b"])
        assert result.exit_code == 0
        # Should show pull info: model name, quant, source, destination
        assert "Pulling" in result.output or "Qwen 3.5 8B" in result.output

    def test_pull_no_model_argument(self) -> None:
        """pull without model argument shows error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["pull"])
        assert result.exit_code != 0

    @patch("mlx_stack.core.pull.download_model")
    @patch("mlx_stack.core.pull.check_disk_space", return_value=(True, 100.0))
    @patch("mlx_stack.core.pull.load_catalog")
    def test_pull_with_bench_flag(
        self,
        mock_catalog: MagicMock,
        mock_space: MagicMock,
        mock_download: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-CROSS-014: --bench flag accepted and runs after download."""
        mock_catalog.return_value = [_make_entry()]

        with patch("mlx_stack.cli.pull._run_post_download_bench") as mock_bench:
            runner = CliRunner()
            result = runner.invoke(cli, ["pull", "qwen3.5-8b", "--bench"])
            assert result.exit_code == 0
            mock_bench.assert_called_once()

    @patch("mlx_stack.core.pull.download_model")
    @patch("mlx_stack.core.pull.check_disk_space", return_value=(True, 100.0))
    @patch("mlx_stack.core.pull.load_catalog")
    def test_pull_bench_calls_run_benchmark_with_save(
        self,
        mock_catalog: MagicMock,
        mock_space: MagicMock,
        mock_download: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """pull --bench calls run_benchmark(save=True) to persist results."""
        from mlx_stack.core.benchmark import BenchmarkResult_

        mock_catalog.return_value = [_make_entry()]
        mock_result = BenchmarkResult_(
            model_id="qwen3.5-8b",
            quant="int4",
            prompt_tps_mean=150.0,
            prompt_tps_std=5.0,
            gen_tps_mean=80.0,
            gen_tps_std=2.0,
        )
        with patch(
            "mlx_stack.core.benchmark.run_benchmark", return_value=mock_result
        ) as mock_bench:
            runner = CliRunner()
            result = runner.invoke(cli, ["pull", "qwen3.5-8b", "--bench"])
            assert result.exit_code == 0
            mock_bench.assert_called_once_with(target="qwen3.5-8b", save=True)

    @patch("mlx_stack.core.pull.download_model")
    @patch("mlx_stack.core.pull.check_disk_space", return_value=(True, 100.0))
    @patch("mlx_stack.core.pull.load_catalog")
    def test_pull_no_traceback_on_error(
        self,
        mock_catalog: MagicMock,
        mock_space: MagicMock,
        mock_download: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """No Python traceback shown for any error scenario."""
        mock_catalog.return_value = [_make_entry()]
        mock_download.side_effect = DownloadError("Test error")

        runner = CliRunner()
        result = runner.invoke(cli, ["pull", "qwen3.5-8b"])
        assert "Traceback" not in result.output

    @patch("mlx_stack.core.pull.load_catalog", side_effect=CatalogError("Catalog broken"))
    def test_pull_catalog_error(
        self,
        mock_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """Catalog error produces clear message."""
        runner = CliRunner()
        result = runner.invoke(cli, ["pull", "some-model"])
        assert result.exit_code == 1
        assert "Catalog error" in result.output


# =========================================================================== #
# Integration: pull + models command visibility
# =========================================================================== #


class TestPullModelsIntegration:
    """Tests that pulled models appear in mlx-stack models output."""

    @patch("mlx_stack.core.pull.download_model")
    @patch("mlx_stack.core.pull.check_disk_space", return_value=(True, 100.0))
    @patch("mlx_stack.core.pull.load_catalog")
    @patch("mlx_stack.core.models.load_catalog")
    def test_pulled_model_appears_in_models(
        self,
        mock_models_catalog: MagicMock,
        mock_pull_catalog: MagicMock,
        mock_space: MagicMock,
        mock_download: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-PULL-007: After pull, model appears in mlx-stack models output."""
        entry = _make_entry()
        mock_pull_catalog.return_value = [entry]
        mock_models_catalog.return_value = [entry]

        runner = CliRunner()

        # Pull the model
        result = runner.invoke(cli, ["pull", "qwen3.5-8b"])
        assert result.exit_code == 0

        # Create model files at expected location to be found by scan
        models_dir = mlx_stack_home / "models"
        model_path = models_dir / "qwen3.5-8b-4bit"
        model_path.mkdir(parents=True, exist_ok=True)
        (model_path / "config.json").write_text("{}")

        # List models — should show the pulled model
        result = runner.invoke(cli, ["models"])
        assert result.exit_code == 0
        # The model directory or catalog name should appear
        assert "qwen3.5-8b-4bit" in result.output or "Qwen 3.5 8B" in result.output
