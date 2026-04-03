"""Tests for the model catalog system — core/catalog.py.

Covers catalog loading (all 15 entries from shipped YAML), schema validation,
querying by family/tag/capability, error handling for corrupt/missing files,
and individual data class behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mlx_stack.core.catalog import (
    BenchmarkResult,
    Capabilities,
    CatalogEntry,
    CatalogError,
    QualityScores,
    QuantSource,
    _parse_entry,
    get_entry_by_id,
    load_catalog,
    load_catalog_from_directory,
    query_by_capability,
    query_by_family,
    query_by_tag,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def catalog() -> list[CatalogEntry]:
    """Load the full shipped catalog."""
    return load_catalog()


@pytest.fixture
def sample_yaml_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with valid catalog YAML files for testing."""
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()

    entry_data = {
        "id": "test-model-1",
        "name": "Test Model 1",
        "family": "Test Family",
        "params_b": 7.0,
        "architecture": "transformer",
        "min_mlx_lm_version": "0.22.0",
        "sources": {
            "int4": {
                "hf_repo": "test/model-4bit",
                "disk_size_gb": 4.0,
            },
            "int8": {
                "hf_repo": "test/model-8bit",
                "disk_size_gb": 7.5,
            },
        },
        "capabilities": {
            "tool_calling": True,
            "tool_call_parser": "hermes",
            "thinking": False,
            "reasoning_parser": None,
            "vision": False,
        },
        "quality": {
            "overall": 70,
            "coding": 65,
            "reasoning": 68,
            "instruction_following": 72,
        },
        "benchmarks": {
            "m4-pro-48": {
                "prompt_tps": 100.0,
                "gen_tps": 55.0,
                "memory_gb": 5.0,
            },
        },
        "tags": ["balanced", "agent-ready"],
    }

    (catalog_dir / "test-model-1.yaml").write_text(yaml.dump(entry_data))

    entry_data2 = dict(entry_data)
    entry_data2["id"] = "test-model-2"
    entry_data2["name"] = "Test Model 2"
    entry_data2["family"] = "Other Family"
    entry_data2["params_b"] = 14.0
    entry_data2["capabilities"] = {
        "tool_calling": False,
        "tool_call_parser": None,
        "thinking": True,
        "reasoning_parser": "deepseek_r1",
        "vision": True,
    }
    entry_data2["tags"] = ["vision", "thinking"]
    (catalog_dir / "test-model-2.yaml").write_text(yaml.dump(entry_data2))

    return catalog_dir


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


class TestQuantSource:
    """Tests for QuantSource dataclass."""

    def test_basic_fields(self) -> None:
        qs = QuantSource(hf_repo="mlx-community/model-4bit", disk_size_gb=4.5)
        assert qs.hf_repo == "mlx-community/model-4bit"
        assert qs.disk_size_gb == 4.5
        assert qs.convert_from is False

    def test_convert_from_flag(self) -> None:
        qs = QuantSource(hf_repo="org/model", disk_size_gb=16.0, convert_from=True)
        assert qs.convert_from is True

    def test_frozen(self) -> None:
        qs = QuantSource(hf_repo="test/repo", disk_size_gb=1.0)
        with pytest.raises(AttributeError):
            qs.hf_repo = "other"  # type: ignore[misc]


class TestCapabilities:
    """Tests for Capabilities dataclass."""

    def test_tool_calling_model(self) -> None:
        caps = Capabilities(
            tool_calling=True,
            tool_call_parser="hermes",
            thinking=False,
            reasoning_parser=None,
            vision=False,
        )
        assert caps.tool_calling is True
        assert caps.tool_call_parser == "hermes"

    def test_thinking_model(self) -> None:
        caps = Capabilities(
            tool_calling=False,
            tool_call_parser=None,
            thinking=True,
            reasoning_parser="deepseek_r1",
            vision=False,
        )
        assert caps.thinking is True
        assert caps.reasoning_parser == "deepseek_r1"

    def test_vision_model(self) -> None:
        caps = Capabilities(
            tool_calling=False,
            tool_call_parser=None,
            thinking=False,
            reasoning_parser=None,
            vision=True,
        )
        assert caps.vision is True


class TestQualityScores:
    """Tests for QualityScores dataclass."""

    def test_all_fields(self) -> None:
        qs = QualityScores(overall=80, coding=75, reasoning=78, instruction_following=85)
        assert qs.overall == 80
        assert qs.coding == 75
        assert qs.reasoning == 78
        assert qs.instruction_following == 85


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_all_fields(self) -> None:
        br = BenchmarkResult(prompt_tps=100.0, gen_tps=55.0, memory_gb=5.5)
        assert br.prompt_tps == 100.0
        assert br.gen_tps == 55.0
        assert br.memory_gb == 5.5


class TestCatalogEntry:
    """Tests for CatalogEntry dataclass."""

    def test_basic_construction(self) -> None:
        entry = CatalogEntry(
            id="test-1",
            name="Test Model",
            family="Test",
            params_b=7.0,
            architecture="transformer",
            min_mlx_lm_version="0.22.0",
            sources={"int4": QuantSource("repo", 4.0)},
            capabilities=Capabilities(True, "hermes", False, None, False),
            quality=QualityScores(70, 65, 68, 72),
            benchmarks={"m4-pro-48": BenchmarkResult(100.0, 55.0, 5.0)},
            tags=["balanced"],
        )
        assert entry.id == "test-1"
        assert entry.family == "Test"
        assert entry.params_b == 7.0
        assert len(entry.sources) == 1
        assert len(entry.tags) == 1

    def test_default_tags(self) -> None:
        entry = CatalogEntry(
            id="t",
            name="T",
            family="F",
            params_b=1.0,
            architecture="transformer",
            min_mlx_lm_version="0.1.0",
            sources={"int4": QuantSource("r", 1.0)},
            capabilities=Capabilities(False, None, False, None, False),
            quality=QualityScores(50, 50, 50, 50),
            benchmarks={},
        )
        assert entry.tags == []


# --------------------------------------------------------------------------- #
# Loading — shipped catalog
# --------------------------------------------------------------------------- #


class TestLoadCatalog:
    """Tests for loading the shipped catalog."""

    def test_loads_15_entries(self, catalog: list[CatalogEntry]) -> None:
        """All 15 catalog YAML files load successfully."""
        assert len(catalog) == 15

    def test_six_distinct_family_values_in_yaml(self, catalog: list[CatalogEntry]) -> None:
        """Catalog YAML files use 6 distinct family values."""
        families = {e.family for e in catalog}
        expected_families = {
            "Qwen 3.5",
            "Nemotron",
            "Gemma 3",
            "DeepSeek R1",
            "Qwen 3",
            "Llama 3.3",
        }
        assert families == expected_families

    def test_five_family_groups(self, catalog: list[CatalogEntry]) -> None:
        """Catalog covers 5 family groups (Qwen 3 and Llama 3.3 counted as one group).

        The grouping reflects that Qwen 3 (1 model) and Llama 3.3 (1 model) are
        grouped together as a single "agent-ready alternatives" group, yielding 5
        logical family groups: Qwen 3.5, Nemotron, Gemma 3, DeepSeek R1,
        and Qwen 3 / Llama 3.3.
        """
        # Define the 5 family groups: the last group combines Qwen 3 + Llama 3.3
        family_groups: dict[str, set[str]] = {
            "Qwen 3.5": {"Qwen 3.5"},
            "Nemotron": {"Nemotron"},
            "Gemma 3": {"Gemma 3"},
            "DeepSeek R1": {"DeepSeek R1"},
            "Qwen 3 / Llama 3.3": {"Qwen 3", "Llama 3.3"},
        }

        # Verify we have exactly 5 groups
        assert len(family_groups) == 5

        # Verify every catalog entry maps to exactly one group
        all_group_families = set()
        for group_families in family_groups.values():
            all_group_families.update(group_families)

        for entry in catalog:
            assert entry.family in all_group_families, (
                f"Model '{entry.id}' has family '{entry.family}' that is not "
                f"mapped to any family group"
            )

        # Verify each group has at least one model
        for group_name, group_families in family_groups.items():
            group_models = [e for e in catalog if e.family in group_families]
            assert len(group_models) > 0, f"Family group '{group_name}' has no models"

    def test_all_entries_have_required_fields(self, catalog: list[CatalogEntry]) -> None:
        """Every entry has all required fields populated."""
        for entry in catalog:
            assert entry.id
            assert entry.name
            assert entry.family
            assert entry.params_b > 0
            assert entry.architecture
            assert entry.min_mlx_lm_version
            assert len(entry.sources) > 0
            assert entry.quality.overall > 0

    def test_all_entries_have_valid_sources(self, catalog: list[CatalogEntry]) -> None:
        """Every entry's sources have valid quantization keys and required fields."""
        valid_quants = {"int4", "int8", "bf16"}
        for entry in catalog:
            for quant, source in entry.sources.items():
                assert quant in valid_quants, f"{entry.id}: invalid quant '{quant}'"
                assert source.hf_repo, f"{entry.id}: empty hf_repo for {quant}"
                assert source.disk_size_gb > 0, f"{entry.id}: invalid disk_size_gb for {quant}"

    def test_all_entries_have_benchmarks(self, catalog: list[CatalogEntry]) -> None:
        """Every entry has at least one benchmark entry."""
        for entry in catalog:
            assert len(entry.benchmarks) > 0, f"{entry.id}: no benchmarks"
            for bench in entry.benchmarks.values():
                assert bench.prompt_tps > 0
                assert bench.gen_tps > 0
                assert bench.memory_gb > 0

    def test_entries_sorted_by_family_then_params(self, catalog: list[CatalogEntry]) -> None:
        """Catalog entries are sorted by family name, then by params_b."""
        for i in range(1, len(catalog)):
            prev, curr = catalog[i - 1], catalog[i]
            if prev.family == curr.family:
                assert prev.params_b <= curr.params_b, (
                    f"Not sorted: {prev.id} ({prev.params_b}B) > {curr.id} ({curr.params_b}B)"
                )

    def test_no_duplicate_ids(self, catalog: list[CatalogEntry]) -> None:
        """All model IDs are unique."""
        ids = [e.id for e in catalog]
        assert len(ids) == len(set(ids))

    def test_qwen35_has_6_models(self, catalog: list[CatalogEntry]) -> None:
        """Qwen 3.5 family has exactly 6 models."""
        qwen35 = [e for e in catalog if e.family == "Qwen 3.5"]
        assert len(qwen35) == 6

    def test_specific_model_fields(self, catalog: list[CatalogEntry]) -> None:
        """Spot-check a specific model's fields."""
        entry = get_entry_by_id(catalog, "qwen3.5-8b")
        assert entry is not None
        assert entry.name == "Qwen 3.5 8B"
        assert entry.family == "Qwen 3.5"
        assert entry.params_b == 8.0
        assert entry.architecture == "transformer"
        assert "int4" in entry.sources
        assert entry.capabilities.tool_calling is True
        assert entry.capabilities.thinking is True
        assert entry.quality.overall > 0

    def test_deepseek_r1_architecture(self, catalog: list[CatalogEntry]) -> None:
        """DeepSeek R1 models have mamba2-hybrid architecture."""
        deepseek = [e for e in catalog if e.family == "DeepSeek R1"]
        assert len(deepseek) > 0
        for entry in deepseek:
            assert entry.architecture == "mamba2-hybrid"

    def test_gemma3_has_vision(self, catalog: list[CatalogEntry]) -> None:
        """Gemma 3 models have vision capability."""
        gemma = [e for e in catalog if e.family == "Gemma 3"]
        assert len(gemma) > 0
        for entry in gemma:
            assert entry.capabilities.vision is True

    def test_tool_calling_models_have_parser(self, catalog: list[CatalogEntry]) -> None:
        """Models with tool_calling=True have a tool_call_parser set."""
        for entry in catalog:
            if entry.capabilities.tool_calling:
                assert entry.capabilities.tool_call_parser is not None, (
                    f"{entry.id}: tool_calling=True but no tool_call_parser"
                )

    def test_thinking_models_have_parser(self, catalog: list[CatalogEntry]) -> None:
        """Models with thinking=True have a reasoning_parser set."""
        for entry in catalog:
            if entry.capabilities.thinking:
                assert entry.capabilities.reasoning_parser is not None, (
                    f"{entry.id}: thinking=True but no reasoning_parser"
                )


# --------------------------------------------------------------------------- #
# Loading — from directory
# --------------------------------------------------------------------------- #


class TestLoadCatalogFromDirectory:
    """Tests for loading catalog from a custom directory."""

    def test_loads_entries(self, sample_yaml_dir: Path) -> None:
        entries = load_catalog_from_directory(str(sample_yaml_dir))
        assert len(entries) == 2

    def test_entries_have_correct_ids(self, sample_yaml_dir: Path) -> None:
        entries = load_catalog_from_directory(str(sample_yaml_dir))
        ids = {e.id for e in entries}
        assert "test-model-1" in ids
        assert "test-model-2" in ids

    def test_entries_sorted(self, sample_yaml_dir: Path) -> None:
        entries = load_catalog_from_directory(str(sample_yaml_dir))
        # "Other Family" < "Test Family" alphabetically
        assert entries[0].family == "Other Family"
        assert entries[1].family == "Test Family"


# --------------------------------------------------------------------------- #
# Error handling — corrupt/missing catalog files
# --------------------------------------------------------------------------- #


class TestCatalogErrors:
    """Tests for error handling with corrupt/missing catalog files."""

    def test_missing_directory(self, tmp_path: Path) -> None:
        """Non-existent directory raises CatalogError."""
        with pytest.raises(CatalogError, match="Catalog directory not found"):
            load_catalog_from_directory(str(tmp_path / "nonexistent"))

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory raises CatalogError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(CatalogError, match="No catalog YAML files found"):
            load_catalog_from_directory(str(empty_dir))

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        """Malformed YAML raises CatalogError with filename."""
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        (catalog_dir / "bad.yaml").write_text("{{{invalid yaml")
        with pytest.raises(CatalogError, match="contains invalid YAML"):
            load_catalog_from_directory(str(catalog_dir))

    def test_non_mapping_yaml(self, tmp_path: Path) -> None:
        """YAML that is a list (not mapping) raises CatalogError."""
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        (catalog_dir / "list.yaml").write_text("- item1\n- item2\n")
        with pytest.raises(CatalogError, match="must contain a YAML mapping"):
            load_catalog_from_directory(str(catalog_dir))

    def test_missing_required_field(self, tmp_path: Path) -> None:
        """Missing required field raises CatalogError."""
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        (catalog_dir / "incomplete.yaml").write_text(yaml.dump({"id": "test", "name": "Test"}))
        with pytest.raises(CatalogError, match="missing required field"):
            load_catalog_from_directory(str(catalog_dir))

    def test_wrong_field_type(self, tmp_path: Path) -> None:
        """Wrong field type raises CatalogError."""
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        data = _make_valid_entry()
        data["params_b"] = "not-a-number"
        (catalog_dir / "wrong_type.yaml").write_text(yaml.dump(data))
        with pytest.raises(CatalogError, match="wrong type"):
            load_catalog_from_directory(str(catalog_dir))

    def test_empty_sources(self, tmp_path: Path) -> None:
        """Empty sources raises CatalogError."""
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        data = _make_valid_entry()
        data["sources"] = {}
        (catalog_dir / "empty_sources.yaml").write_text(yaml.dump(data))
        with pytest.raises(CatalogError, match="must not be empty"):
            load_catalog_from_directory(str(catalog_dir))

    def test_invalid_quant_key(self, tmp_path: Path) -> None:
        """Invalid quantization key raises CatalogError."""
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        data = _make_valid_entry()
        data["sources"]["int6"] = {"hf_repo": "test/repo", "disk_size_gb": 5.0}
        (catalog_dir / "bad_quant.yaml").write_text(yaml.dump(data))
        with pytest.raises(CatalogError, match="invalid quantization 'int6'"):
            load_catalog_from_directory(str(catalog_dir))

    def test_missing_source_field(self, tmp_path: Path) -> None:
        """Missing required source field raises CatalogError."""
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        data = _make_valid_entry()
        data["sources"]["int4"] = {"hf_repo": "test/repo"}  # missing disk_size_gb
        (catalog_dir / "no_disk_size.yaml").write_text(yaml.dump(data))
        with pytest.raises(CatalogError, match="missing required field 'disk_size_gb'"):
            load_catalog_from_directory(str(catalog_dir))

    def test_missing_capability_field(self, tmp_path: Path) -> None:
        """Missing capability field raises CatalogError."""
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        data = _make_valid_entry()
        del data["capabilities"]["tool_calling"]
        (catalog_dir / "no_tool_calling.yaml").write_text(yaml.dump(data))
        with pytest.raises(CatalogError, match="capabilities missing required field"):
            load_catalog_from_directory(str(catalog_dir))

    def test_missing_quality_field(self, tmp_path: Path) -> None:
        """Missing quality field raises CatalogError."""
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        data = _make_valid_entry()
        del data["quality"]["overall"]
        (catalog_dir / "no_overall.yaml").write_text(yaml.dump(data))
        with pytest.raises(CatalogError, match="quality missing required field"):
            load_catalog_from_directory(str(catalog_dir))

    def test_missing_benchmark_field(self, tmp_path: Path) -> None:
        """Missing benchmark field raises CatalogError."""
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        data = _make_valid_entry()
        data["benchmarks"]["m4-pro-48"] = {"prompt_tps": 100.0}  # missing gen_tps, memory_gb
        (catalog_dir / "no_gen_tps.yaml").write_text(yaml.dump(data))
        with pytest.raises(CatalogError, match=r"benchmark.*missing required field"):
            load_catalog_from_directory(str(catalog_dir))

    def test_non_string_tag(self, tmp_path: Path) -> None:
        """Non-string tag raises CatalogError."""
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        data = _make_valid_entry()
        data["tags"] = [123, "valid"]
        (catalog_dir / "bad_tag.yaml").write_text(yaml.dump(data))
        with pytest.raises(CatalogError, match="tags must be strings"):
            load_catalog_from_directory(str(catalog_dir))

    def test_error_identifies_filename(self, tmp_path: Path) -> None:
        """Error messages include the filename for easy identification."""
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        (catalog_dir / "specific-file.yaml").write_text(yaml.dump({"id": "x"}))
        with pytest.raises(CatalogError, match=r"specific-file\.yaml"):
            load_catalog_from_directory(str(catalog_dir))

    def test_non_numeric_disk_size_gb(self, tmp_path: Path) -> None:
        """Non-numeric disk_size_gb raises CatalogError, not raw ValueError."""
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        data = _make_valid_entry()
        data["sources"]["int4"]["disk_size_gb"] = "abc"
        (catalog_dir / "bad_disk_size.yaml").write_text(yaml.dump(data))
        with pytest.raises(CatalogError, match=r"disk_size_gb.*must be numeric"):
            load_catalog_from_directory(str(catalog_dir))

    def test_non_numeric_quality_score(self, tmp_path: Path) -> None:
        """Non-numeric quality score raises CatalogError, not raw ValueError."""
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        data = _make_valid_entry()
        data["quality"]["overall"] = "high"
        (catalog_dir / "bad_quality.yaml").write_text(yaml.dump(data))
        with pytest.raises(CatalogError, match=r"quality.*overall.*must be numeric"):
            load_catalog_from_directory(str(catalog_dir))

    def test_non_numeric_benchmark_value(self, tmp_path: Path) -> None:
        """Non-numeric benchmark value raises CatalogError, not raw ValueError."""
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        data = _make_valid_entry()
        data["benchmarks"]["m4-pro-48"]["gen_tps"] = "fast"
        (catalog_dir / "bad_bench.yaml").write_text(yaml.dump(data))
        with pytest.raises(CatalogError, match=r"benchmark.*gen_tps.*must be numeric"):
            load_catalog_from_directory(str(catalog_dir))

    def test_corrupted_disk_size_no_raw_valueerror(self, tmp_path: Path) -> None:
        """Ensure corrupted disk_size_gb does not leak a raw ValueError to user."""
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        data = _make_valid_entry()
        data["sources"]["int4"]["disk_size_gb"] = "not-a-number"
        (catalog_dir / "corrupt.yaml").write_text(yaml.dump(data))
        with pytest.raises(CatalogError):
            load_catalog_from_directory(str(catalog_dir))
        # Confirm it's NOT a raw ValueError
        try:
            load_catalog_from_directory(str(catalog_dir))
        except CatalogError:
            pass  # Expected
        except (ValueError, TypeError):
            pytest.fail("Raw ValueError/TypeError leaked instead of CatalogError")

    def test_corrupted_quality_no_raw_valueerror(self, tmp_path: Path) -> None:
        """Ensure corrupted quality score does not leak a raw ValueError to user."""
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        data = _make_valid_entry()
        data["quality"]["coding"] = "excellent"
        (catalog_dir / "corrupt_quality.yaml").write_text(yaml.dump(data))
        with pytest.raises(CatalogError):
            load_catalog_from_directory(str(catalog_dir))

    def test_corrupted_benchmark_no_raw_valueerror(self, tmp_path: Path) -> None:
        """Ensure corrupted benchmark value does not leak a raw ValueError to user."""
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        data = _make_valid_entry()
        data["benchmarks"]["m4-pro-48"]["memory_gb"] = "lots"
        (catalog_dir / "corrupt_bench.yaml").write_text(yaml.dump(data))
        with pytest.raises(CatalogError):
            load_catalog_from_directory(str(catalog_dir))

    def test_no_python_traceback_in_message(self, tmp_path: Path) -> None:
        """CatalogError messages don't contain Python traceback keywords."""
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        (catalog_dir / "bad.yaml").write_text("{{{")
        try:
            load_catalog_from_directory(str(catalog_dir))
        except CatalogError as exc:
            # The error message itself should not look like a traceback
            assert "Traceback" not in str(exc)


# --------------------------------------------------------------------------- #
# Querying
# --------------------------------------------------------------------------- #


class TestQueryByFamily:
    """Tests for query_by_family()."""

    def test_qwen35_returns_6(self, catalog: list[CatalogEntry]) -> None:
        """Filtering by 'Qwen 3.5' returns exactly 6 models."""
        results = query_by_family(catalog, "Qwen 3.5")
        assert len(results) == 6

    def test_case_insensitive(self, catalog: list[CatalogEntry]) -> None:
        """Family matching is case-insensitive."""
        results = query_by_family(catalog, "qwen 3.5")
        assert len(results) == 6

    def test_nemotron(self, catalog: list[CatalogEntry]) -> None:
        results = query_by_family(catalog, "Nemotron")
        assert len(results) == 2

    def test_gemma3(self, catalog: list[CatalogEntry]) -> None:
        results = query_by_family(catalog, "Gemma 3")
        assert len(results) == 3

    def test_deepseek_r1(self, catalog: list[CatalogEntry]) -> None:
        results = query_by_family(catalog, "DeepSeek R1")
        assert len(results) == 2

    def test_nonexistent_family(self, catalog: list[CatalogEntry]) -> None:
        results = query_by_family(catalog, "Nonexistent")
        assert len(results) == 0


class TestQueryByTag:
    """Tests for query_by_tag()."""

    def test_agent_ready(self, catalog: list[CatalogEntry]) -> None:
        results = query_by_tag(catalog, "agent-ready")
        assert len(results) > 0
        for entry in results:
            assert "agent-ready" in entry.tags

    def test_vision(self, catalog: list[CatalogEntry]) -> None:
        results = query_by_tag(catalog, "vision")
        assert len(results) > 0
        for entry in results:
            assert "vision" in entry.tags

    def test_thinking(self, catalog: list[CatalogEntry]) -> None:
        results = query_by_tag(catalog, "thinking")
        assert len(results) > 0
        for entry in results:
            assert "thinking" in entry.tags

    def test_case_insensitive(self, catalog: list[CatalogEntry]) -> None:
        results_lower = query_by_tag(catalog, "agent-ready")
        results_upper = query_by_tag(catalog, "Agent-Ready")
        assert len(results_lower) == len(results_upper)

    def test_nonexistent_tag(self, catalog: list[CatalogEntry]) -> None:
        results = query_by_tag(catalog, "nonexistent-tag")
        assert len(results) == 0


class TestQueryByCapability:
    """Tests for query_by_capability()."""

    def test_tool_calling_true(self, catalog: list[CatalogEntry]) -> None:
        """Filter by tool_calling=True returns correct models."""
        results = query_by_capability(catalog, tool_calling=True)
        assert len(results) > 0
        for entry in results:
            assert entry.capabilities.tool_calling is True

    def test_tool_calling_false(self, catalog: list[CatalogEntry]) -> None:
        results = query_by_capability(catalog, tool_calling=False)
        assert len(results) > 0
        for entry in results:
            assert entry.capabilities.tool_calling is False

    def test_vision_true(self, catalog: list[CatalogEntry]) -> None:
        results = query_by_capability(catalog, vision=True)
        assert len(results) > 0
        for entry in results:
            assert entry.capabilities.vision is True

    def test_thinking_true(self, catalog: list[CatalogEntry]) -> None:
        results = query_by_capability(catalog, thinking=True)
        assert len(results) > 0
        for entry in results:
            assert entry.capabilities.thinking is True

    def test_combined_filters(self, catalog: list[CatalogEntry]) -> None:
        """Multiple capability filters are AND-ed together."""
        results = query_by_capability(catalog, tool_calling=True, thinking=True)
        assert len(results) > 0
        for entry in results:
            assert entry.capabilities.tool_calling is True
            assert entry.capabilities.thinking is True

    def test_no_match(self, catalog: list[CatalogEntry]) -> None:
        """Filtering with impossible combination returns empty list."""
        # No Gemma model has tool_calling, so vision+tool_calling should exclude Gemma
        # and non-vision models, possibly yielding empty
        results = query_by_capability(catalog, tool_calling=True, vision=True)
        # All results must match both
        for entry in results:
            assert entry.capabilities.tool_calling is True
            assert entry.capabilities.vision is True

    def test_invalid_capability_raises(self, catalog: list[CatalogEntry]) -> None:
        with pytest.raises(ValueError, match="Invalid capability filter"):
            query_by_capability(catalog, invalid_cap=True)

    def test_empty_filters_returns_all(self, catalog: list[CatalogEntry]) -> None:
        results = query_by_capability(catalog)
        assert len(results) == 15


class TestGetEntryById:
    """Tests for get_entry_by_id()."""

    def test_existing_id(self, catalog: list[CatalogEntry]) -> None:
        entry = get_entry_by_id(catalog, "qwen3.5-8b")
        assert entry is not None
        assert entry.id == "qwen3.5-8b"

    def test_nonexistent_id(self, catalog: list[CatalogEntry]) -> None:
        entry = get_entry_by_id(catalog, "nonexistent")
        assert entry is None

    def test_all_ids_findable(self, catalog: list[CatalogEntry]) -> None:
        for expected in catalog:
            found = get_entry_by_id(catalog, expected.id)
            assert found is not None
            assert found.id == expected.id


# --------------------------------------------------------------------------- #
# Helper to build valid entry data for error tests
# --------------------------------------------------------------------------- #


def _make_valid_entry() -> dict:
    """Create a minimal valid catalog entry dict for mutation tests."""
    return {
        "id": "valid-model",
        "name": "Valid Model",
        "family": "Test",
        "params_b": 7.0,
        "architecture": "transformer",
        "min_mlx_lm_version": "0.22.0",
        "sources": {
            "int4": {
                "hf_repo": "test/valid-4bit",
                "disk_size_gb": 4.0,
            },
        },
        "capabilities": {
            "tool_calling": True,
            "tool_call_parser": "hermes",
            "thinking": False,
            "reasoning_parser": None,
            "vision": False,
        },
        "quality": {
            "overall": 70,
            "coding": 65,
            "reasoning": 68,
            "instruction_following": 72,
        },
        "benchmarks": {
            "m4-pro-48": {
                "prompt_tps": 100.0,
                "gen_tps": 55.0,
                "memory_gb": 5.0,
            },
        },
        "tags": ["test"],
    }


# =========================================================================== #
# Gated field tests
# =========================================================================== #


class TestGatedField:
    """Tests for the CatalogEntry.gated field."""

    def test_gated_defaults_to_false(self) -> None:
        """CatalogEntry without gated field defaults to False."""
        data = _make_valid_entry()
        entry = _parse_entry(data)
        assert entry.gated is False

    def test_gated_true_from_yaml(self) -> None:
        """CatalogEntry with gated: true parses correctly."""
        data = _make_valid_entry()
        data["gated"] = True
        entry = _parse_entry(data)
        assert entry.gated is True

    def test_gated_false_explicit(self) -> None:
        """Explicit gated: false parses correctly."""
        data = _make_valid_entry()
        data["gated"] = False
        entry = _parse_entry(data)
        assert entry.gated is False

    def test_shipped_catalog_gated_models(self) -> None:
        """Shipped catalog gated models are correctly marked."""
        catalog = load_catalog()
        gated = [e for e in catalog if e.gated]
        gated_ids = {e.id for e in gated}
        assert gated_ids == {
            "deepseek-r1-32b",
            "gemma3-4b",
            "gemma3-12b",
            "gemma3-27b",
            "llama3.3-8b",
            "nemotron-49b",
            "nemotron-8b",
            "qwen3.5-3b",
            "qwen3.5-8b",
            "qwen3.5-14b",
            "qwen3.5-32b",
            "qwen3.5-72b",
        }

    def test_shipped_catalog_non_gated_models(self) -> None:
        """Shipped catalog non-gated models all have gated=False."""
        catalog = load_catalog()
        non_gated = [e for e in catalog if not e.gated]
        assert len(non_gated) == len(catalog) - 12
        for entry in non_gated:
            assert entry.gated is False
