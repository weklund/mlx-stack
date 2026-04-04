"""Tests for model inventory and pull integration with models module.

Validates:
- VAL-PULL-007: Model inventory tracking in models.json
- Integration between pull inventory and models command scan
"""

from __future__ import annotations

import json
from pathlib import Path

from mlx_stack.core.pull import (
    ModelInventoryEntry,
    add_to_inventory,
    find_in_inventory,
    load_inventory,
    save_inventory,
)

# =========================================================================== #
# Inventory file I/O tests
# =========================================================================== #


class TestInventoryIO:
    """Tests for inventory file read/write operations."""

    def test_empty_inventory_on_fresh_install(self, mlx_stack_home: Path) -> None:
        """No models.json → empty inventory."""
        assert load_inventory() == []

    def test_save_creates_file(self, mlx_stack_home: Path) -> None:
        """save_inventory creates models.json."""
        # Act
        entries = [{"model_id": "test", "quant": "int4", "name": "Test"}]
        save_inventory(entries)

        # Assert
        inv_path = mlx_stack_home / "models.json"
        assert inv_path.exists()

    def test_round_trip_preserves_data(self, mlx_stack_home: Path) -> None:
        """Save → load round-trip preserves all fields."""
        # Arrange
        entry = {
            "model_id": "qwen3.5-8b",
            "name": "Qwen 3.5 8B",
            "quant": "int4",
            "source_type": "mlx-community",
            "hf_repo": "mlx-community/qwen3.5-8b-4bit",
            "local_path": "/tmp/models/qwen3.5-8b-4bit",
            "disk_size_gb": 4.5,
            "downloaded_at": "2025-01-01T00:00:00+00:00",
        }

        # Act
        save_inventory([entry])
        loaded = load_inventory()

        # Assert
        assert len(loaded) == 1
        for key, value in entry.items():
            assert loaded[0][key] == value

    def test_multiple_entries(self, mlx_stack_home: Path) -> None:
        """Multiple entries saved and loaded correctly."""
        entries = [
            {"model_id": "model-a", "quant": "int4"},
            {"model_id": "model-b", "quant": "int8"},
            {"model_id": "model-c", "quant": "bf16"},
        ]
        save_inventory(entries)
        loaded = load_inventory()
        assert len(loaded) == 3
        assert [e["model_id"] for e in loaded] == ["model-a", "model-b", "model-c"]

    def test_inventory_file_valid_json(self, mlx_stack_home: Path) -> None:
        """Saved file is valid JSON."""
        save_inventory([{"model_id": "test", "quant": "int4"}])
        inv_path = mlx_stack_home / "models.json"
        content = inv_path.read_text(encoding="utf-8")
        parsed = json.loads(content)
        assert isinstance(parsed, list)

    def test_corrupt_json_returns_empty(self, mlx_stack_home: Path) -> None:
        """Corrupt JSON returns empty list."""
        inv_path = mlx_stack_home / "models.json"
        inv_path.write_text("{{{not valid", encoding="utf-8")
        assert load_inventory() == []

    def test_non_list_json_returns_empty(self, mlx_stack_home: Path) -> None:
        """JSON that's not a list returns empty list."""
        inv_path = mlx_stack_home / "models.json"
        inv_path.write_text('{"not": "a list"}', encoding="utf-8")
        assert load_inventory() == []


# =========================================================================== #
# Inventory add/find operations
# =========================================================================== #


class TestInventoryOperations:
    """Tests for add_to_inventory and find_in_inventory."""

    def test_add_new_entry(self, mlx_stack_home: Path) -> None:
        """Adding a new entry creates it."""
        # Arrange
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

        # Act
        add_to_inventory(entry)

        # Assert
        loaded = load_inventory()
        assert len(loaded) == 1

    def test_add_replaces_same_model_quant(self, mlx_stack_home: Path) -> None:
        """Duplicate model_id+quant replaces rather than duplicates."""
        for i in range(3):
            entry = ModelInventoryEntry(
                model_id="qwen3.5-8b",
                name="Qwen 3.5 8B",
                quant="int4",
                source_type="mlx-community",
                hf_repo="mlx-community/qwen3.5-8b-4bit",
                local_path=f"/tmp/models/path-{i}",
                disk_size_gb=4.5,
                downloaded_at=f"2025-01-0{i + 1}T00:00:00+00:00",
            )
            add_to_inventory(entry)

        loaded = load_inventory()
        assert len(loaded) == 1
        assert loaded[0]["local_path"] == "/tmp/models/path-2"

    def test_different_quants_preserved(self, mlx_stack_home: Path) -> None:
        """Same model with different quants kept as separate entries."""
        for quant in ["int4", "int8", "bf16"]:
            entry = ModelInventoryEntry(
                model_id="qwen3.5-8b",
                name="Qwen 3.5 8B",
                quant=quant,
                source_type="mlx-community",
                hf_repo=f"mlx-community/qwen3.5-8b-{quant}",
                local_path=f"/tmp/models/qwen3.5-8b-{quant}",
                disk_size_gb=4.5,
                downloaded_at="2025-01-01T00:00:00+00:00",
            )
            add_to_inventory(entry)

        loaded = load_inventory()
        assert len(loaded) == 3

    def test_find_existing_entry(self, mlx_stack_home: Path) -> None:
        """Finding an existing entry returns it."""
        # Arrange
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

        # Act
        found = find_in_inventory("qwen3.5-8b", "int4")

        # Assert
        assert found is not None
        assert found["name"] == "Qwen 3.5 8B"

    def test_find_nonexistent_returns_none(self, mlx_stack_home: Path) -> None:
        """Finding a nonexistent model returns None."""
        assert find_in_inventory("nonexistent", "int4") is None

    def test_find_wrong_quant_returns_none(self, mlx_stack_home: Path) -> None:
        """Finding with wrong quant returns None."""
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

        assert find_in_inventory("qwen3.5-8b", "int8") is None


# =========================================================================== #
# Inventory entry data class
# =========================================================================== #


class TestModelInventoryEntry:
    """Tests for ModelInventoryEntry data class."""

    def test_create_entry(self) -> None:
        """Can create an inventory entry."""
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
        assert entry.model_id == "qwen3.5-8b"
        assert entry.quant == "int4"

    def test_entry_is_frozen(self) -> None:
        """Entry is immutable."""
        import pytest

        entry = ModelInventoryEntry(
            model_id="test",
            name="Test",
            quant="int4",
            source_type="mlx-community",
            hf_repo="test/repo",
            local_path="/tmp/test",
            disk_size_gb=1.0,
            downloaded_at="2025-01-01T00:00:00+00:00",
        )
        with pytest.raises(AttributeError):
            entry.model_id = "changed"  # type: ignore[misc]
