"""Tests for auto-creation of ~/.mlx-stack/ data directory on first use.

Validates VAL-SETUP-004: On a clean system where ~/.mlx-stack/ does not exist,
running any state-writing command creates the directory with correct permissions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mlx_stack.core.paths import ensure_data_home


class TestDataDirectoryAutoCreation:
    """Test that the data directory is auto-created on first use."""

    def test_ensure_creates_directory(self, clean_mlx_stack_home: Path) -> None:
        """Data directory is created when it doesn't exist."""
        assert not clean_mlx_stack_home.exists()
        ensure_data_home()
        assert clean_mlx_stack_home.exists()
        assert clean_mlx_stack_home.is_dir()

    def test_ensure_is_idempotent(self, mlx_stack_home: Path) -> None:
        """Data directory creation is idempotent."""
        ensure_data_home()
        ensure_data_home()
        assert mlx_stack_home.is_dir()

    def test_ensure_creates_parent_dirs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nested parent directories are created automatically."""
        deep_path = tmp_path / "a" / "b" / "c" / ".mlx-stack"
        monkeypatch.setenv("MLX_STACK_HOME", str(deep_path))
        ensure_data_home()
        assert deep_path.exists()
        assert deep_path.is_dir()

    def test_directory_has_correct_permissions(self, clean_mlx_stack_home: Path) -> None:
        """Created directory has standard user permissions."""
        ensure_data_home()
        mode = clean_mlx_stack_home.stat().st_mode & 0o777
        # Owner should have rwx
        assert mode & 0o700 == 0o700
