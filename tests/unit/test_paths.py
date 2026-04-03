"""Tests for the paths module — data directory management."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlx_stack.core.paths import (
    ensure_data_home,
    get_benchmarks_dir,
    get_config_path,
    get_data_home,
    get_lock_path,
    get_logs_dir,
    get_models_dir,
    get_pids_dir,
    get_profile_path,
    get_stacks_dir,
)


class TestGetDataHome:
    """Tests for get_data_home()."""

    def test_uses_env_var(self, mlx_stack_home: Path) -> None:
        result = get_data_home()
        assert result == mlx_stack_home

    def test_default_is_home_dot_mlx_stack(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MLX_STACK_HOME", raising=False)
        result = get_data_home()
        assert result == Path.home() / ".mlx-stack"


class TestEnsureDataHome:
    """Tests for ensure_data_home() — directory auto-creation."""

    def test_creates_directory_if_missing(self, clean_mlx_stack_home: Path) -> None:
        assert not clean_mlx_stack_home.exists()
        result = ensure_data_home()
        assert result == clean_mlx_stack_home
        assert clean_mlx_stack_home.exists()
        assert clean_mlx_stack_home.is_dir()

    def test_idempotent_when_exists(self, mlx_stack_home: Path) -> None:
        result = ensure_data_home()
        assert result == mlx_stack_home
        assert mlx_stack_home.is_dir()

    def test_directory_permissions(self, clean_mlx_stack_home: Path) -> None:
        ensure_data_home()
        mode = clean_mlx_stack_home.stat().st_mode & 0o777
        # Should be readable and writable by owner
        assert mode & 0o700 == 0o700


class TestPathHelpers:
    """Tests for path helper functions."""

    def test_profile_path(self, mlx_stack_home: Path) -> None:
        assert get_profile_path() == mlx_stack_home / "profile.json"

    def test_config_path(self, mlx_stack_home: Path) -> None:
        assert get_config_path() == mlx_stack_home / "config.yaml"

    def test_stacks_dir(self, mlx_stack_home: Path) -> None:
        assert get_stacks_dir() == mlx_stack_home / "stacks"

    def test_models_dir(self, mlx_stack_home: Path) -> None:
        assert get_models_dir() == mlx_stack_home / "models"

    def test_logs_dir(self, mlx_stack_home: Path) -> None:
        assert get_logs_dir() == mlx_stack_home / "logs"

    def test_pids_dir(self, mlx_stack_home: Path) -> None:
        assert get_pids_dir() == mlx_stack_home / "pids"

    def test_benchmarks_dir(self, mlx_stack_home: Path) -> None:
        assert get_benchmarks_dir() == mlx_stack_home / "benchmarks"

    def test_lock_path(self, mlx_stack_home: Path) -> None:
        assert get_lock_path() == mlx_stack_home / "lock"
