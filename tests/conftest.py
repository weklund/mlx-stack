"""Shared pytest fixtures for mlx-stack tests.

Provides MLX_STACK_HOME isolation via tmp_path to ensure tests never touch
the real ~/.mlx-stack/ directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def mlx_stack_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide an isolated MLX_STACK_HOME directory for testing.

    Sets the MLX_STACK_HOME environment variable to a temporary directory,
    ensuring no test touches the real ~/.mlx-stack/ directory.

    Returns:
        Path to the temporary mlx-stack home directory.
    """
    home = tmp_path / ".mlx-stack"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MLX_STACK_HOME", str(home))
    return home


@pytest.fixture()
def clean_mlx_stack_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide a clean (non-existent) MLX_STACK_HOME for testing auto-creation.

    Sets MLX_STACK_HOME to a path that does NOT yet exist, so tests can
    verify that the directory is auto-created on first use.

    Returns:
        Path to the non-existent mlx-stack home directory.
    """
    home = tmp_path / ".mlx-stack"
    # Intentionally do NOT create the directory
    monkeypatch.setenv("MLX_STACK_HOME", str(home))
    return home
