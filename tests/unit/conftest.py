"""Unit-test-specific fixtures.

Provides ``fake_services``, ``stack_on_disk``, and directory helpers that
build on the shared ``mlx_stack_home`` fixture from ``tests/conftest.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.factories import write_litellm_yaml, write_stack_yaml
from tests.fakes import FakeServiceLayer

# --------------------------------------------------------------------------- #
# Directory helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def pids_dir(mlx_stack_home: Path) -> Path:
    """Create and return the ``pids/`` subdirectory."""
    d = mlx_stack_home / "pids"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def logs_dir(mlx_stack_home: Path) -> Path:
    """Create and return the ``logs/`` subdirectory."""
    d = mlx_stack_home / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Stack-on-disk fixture
# --------------------------------------------------------------------------- #


@pytest.fixture
def stack_on_disk(mlx_stack_home: Path) -> Path:
    """Write the default stack + litellm YAML and return the home path."""
    write_stack_yaml(mlx_stack_home)
    write_litellm_yaml(mlx_stack_home)
    return mlx_stack_home


# --------------------------------------------------------------------------- #
# FakeServiceLayer fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_services(monkeypatch: pytest.MonkeyPatch) -> FakeServiceLayer:
    """Provide a ``FakeServiceLayer`` patched into ``mlx_stack.core.stack_up``.

    The default configuration produces a fully successful startup.  Tests
    call configuration methods (e.g. ``fake_services.fail_port(8000)``)
    in the *Arrange* phase to inject specific failure scenarios.
    """
    layer = FakeServiceLayer()
    _patch_stack_up(monkeypatch, layer)
    return layer


@pytest.fixture
def fake_watchdog_services(monkeypatch: pytest.MonkeyPatch) -> FakeServiceLayer:
    """Provide a ``FakeServiceLayer`` patched into ``mlx_stack.core.watchdog``.

    Same as ``fake_services`` but targets the watchdog module's imports.
    """
    layer = FakeServiceLayer()
    _patch_watchdog(monkeypatch, layer)
    return layer


# --------------------------------------------------------------------------- #
# Internal patch helpers
# --------------------------------------------------------------------------- #


def _patch_stack_up(mp: pytest.MonkeyPatch, layer: FakeServiceLayer) -> None:
    """Apply monkeypatches for ``mlx_stack.core.stack_up``."""
    prefix = "mlx_stack.core.stack_up"
    mp.setattr(f"{prefix}.start_service", layer.start_service)
    mp.setattr(f"{prefix}.wait_for_healthy", layer.wait_for_healthy)
    mp.setattr(f"{prefix}.check_port_conflict", layer.check_port_conflict)
    mp.setattr(f"{prefix}.is_process_alive", layer.is_process_alive)
    mp.setattr(f"{prefix}.cleanup_stale_pid", layer.cleanup_stale_pid)
    mp.setattr(f"{prefix}.acquire_lock", layer.acquire_lock)
    mp.setattr(f"{prefix}.ensure_dependency", layer.ensure_dependency)
    mp.setattr(f"{prefix}.shutil.which", layer.which)
    mp.setattr(f"{prefix}.check_local_model_exists", layer.check_local_model_exists)
    mp.setattr(f"{prefix}.load_catalog", layer.load_catalog)
    mp.setattr(f"{prefix}.get_value", layer.get_value)


def _patch_watchdog(mp: pytest.MonkeyPatch, layer: FakeServiceLayer) -> None:
    """Apply monkeypatches for ``mlx_stack.core.watchdog``."""
    prefix = "mlx_stack.core.watchdog"
    mp.setattr(f"{prefix}.start_service", layer.start_service)
    mp.setattr(f"{prefix}.acquire_lock", layer.acquire_lock)
    mp.setattr(f"{prefix}.is_process_alive", layer.is_process_alive)
    mp.setattr(f"{prefix}.get_value", layer.get_value)
