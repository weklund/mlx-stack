"""End-to-end integration test for the launchd lifecycle.

Tests the full install → status → uninstall lifecycle using real
launchctl commands on macOS. This test creates an actual LaunchAgent
plist, loads it via launchctl, verifies status, then cleans up.

**Requirements:**
  - macOS (darwin) — skipped on other platforms
  - No existing mlx-stack watchdog plist (won't overwrite a real install)
  - Run explicitly with: ``pytest -m integration``

**What it tests:**
  1. ``install_agent()`` creates a valid plist at the expected path
  2. Plist content has correct Label, RunAtLoad, KeepAlive, ProgramArguments
  3. Plist file permissions are 0o644
  4. ``get_agent_status()`` reports installed=True after install
  5. ``uninstall_agent()`` removes the plist and unloads the agent
  6. ``get_agent_status()`` reports installed=False after uninstall
  7. ``launchctl list`` no longer shows the agent after uninstall

The test uses a tmp_path-based MLX_STACK_HOME with a minimal stack
definition so that ``install_agent()``'s prerequisite check passes.
A try/finally block guarantees cleanup even on assertion failures.
"""

from __future__ import annotations

import contextlib
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from mlx_stack.core.launchd import (
    LAUNCHD_LABEL,
    PLIST_PERMISSIONS,
    get_agent_status,
    get_plist_path,
    install_agent,
    uninstall_agent,
    unload_agent,
)

# ---------------------------------------------------------------------------
# Markers and skip conditions
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.integration

_PLIST_PATH = get_plist_path()

_skip_not_darwin = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="launchd integration tests require macOS",
)

_skip_existing_install = pytest.mark.skipif(
    _PLIST_PATH.exists(),
    reason=(
        f"Skipping to avoid overwriting existing plist at {_PLIST_PATH}. "
        "Uninstall the real agent first if you want to run this test."
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_mlx_stack_binary() -> str:
    """Resolve the mlx-stack binary for use in the plist.

    Checks ``shutil.which`` first, then falls back to the directory
    containing the current Python interpreter.
    """
    binary = shutil.which("mlx-stack")
    if binary is not None:
        return binary

    exe_dir = Path(sys.executable).parent
    candidate = exe_dir / "mlx-stack"
    if candidate.exists():
        return str(candidate)

    pytest.skip("mlx-stack binary not found on PATH")
    # unreachable, but keeps type-checkers happy
    return ""  # pragma: no cover


def _launchctl_lists_agent() -> bool:
    """Return True if ``launchctl list`` shows the watchdog agent."""
    try:
        result = subprocess.run(
            ["launchctl", "list", LAUNCHD_LABEL],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _force_cleanup() -> None:
    """Best-effort cleanup: unload agent and remove plist file."""
    plist_path = get_plist_path()

    # Try to unload via launchctl bootout
    with contextlib.suppress(Exception):
        unload_agent(plist_path)

    # Remove plist file if it exists
    try:
        if plist_path.exists():
            plist_path.unlink()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@_skip_not_darwin
@_skip_existing_install
class TestLaunchdE2ELifecycle:
    """End-to-end launchd lifecycle: install → verify → uninstall → verify.

    Uses a tmp_path-based MLX_STACK_HOME with a minimal stack definition
    so ``install_agent()`` passes its prerequisite check. The real plist
    is written to ``~/Library/LaunchAgents/`` and loaded via ``launchctl``.

    A ``try/finally`` ensures cleanup always runs, even on test failure.

    **How to run**::

        pytest -m integration tests/integration/test_launchd_e2e.py -v
    """

    def test_full_lifecycle(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test install → status → uninstall lifecycle with real launchctl.

        Steps:
            1. Set up a tmp_path MLX_STACK_HOME with minimal default.yaml
            2. Call ``install_agent()`` with the resolved binary path
            3. Verify plist exists, content, permissions, and status
            4. Call ``uninstall_agent()``
            5. Verify plist removed and status reports not installed
            6. Verify launchctl no longer lists the agent

        Cleanup:
            Always attempts unload + plist removal in a finally block.
        """
        # ---- Setup: isolated MLX_STACK_HOME with minimal stack ----
        mlx_home = tmp_path / ".mlx-stack"
        stacks_dir = mlx_home / "stacks"
        stacks_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = mlx_home / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Minimal valid stack definition (empty tiers)
        stack_def = {
            "schema_version": 1,
            "name": "default",
            "hardware_profile": "test-profile",
            "intent": "balanced",
            "created": "2025-01-01T00:00:00",
            "tiers": [],
        }
        (stacks_dir / "default.yaml").write_text(yaml.dump(stack_def))

        monkeypatch.setenv("MLX_STACK_HOME", str(mlx_home))

        # Resolve the binary path
        binary_path = _resolve_mlx_stack_binary()

        try:
            # ---- Step 1: Install the agent ----
            _plist_path, was_reinstall = install_agent(binary_path)
            assert not was_reinstall, "Expected fresh install, not reinstall"

            # ---- Step 2: Verify plist file exists ----
            canonical_path = get_plist_path()
            assert canonical_path.exists(), f"Plist file not found at {canonical_path}"

            # ---- Step 3: Verify plist content ----
            with open(canonical_path, "rb") as f:
                plist_data = plistlib.load(f)

            assert plist_data["Label"] == LAUNCHD_LABEL, (
                f"Expected Label={LAUNCHD_LABEL!r}, got {plist_data.get('Label')!r}"
            )
            assert plist_data["RunAtLoad"] is True, "Expected RunAtLoad=True"
            assert plist_data["KeepAlive"] is True, "Expected KeepAlive=True"

            prog_args = plist_data["ProgramArguments"]
            assert isinstance(prog_args, list), "ProgramArguments should be a list"
            assert len(prog_args) >= 2, "ProgramArguments should have at least 2 elements"
            assert prog_args[0] == binary_path, (
                f"Expected binary path {binary_path!r}, got {prog_args[0]!r}"
            )
            assert prog_args[1] == "watch", f"Expected 'watch' subcommand, got {prog_args[1]!r}"

            # ---- Step 4: Verify file permissions ----
            mode = canonical_path.stat().st_mode & 0o777
            assert mode == PLIST_PERMISSIONS, (
                f"Expected permissions {oct(PLIST_PERMISSIONS)}, got {oct(mode)}"
            )

            # ---- Step 5: Verify agent status reports installed ----
            status = get_agent_status()
            assert status.installed is True, (
                "get_agent_status() should report installed=True after install"
            )

            # ---- Step 6: Uninstall the agent ----
            result = uninstall_agent()
            assert result is True, "uninstall_agent() should return True"

            # ---- Step 7: Verify plist removed ----
            assert not canonical_path.exists(), (
                f"Plist file should be removed after uninstall: {canonical_path}"
            )

            # ---- Step 8: Verify status reports not installed ----
            status_after = get_agent_status()
            assert status_after.installed is False, (
                "get_agent_status() should report installed=False after uninstall"
            )

            # ---- Step 9: Verify launchctl no longer lists the agent ----
            assert not _launchctl_lists_agent(), (
                f"launchctl list should not show {LAUNCHD_LABEL} after uninstall"
            )

        finally:
            # ---- Cleanup: always attempt to remove the agent ----
            _force_cleanup()
