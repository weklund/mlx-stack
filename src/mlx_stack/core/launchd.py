"""launchd integration for mlx-stack.

Implements plist generation and launchctl management for running the
watchdog health monitor as a macOS LaunchAgent. Uses plistlib (stdlib)
for plist generation.

Provides:
- generate_plist(): Generate a plist dict for the watchdog agent
- write_plist(): Write the plist to ~/Library/LaunchAgents/
- load_agent(): Load the agent via launchctl bootstrap
- unload_agent(): Unload the agent via launchctl bootout
- get_agent_status(): Check if the agent is loaded and get PID
- get_plist_path(): Get the canonical plist file path
"""

from __future__ import annotations

import contextlib
import os
import plistlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx_stack.core.paths import get_logs_dir, get_stacks_dir

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

LAUNCHD_LABEL = "com.mlx-stack.watchdog"
PLIST_FILENAME = f"{LAUNCHD_LABEL}.plist"
PLIST_PERMISSIONS = 0o644


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class LaunchdError(Exception):
    """Raised when a launchd operation fails."""


class PlatformError(LaunchdError):
    """Raised when running on a non-macOS platform."""


class PrerequisiteError(LaunchdError):
    """Raised when a prerequisite is not met (e.g., init not run)."""


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AgentStatus:
    """Status of the launchd agent."""

    installed: bool
    running: bool
    pid: int | None
    label: str = LAUNCHD_LABEL

    @property
    def message(self) -> str:
        """Return a human-readable status message."""
        if not self.installed:
            return "not installed"
        if self.running and self.pid is not None:
            return f"installed and running (PID {self.pid})"
        return "installed but not running"


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #


def get_plist_path() -> Path:
    """Return the canonical path for the watchdog plist file.

    Returns:
        ~/Library/LaunchAgents/com.mlx-stack.watchdog.plist
    """
    return Path.home() / "Library" / "LaunchAgents" / PLIST_FILENAME


# --------------------------------------------------------------------------- #
# Platform and prerequisite checks
# --------------------------------------------------------------------------- #


def check_platform() -> None:
    """Check that we're running on macOS.

    Raises:
        PlatformError: If not running on macOS (darwin).
    """
    if sys.platform != "darwin":
        msg = f"launchd integration is only available on macOS. Current platform: {sys.platform}"
        raise PlatformError(msg)


def check_init_prerequisite() -> None:
    """Check that mlx-stack init has been run.

    Verifies that a stack definition exists at
    ~/.mlx-stack/stacks/default.yaml.

    Raises:
        PrerequisiteError: If init has not been run.
    """
    stack_path = get_stacks_dir() / "default.yaml"
    if not stack_path.exists():
        msg = "No stack configuration found. Run 'mlx-stack init' first."
        raise PrerequisiteError(msg)


# --------------------------------------------------------------------------- #
# Plist generation
# --------------------------------------------------------------------------- #


def _resolve_mlx_stack_binary() -> str:
    """Resolve the full path to the mlx-stack binary.

    Returns:
        The full path to the mlx-stack executable.

    Raises:
        LaunchdError: If the binary cannot be found.
    """
    binary = shutil.which("mlx-stack")
    if binary is not None:
        return binary

    # Fallback: try sys.executable-based resolution
    # (e.g., when installed in a venv, the binary is next to python)
    exe_dir = Path(sys.executable).parent
    candidate = exe_dir / "mlx-stack"
    if candidate.exists():
        return str(candidate)

    msg = "Could not find the mlx-stack binary on PATH. Ensure mlx-stack is properly installed."
    raise LaunchdError(msg)


def _build_environment_variables(mlx_stack_binary: str) -> dict[str, str]:
    """Build the EnvironmentVariables dict for the plist.

    Always includes PATH (with the directory containing the mlx-stack
    binary). Includes MLX_STACK_HOME only if a custom (non-default)
    value is set via environment variable.

    Args:
        mlx_stack_binary: Full path to the mlx-stack binary.

    Returns:
        Dict of environment variable name → value.
    """
    env: dict[str, str] = {}

    # Build PATH: include the binary's directory plus standard paths
    binary_dir = str(Path(mlx_stack_binary).parent)
    standard_paths = [
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
        "/opt/homebrew/bin",
    ]

    # Ensure binary_dir is first, then add standard paths not already present
    path_components = [binary_dir, *(p for p in standard_paths if p != binary_dir)]

    env["PATH"] = ":".join(path_components)

    # Include MLX_STACK_HOME only if custom (non-default)
    custom_home = os.environ.get("MLX_STACK_HOME")
    if custom_home:
        env["MLX_STACK_HOME"] = custom_home

    return env


def generate_plist(mlx_stack_binary: str | None = None) -> dict[str, Any]:
    """Generate the launchd plist dictionary for the watchdog agent.

    Args:
        mlx_stack_binary: Full path to the mlx-stack binary.
            If None, resolves automatically.

    Returns:
        A dict suitable for writing with plistlib.

    Raises:
        LaunchdError: If the binary cannot be resolved.
    """
    if mlx_stack_binary is None:
        mlx_stack_binary = _resolve_mlx_stack_binary()

    logs_dir = get_logs_dir()

    plist: dict[str, Any] = {
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": [mlx_stack_binary, "watch"],
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(logs_dir / "watchdog.stdout.log"),
        "StandardErrorPath": str(logs_dir / "watchdog.stderr.log"),
        "EnvironmentVariables": _build_environment_variables(mlx_stack_binary),
    }

    return plist


# --------------------------------------------------------------------------- #
# Plist file management
# --------------------------------------------------------------------------- #


def write_plist(plist_data: dict[str, Any], plist_path: Path | None = None) -> Path:
    """Write the plist dict to the LaunchAgents directory.

    Creates the ~/Library/LaunchAgents/ directory if it doesn't exist.
    Sets file permissions to 0o644.

    Args:
        plist_data: The plist dictionary to write.
        plist_path: Override path for testing. Defaults to get_plist_path().

    Returns:
        The path where the plist was written.

    Raises:
        LaunchdError: If the plist cannot be written.
    """
    if plist_path is None:
        plist_path = get_plist_path()

    try:
        # Create LaunchAgents directory if needed
        plist_path.parent.mkdir(parents=True, exist_ok=True)

        # Write plist using plistlib
        with open(plist_path, "wb") as f:
            plistlib.dump(plist_data, f)

        # Set permissions
        plist_path.chmod(PLIST_PERMISSIONS)

    except OSError as exc:
        msg = f"Failed to write plist to {plist_path}: {exc}"
        raise LaunchdError(msg) from None

    return plist_path


# --------------------------------------------------------------------------- #
# launchctl operations
# --------------------------------------------------------------------------- #


def _get_gui_uid() -> int:
    """Get the current user's UID for launchctl gui/ domain.

    Returns:
        The current user's UID.
    """
    return os.getuid()


def load_agent(plist_path: Path | None = None) -> None:
    """Load the watchdog agent via launchctl bootstrap.

    Runs: launchctl bootstrap gui/<uid> <plist_path>

    Args:
        plist_path: Path to the plist file. Defaults to get_plist_path().

    Raises:
        LaunchdError: If launchctl bootstrap fails.
    """
    if plist_path is None:
        plist_path = get_plist_path()

    uid = _get_gui_uid()
    cmd = ["launchctl", "bootstrap", f"gui/{uid}", str(plist_path)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            msg = f"launchctl bootstrap failed (exit {result.returncode}): {stderr}"
            raise LaunchdError(msg)
    except subprocess.TimeoutExpired:
        msg = "launchctl bootstrap timed out after 30 seconds"
        raise LaunchdError(msg) from None
    except FileNotFoundError:
        msg = "launchctl not found — is this macOS?"
        raise LaunchdError(msg) from None


def unload_agent(plist_path: Path | None = None) -> None:
    """Unload the watchdog agent via launchctl bootout.

    Runs: launchctl bootout gui/<uid> <plist_path>

    Args:
        plist_path: Path to the plist file. Defaults to get_plist_path().

    Raises:
        LaunchdError: If launchctl bootout fails.
    """
    if plist_path is None:
        plist_path = get_plist_path()

    uid = _get_gui_uid()
    cmd = ["launchctl", "bootout", f"gui/{uid}", str(plist_path)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        # bootout returns non-zero if the service isn't loaded;
        # we treat that as non-fatal since we're just trying to
        # ensure it's unloaded
        if result.returncode != 0:
            stderr = result.stderr.strip()
            # Error 3 = "No such process" (already unloaded) — non-fatal
            if "3:" not in stderr and "No such process" not in stderr:
                msg = f"launchctl bootout failed (exit {result.returncode}): {stderr}"
                raise LaunchdError(msg)
    except subprocess.TimeoutExpired:
        msg = "launchctl bootout timed out after 30 seconds"
        raise LaunchdError(msg) from None
    except FileNotFoundError:
        msg = "launchctl not found — is this macOS?"
        raise LaunchdError(msg) from None


# --------------------------------------------------------------------------- #
# Status checking
# --------------------------------------------------------------------------- #


def get_agent_status() -> AgentStatus:
    """Check the current status of the launchd agent.

    Checks:
    1. Whether the plist file exists (installed)
    2. Whether launchctl list shows the agent (running + PID)

    Returns:
        AgentStatus with installed, running, and pid fields.
    """
    plist_path = get_plist_path()
    installed = plist_path.exists()

    if not installed:
        return AgentStatus(installed=False, running=False, pid=None)

    # Check launchctl list for the agent
    pid = _get_agent_pid()

    return AgentStatus(
        installed=True,
        running=pid is not None,
        pid=pid,
    )


def _get_agent_pid() -> int | None:
    """Query launchctl for the agent's PID.

    Runs: launchctl list com.mlx-stack.watchdog

    Returns:
        The PID if the agent is loaded and running, None otherwise.
    """
    try:
        result = subprocess.run(
            ["launchctl", "list", LAUNCHD_LABEL],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        # Parse the output — launchctl list <label> produces a
        # key-value output. Look for the "PID" key.
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith('"PID"'):
                # Format: "PID" = <number>;
                parts = line.split("=")
                if len(parts) >= 2:
                    pid_str = parts[1].strip().rstrip(";").strip()
                    try:
                        return int(pid_str)
                    except ValueError:
                        return None

        return None

    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


# --------------------------------------------------------------------------- #
# High-level operations
# --------------------------------------------------------------------------- #


def install_agent(mlx_stack_binary: str | None = None) -> tuple[Path, bool]:
    """Install the watchdog as a launchd agent.

    Performs:
    1. Platform check (macOS only)
    2. Prerequisite check (init must have been run)
    3. Generate plist
    4. If already installed, bootout old agent
    5. Write new plist (with 0o644 permissions)
    6. Bootstrap new agent

    Args:
        mlx_stack_binary: Path to the mlx-stack binary (auto-resolved if None).

    Returns:
        Tuple of (plist_path, was_reinstall).

    Raises:
        PlatformError: If not on macOS.
        PrerequisiteError: If init has not been run.
        LaunchdError: If any launchd operation fails.
    """
    check_platform()
    check_init_prerequisite()

    # Ensure logs dir exists for stdout/stderr paths
    logs_dir = get_logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)

    plist_data = generate_plist(mlx_stack_binary)
    plist_path = get_plist_path()

    # Check if already installed
    was_reinstall = plist_path.exists()
    if was_reinstall:
        # Bootout old agent before writing new plist
        with contextlib.suppress(LaunchdError):
            unload_agent(plist_path)

    # Write new plist
    write_plist(plist_data, plist_path)

    # Bootstrap new agent
    load_agent(plist_path)

    return plist_path, was_reinstall


def uninstall_agent() -> bool:
    """Uninstall the watchdog launchd agent.

    Performs:
    1. Platform check (macOS only)
    2. Check if installed
    3. Bootout the agent
    4. Remove the plist file

    Returns:
        True if uninstalled, False if not installed.

    Raises:
        PlatformError: If not on macOS.
        LaunchdError: If launchctl bootout fails.
    """
    check_platform()

    plist_path = get_plist_path()
    if not plist_path.exists():
        return False

    # Bootout the agent — only suppress "No such process" (already unloaded)
    try:
        unload_agent(plist_path)
    except LaunchdError as exc:
        # "No such process" means the agent wasn't loaded — safe to ignore.
        # Any other launchctl error is fatal and should propagate.
        err_msg = str(exc)
        if "No such process" not in err_msg and "3:" not in err_msg:
            raise

    # Remove plist file
    try:
        plist_path.unlink()
    except OSError as exc:
        msg = f"Failed to remove plist file {plist_path}: {exc}"
        raise LaunchdError(msg) from None

    return True
