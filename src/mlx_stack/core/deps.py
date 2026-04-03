"""Dependency management module for mlx-stack.

Checks for and auto-installs pinned versions of vllm-mlx and litellm
as uv tools. Performs PATH lookup to detect installed tools, auto-installs
via ``uv tool install <tool>==<version>`` when missing, shows progress
during installation, verifies post-install availability, detects version
mismatches with warnings, and provides clear manual install instructions
on failure.

Read-only commands (profile, config, recommend, models) do NOT trigger
dependency checks.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass

from rich.console import Console

# --------------------------------------------------------------------------- #
# Pinned versions
# --------------------------------------------------------------------------- #

PINNED_VERSIONS: dict[str, str] = {
    "vllm-mlx": "0.2.6",
    "litellm": "1.83.0",  # first post-CVE-2026-33634 verified release
}

# Extras required for specific tools (e.g. litellm needs [proxy] for the
# proxy server dependencies like backoff, prisma, etc.)
_TOOL_EXTRAS: dict[str, str] = {
    "litellm": "proxy",
}

# Map tool name to the CLI binary name used for PATH lookup
_TOOL_BINARY_MAP: dict[str, str] = {
    "vllm-mlx": "vllm-mlx",
    "litellm": "litellm",
}

# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class DependencyError(Exception):
    """Raised when a required dependency cannot be found or installed."""


class DependencyInstallError(DependencyError):
    """Raised when ``uv tool install`` fails."""


class DependencyVersionMismatchWarning(Exception):
    """Sentinel for version mismatch warnings (not raised, used for typing)."""


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DependencyStatus:
    """Status of a single managed dependency.

    Attributes:
        name: Package name (e.g. ``vllm-mlx``).
        pinned_version: The version we want installed.
        installed: Whether the tool binary is available on PATH.
        installed_version: The detected installed version, or ``None``.
        version_match: ``True`` when installed version matches pinned.
    """

    name: str
    pinned_version: str
    installed: bool
    installed_version: str | None
    version_match: bool | None


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

_console = Console(stderr=True)


def _build_install_spec(tool: str, version: str) -> str:
    """Build the pip/uv install specifier for a tool, including extras."""
    extra = _TOOL_EXTRAS.get(tool)
    if extra:
        return f"{tool}[{extra}]=={version}"
    return f"{tool}=={version}"


def _find_binary(tool: str) -> str | None:
    """Return the full path to a tool's binary, or ``None`` if not found.

    Uses :func:`shutil.which` for PATH lookup.
    """
    binary = _TOOL_BINARY_MAP.get(tool, tool)
    return shutil.which(binary)


def _get_installed_version(tool: str) -> str | None:
    """Attempt to determine the installed version of a uv tool.

    Runs ``uv tool list`` and parses the output for the tool name and
    version.  Returns ``None`` when the tool is not found or the output
    cannot be parsed.
    """
    uv_path = shutil.which("uv")
    if uv_path is None:
        return None

    try:
        env = {**os.environ, "NO_COLOR": "1"}
        result = subprocess.run(
            [uv_path, "tool", "list"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None

    if result.returncode != 0:
        return None

    # Parse lines like "vllm-mlx v0.2.6" or "litellm v1.83.0"
    for line in result.stdout.splitlines():
        # Match "<tool> v<version>" pattern
        pattern = rf"^{re.escape(tool)}\s+v?(\S+)"
        match = re.match(pattern, line.strip())
        if match:
            return match.group(1)

    return None


def _install_tool(tool: str, version: str) -> None:
    """Install a tool at a pinned version via ``uv tool install``.

    Args:
        tool: Package name (e.g. ``vllm-mlx``).
        version: Exact version to install.

    Raises:
        DependencyInstallError: When ``uv tool install`` fails.
        DependencyError: When ``uv`` itself is not available.
    """
    uv_path = shutil.which("uv")
    if uv_path is None:
        msg = (
            "uv is not available on PATH. Install it from https://docs.astral.sh/uv/ and try again."
        )
        raise DependencyError(msg)

    install_spec = _build_install_spec(tool, version)
    cmd = [uv_path, "tool", "install", install_spec]
    cmd_str = f"uv tool install {install_spec}"

    _console.print(f"[cyan]Installing {tool} v{version}...[/cyan]")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        msg = f"Installation timed out: {cmd_str}\n\nInstall manually with: {cmd_str}"
        raise DependencyInstallError(msg) from None
    except OSError as exc:
        msg = f"Failed to run: {cmd_str}\nError: {exc}\n\nInstall manually with: {cmd_str}"
        raise DependencyInstallError(msg) from None

    if result.returncode != 0:
        stderr = result.stderr.strip()
        msg = (
            f"Failed to install {tool}:\n"
            f"  Command: {cmd_str}\n"
            f"  Error: {stderr}\n\n"
            f"Install manually with: {cmd_str}"
        )
        raise DependencyInstallError(msg)

    _console.print(f"[green]✓ {tool} v{version} installed successfully.[/green]")


def _verify_post_install(tool: str) -> bool:
    """Verify that a tool is available on PATH after installation.

    Returns ``True`` if the binary is found.
    """
    return _find_binary(tool) is not None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def check_dependency(tool: str) -> DependencyStatus:
    """Check the installation status of a single dependency.

    Performs PATH lookup and version detection without installing anything.

    Args:
        tool: Package name (must be a key in :data:`PINNED_VERSIONS`).

    Returns:
        A :class:`DependencyStatus` with installation details.

    Raises:
        ValueError: If *tool* is not a known dependency.
    """
    if tool not in PINNED_VERSIONS:
        msg = f"Unknown dependency '{tool}'. Known: {', '.join(sorted(PINNED_VERSIONS))}"
        raise ValueError(msg)

    pinned = PINNED_VERSIONS[tool]
    binary_path = _find_binary(tool)
    installed = binary_path is not None

    installed_version: str | None = None
    version_match: bool | None = None

    if installed:
        installed_version = _get_installed_version(tool)
        if installed_version is not None:
            version_match = installed_version == pinned

    return DependencyStatus(
        name=tool,
        pinned_version=pinned,
        installed=installed,
        installed_version=installed_version,
        version_match=version_match,
    )


def ensure_dependency(tool: str) -> DependencyStatus:
    """Ensure a dependency is installed at the pinned version.

    1. Check if the tool is already available on PATH.
    2. If missing, auto-install via ``uv tool install <tool>==<version>``.
    3. Verify post-install availability.
    4. Warn on version mismatch (but do not block).

    Args:
        tool: Package name (must be a key in :data:`PINNED_VERSIONS`).

    Returns:
        A :class:`DependencyStatus` reflecting the final state.

    Raises:
        ValueError: If *tool* is not a known dependency.
        DependencyError: If ``uv`` is not available.
        DependencyInstallError: If auto-install fails.
    """
    status = check_dependency(tool)

    if not status.installed:
        # Auto-install
        _install_tool(tool, status.pinned_version)

        # Verify post-install
        if not _verify_post_install(tool):
            cmd_str = f"uv tool install {_build_install_spec(tool, status.pinned_version)}"
            msg = (
                f"{tool} was not found on PATH after installation.\n"
                f"This may be because the uv tool bin directory is not in your PATH.\n\n"
                f"Try running:\n"
                f"  {cmd_str}\n"
                f'  export PATH="$HOME/.local/bin:$PATH"'
            )
            raise DependencyInstallError(msg)

        # Re-check status after install
        status = check_dependency(tool)

    # Warn on version mismatch
    if status.installed and status.version_match is False:
        _warn_version_mismatch(tool, status)

    return status


def ensure_all_dependencies() -> list[DependencyStatus]:
    """Ensure all managed dependencies are installed.

    Calls :func:`ensure_dependency` for each tool in :data:`PINNED_VERSIONS`.

    Returns:
        A list of :class:`DependencyStatus` for each dependency.

    Raises:
        DependencyError: If any dependency cannot be installed.
        DependencyInstallError: If any auto-install fails.
    """
    return [ensure_dependency(tool) for tool in PINNED_VERSIONS]


def _warn_version_mismatch(tool: str, status: DependencyStatus) -> None:
    """Display a Rich warning about a version mismatch.

    Args:
        tool: The package name.
        status: The dependency status with version info.
    """
    pinned = status.pinned_version
    installed = status.installed_version or "unknown"
    cmd = f"uv tool install {_build_install_spec(tool, pinned)}"
    _console.print(
        f"[yellow]⚠ {tool} version mismatch: "
        f"installed v{installed}, expected v{pinned}.[/yellow]\n"
        f"  Upgrade/downgrade with: {cmd}"
    )
