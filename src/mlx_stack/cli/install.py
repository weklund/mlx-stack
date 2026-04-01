"""CLI commands for launchd integration — `mlx-stack install` and `mlx-stack uninstall`.

Manages the watchdog as a macOS LaunchAgent, enabling automatic startup
on login and persistent health monitoring.
"""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.text import Text

from mlx_stack.core.launchd import (
    AgentStatus,
    LaunchdError,
    PlatformError,
    PrerequisiteError,
    get_agent_status,
    install_agent,
    uninstall_agent,
)

console = Console(stderr=True)


# --------------------------------------------------------------------------- #
# Status display helper
# --------------------------------------------------------------------------- #


def _display_status(status: AgentStatus) -> None:
    """Display the agent status in Rich format.

    Args:
        status: The current agent status.
    """
    out = Console()

    if not status.installed:
        out.print(Text("Status: not installed", style="dim"))
    elif status.running and status.pid is not None:
        out.print(
            Text(f"Status: installed and running (PID {status.pid})", style="green")
        )
    else:
        out.print(Text("Status: installed but not running", style="yellow"))


# --------------------------------------------------------------------------- #
# install command
# --------------------------------------------------------------------------- #


@click.command()
@click.option(
    "--status",
    "show_status",
    is_flag=True,
    default=False,
    help="Show the current launchd agent status without installing.",
)
def install(show_status: bool) -> None:
    """Install the watchdog as a macOS LaunchAgent.

    Generates a launchd plist that runs `mlx-stack watch` automatically
    on login. The watchdog monitors stack health and auto-restarts
    crashed services.

    \b
    Requires:
      • macOS (launchd is macOS-only)
      • mlx-stack init (stack must be configured first)

    \b
    Behavior:
      • Creates plist at ~/Library/LaunchAgents/com.mlx-stack.watchdog.plist
      • Loads the agent immediately via launchctl bootstrap
      • If already installed, updates the plist and reloads

    \b
    Examples:
        mlx-stack install              Install and start the watchdog agent
        mlx-stack install --status     Check if the agent is installed
    """
    if show_status:
        try:
            status = get_agent_status()
            _display_status(status)
        except PlatformError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            sys.exit(1)
        return

    try:
        plist_path, was_reinstall = install_agent()

        out = Console()
        if was_reinstall:
            out.print(Text("Watchdog agent updated and reloaded.", style="bold green"))
        else:
            out.print(Text("Watchdog agent installed successfully.", style="bold green"))

        out.print(Text(f"  Plist: {plist_path}", style="dim"))
        out.print(
            Text(
                "  The watchdog will start automatically on login.",
                style="dim",
            )
        )

    except PlatformError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    except PrerequisiteError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    except LaunchdError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


# --------------------------------------------------------------------------- #
# uninstall command
# --------------------------------------------------------------------------- #


@click.command()
def uninstall() -> None:
    """Uninstall the watchdog LaunchAgent.

    Stops the watchdog agent and removes the launchd plist. Running
    services are not affected — only the watchdog auto-monitoring
    is disabled.

    \b
    Examples:
        mlx-stack uninstall            Remove the watchdog agent
    """
    try:
        removed = uninstall_agent()

        out = Console()
        if removed:
            out.print(Text("Watchdog agent uninstalled.", style="bold green"))
            out.print(
                Text(
                    "  Running services are not affected.",
                    style="dim",
                )
            )
        else:
            out.print(Text("Watchdog agent is not installed.", style="yellow"))

    except PlatformError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    except LaunchdError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
