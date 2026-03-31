"""CLI command for the watchdog health monitor — `mlx-stack watch`.

Long-running health monitor that polls service status, auto-restarts
crashed services, detects flapping, and triggers log rotation.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

from mlx_stack.core.watchdog import (
    DEFAULT_INTERVAL,
    DEFAULT_MAX_RESTARTS,
    DEFAULT_RESTART_DELAY,
    PollResult,
    RestartRecord,
    WatchdogError,
    WatchdogState,
    run_watchdog,
)

console = Console(stderr=True)

# --------------------------------------------------------------------------- #
# Status color map
# --------------------------------------------------------------------------- #

_STATUS_STYLES: dict[str, str] = {
    "healthy": "green",
    "degraded": "yellow",
    "down": "red",
    "crashed": "bold red",
    "stopped": "dim",
}


# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #


def _format_status_table(result: PollResult, state: WatchdogState) -> None:
    """Print a Rich-formatted status table for a poll cycle.

    Args:
        result: The poll cycle result.
        state: Current watchdog state.
    """
    out = Console()
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    out.print()
    out.print(Text(f"[Cycle {state.cycle_count}] {now}", style="bold cyan"))

    table = Table(
        show_header=True,
        header_style="bold",
    )
    table.add_column("Service", style="bold", min_width=12)
    table.add_column("Status", min_width=10)
    table.add_column("Port", justify="right", min_width=6)

    for svc in result.statuses:
        style = _STATUS_STYLES.get(svc.status, "")
        status_text = Text(svc.status, style=style)

        # Annotate flapping services
        if svc.tier in result.flapping_services:
            status_text.append(" [FLAPPING]", style="bold magenta")

        table.add_row(
            svc.tier,
            status_text,
            str(svc.port),
        )

    out.print(table)

    # Show rotation info
    if result.rotations_performed > 0:
        out.print(
            Text(
                f"  Log rotation: {result.rotations_performed} file(s) rotated",
                style="dim",
            )
        )

    # Show restart summary
    if result.restarts_attempted > 0:
        out.print(
            Text(
                f"  Restarts: {result.restarts_succeeded}/{result.restarts_attempted} succeeded",
                style=(
                    "yellow"
                    if result.restarts_succeeded < result.restarts_attempted
                    else "green"
                ),
            )
        )


def _format_restart_event(record: RestartRecord) -> None:
    """Print a Rich-formatted restart event.

    Args:
        record: The restart record.
    """
    out = Console()
    ts = datetime.fromtimestamp(record.timestamp, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    status = "✓" if record.success else "✗"
    style = "green" if record.success else "red"

    out.print(
        Text(
            f"  [{status}] {ts} Restart {record.service} "
            f"(attempt #{record.attempt}): {record.reason}",
            style=style,
        )
    )


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


def _validate_positive_int(
    ctx: click.Context,
    param: click.Parameter,
    value: int,
) -> int:
    """Validate that a value is a positive integer.

    Args:
        ctx: Click context.
        param: Click parameter.
        value: The value to validate.

    Returns:
        The validated value.

    Raises:
        click.BadParameter: If value is not positive.
    """
    if value < 1:
        raise click.BadParameter(
            f"Must be a positive integer (got {value})."
        )
    return value


# --------------------------------------------------------------------------- #
# CLI command
# --------------------------------------------------------------------------- #


@click.command()
@click.option(
    "--interval",
    type=int,
    default=DEFAULT_INTERVAL,
    show_default=True,
    callback=_validate_positive_int,
    help="Seconds between health polls.",
)
@click.option(
    "--max-restarts",
    type=int,
    default=DEFAULT_MAX_RESTARTS,
    show_default=True,
    callback=_validate_positive_int,
    help="Maximum restarts before marking service as flapping.",
)
@click.option(
    "--restart-delay",
    type=int,
    default=DEFAULT_RESTART_DELAY,
    show_default=True,
    callback=_validate_positive_int,
    help="Base delay (seconds) before restart with exponential backoff.",
)
@click.option(
    "--daemon",
    is_flag=True,
    default=False,
    help="Run in background as a daemon.",
)
def watch(
    interval: int,
    max_restarts: int,
    restart_delay: int,
    daemon: bool,
) -> None:
    """Monitor stack health and auto-restart crashed services.

    Polls service health at regular intervals (default 30s) and
    auto-restarts any crashed services (PID file exists, process dead).
    Services that are intentionally stopped (no PID file) are NOT
    restarted.

    \b
    Features:
      • Rich-formatted status display each poll cycle
      • Auto-restart of crashed services
      • Flap detection (stops restarting after threshold)
      • Exponential backoff on restart delay
      • Log rotation each poll cycle
      • Daemon mode with PID file

    \b
    Examples:
        mlx-stack watch                     Start monitoring (foreground)
        mlx-stack watch --daemon            Start as background daemon
        mlx-stack watch --interval 60       Poll every 60 seconds
        mlx-stack watch --max-restarts 3    Flap after 3 restarts
    """
    try:
        if daemon:
            console.print(
                Text("Starting watchdog in daemon mode...", style="bold cyan")
            )

        run_watchdog(
            interval=interval,
            max_restarts=max_restarts,
            restart_delay=restart_delay,
            daemon=daemon,
            status_callback=_format_status_table,
            restart_callback=_format_restart_event,
        )

        # Clean exit
        if not daemon:
            out = Console()
            out.print()
            out.print(Text("Watchdog stopped.", style="bold cyan"))

    except WatchdogError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        out = Console()
        out.print()
        out.print(Text("Watchdog stopped.", style="bold cyan"))
