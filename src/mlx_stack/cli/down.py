"""CLI command for stopping services — `mlx-stack down`.

Stops all managed services, or a single tier with --tier.
Reports per-service shutdown results including graceful vs forced.
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

from mlx_stack.core.process import LockError
from mlx_stack.core.stack_down import DownError, DownResult, run_down

console = Console(stderr=True)


def _display_results(result: DownResult) -> None:
    """Display shutdown results.

    Shows a table of stopped services with per-service shutdown method
    (graceful vs forced) and cleanup information.

    Args:
        result: The DownResult from shutdown.
    """
    out = Console()
    out.print()

    # Warnings
    for warning in result.warnings:
        out.print(f"[yellow]⚠ {warning}[/yellow]")

    if result.warnings:
        out.print()

    # Results table
    table = Table(
        title="Shutdown Summary",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Service", style="bold", min_width=12)
    table.add_column("PID", justify="right", min_width=8)
    table.add_column("Status", min_width=14)
    table.add_column("Method", min_width=10)

    for svc in result.services:
        pid_str = str(svc.pid) if svc.pid is not None else "-"

        # Status styling
        if svc.status == "stopped":
            status_display = "[bold green]stopped[/bold green]"
        elif svc.status == "stale":
            status_display = "[yellow]stale (cleaned)[/yellow]"
        elif svc.status == "corrupt":
            status_display = "[yellow]corrupt (cleaned)[/yellow]"
        elif svc.status == "not-running":
            status_display = "[dim]not running[/dim]"
        else:
            status_display = svc.status

        # Method display
        if svc.graceful is True:
            method_display = "[green]graceful (SIGTERM)[/green]"
        elif svc.graceful is False:
            method_display = "[red]forced (SIGKILL)[/red]"
        else:
            method_display = "-"

        table.add_row(svc.name, pid_str, status_display, method_display)

    out.print(table)
    out.print()


@click.command()
@click.option("--tier", "tier_filter", type=str, help="Stop only the specified tier.")
def down(tier_filter: str | None) -> None:
    """Stop all managed services.

    Terminates processes in correct order: LiteLLM first, then model
    servers in reverse startup order. Uses SIGTERM with a 10-second
    grace period, escalating to SIGKILL if needed.

    Use --tier to stop only a specific tier while leaving others running.
    """
    try:
        result = run_down(
            tier_filter=tier_filter,
        )
    except DownError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None
    except LockError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None

    if result.nothing_to_stop:
        out = Console()
        out.print()
        out.print(Text("Nothing to stop.", style="bold yellow"))
        out.print("[dim]No PID files found — no managed services are running.[/dim]")
        out.print()
        return

    _display_results(result)
