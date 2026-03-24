"""CLI command for service status — `mlx-stack status`.

Displays the health and metrics for all managed services in a
formatted Rich table or as JSON (with --json). Read-only: does not
modify any files or acquire the lockfile.
"""

from __future__ import annotations

import json

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

from mlx_stack.core.stack_status import StatusResult, run_status, status_to_dict

console = Console(stderr=True)

# Status display styling — maps state to Rich markup
_STATUS_STYLES: dict[str, str] = {
    "healthy": "[bold green]healthy[/bold green]",
    "degraded": "[bold yellow]degraded[/bold yellow]",
    "down": "[bold red]down[/bold red]",
    "crashed": "[bold red]crashed[/bold red]",
    "stopped": "[dim]stopped[/dim]",
}


def _display_table(result: StatusResult) -> None:
    """Display service statuses as a Rich table.

    Columns: Tier, Model, Port, Status, Uptime.

    Args:
        result: The StatusResult to display.
    """
    out = Console()
    out.print()

    table = Table(
        title="Service Status",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Tier", style="bold", min_width=12)
    table.add_column("Model", min_width=20)
    table.add_column("Port", justify="right", min_width=6)
    table.add_column("Status", min_width=10)
    table.add_column("Uptime", justify="right", min_width=10)

    for svc in result.services:
        status_display = _STATUS_STYLES.get(svc.status, svc.status)
        table.add_row(
            svc.tier,
            svc.model,
            str(svc.port),
            status_display,
            svc.uptime_display,
        )

    out.print(table)
    out.print()


def _display_json(result: StatusResult) -> None:
    """Display service statuses as JSON to stdout.

    Args:
        result: The StatusResult to display.
    """
    data = status_to_dict(result)
    click.echo(json.dumps(data, indent=2))


@click.command()
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format.")
def status(json_output: bool) -> None:
    """Show health and status of all services.

    Reports the current state of each managed service: healthy, degraded,
    down, crashed, or stopped. Displays a formatted table by default, or
    valid JSON with --json.

    This command is read-only and safe to run concurrently with other
    mlx-stack commands.
    """
    result = run_status()

    # Handle no-stack scenario
    if result.no_stack:
        if json_output:
            _display_json(result)
        else:
            out = Console()
            out.print()
            out.print(
                Text(
                    result.message or "No stack configured — run 'mlx-stack init'.",
                    style="yellow",
                )
            )
            out.print()
        return

    # Display results
    if json_output:
        _display_json(result)
    else:
        _display_table(result)
