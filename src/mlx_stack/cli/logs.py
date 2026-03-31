"""CLI command for viewing and managing service logs — `mlx-stack logs`.

View log content, follow in real-time, list available log files,
trigger on-demand rotation, and view archived logs.
"""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

from mlx_stack.core.log_viewer import (
    DEFAULT_TAIL_LINES,
    follow_log,
    get_available_services,
    get_log_path,
    list_log_files,
    read_all_logs,
    read_log_tail,
    rotate_all_logs,
    rotate_service_log,
)

console = Console(stderr=True)


def _display_log_list() -> None:
    """Display a Rich table listing all available log files."""
    log_files = list_log_files()

    out = Console()
    out.print()

    if not log_files:
        out.print(
            Text(
                "No log files found in logs directory.",
                style="yellow",
            )
        )
        out.print()
        return

    table = Table(
        title="Available Log Files",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Name", style="bold", min_width=20)
    table.add_column("Size", justify="right", min_width=10)
    table.add_column("Modified", min_width=20)

    for info in log_files:
        table.add_row(info.name, info.size_display, info.modified_display)

    out.print(table)
    out.print()


def _display_rotation_results(results: list) -> None:
    """Display rotation results with Rich formatting.

    Args:
        results: List of RotationResult objects.
    """
    out = Console()
    out.print()

    any_rotated = False
    for result in results:
        if result.error:
            out.print(f"[red]✗[/red] {result.service}: {result.error}")
        elif result.rotated:
            out.print(f"[green]✓[/green] {result.service}: rotated")
            any_rotated = True
        else:
            out.print(f"[dim]–[/dim] {result.service}: no rotation needed")

    if not results:
        out.print(Text("No log files found to rotate.", style="yellow"))
    elif not any_rotated and not any(r.error for r in results):
        out.print()
        out.print(Text("No rotation needed.", style="dim"))

    out.print()


@click.command()
@click.argument("service", required=False, default=None)
@click.option(
    "--tail",
    "tail_lines",
    type=int,
    default=None,
    help=f"Show last N lines (default {DEFAULT_TAIL_LINES}).",
)
@click.option(
    "--follow",
    "-f",
    is_flag=True,
    default=False,
    help="Follow log output in real-time (tail -f behavior).",
)
@click.option(
    "--service",
    "service_filter",
    type=str,
    default=None,
    help="Filter to a specific service's log.",
)
@click.option(
    "--rotate",
    is_flag=True,
    default=False,
    help="Rotate eligible log files.",
)
@click.option(
    "--all",
    "show_all",
    is_flag=True,
    default=False,
    help="Show archived and current logs in chronological order.",
)
def logs(
    service: str | None,
    tail_lines: int | None,
    follow: bool,
    service_filter: str | None,
    rotate: bool,
    show_all: bool,
) -> None:
    """View and manage service logs.

    Without arguments, lists all available log files with sizes and
    modification times.

    \b
    Examples:
        mlx-stack logs                  List all log files
        mlx-stack logs fast             Show last 50 lines of fast.log
        mlx-stack logs fast --tail 100  Show last 100 lines
        mlx-stack logs fast --follow    Stream new log content in real-time
        mlx-stack logs --rotate         Rotate all eligible log files
        mlx-stack logs fast --all       Show archived + current logs
    """
    # Resolve effective service name: argument takes precedence over --service
    effective_service = service or service_filter

    # Handle --rotate mode
    if rotate:
        if effective_service:
            # Rotate only the specified service
            available = get_available_services()
            if available and effective_service not in available:
                console.print(
                    f"[red]Error:[/red] No log file found for service "
                    f"'{effective_service}'. "
                    f"Available services: {', '.join(available)}"
                )
                sys.exit(1)

            result = rotate_service_log(effective_service)
            _display_rotation_results([result])
        else:
            # Rotate all eligible logs
            results = rotate_all_logs()
            _display_rotation_results(results)
        return

    # No service specified — list log files
    if effective_service is None:
        _display_log_list()
        return

    # Validate service name
    log_path = get_log_path(effective_service)
    if log_path is None:
        available = get_available_services()
        if available:
            console.print(
                f"[red]Error:[/red] No log file found for service "
                f"'{effective_service}'. "
                f"Available services: {', '.join(available)}"
            )
        else:
            console.print(
                f"[red]Error:[/red] No log file found for service "
                f"'{effective_service}'. No log files exist."
            )
        sys.exit(1)

    # Handle --all mode: show archives + current
    if show_all:
        content = read_all_logs(effective_service)
        if content:
            click.echo(content)
        else:
            console.print(
                Text(
                    f"No log content found for service '{effective_service}'.",
                    style="yellow",
                )
            )
        return

    # Handle --follow mode
    if follow:
        num = tail_lines if tail_lines is not None else DEFAULT_TAIL_LINES
        try:
            follow_log(log_path, num_lines=num, output_callback=click.echo)
        except KeyboardInterrupt:
            # Belt-and-suspenders: ensure clean exit
            pass
        return

    # Default: show tail of log
    num = tail_lines if tail_lines is not None else DEFAULT_TAIL_LINES
    content = read_log_tail(log_path, num_lines=num)
    if content:
        click.echo(content)
    else:
        console.print(
            Text(
                f"Log file for service '{effective_service}' is empty.",
                style="yellow",
            )
        )
