"""CLI command for starting services — `mlx-stack up`.

Starts all services defined in the active stack, or a single tier
with --tier. Supports --dry-run to preview commands without executing.
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

from mlx_stack.core.process import LockError
from mlx_stack.core.stack_up import UpError, UpResult, run_up

console = Console(stderr=True)


def _display_dry_run(result: UpResult) -> None:
    """Display dry-run commands.

    Shows the exact shell commands that would be executed for each
    vllm-mlx instance and LiteLLM without actually running them.

    Args:
        result: The UpResult from the dry-run.
    """
    out = Console()
    out.print()
    out.print(Text("Dry run — commands that would be executed:", style="bold cyan"))
    out.print()

    for cmd_info in result.dry_run_commands:
        service = cmd_info["service"]
        command = cmd_info["command"]
        svc_type = cmd_info["type"]

        label = f"[bold]{service}[/bold]" if svc_type == "vllm-mlx" else "[bold]litellm[/bold]"
        out.print(f"  {label}:")
        out.print(f"    [green]{command}[/green]")
        out.print()


def _display_summary(result: UpResult) -> None:
    """Display a summary table of service statuses.

    Shows tier name, model, port, and status for each service plus
    LiteLLM.

    Args:
        result: The UpResult from startup.
    """
    out = Console()
    out.print()

    if result.already_running:
        out.print(Text("All services are already running.", style="bold yellow"))
        out.print()

    # Warnings
    for warning in result.warnings:
        out.print(f"[yellow]⚠ {warning}[/yellow]")

    if result.warnings:
        out.print()

    # Summary table
    table = Table(
        title="Service Summary",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Service", style="bold", min_width=12)
    table.add_column("Model", min_width=20)
    table.add_column("Port", justify="right", min_width=6)
    table.add_column("Status", min_width=10)

    # Status styling
    status_styles = {
        "healthy": "[bold green]healthy[/bold green]",
        "already-running": "[bold green]already-running[/bold green]",
        "failed": "[bold red]failed[/bold red]",
        "skipped": "[yellow]skipped[/yellow]",
        "dry-run": "[cyan]dry-run[/cyan]",
    }

    for tier in result.tiers:
        status_display = status_styles.get(tier.status, tier.status)
        if tier.error:
            status_display += f"\n[dim]{tier.error}[/dim]"
        table.add_row(
            tier.name,
            tier.model,
            str(tier.port),
            status_display,
        )

    # LiteLLM row
    if result.litellm:
        litellm = result.litellm
        status_display = status_styles.get(litellm.status, litellm.status)
        if litellm.error:
            status_display += f"\n[dim]{litellm.error}[/dim]"
        table.add_row(
            litellm.name,
            litellm.model,
            str(litellm.port),
            status_display,
        )

    out.print(table)
    out.print()

    # Next steps for healthy stacks
    any_healthy = any(t.status in ("healthy", "already-running") for t in result.tiers)
    if any_healthy:
        litellm_port = result.litellm.port if result.litellm else 4000
        out.print(f"[dim]Endpoint: http://localhost:{litellm_port}/v1[/dim]")
        out.print()


@click.command()
@click.option("--dry-run", is_flag=True, help="Show commands without executing.")
@click.option("--tier", "tier_filter", type=str, help="Start only the specified tier.")
def up(dry_run: bool, tier_filter: str | None) -> None:
    """Start all services in the active stack.

    Reads the stack definition from ~/.mlx-stack/stacks/default.yaml
    and starts one vllm-mlx subprocess per tier plus a LiteLLM proxy.

    Use --dry-run to preview the exact commands without starting anything.
    Use --tier to start only a specific tier (plus LiteLLM if needed).
    """
    try:
        result = run_up(
            dry_run=dry_run,
            tier_filter=tier_filter,
        )
    except UpError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None
    except LockError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None

    if result.dry_run:
        _display_dry_run(result)
    else:
        _display_summary(result)

    # Exit with non-zero if all tiers failed
    any_success = any(t.status in ("healthy", "already-running", "dry-run") for t in result.tiers)
    if not any_success and not result.dry_run:
        raise SystemExit(1)
