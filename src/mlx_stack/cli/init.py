"""CLI command for stack initialization — `mlx-stack init`.

Generates stack definition and LiteLLM configuration files from a
hardware profile and recommendation. Supports --accept-defaults for
non-interactive mode, --intent, --add/--remove for customization,
and --force for overwriting existing configs.
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

from mlx_stack.core.stack_init import InitError, run_init

console = Console(stderr=True)


def _display_summary(result: dict) -> None:
    """Display a summary of the generated configuration.

    Shows file paths, tier assignments, total estimated memory,
    and next-step instructions.

    Args:
        result: The result dict from run_init.
    """
    out = Console()
    stack = result["stack"]
    profile = result["profile"]

    out.print()
    out.print(Text("✅ Stack initialized successfully!", style="bold green"))
    out.print()

    # File paths
    out.print(Text("Generated files:", style="bold"))
    out.print(f"  Stack:   {result['stack_path']}")
    out.print(f"  LiteLLM: {result['litellm_path']}")
    out.print()

    # Tier assignments table
    out.print(Text("Tier assignments:", style="bold"))
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Tier", style="bold", min_width=12)
    table.add_column("Model", min_width=20)
    table.add_column("Quant", min_width=6)
    table.add_column("Port", justify="right", min_width=6)

    for tier in stack["tiers"]:
        table.add_row(
            tier["name"],
            tier["model"],
            tier["quant"],
            str(tier["port"]),
        )

    out.print(table)

    # Hardware and memory summary
    out.print()
    budget_gb = result["memory_budget_gb"]
    total_memory_gb = result.get("total_memory_gb", 0.0)
    out.print(
        f"[dim]Hardware: {profile.chip} ({profile.memory_gb} GB) · "
        f"Budget: {budget_gb:.1f} GB[/dim]"
    )
    if total_memory_gb > 0:
        out.print(
            f"[dim]Total estimated memory: {total_memory_gb:.1f} GB[/dim]"
        )

    # Warnings (e.g., memory budget exceeded with --add)
    init_warnings = result.get("warnings", [])
    if init_warnings:
        out.print()
        for warning in init_warnings:
            out.print(f"[yellow]⚠ {warning}[/yellow]")

    # Cloud fallback indicator
    if stack.get("cloud_fallback"):
        out.print()
        out.print(
            "[bold green]☁ Cloud Fallback[/bold green]  "
            "Premium tier via OpenRouter configured"
        )

    # Missing models warning
    missing = result.get("missing_models", [])
    if missing:
        out.print()
        out.print("[yellow]⚠ Missing local models:[/yellow]")
        for model_id in missing:
            out.print(f"  • {model_id}")
        out.print()
        out.print(
            "  Run [bold]mlx-stack pull[/bold] to download missing models "
            "before starting the stack."
        )

    # Next steps
    out.print()
    out.print(Text("Next steps:", style="bold"))
    if missing:
        out.print("  1. [bold]mlx-stack pull[/bold]  — Download missing models")
        out.print("  2. [bold]mlx-stack up[/bold]    — Start all services")
    else:
        out.print("  1. [bold]mlx-stack up[/bold]  — Start all services")
    out.print()


@click.command()
@click.option(
    "--accept-defaults",
    is_flag=True,
    default=False,
    help="Use defaults without prompting (balanced intent, default budget).",
)
@click.option(
    "--intent",
    type=str,
    default=None,
    help="Recommendation intent: balanced (default) or agent-fleet.",
)
@click.option(
    "--add",
    "add_models",
    multiple=True,
    help="Add a model to the stack (can be specified multiple times).",
)
@click.option(
    "--remove",
    "remove_tiers",
    multiple=True,
    help="Remove a tier from the stack (can be specified multiple times).",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing stack configuration.",
)
def init(
    accept_defaults: bool,
    intent: str | None,
    add_models: tuple[str, ...],
    remove_tiers: tuple[str, ...],
    force: bool,
) -> None:
    """Generate stack definition and LiteLLM config.

    Creates ~/.mlx-stack/stacks/default.yaml with tier assignments
    and ~/.mlx-stack/litellm.yaml with proxy configuration.

    Use --accept-defaults for non-interactive mode. Combine with
    --intent to specify the optimization strategy.

    Use --add to include additional models and --remove to exclude
    specific tiers from the default recommendation.

    Requires --force to overwrite an existing stack configuration.
    """
    # Default intent
    if intent is None:
        intent = "balanced"

    try:
        result = run_init(
            intent=intent,
            add_models=list(add_models) if add_models else None,
            remove_tiers=list(remove_tiers) if remove_tiers else None,
            force=force,
        )
    except InitError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None

    _display_summary(result)
