"""CLI command for model listing — `mlx-stack models`.

Lists locally downloaded models with disk size, quantization, and source type.
Active stack models are marked with a visual indicator. The --catalog flag
shows all 15 catalog models with hardware-specific benchmark data.

Output is formatted as a Rich table with human-readable names.
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

from mlx_stack.core.catalog import (
    CatalogError,
    load_catalog,
    query_by_capability,
    query_by_family,
    query_by_tag,
)
from mlx_stack.core.hardware import load_profile
from mlx_stack.core.models import (
    ModelsError,
    format_size,
    get_models_directory,
    get_remote_stack_models,
    list_catalog_models,
    scan_local_models,
)

console = Console(stderr=True)


# --------------------------------------------------------------------------- #
# Local models display
# --------------------------------------------------------------------------- #


def _display_local_models() -> None:
    """Display locally downloaded models in a Rich table."""
    out = Console()

    models_dir = get_models_directory()
    try:
        catalog = load_catalog()
    except CatalogError:
        catalog = []

    local_models = scan_local_models(models_dir=models_dir, catalog=catalog)
    remote_models = get_remote_stack_models(local_models=local_models, catalog=catalog)

    if not local_models and not remote_models:
        out.print()
        out.print(
            "[yellow]No models found.[/yellow] "
            "Run [bold]mlx-stack pull[/bold] to download a model, "
            "or [bold]mlx-stack init[/bold] to set up a stack."
        )
        out.print()
        return

    out.print()
    out.print(Text("Local Models", style="bold cyan"))
    out.print()

    if local_models:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("", min_width=2)  # Active indicator
        table.add_column("Model", min_width=20)
        table.add_column("Size", justify="right", min_width=8)
        table.add_column("Quant", min_width=6)
        table.add_column("Source", min_width=14)

        for model in local_models:
            # Active indicator
            indicator = "✓" if model.is_active else ""
            indicator_style = "bold green" if model.is_active else ""

            # Display name: prefer catalog name, fall back to directory name
            display_name = model.catalog_name if model.catalog_name else model.name

            # Size
            size_str = format_size(model.disk_size_bytes)

            table.add_row(
                Text(indicator, style=indicator_style),
                display_name,
                size_str,
                model.quant,
                model.source_type,
            )

        out.print(table)
    else:
        out.print("[dim]No local models downloaded yet.[/dim]")

    # Show remote-only stack models
    if remote_models:
        out.print()
        out.print(Text("Stack Models (not downloaded)", style="bold yellow"))
        out.print()

        remote_table = Table(show_header=True, header_style="bold yellow")
        remote_table.add_column("", min_width=2)
        remote_table.add_column("Model", min_width=20)
        remote_table.add_column("Tier", min_width=10)
        remote_table.add_column("Quant", min_width=6)
        remote_table.add_column("Source", min_width=10)
        remote_table.add_column("Est. Size", justify="right", min_width=10)

        for rm in remote_models:
            est_size = f"{rm['est_size_gb']:.1f} GB" if rm.get("est_size_gb") else "—"
            remote_table.add_row(
                Text("✓", style="bold green"),
                rm["catalog_name"],
                rm["tier"],
                rm["quant"],
                "remote",
                est_size,
            )

        out.print(remote_table)

    out.print()
    out.print(f"[dim]Models directory: {models_dir}[/dim]")
    if any(m.is_active for m in local_models) or remote_models:
        out.print("[dim]✓ = active in current stack[/dim]")
    out.print()


# --------------------------------------------------------------------------- #
# Catalog display
# --------------------------------------------------------------------------- #


def _display_catalog(
    family: str | None = None,
    tag: str | None = None,
    tool_calling: bool = False,
) -> None:
    """Display the full model catalog with hardware-specific benchmark data.

    Args:
        family: Optional family name filter (case-insensitive).
        tag: Optional tag filter (case-insensitive).
        tool_calling: If True, filter to tool-calling-capable models only.
    """
    out = Console()

    try:
        catalog = load_catalog()
    except CatalogError as exc:
        console.print(f"[bold red]Error:[/bold red] Could not load catalog: {exc}")
        raise SystemExit(1) from None

    # Apply filters
    filtered = catalog
    if family:
        filtered = query_by_family(filtered, family)
    if tag:
        filtered = query_by_tag(filtered, tag)
    if tool_calling:
        filtered = query_by_capability(filtered, tool_calling=True)

    if not filtered:
        out.print()
        filter_parts: list[str] = []
        if family:
            filter_parts.append(f"family={family}")
        if tag:
            filter_parts.append(f"tag={tag}")
        if tool_calling:
            filter_parts.append("tool-calling")
        filter_desc = ", ".join(filter_parts) if filter_parts else "filters"
        out.print(
            f"[yellow]No models match the given filters ({filter_desc}).[/yellow] "
            "Run [bold]mlx-stack models --catalog[/bold] to see all models."
        )
        out.print()
        return

    profile = load_profile()
    local_models = scan_local_models(catalog=catalog)
    catalog_models = list_catalog_models(
        catalog=filtered, profile=profile, local_models=local_models
    )

    out.print()
    out.print(Text("Model Catalog", style="bold cyan"))

    if profile:
        out.print(f"[dim]Hardware: {profile.chip} ({profile.memory_gb} GB)[/dim]")
    else:
        out.print(
            "[dim]No hardware profile — run 'mlx-stack profile' for hardware-specific data[/dim]"
        )

    out.print()

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("", width=1)  # Local indicator
    table.add_column("Name", min_width=14, no_wrap=True)
    table.add_column("Family", min_width=8)
    table.add_column("Params", justify="right", min_width=5)
    table.add_column("Quants", min_width=10)

    if profile:
        table.add_column("Gen t/s", justify="right", min_width=7)
        table.add_column("Mem GB", justify="right", min_width=6)

    for cm in catalog_models:
        # Local indicator
        local_indicator = "●" if cm.is_local else ""
        local_style = "bold green" if cm.is_local else ""

        # Parameters
        params_str = f"{cm.params_b:.1f}B" if cm.params_b >= 1.0 else f"{cm.params_b:.1f}B"

        # Quantizations
        quants_str = ", ".join(cm.quants)

        row: list[str | Text] = [
            Text(local_indicator, style=local_style),
            cm.name,
            cm.family,
            params_str,
            quants_str,
        ]

        if profile:
            # Gen t/s
            if cm.gen_tps is not None:
                tps_str = f"{cm.gen_tps:.0f}"
                if cm.is_estimated:
                    tps_str += "~"
            else:
                tps_str = "—"

            # Memory
            if cm.memory_gb is not None:
                mem_str = f"{cm.memory_gb:.1f}"
                if cm.is_estimated:
                    mem_str += "~"
            else:
                mem_str = "—"

            row.extend([tps_str, mem_str])

        table.add_row(*row)

    out.print(table)

    out.print()
    if profile and any(cm.is_estimated for cm in catalog_models):
        out.print("[dim]~ = estimated values (run 'mlx-stack bench --save' to calibrate)[/dim]")
    out.print("[dim]● = available locally[/dim]")
    out.print()


# --------------------------------------------------------------------------- #
# Click command
# --------------------------------------------------------------------------- #


@click.command()
@click.option("--catalog", is_flag=True, help="Show full catalog with benchmark data.")
@click.option("--family", default=None, help="Filter catalog by model family (e.g., 'qwen3.5').")
@click.option("--tag", default=None, help="Filter catalog by tag (e.g., 'agent-ready').")
@click.option(
    "--tool-calling", "tool_calling", is_flag=True,
    help="Filter catalog to tool-calling-capable models only.",
)
def models(
    catalog: bool,
    family: str | None,
    tag: str | None,
    tool_calling: bool,
) -> None:
    """List local models or browse the catalog.

    Without flags, shows locally downloaded models with disk size,
    quantization, and source type. Active stack models are marked
    with a visual indicator.

    Use --catalog to display all 15 catalog models with hardware-specific
    benchmark data (gen_tps, memory) for your detected hardware profile.

    Filter flags (--family, --tag, --tool-calling) require --catalog.
    """
    try:
        # If filter flags are used without --catalog, enable catalog mode
        if (family or tag or tool_calling) and not catalog:
            catalog = True

        if catalog:
            _display_catalog(family=family, tag=tag, tool_calling=tool_calling)
        else:
            _display_local_models()
    except ModelsError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None
