"""CLI command for model listing — `mlx-stack models`.

Lists locally downloaded models with disk size, quantization, and source type.
Active stack models are marked with a visual indicator. The --catalog flag
shows all 15 catalog models with hardware-specific benchmark data.

The --recommend flag shows scored tier recommendations (absorbed from the
old ``recommend`` command). The --available flag queries the HuggingFace
API and shows an enriched model list.

Output is formatted as a Rich table with human-readable names.
"""

from __future__ import annotations

import json
import re
from typing import Any

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
from mlx_stack.core.config import ConfigCorruptError, get_value
from mlx_stack.core.hardware import (
    HardwareError,
    HardwareProfile,
    detect_hardware,
    load_profile,
)
from mlx_stack.core.models import (
    ModelsError,
    format_size,
    get_models_directory,
    get_remote_stack_models,
    list_catalog_models,
    scan_local_models,
)
from mlx_stack.core.paths import get_benchmarks_dir
from mlx_stack.core.scoring import (
    VALID_INTENTS,
    RecommendationResult,
    ScoringError,
)
from mlx_stack.core.scoring import (
    recommend as run_recommend,
)

console = Console(stderr=True)


# --------------------------------------------------------------------------- #
# Budget parsing (moved from cli/recommend.py)
# --------------------------------------------------------------------------- #

_BUDGET_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)\s*(gb|GB|Gb|gB)?$")


def parse_budget(raw: str) -> float:
    """Parse a budget string like '30gb', '30GB', '30' into GB float.

    Args:
        raw: The raw budget string from CLI.

    Returns:
        Budget in GB as a float.

    Raises:
        click.BadParameter: If the budget format is invalid or value is non-positive.
    """
    match = _BUDGET_PATTERN.match(raw.strip())
    if not match:
        msg = (
            f"Invalid budget format '{raw}'. "
            f"Expected a positive number with optional 'gb' suffix (e.g., '30gb', '16')."
        )
        raise click.BadParameter(msg, param_hint="'--budget'")

    value = float(match.group(1))
    if value <= 0:
        msg = f"Invalid budget '{raw}'. Budget must be a positive value."
        raise click.BadParameter(msg, param_hint="'--budget'")

    return value


# --------------------------------------------------------------------------- #
# Hardware profile resolution (moved from cli/recommend.py)
# --------------------------------------------------------------------------- #


def _resolve_profile() -> HardwareProfile:
    """Load existing profile or auto-detect hardware.

    Returns:
        A HardwareProfile instance.

    Raises:
        SystemExit: If hardware detection fails.
    """
    profile = load_profile()
    if profile is not None:
        return profile

    # Auto-detect (in-memory only — recommend is display-only, no file writes)
    console.print("[dim]No saved profile found — detecting hardware...[/dim]")
    try:
        return detect_hardware()
    except HardwareError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None


# --------------------------------------------------------------------------- #
# Saved benchmarks loading (moved from cli/recommend.py)
# --------------------------------------------------------------------------- #


def _load_saved_benchmarks(profile_id: str) -> dict[str, Any] | None:
    """Load saved benchmark data for the given profile, if available.

    Reads from ~/.mlx-stack/benchmarks/<profile_id>.json.

    Args:
        profile_id: The hardware profile ID.

    Returns:
        Dict mapping model_id -> benchmark data, or None if no data.
    """
    benchmarks_dir = get_benchmarks_dir()
    benchmark_file = benchmarks_dir / f"{profile_id}.json"

    if not benchmark_file.exists():
        return None

    try:
        data = json.loads(benchmark_file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        console.print(
            f"[yellow]⚠ Warning:[/yellow] Could not parse saved benchmarks "
            f"at {benchmark_file}. Falling back to catalog data."
        )

    return None


# --------------------------------------------------------------------------- #
# Recommend display helpers (moved from cli/recommend.py)
# --------------------------------------------------------------------------- #


def _format_tps(tps: float, is_estimated: bool) -> str:
    """Format tokens per second with optional estimated label."""
    formatted = f"{tps:.1f} tok/s"
    if is_estimated:
        formatted += " (est.)"
    return formatted


def _format_memory(memory_gb: float) -> str:
    """Format memory usage in GB."""
    return f"{memory_gb:.1f} GB"


def _display_tier_table(result: RecommendationResult) -> None:
    """Display the recommended tiers as a Rich table."""
    out = Console()

    out.print()
    title = Text("Recommended Stack", style="bold cyan")
    title.append(f"  ({result.intent})")
    out.print(title)
    out.print(
        f"[dim]Hardware: {result.hardware_profile.chip} "
        f"({result.hardware_profile.memory_gb} GB) · "
        f"Budget: {result.memory_budget_gb:.1f} GB[/dim]"
    )
    out.print()

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Tier", style="bold", min_width=10)
    table.add_column("Model", min_width=20)
    table.add_column("Quant", min_width=6)
    table.add_column("Gen TPS", justify="right", min_width=15)
    table.add_column("Memory", justify="right", min_width=10)

    for tier_assign in result.tiers:
        table.add_row(
            tier_assign.tier,
            tier_assign.model.entry.name,
            tier_assign.quant,
            _format_tps(tier_assign.model.gen_tps, tier_assign.model.is_estimated),
            _format_memory(tier_assign.model.memory_gb),
        )

    out.print(table)

    # Cloud fallback row if OpenRouter key is configured
    try:
        openrouter_key = get_value("openrouter-key")
    except (ConfigCorruptError, Exception):
        openrouter_key = ""

    if openrouter_key:
        out.print()
        out.print(
            "[bold green]☁ Cloud Fallback[/bold green]  "
            "Premium tier via OpenRouter (GPT-4o / Claude Sonnet)"
        )

    # Estimated warning
    has_estimates = any(t.model.is_estimated for t in result.tiers)
    if has_estimates:
        out.print()
        out.print("[yellow]⚠ Some performance values are estimated from bandwidth ratio.[/yellow]")
        out.print("  Run [bold]mlx-stack bench --save[/bold] to calibrate with real measurements.")

    out.print()
    out.print("[dim]This is a recommendation only — no files were written.[/dim]")
    out.print("[dim]Run [bold]mlx-stack setup[/bold] to generate stack configuration.[/dim]")


def _display_all_models(result: RecommendationResult) -> None:
    """Display all budget-fitting models sorted by composite score."""
    out = Console()

    out.print()
    title = Text("All Budget-Fitting Models", style="bold cyan")
    title.append(f"  ({result.intent})")
    out.print(title)
    out.print(
        f"[dim]Hardware: {result.hardware_profile.chip} "
        f"({result.hardware_profile.memory_gb} GB) · "
        f"Budget: {result.memory_budget_gb:.1f} GB[/dim]"
    )
    out.print()

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", justify="right", style="dim", min_width=3)
    table.add_column("Model", min_width=20)
    table.add_column("Family", min_width=10)
    table.add_column("Params", justify="right", min_width=8)
    table.add_column("Score", justify="right", min_width=8)
    table.add_column("Gen TPS", justify="right", min_width=15)
    table.add_column("Memory", justify="right", min_width=10)

    for idx, scored in enumerate(result.all_scored, 1):
        table.add_row(
            str(idx),
            scored.entry.name,
            scored.entry.family,
            f"{scored.entry.params_b:.1f}B",
            f"{scored.composite_score:.3f}",
            _format_tps(scored.gen_tps, scored.is_estimated),
            _format_memory(scored.memory_gb),
        )

    out.print(table)
    out.print()
    count = len(result.all_scored)
    budget = f"{result.memory_budget_gb:.1f}"
    out.print(f"[dim]{count} models fit within the {budget} GB budget.[/dim]")

    # Cloud fallback note
    try:
        openrouter_key = get_value("openrouter-key")
    except (ConfigCorruptError, Exception):
        openrouter_key = ""

    if openrouter_key:
        out.print()
        out.print(
            "[bold green]☁ Cloud Fallback[/bold green]  Premium tier via OpenRouter also available."
        )

    # Estimated warning
    has_estimates = any(m.is_estimated for m in result.all_scored)
    if has_estimates:
        out.print()
        out.print("[yellow]⚠ Some performance values are estimated from bandwidth ratio.[/yellow]")
        out.print("  Run [bold]mlx-stack bench --save[/bold] to calibrate with real measurements.")

    out.print()
    out.print("[dim]This is a recommendation only — no files were written.[/dim]")


# --------------------------------------------------------------------------- #
# Available models display
# --------------------------------------------------------------------------- #


def _display_available_models() -> None:
    """Query the HuggingFace API and display discovered models."""
    from mlx_stack.core.discovery import DiscoveryError, discover_models

    out = Console()

    profile = load_profile()
    hardware_profile_id = profile.profile_id if profile else None

    try:
        discovered = discover_models(hardware_profile_id=hardware_profile_id)
    except DiscoveryError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None

    out.print()
    out.print(Text("Available Models", style="bold cyan"))
    if profile:
        out.print(f"[dim]Hardware: {profile.chip} ({profile.memory_gb} GB)[/dim]")
    out.print(f"[dim]Source: HuggingFace mlx-community · {len(discovered)} models[/dim]")
    out.print()

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Model", min_width=20)
    table.add_column("Params", justify="right", min_width=8)
    table.add_column("Quant", min_width=6)
    table.add_column("Downloads", justify="right", min_width=10)

    if any(d.gen_tps is not None for d in discovered):
        table.add_column("Gen t/s", justify="right", min_width=8)
        table.add_column("Mem GB", justify="right", min_width=7)
        has_perf_cols = True
    else:
        has_perf_cols = False

    for model in discovered:
        params_str = f"{model.params_b:.1f}B" if model.params_b > 0 else "—"
        dl_str = f"{model.downloads:,}" if model.downloads > 0 else "—"

        row: list[str] = [model.display_name, params_str, model.quant, dl_str]

        if has_perf_cols:
            tps_str = f"{model.gen_tps:.0f}" if model.gen_tps is not None else "—"
            mem_str = f"{model.memory_gb:.1f}" if model.memory_gb is not None else "—"
            row.extend([tps_str, mem_str])

        table.add_row(*row)

    out.print(table)
    out.print()


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
            "or [bold]mlx-stack setup[/bold] to set up a stack."
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
            display_name = model.catalog_name or model.name

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
            "[dim]No hardware profile — run 'mlx-stack setup' for hardware-specific data[/dim]"
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
        params_str = f"{cm.params_b:.1f}B"

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
# Recommend logic
# --------------------------------------------------------------------------- #


def _run_recommend(
    budget: str | None,
    intent: str | None,
    show_all: bool,
) -> None:
    """Execute the recommend logic (absorbed from old recommend command).

    Args:
        budget: Optional budget string (e.g. '30gb').
        intent: Optional intent string ('balanced' or 'agent-fleet').
        show_all: If True, show ranked list instead of tier table.
    """
    # --- Validate intent ---
    if intent is None:
        intent = "balanced"
    elif intent not in VALID_INTENTS:
        valid = ", ".join(sorted(VALID_INTENTS))
        console.print(
            f"[bold red]Error:[/bold red] Invalid intent '{intent}'. Valid intents: {valid}"
        )
        raise SystemExit(1)

    # --- Parse budget ---
    budget_gb_override: float | None = None
    if budget is not None:
        try:
            budget_gb_override = parse_budget(budget)
        except click.BadParameter as exc:
            console.print(f"[bold red]Error:[/bold red] {exc.format_message()}")
            raise SystemExit(1) from None

    # --- Resolve hardware profile ---
    profile = _resolve_profile()

    # --- Read memory-budget-pct from config (used when no --budget override) ---
    budget_pct = 40
    if budget_gb_override is None:
        try:
            budget_pct = int(get_value("memory-budget-pct"))
        except (ConfigCorruptError, ValueError):
            budget_pct = 40

    # --- Load catalog ---
    try:
        catalog = load_catalog()
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] Could not load model catalog: {exc}")
        raise SystemExit(1) from None

    # --- Load saved benchmarks ---
    saved_benchmarks = _load_saved_benchmarks(profile.profile_id)

    # --- Run recommendation ---
    try:
        result = run_recommend(
            catalog=catalog,
            profile=profile,
            intent=intent,
            budget_pct=budget_pct,
            budget_gb_override=budget_gb_override,
            saved_benchmarks=saved_benchmarks,
        )
    except ScoringError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None

    # --- Check for zero results ---
    if not result.all_scored:
        console.print(
            f"[bold red]Error:[/bold red] No models fit within the "
            f"{result.memory_budget_gb:.1f} GB budget."
        )
        console.print(
            "[dim]Try increasing the budget with --budget or "
            "adjusting memory-budget-pct in config.[/dim]"
        )
        raise SystemExit(1)

    # --- Display results ---
    if show_all:
        _display_all_models(result)
    else:
        _display_tier_table(result)


# --------------------------------------------------------------------------- #
# Click command
# --------------------------------------------------------------------------- #


@click.command()
@click.option("--catalog", is_flag=True, help="Show full catalog with benchmark data.")
@click.option(
    "--recommend",
    "recommend_flag",
    is_flag=True,
    help="Show scored tier recommendations for your hardware.",
)
@click.option(
    "--available",
    is_flag=True,
    help="Query HuggingFace API and show enriched model list.",
)
@click.option(
    "--budget",
    type=str,
    default=None,
    help="Memory budget override (e.g., '30gb', '16'). Requires --recommend.",
)
@click.option(
    "--intent",
    type=str,
    default=None,
    help="Recommendation intent: balanced (default) or agent-fleet. Requires --recommend.",
)
@click.option(
    "--show-all",
    is_flag=True,
    default=False,
    help="Show all budget-fitting models sorted by score. Requires --recommend.",
)
@click.option("--family", default=None, help="Filter catalog by model family (e.g., 'qwen3.5').")
@click.option("--tag", default=None, help="Filter catalog by tag (e.g., 'agent-ready').")
@click.option(
    "--tool-calling",
    "tool_calling",
    is_flag=True,
    help="Filter catalog to tool-calling-capable models only.",
)
def models(
    catalog: bool,
    recommend_flag: bool,
    available: bool,
    budget: str | None,
    intent: str | None,
    show_all: bool,
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

    Use --recommend to show scored tier recommendations for your hardware.
    Combine with --budget, --intent, and --show-all for more control.

    Use --available to query the HuggingFace API and browse available models.

    Filter flags (--family, --tag, --tool-calling) require --catalog.
    """
    try:
        # --- Mutual exclusivity check ---
        mode_flags = []
        if recommend_flag:
            mode_flags.append("--recommend")
        if available:
            mode_flags.append("--available")
        if catalog or family or tag or tool_calling:
            mode_flags.append("--catalog")

        if len(mode_flags) > 1:
            flags_str = " and ".join(mode_flags)
            console.print(
                f"[bold red]Error:[/bold red] {flags_str} are mutually exclusive. "
                "Use only one at a time."
            )
            raise SystemExit(1)

        # --- Recommend-dependent flag checks ---
        if not recommend_flag and (budget is not None or intent is not None or show_all):
            dependent_flags = []
            if budget is not None:
                dependent_flags.append("--budget")
            if intent is not None:
                dependent_flags.append("--intent")
            if show_all:
                dependent_flags.append("--show-all")
            flags_str = ", ".join(dependent_flags)
            console.print(
                f"[bold red]Error:[/bold red] {flags_str} "
                "can only be used with --recommend."
            )
            raise SystemExit(1)

        # --- Route to the appropriate display function ---
        if recommend_flag:
            _run_recommend(budget=budget, intent=intent, show_all=show_all)
        elif available:
            _display_available_models()
        elif catalog or family or tag or tool_calling:
            _display_catalog(family=family, tag=tag, tool_calling=tool_calling)
        else:
            _display_local_models()
    except ModelsError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None
