"""CLI command for model recommendation — `mlx-stack recommend`.

Recommends an optimal model stack based on hardware profile and user intent.
Reads existing profile or auto-detects hardware. Display-only — no files written.

Supports --budget, --intent (balanced/agent-fleet), and --show-all flags.
"""

from __future__ import annotations

import json
import re
from typing import Any

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

from mlx_stack.core.catalog import load_catalog
from mlx_stack.core.config import ConfigCorruptError, get_value
from mlx_stack.core.hardware import (
    HardwareError,
    HardwareProfile,
    detect_hardware,
    load_profile,
    save_profile,
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
# Budget parsing
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
        msg = (
            f"Invalid budget '{raw}'. Budget must be a positive value."
        )
        raise click.BadParameter(msg, param_hint="'--budget'")

    return value


# --------------------------------------------------------------------------- #
# Hardware profile resolution
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

    # Auto-detect
    console.print("[dim]No saved profile found — detecting hardware...[/dim]")
    try:
        profile = detect_hardware()
        save_profile(profile)
        return profile
    except HardwareError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None


# --------------------------------------------------------------------------- #
# Saved benchmarks loading
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
        pass

    return None


# --------------------------------------------------------------------------- #
# Display helpers
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
        out.print(
            "[yellow]⚠ Some performance values are estimated from bandwidth ratio.[/yellow]"
        )
        out.print(
            "  Run [bold]mlx-stack bench --save[/bold] to calibrate with real measurements."
        )

    out.print()
    out.print("[dim]This is a recommendation only — no files were written.[/dim]")
    out.print("[dim]Run [bold]mlx-stack init[/bold] to generate stack configuration.[/dim]")


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
            "[bold green]☁ Cloud Fallback[/bold green]  "
            "Premium tier via OpenRouter also available."
        )

    # Estimated warning
    has_estimates = any(m.is_estimated for m in result.all_scored)
    if has_estimates:
        out.print()
        out.print(
            "[yellow]⚠ Some performance values are estimated from bandwidth ratio.[/yellow]"
        )
        out.print(
            "  Run [bold]mlx-stack bench --save[/bold] to calibrate with real measurements."
        )

    out.print()
    out.print("[dim]This is a recommendation only — no files were written.[/dim]")


# --------------------------------------------------------------------------- #
# Click command
# --------------------------------------------------------------------------- #


@click.command()
@click.option(
    "--budget",
    type=str,
    default=None,
    help="Memory budget override (e.g., '30gb', '16'). Defaults to 40%% of unified memory.",
)
@click.option(
    "--intent",
    type=str,
    default=None,
    help="Recommendation intent: balanced (default) or agent-fleet.",
)
@click.option(
    "--show-all",
    is_flag=True,
    default=False,
    help="Show all budget-fitting models sorted by score instead of tier assignments.",
)
def recommend(budget: str | None, intent: str | None, show_all: bool) -> None:
    """Recommend an optimal model stack for your hardware.

    Analyzes your hardware profile and the model catalog to recommend
    an optimal stack with tier assignments (standard, fast, longctx).

    Uses 40% of unified memory as the default budget. Override with --budget.
    Supports --intent to change optimization strategy (balanced or agent-fleet).
    Use --show-all to see all budget-fitting models ranked by composite score.

    This command is display-only — no configuration files are written.
    """
    # --- Validate intent ---
    if intent is None:
        intent = "balanced"
    elif intent not in VALID_INTENTS:
        valid = ", ".join(sorted(VALID_INTENTS))
        console.print(
            f"[bold red]Error:[/bold red] Invalid intent '{intent}'. "
            f"Valid intents: {valid}"
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
