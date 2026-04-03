"""Interactive guided setup for mlx-stack.

Walks through hardware detection, model selection, and stack startup
in a single command. Replaces the profile -> recommend -> init -> pull -> up
flow with a guided experience.

Use ``--accept-defaults`` for non-interactive CI/scripting mode.

**How to run**::

    mlx-stack setup
    mlx-stack setup --accept-defaults
    mlx-stack setup --intent agent-fleet --budget-pct 60
"""

from __future__ import annotations

from typing import Any

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

from mlx_stack.core.discovery import DiscoveredModel, DiscoveryError, discover_models
from mlx_stack.core.onboarding import (
    OnboardingError,
    ScoredDiscoveredModel,
    assign_tiers,
    detect_and_budget,
    generate_config,
    pull_setup_models,
    score_and_filter,
    select_defaults,
    start_stack,
)

console = Console(stderr=True)
out = Console()

# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #


def _format_downloads(n: int) -> str:
    """Format download count with K/M suffixes."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _format_tps(tps: float | None) -> str:
    """Format tokens/sec or '--' if unavailable."""
    if tps is None:
        return "--"
    return f"{tps:.0f} t/s"


def _format_quality(q: float | None) -> str:
    """Format quality pass rate or '--' if unavailable."""
    if q is None:
        return "--"
    return f"{q:.0%}"


def _format_memory(gb: float | None) -> str:
    """Format memory in GB or '--'."""
    if gb is None:
        return "--"
    return f"{gb:.1f} GB"


# --------------------------------------------------------------------------- #
# Step displays
# --------------------------------------------------------------------------- #


def _display_hardware(profile: Any, budget_gb: float, budget_pct: int) -> None:
    """Display hardware detection results."""
    out.print()
    out.print(Text("  Hardware", style="bold cyan"))
    out.print("  " + "─" * 40)
    out.print(f"  Chip:       {profile.chip}")
    out.print(f"  GPU Cores:  {profile.gpu_cores}")
    out.print(f"  Memory:     {profile.memory_gb} GB unified")
    out.print(f"  Bandwidth:  {profile.bandwidth_gbps:.1f} GB/s")
    out.print()
    out.print(f"  Model budget: ~{budget_gb:.1f} GB ({budget_pct}% of memory)")
    out.print()


def _prompt_intent(accept_defaults: bool, intent_override: str | None) -> str:
    """Prompt for intent selection. Returns 'balanced' or 'agent-fleet'."""
    if intent_override:
        return intent_override
    if accept_defaults:
        return "balanced"

    out.print(Text("  Use Case", style="bold cyan"))
    out.print("  " + "─" * 40)
    out.print("  How will you use your stack?")
    out.print()
    out.print("    1. General purpose — balanced quality and speed")
    out.print("    2. Coding agents — optimized for tool calling")
    out.print()

    choice = click.prompt("  Select", type=click.IntRange(1, 2), default=1)
    out.print()

    return "balanced" if choice == 1 else "agent-fleet"


def _display_models_table(
    scored: list[ScoredDiscoveredModel],
    budget_gb: float,
) -> None:
    """Display the model selection table."""
    table = Table(show_header=True, header_style="bold cyan", padding=(0, 1))
    table.add_column("#", style="dim", width=4)
    table.add_column("Model", min_width=24)
    table.add_column("Quant", width=5)
    table.add_column("Size", justify="right", width=7)
    table.add_column("Speed", justify="right", width=8)
    table.add_column("Quality", justify="right", width=7)
    table.add_column("Tools", justify="center", width=5)
    table.add_column("Downloads", justify="right", width=9)

    for i, s in enumerate(scored, 1):
        m = s.model
        marker = "*" if s.is_recommended else " "
        idx = f"{i} {marker}"

        name_style = "bold" if s.is_recommended else ""
        tools = "Yes" if m.tool_calling else ("--" if not m.has_benchmark else "No")

        table.add_row(
            idx,
            Text(m.display_name, style=name_style),
            m.quant,
            _format_memory(m.memory_gb),
            _format_tps(m.gen_tps),
            _format_quality(m.quality_overall),
            tools,
            _format_downloads(m.downloads),
        )

    out.print(table)

    # Show recommended total
    recommended = [s for s in scored if s.is_recommended]
    if recommended:
        total_mem = sum(s.model.memory_gb or 0.0 for s in recommended)
        out.print(
            f"\n  [dim]* = recommended "
            f"({total_mem:.1f} GB total, within {budget_gb:.1f} GB budget)[/dim]"
        )


def _prompt_model_selection(
    scored: list[ScoredDiscoveredModel],
    accept_defaults: bool,
) -> list[tuple[int, str]]:
    """Prompt for model selection. Returns list of (index, quant) tuples.

    Empty input = accept defaults (recommended models with default quant).
    Input like '1,3,5' = specific models with default quant.
    Input like '1:int8,3' = model 1 as int8, model 3 as default quant.
    """
    if accept_defaults:
        return [
            (i, s.model.quant) for i, s in enumerate(scored) if s.is_recommended
        ]

    out.print()
    raw = click.prompt(
        "  Enter to accept defaults, or type model numbers (e.g. 1,3,5 or 1:int8,3)",
        default="",
        show_default=False,
    )
    out.print()

    if not raw.strip():
        # Accept defaults
        return [
            (i, s.model.quant) for i, s in enumerate(scored) if s.is_recommended
        ]

    # Parse input
    selections: list[tuple[int, str]] = []
    parts = [p.strip() for p in raw.split(",") if p.strip()]

    for part in parts:
        if ":" in part:
            idx_str, quant = part.split(":", 1)
            idx = int(idx_str) - 1  # Convert to 0-based
            selections.append((idx, quant.strip()))
        else:
            idx = int(part) - 1
            selections.append((idx, scored[idx].model.quant))

    return selections


def _display_tier_assignment(tiers: list[Any], budget_gb: float) -> None:
    """Display tier assignments."""
    out.print(Text("  Tier Assignment", style="bold cyan"))
    out.print("  " + "─" * 40)

    table = Table(show_header=True, header_style="bold cyan", padding=(0, 1))
    table.add_column("Tier", min_width=10)
    table.add_column("Model", min_width=24)
    table.add_column("Role", min_width=20)

    roles = {
        "standard": "highest quality",
        "fast": "fastest responses",
    }

    total_mem = 0.0
    for t in tiers:
        role = roles.get(t.tier_name, "additional model")
        table.add_row(t.tier_name, t.model.display_name, role)
        total_mem += t.model.memory_gb or 0.0

    out.print(table)
    out.print(f"\n  Total: {total_mem:.1f} GB of {budget_gb:.1f} GB budget")
    out.print()


def _display_final_status(tiers: list[Any], litellm_port: int) -> None:
    """Display final success message with endpoint info."""
    tier_names = ", ".join(t.tier_name for t in tiers)

    out.print()
    out.print(Text("  Your stack is running!", style="bold green"))
    out.print()
    out.print(f"  API endpoint:  http://localhost:{litellm_port}/v1")
    out.print("  API key:       dummy")
    out.print(f"  Models:        {tier_names}")
    out.print()
    out.print("  [dim]Quick test:[/dim]")
    out.print(
        f"    curl http://localhost:{litellm_port}/v1/chat/completions \\\n"
        f"      -H 'Content-Type: application/json' \\\n"
        f"      -d '{{\"model\":\"{tiers[0].tier_name}\","
        f"\"messages\":[{{\"role\":\"user\",\"content\":\"Hello!\"}}]}}'"
    )
    out.print()
    out.print("  [dim]Manage your stack:[/dim]")
    out.print("    mlx-stack status    — check health")
    out.print("    mlx-stack down      — stop services")
    out.print()


def _prompt_always_on(accept_defaults: bool) -> bool:
    """Prompt whether to install LaunchAgent for 24/7 operation."""
    if accept_defaults:
        return False

    out.print(Text("  Always-On", style="bold cyan"))
    out.print("  " + "─" * 40)
    out.print("  Run your stack 24/7 as a background service?")
    out.print("  This installs a macOS LaunchAgent that starts on login")
    out.print("  and auto-restarts if a model crashes.")
    out.print()

    return click.confirm("  Install LaunchAgent?", default=False)


# --------------------------------------------------------------------------- #
# Main command
# --------------------------------------------------------------------------- #


@click.command()
@click.option(
    "--accept-defaults",
    is_flag=True,
    default=False,
    help="Skip all prompts and use recommended defaults.",
)
@click.option(
    "--intent",
    "intent_override",
    type=click.Choice(["balanced", "agent-fleet"], case_sensitive=False),
    default=None,
    help="Use case intent (prompted if not provided).",
)
@click.option(
    "--budget-pct",
    type=click.IntRange(10, 90),
    default=None,
    help="Memory budget as percentage of unified memory (default: 40).",
)
def setup(
    accept_defaults: bool,
    intent_override: str | None,
    budget_pct: int | None,
) -> None:
    """Interactive guided setup for your local LLM stack.

    Walks through hardware detection, model selection, and stack startup
    in a single command.

    Use --accept-defaults for non-interactive CI/scripting mode.
    """
    # Auto-detect non-interactive terminals
    try:
        is_tty = click.get_text_stream("stdin").isatty()
    except (AttributeError, OSError):
        is_tty = False

    if not is_tty and not accept_defaults:
        out.print("[yellow]Non-interactive terminal detected, using defaults.[/yellow]")
        accept_defaults = True

    effective_budget_pct = budget_pct or 40

    # ── Step 1: Hardware ─────────────────────────────────────────────────
    try:
        profile, budget_gb = detect_and_budget(effective_budget_pct)
    except OnboardingError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None

    _display_hardware(profile, budget_gb, effective_budget_pct)

    # ── Step 2: Intent ───────────────────────────────────────────────────
    intent = _prompt_intent(accept_defaults, intent_override)

    # ── Step 3: Model discovery + selection ───────────────────────────────
    out.print(Text("  Model Selection", style="bold cyan"))
    out.print("  " + "─" * 40)
    out.print("  [dim]Querying mlx-community for available models...[/dim]")

    try:
        all_models = discover_models(hardware_profile_id=profile.profile_id)
    except DiscoveryError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None

    if not all_models:
        console.print(
            "[bold red]Error:[/bold red] No models found. "
            "Check your network connection."
        )
        raise SystemExit(1) from None

    scored = score_and_filter(all_models, intent, budget_gb)

    if not scored:
        console.print(
            f"[bold red]Error:[/bold red] No models fit within the "
            f"{budget_gb:.1f} GB budget. Try increasing --budget-pct."
        )
        raise SystemExit(1) from None

    scored = select_defaults(scored, budget_gb)

    out.print(f"  Found {len(scored)} models that fit your {budget_gb:.1f} GB budget.")
    out.print()
    _display_models_table(scored, budget_gb)

    selections = _prompt_model_selection(scored, accept_defaults)

    if not selections:
        console.print("[bold red]Error:[/bold red] No models selected.")
        raise SystemExit(1) from None

    # Resolve selections to models
    selected_models: list[ScoredDiscoveredModel] = []
    for idx, quant in selections:
        if 0 <= idx < len(scored):
            s = scored[idx]
            # If quant differs from default, update the model's hf_repo
            if quant != s.model.quant:
                # Replace quant suffix in repo name
                base = s.model.hf_repo
                for suffix in ["-4bit", "-8bit", "-bf16", "-fp16", "-4bits"]:
                    if base.endswith(suffix):
                        base = base[: -len(suffix)]
                        break
                quant_suffix = {"int4": "-4bit", "int8": "-8bit", "bf16": "-bf16"}.get(
                    quant, f"-{quant}"
                )
                new_model = DiscoveredModel(
                    hf_repo=base + quant_suffix,
                    display_name=s.model.display_name,
                    params_b=s.model.params_b,
                    quant=quant,
                    downloads=s.model.downloads,
                    gen_tps=s.model.gen_tps,
                    prompt_tps=s.model.prompt_tps,
                    memory_gb=s.model.memory_gb,
                    quality_overall=s.model.quality_overall,
                    tool_calling=s.model.tool_calling,
                    thinking=s.model.thinking,
                    has_benchmark=s.model.has_benchmark,
                )
                selected_models.append(ScoredDiscoveredModel(
                    model=new_model,
                    composite_score=s.composite_score,
                    speed_score=s.speed_score,
                    quality_score=s.quality_score,
                    tool_calling_score=s.tool_calling_score,
                    memory_efficiency_score=s.memory_efficiency_score,
                    is_recommended=True,
                ))
            else:
                selected_models.append(ScoredDiscoveredModel(
                    model=s.model,
                    composite_score=s.composite_score,
                    speed_score=s.speed_score,
                    quality_score=s.quality_score,
                    tool_calling_score=s.tool_calling_score,
                    memory_efficiency_score=s.memory_efficiency_score,
                    is_recommended=True,
                ))

    # ── Step 4: Tier assignment ──────────────────────────────────────────
    tiers = assign_tiers(selected_models)
    _display_tier_assignment(tiers, budget_gb)

    if not accept_defaults and len(tiers) > 1:
        click.prompt("  Press Enter to confirm", default="", show_default=False)
    out.print()

    # ── Step 5: Generate config + pull + start ───────────────────────────
    # Check for existing stack
    from mlx_stack.core.paths import get_stacks_dir

    stack_path = get_stacks_dir() / "default.yaml"
    if stack_path.exists() and not accept_defaults:
        overwrite = click.confirm(
            "  An existing stack configuration was found. Overwrite?",
            default=True,
        )
        if not overwrite:
            out.print("[yellow]  Setup cancelled.[/yellow]")
            raise SystemExit(0) from None

    try:
        stack_path, litellm_path = generate_config(
            profile=profile,
            intent=intent,
            tier_mappings=tiers,
            budget_gb=budget_gb,
        )
    except Exception as exc:
        console.print(f"[bold red]Error generating config:[/bold red] {exc}")
        raise SystemExit(1) from None

    # Pull models
    out.print(Text("  Downloading Models", style="bold cyan"))
    out.print("  " + "─" * 40)

    models_to_pull = [t.model for t in tiers]
    for i, model in enumerate(models_to_pull, 1):
        out.print(f"  [bold][{i}/{len(models_to_pull)}][/bold]", end=" ")

    try:
        pull_setup_models(models_to_pull, out)
    except Exception as exc:
        console.print(f"[bold red]Download error:[/bold red] {exc}")
        out.print("[yellow]  Some models failed. Run 'mlx-stack pull' to retry.[/yellow]")

    # Start stack
    out.print()
    out.print(Text("  Starting Stack", style="bold cyan"))
    out.print("  " + "─" * 40)

    try:
        up_result = start_stack()

        for tier in up_result.tiers:
            icon = (
                "[bold green]✓[/bold green]"
                if tier.status == "healthy"
                else "[bold red]✗[/bold red]"
            )
            out.print(f"  {icon} {tier.name} ({tier.model}) on port {tier.port}")

        if up_result.litellm:
            icon = (
                "[bold green]✓[/bold green]"
                if up_result.litellm.status == "healthy"
                else "[bold red]✗[/bold red]"
            )
            out.print(f"  {icon} LiteLLM proxy on port {up_result.litellm.port}")

    except Exception as exc:
        console.print(f"[bold red]Startup error:[/bold red] {exc}")
        out.print("[yellow]  Stack may be partially started. Check 'mlx-stack status'.[/yellow]")
        raise SystemExit(1) from None

    litellm_port = int(
        next(
            (t.port for t in (up_result.tiers + ([up_result.litellm] if up_result.litellm else []))
             if t.name == "litellm"),
            4000,
        )
    )
    _display_final_status(tiers, litellm_port)

    # ── Step 6: Always-on ────────────────────────────────────────────────
    if _prompt_always_on(accept_defaults):
        try:
            from mlx_stack.core.launchd import generate_plist, load_agent, write_plist

            plist_data = generate_plist()
            plist_path = write_plist(plist_data)
            load_agent(plist_path)
            out.print()
            out.print(
                "[bold green]  ✓ LaunchAgent installed.[/bold green] "
                "Your stack will start automatically on login."
            )
            out.print("    Uninstall anytime with: mlx-stack uninstall")
        except Exception as exc:
            console.print(
                f"[bold red]LaunchAgent error:[/bold red] {exc}\n"
                "  You can install it later with: mlx-stack install"
            )

    out.print()
