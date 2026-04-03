"""CLI commands for configuration management — `mlx-stack config`.

Provides set, get, list, and reset subcommands for managing
persistent configuration in ~/.mlx-stack/config.yaml.
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from mlx_stack.core.config import (
    ConfigCorruptError,
    ConfigError,
    ConfigValidationError,
    get_all_config,
    get_value,
    mask_value,
    reset_config,
    set_value,
)

console = Console(stderr=True)


@click.group()
def config() -> None:
    """Manage mlx-stack configuration."""


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a configuration value.

    KEY is the config key name (e.g., default-quant, litellm-port).
    VALUE is the value to set.

    Valid keys: openrouter-key, default-quant, memory-budget-pct,
    litellm-port, model-dir, auto-health-check.
    """
    try:
        result = set_value(key, value)
        display = mask_value(key, result)
        console.print(f"[green]✓[/green] Set [bold]{key}[/bold] = {display}")
    except ConfigValidationError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None
    except ConfigCorruptError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None
    except ConfigError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None


@config.command("get")
@click.argument("key")
def config_get(key: str) -> None:
    """Get a configuration value.

    KEY is the config key name to retrieve.

    Valid keys: openrouter-key, default-quant, memory-budget-pct,
    litellm-port, model-dir, auto-health-check.
    """
    try:
        value = get_value(key)
        display = mask_value(key, value)
        console.print(display)
    except ConfigCorruptError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None
    except ConfigError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None


@config.command("list")
def config_list() -> None:
    """List all configuration values.

    Shows a table of all config keys with their current values,
    defaults, and whether each value is user-set or default.
    The openrouter-key is masked for security.
    """
    try:
        entries = get_all_config()
    except ConfigCorruptError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None

    out = Console()
    table = Table(title="Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Key", style="bold")
    table.add_column("Value")
    table.add_column("Default")
    table.add_column("Source", style="dim")

    for entry in entries:
        source_style = "[dim]default[/dim]" if entry["is_default"] else "[green]user-set[/green]"
        value_display = entry["masked_value"]

        # Show empty string as (not set) for clarity
        if entry["name"] == "openrouter-key" and entry["is_default"]:
            value_display = "[dim](not set)[/dim]"

        default_display = str(entry["default"])
        if entry["name"] == "openrouter-key":
            default_display = "(not set)"

        table.add_row(
            entry["name"],
            value_display,
            default_display,
            source_style,
        )

    out.print()
    out.print(table)
    out.print()


@config.command("reset")
@click.option("--yes", is_flag=True, help="Confirm reset without prompting.")
@click.option("--force", is_flag=True, help="Alias for --yes.")
def config_reset(yes: bool, force: bool) -> None:
    """Reset configuration to defaults.

    Removes all user-set values, restoring defaults for all keys.
    Requires --yes or --force confirmation in non-interactive mode.
    """
    confirmed = yes or force

    if not confirmed:
        # Check if stdin is a TTY for interactive confirmation
        try:
            if click.get_text_stream("stdin").isatty():
                confirmed = click.confirm("Reset all configuration to defaults?", default=False)
            else:
                console.print(
                    "[bold red]Error:[/bold red] Reset requires --yes or --force flag "
                    "in non-interactive mode."
                )
                raise SystemExit(1)
        except (AttributeError, OSError):
            console.print(
                "[bold red]Error:[/bold red] Reset requires --yes or --force flag "
                "in non-interactive mode."
            )
            raise SystemExit(1) from None

    if not confirmed:
        console.print("[yellow]Reset cancelled.[/yellow]")
        return

    try:
        reset_config()
        console.print("[green]✓[/green] Configuration reset to defaults.")
    except ConfigError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None
