"""Main CLI entry point for mlx-stack.

Provides the top-level Click command group with --help, --version,
Rich-formatted output, and typo suggestions for unknown subcommands.
"""

from __future__ import annotations

import difflib

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

from mlx_stack import __version__
from mlx_stack.cli.config import config as config_group
from mlx_stack.cli.profile import profile as profile_command
from mlx_stack.cli.recommend import recommend as recommend_command
from mlx_stack.core.paths import ensure_data_home

console = Console(stderr=True)


class RichGroup(click.Group):
    """Custom Click Group with Rich-formatted help and typo suggestions."""

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Format help text using Rich tables."""
        console_out = Console()

        # Title
        console_out.print()
        title = Text("mlx-stack", style="bold cyan")
        title.append(" — CLI control plane for local LLM infrastructure on Apple Silicon")
        console_out.print(title)
        console_out.print()

        # Usage
        usage = Text("Usage: ", style="bold") + Text("mlx-stack [OPTIONS] COMMAND [ARGS]...")
        console_out.print(usage)
        console_out.print()

        # Options
        console_out.print(Text("Options:", style="bold yellow"))
        options_table = Table(show_header=False, box=None, padding=(0, 2))
        options_table.add_column(style="green", min_width=20)
        options_table.add_column()
        options_table.add_row("--version", "Show version and exit.")
        options_table.add_row("--help", "Show this message and exit.")
        console_out.print(options_table)
        console_out.print()

        # Commands grouped by category
        commands = self.list_commands(ctx)
        if commands:
            # Group commands by category
            categories: dict[str, list[tuple[str, str]]] = {
                "Setup & Configuration": [],
                "Model Management": [],
                "Stack Lifecycle": [],
                "Diagnostics": [],
            }

            command_categories = {
                "profile": "Setup & Configuration",
                "config": "Setup & Configuration",
                "init": "Setup & Configuration",
                "recommend": "Model Management",
                "models": "Model Management",
                "pull": "Model Management",
                "up": "Stack Lifecycle",
                "down": "Stack Lifecycle",
                "status": "Stack Lifecycle",
                "bench": "Diagnostics",
            }

            for cmd_name in commands:
                cmd = self.get_command(ctx, cmd_name)
                if cmd is None:
                    continue
                help_text = cmd.get_short_help_str(limit=80)
                category = command_categories.get(cmd_name, "Other")
                if category in categories:
                    categories[category].append((cmd_name, help_text))
                else:
                    categories.setdefault("Other", []).append((cmd_name, help_text))

            for category_name, cmds in categories.items():
                if not cmds:
                    continue
                console_out.print(Text(f"{category_name}:", style="bold yellow"))
                cmd_table = Table(show_header=False, box=None, padding=(0, 2))
                cmd_table.add_column(style="green", min_width=20)
                cmd_table.add_column()
                for cmd_name, help_text in cmds:
                    cmd_table.add_row(cmd_name, help_text)
                console_out.print(cmd_table)
                console_out.print()

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        """Override resolve_command to provide typo suggestions."""
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError:
            # Get the attempted command name
            if args:
                cmd_name = args[0]
                available = self.list_commands(ctx)
                matches = difflib.get_close_matches(cmd_name, available, n=3, cutoff=0.5)

                error_msg = f"Error: No such command '{cmd_name}'."
                if matches:
                    suggestions = ", ".join(f"'{m}'" for m in matches)
                    error_msg += f"\n\nDid you mean one of these?\n    {suggestions}"
                error_msg += "\n\nRun 'mlx-stack --help' for a list of available commands."

                console.print(f"[red]{error_msg}[/red]")
                ctx.exit(2)
                raise SystemExit(2)  # noqa: B904 — we want to exit, not chain
            raise


def version_callback(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    """Print version and exit."""
    if not value or ctx.resilient_parsing:
        return
    click.echo(f"mlx-stack, version {__version__}")
    ctx.exit(0)


@click.group(cls=RichGroup, invoke_without_command=True)
@click.option(
    "--version",
    is_flag=True,
    callback=version_callback,
    expose_value=False,
    is_eager=True,
    help="Show version and exit.",
)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """CLI control plane for local LLM infrastructure on Apple Silicon."""
    ensure_data_home()
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# --- Placeholder commands for planned features ---
# These will be replaced by real implementations in subsequent features.


cli.add_command(profile_command, "profile")
cli.add_command(recommend_command, "recommend")


@cli.command()
@click.option("--accept-defaults", is_flag=True, help="Use defaults without prompting.")
@click.option("--intent", type=str, help="Recommendation intent (balanced, agent-fleet).")
@click.option("--force", is_flag=True, help="Overwrite existing stack configuration.")
def init(accept_defaults: bool, intent: str | None, force: bool) -> None:
    """Generate stack definition and LiteLLM config."""
    console.print("[yellow]Not yet implemented.[/yellow] Coming in the init-command feature.")
    raise SystemExit(1)


@cli.command()
@click.argument("model", required=False)
@click.option("--quant", type=str, help="Quantization level (int4, int8, bf16).")
@click.option("--bench", is_flag=True, help="Run benchmark after download.")
@click.option("--force", is_flag=True, help="Re-download even if model exists.")
def pull(model: str | None, quant: str | None, bench: bool, force: bool) -> None:
    """Download a model from the catalog."""
    console.print("[yellow]Not yet implemented.[/yellow] Coming in the pull-command feature.")
    raise SystemExit(1)


@cli.command()
@click.option("--catalog", is_flag=True, help="Show full catalog with benchmark data.")
def models(catalog: bool) -> None:
    """List local models or browse the catalog."""
    console.print("[yellow]Not yet implemented.[/yellow] Coming in the models-command feature.")
    raise SystemExit(1)


@cli.command()
@click.option("--dry-run", is_flag=True, help="Show commands without executing.")
@click.option("--tier", type=str, help="Start only the specified tier.")
def up(dry_run: bool, tier: str | None) -> None:
    """Start all services in the active stack."""
    console.print("[yellow]Not yet implemented.[/yellow] Coming in the up-command feature.")
    raise SystemExit(1)


@cli.command()
@click.option("--tier", type=str, help="Stop only the specified tier.")
def down(tier: str | None) -> None:
    """Stop all managed services."""
    console.print("[yellow]Not yet implemented.[/yellow] Coming in the down-command feature.")
    raise SystemExit(1)


@cli.command()
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format.")
def status(json_output: bool) -> None:
    """Show health and status of all services."""
    console.print("[yellow]Not yet implemented.[/yellow] Coming in the status-command feature.")
    raise SystemExit(1)


@cli.command()
@click.argument("target", required=False)
@click.option("--save", is_flag=True, help="Save benchmark results.")
def bench(target: str | None, save: bool) -> None:
    """Benchmark a tier or model."""
    console.print("[yellow]Not yet implemented.[/yellow] Coming in the bench-command feature.")
    raise SystemExit(1)


cli.add_command(config_group, "config")
