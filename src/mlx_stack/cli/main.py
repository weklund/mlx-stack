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
from mlx_stack.cli.bench import bench as bench_command
from mlx_stack.cli.config import config as config_group
from mlx_stack.cli.down import down as down_command
from mlx_stack.cli.init import init as init_command
from mlx_stack.cli.install import install as install_command
from mlx_stack.cli.install import uninstall as uninstall_command
from mlx_stack.cli.logs import logs as logs_command
from mlx_stack.cli.models import models as models_command
from mlx_stack.cli.pull import pull as pull_command
from mlx_stack.cli.recommend import recommend as recommend_command
from mlx_stack.cli.setup import setup as setup_command
from mlx_stack.cli.status import status as status_command
from mlx_stack.cli.up import up as up_command
from mlx_stack.cli.watch import watch as watch_command

console = Console(stderr=True)

# ASCII banner — block-character logo, each row is (color, text) pairs
# Visual width: ~78 columns, fits 80-col terminals
_BANNER_LINES = [
    ("cyan", "  ███╗   ███╗ ██╗     ██╗  ██╗"),
    ("white", "  ███████╗████████╗ █████╗  ██████╗██╗  ██╗"),
    ("cyan", "  ████╗ ████║ ██║     ╚██╗██╔╝"),
    ("white", "  ██╔════╝╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝"),
    ("cyan", "  ██╔████╔██║ ██║      ╚███╔╝ "),
    ("white", "  ███████╗   ██║   ███████║██║     █████╔╝ "),
    ("cyan", "  ██║╚██╔╝██║ ██║      ██╔██╗ "),
    ("white", "  ╚════██║   ██║   ██╔══██║██║     ██╔═██╗ "),
    ("cyan", "  ██║ ╚═╝ ██║ ███████╗██╔╝ ██╗"),
    ("white", " ███████║   ██║   ██║  ██║╚██████╗██║  ██╗"),
    ("cyan", "  ╚═╝     ╚═╝ ╚══════╝╚═╝  ╚═╝"),
    ("white", " ╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝"),
]

# Command categories and their members
_COMMAND_CATEGORIES: dict[str, list[str]] = {
    "Setup & Configuration": ["setup", "config", "init"],
    "Model Management": ["recommend", "models", "pull"],
    "Stack Lifecycle": ["up", "down", "status", "watch", "install", "uninstall"],
    "Diagnostics": ["bench", "logs"],
}


def _get_quick_status() -> list[tuple[str, str]]:
    """Gather lightweight, read-only status info.

    Only reads files that already exist — never creates directories.
    Returns a list of (label, value) pairs for display.
    """
    items: list[tuple[str, str]] = []

    # Chip — read profile.json if it exists
    try:
        from mlx_stack.core.paths import get_profile_path

        profile_path = get_profile_path()
        if profile_path.exists():
            from mlx_stack.core.hardware import load_profile

            profile = load_profile()
            if profile:
                mem = f"{profile.memory_gb}GB"
                items.append(("Chip", f"{profile.chip}  {mem} unified"))
    except Exception:
        pass

    # Stack — check if a stack definition exists
    try:
        from mlx_stack.core.paths import get_stacks_dir

        stack_dir = get_stacks_dir()
        if stack_dir.exists() and (stack_dir / "default.yaml").exists():
            items.append(("Stack", "configured"))
        else:
            items.append(("Stack", "not configured"))
    except Exception:
        pass

    return items


def _render_welcome(ctx: click.Context, group: RichGroup) -> None:
    """Render the branded welcome screen for bare `mlx-stack` invocation."""
    out = Console()

    # Banner
    out.print()
    for i in range(0, len(_BANNER_LINES), 2):
        left_color, left_text = _BANNER_LINES[i]
        right_color, right_text = _BANNER_LINES[i + 1]
        line = Text(left_text, style=f"bold {left_color}")
        line.append(right_text, style=f"bold {right_color}")
        out.print(line, highlight=False)
    out.print()

    # Version line
    ver_line = Text("  v" + __version__, style="dim")
    ver_line.append("  ")
    ver_line.append("Local LLM infrastructure on Apple Silicon", style="italic")
    out.print(ver_line)
    out.print()

    # Quick status (read-only, best-effort)
    status_items = _get_quick_status()
    if status_items:
        for label, value in status_items:
            line = Text(f"  {label}: ", style="bold")
            style = "green" if value != "not configured" else "yellow"
            line.append(value, style=style)
            out.print(line)
        out.print()

    # Grouped commands
    commands = group.list_commands(ctx)
    if commands:
        for category_name, cmd_names in _COMMAND_CATEGORIES.items():
            cmds_in_cat = []
            for cmd_name in cmd_names:
                if cmd_name in commands:
                    cmd = group.get_command(ctx, cmd_name)
                    if cmd:
                        cmds_in_cat.append(
                            (cmd_name, cmd.get_short_help_str(limit=60))
                        )
            if not cmds_in_cat:
                continue

            out.print(Text(f"  {category_name}", style="bold yellow"))
            cmd_table = Table(
                show_header=False,
                box=None,
                padding=(0, 2),
                pad_edge=False,
            )
            cmd_table.add_column(style="green", min_width=14)
            cmd_table.add_column(style="dim")
            for name, help_text in cmds_in_cat:
                cmd_table.add_row(f"    {name}", help_text)
            out.print(cmd_table)
            out.print()

    # Get started nudge
    if not status_items or all(v == "not configured" for _, v in status_items):
        out.print(
            Text("  Get started: ", style="bold")
            + Text("mlx-stack setup", style="bold cyan")
        )
    else:
        out.print(
            Text("  Run ", style="dim")
            + Text("mlx-stack <command> --help", style="dim bold")
            + Text(" for details on any command.", style="dim")
        )
    out.print()


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
            for category_name, cmd_names in _COMMAND_CATEGORIES.items():
                cmds_in_cat = []
                for cmd_name in cmd_names:
                    if cmd_name in commands:
                        cmd = self.get_command(ctx, cmd_name)
                        if cmd:
                            cmds_in_cat.append(
                                (cmd_name, cmd.get_short_help_str(limit=80))
                            )
                if not cmds_in_cat:
                    continue

                console_out.print(Text(f"{category_name}:", style="bold yellow"))
                cmd_table = Table(show_header=False, box=None, padding=(0, 2))
                cmd_table.add_column(style="green", min_width=20)
                cmd_table.add_column()
                for name, help_text in cmds_in_cat:
                    cmd_table.add_row(name, help_text)
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
    if ctx.invoked_subcommand is None:
        _render_welcome(ctx, ctx.command)  # type: ignore[arg-type]


# --- Placeholder commands for planned features ---
# These will be replaced by real implementations in subsequent features.


cli.add_command(setup_command, "setup")
cli.add_command(recommend_command, "recommend")
cli.add_command(init_command, "init")


cli.add_command(pull_command, "pull")
cli.add_command(models_command, "models")
cli.add_command(up_command, "up")
cli.add_command(down_command, "down")
cli.add_command(status_command, "status")


cli.add_command(watch_command, "watch")
cli.add_command(install_command, "install")
cli.add_command(uninstall_command, "uninstall")

cli.add_command(bench_command, "bench")
cli.add_command(logs_command, "logs")

cli.add_command(config_group, "config")
