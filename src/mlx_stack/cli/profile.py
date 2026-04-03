"""CLI command for hardware detection — `mlx-stack profile`.

Detects Apple Silicon hardware, displays results as a Rich table,
and writes the profile to ~/.mlx-stack/profile.json.
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from mlx_stack.core.hardware import HardwareError, detect_hardware, save_profile

console = Console(stderr=True)


@click.command()
def profile() -> None:
    """Detect Apple Silicon hardware and write profile."""
    try:
        hw = detect_hardware()
    except HardwareError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None

    # Save profile to disk
    try:
        save_profile(hw)
    except OSError as exc:
        console.print(f"[bold red]Error:[/bold red] Could not write profile: {exc}")
        raise SystemExit(1) from None

    # Display results as a Rich table
    out = Console()
    table = Table(title="Hardware Profile", show_header=True, header_style="bold cyan")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Chip", hw.chip)
    table.add_row("GPU Cores", str(hw.gpu_cores))
    table.add_row("Unified Memory", f"{hw.memory_gb} GB")

    bandwidth_str = f"{hw.bandwidth_gbps} GB/s"
    if hw.is_estimate:
        bandwidth_str += " (estimate)"
    table.add_row("Memory Bandwidth", bandwidth_str)
    table.add_row("Profile ID", hw.profile_id)

    out.print()
    out.print(table)

    if hw.is_estimate:
        out.print()
        out.print("[yellow]⚠ Bandwidth is estimated for unknown chip.[/yellow]")
        out.print("  Run [bold]mlx-stack bench --save[/bold] to calibrate with real measurements.")

    out.print()
    from mlx_stack.core.paths import get_profile_path

    out.print(f"[dim]Profile saved to {get_profile_path()}[/dim]")
