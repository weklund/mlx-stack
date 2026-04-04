"""CLI command for service status — `mlx-stack status`.

Displays hardware info (when available) and the health/metrics for all
managed services in a formatted Rich table or as JSON (with --json).
Read-only: does not modify any files or acquire the lockfile.
"""

from __future__ import annotations

import json
from typing import Any

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

from mlx_stack.core.hardware import HardwareProfile, load_profile
from mlx_stack.core.stack_status import (
    ServiceHealth,
    StatusResult,
    run_status,
    status_to_dict,
)

console = Console(stderr=True)

# Status display styling — maps state to Rich markup
_STATUS_STYLES: dict[ServiceHealth, str] = {
    ServiceHealth.HEALTHY: "[bold green]healthy[/bold green]",
    ServiceHealth.DEGRADED: "[bold yellow]degraded[/bold yellow]",
    ServiceHealth.DOWN: "[bold red]down[/bold red]",
    ServiceHealth.CRASHED: "[bold red]crashed[/bold red]",
    ServiceHealth.STOPPED: "[dim]stopped[/dim]",
}


def _load_hardware_profile() -> HardwareProfile | None:
    """Load hardware profile from disk, returning None on any error.

    This is a thin wrapper around ``load_profile()`` that additionally
    catches unexpected exceptions so a corrupt profile never crashes
    the status command.
    """
    try:
        return load_profile()
    except Exception:
        return None


def _display_hardware(hw: HardwareProfile) -> None:
    """Display hardware profile as a Rich table.

    Args:
        hw: The hardware profile to display.
    """
    out = Console()

    table = Table(
        title="Hardware",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Chip", hw.chip)
    table.add_row("GPU Cores", str(hw.gpu_cores))
    table.add_row("Memory", f"{hw.memory_gb} GB")

    bandwidth_str = f"{hw.bandwidth_gbps} GB/s"
    if hw.is_estimate:
        bandwidth_str += " (estimate)"
    table.add_row("Bandwidth", bandwidth_str)

    out.print(table)


def _hardware_to_dict(hw: HardwareProfile) -> dict[str, Any]:
    """Convert a HardwareProfile to a JSON-serialisable dict.

    Args:
        hw: The hardware profile to convert.

    Returns:
        A dict with chip, gpu_cores, memory_gb, bandwidth_gbps, is_estimate,
        profile_id.
    """
    return {
        "chip": hw.chip,
        "gpu_cores": hw.gpu_cores,
        "memory_gb": hw.memory_gb,
        "bandwidth_gbps": hw.bandwidth_gbps,
        "is_estimate": hw.is_estimate,
        "profile_id": hw.profile_id,
    }


def _display_table(result: StatusResult, hw: HardwareProfile | None) -> None:
    """Display hardware info and service statuses as Rich tables.

    Args:
        result: The StatusResult to display.
        hw: Optional hardware profile to display above the service table.
    """
    out = Console()
    out.print()

    if hw is not None:
        _display_hardware(hw)
        out.print()

    table = Table(
        title="Service Status",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Tier", style="bold", min_width=12)
    table.add_column("Model", min_width=20)
    table.add_column("Port", justify="right", min_width=6)
    table.add_column("Status", min_width=10)
    table.add_column("Uptime", justify="right", min_width=10)

    for svc in result.services:
        status_display = _STATUS_STYLES.get(svc.status, svc.status)
        table.add_row(
            svc.tier,
            svc.model,
            str(svc.port),
            status_display,
            svc.uptime_display,
        )

    out.print(table)
    out.print()


def _display_json(result: StatusResult, hw: HardwareProfile | None) -> None:
    """Display service statuses as JSON to stdout.

    Args:
        result: The StatusResult to display.
        hw: Optional hardware profile to include in output.
    """
    data = status_to_dict(result)
    data["hardware"] = _hardware_to_dict(hw) if hw is not None else None
    click.echo(json.dumps(data, indent=2))


@click.command()
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format.")
def status(json_output: bool) -> None:
    """Show hardware info and service health.

    Displays the detected Apple Silicon hardware profile (chip, GPU cores,
    memory, bandwidth) when available, followed by the current state of each
    managed service: healthy, degraded, down, crashed, or stopped.

    Outputs a formatted table by default, or valid JSON with --json.

    This command is read-only and safe to run concurrently with other
    mlx-stack commands.
    """
    hw = _load_hardware_profile()
    result = run_status()

    # Handle no-stack scenario
    if result.no_stack:
        if json_output:
            _display_json(result, hw)
        else:
            out = Console()
            out.print()
            if hw is not None:
                _display_hardware(hw)
                out.print()
            out.print(
                Text(
                    result.message or "No stack configured — run 'mlx-stack setup'.",
                    style="yellow",
                )
            )
            out.print()
        return

    # Display results
    if json_output:
        _display_json(result, hw)
    else:
        _display_table(result, hw)
