"""CLI command for benchmarking — ``mlx-stack bench``.

Runs 3 iterations of 1024-token prompt + 100-token generation against
a running tier's vllm-mlx instance or a temporary instance for a local
model. Reports mean ± std dev for prompt_tps and gen_tps. Compares
against catalog thresholds: PASS (<15%), WARN (15-30%), FAIL (>30%).

Supports --save to persist results to ~/.mlx-stack/benchmarks/<profile_id>.json.
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

from mlx_stack.core.benchmark import (
    CLASSIFICATION_PASS,
    CLASSIFICATION_WARN,
    BenchmarkError,
    BenchmarkResult_,
    BenchmarkRunError,
    BenchmarkTargetError,
)
from mlx_stack.core.catalog import CatalogError
from mlx_stack.core.deps import DependencyError, DependencyInstallError

console = Console(stderr=True)


@click.command()
@click.argument("target", required=True)
@click.option("--save", is_flag=True, default=False, help="Save benchmark results.")
def bench(target: str, save: bool) -> None:
    """Benchmark a tier or model.

    TARGET is a running tier name (e.g., 'fast', 'standard') or a
    catalog model ID (e.g., 'qwen3.5-8b'). For running tiers, targets
    the existing vllm-mlx instance. For local models, starts a temporary
    instance with full cleanup.

    Runs 3 iterations of 1024-token prompt + 100-token generation and
    reports mean ± std dev for prompt_tps and gen_tps.

    Compares against catalog benchmarks: PASS (within 15%), WARN (15-30%
    below), FAIL (>30% below) per metric with delta percentage.

    Use --save to persist results to ~/.mlx-stack/benchmarks/<profile_id>.json.
    """
    out = Console()

    try:
        from mlx_stack.core.benchmark import run_benchmark

        result = run_benchmark(target=target, save=save)
        _display_results(result, out, save=save)

    except BenchmarkTargetError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None
    except BenchmarkRunError as exc:
        console.print(f"[bold red]Benchmark error:[/bold red] {exc}")
        raise SystemExit(1) from None
    except BenchmarkError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None
    except DependencyInstallError as exc:
        console.print(f"[bold red]Dependency error:[/bold red] {exc}")
        raise SystemExit(1) from None
    except DependencyError as exc:
        console.print(f"[bold red]Dependency error:[/bold red] {exc}")
        raise SystemExit(1) from None
    except CatalogError as exc:
        console.print(f"[bold red]Catalog error:[/bold red] {exc}")
        raise SystemExit(1) from None


def _display_results(result: BenchmarkResult_, out: Console, save: bool = False) -> None:
    """Display benchmark results with Rich formatting.

    Args:
        result: The benchmark result to display.
        out: Rich console for output.
        save: Whether results were saved.
    """
    out.print()

    # Header
    header = Text("Benchmark Results", style="bold cyan")
    header.append(f" — {result.model_id} ({result.quant})")
    out.print(header)
    out.print()

    # Instance type indicator
    if result.used_temporary_instance:
        out.print("[dim]Using temporary vllm-mlx instance[/dim]")
    else:
        out.print("[dim]Using running tier instance[/dim]")
    out.print()

    # Performance results table
    perf_table = Table(title="Performance", show_header=True, header_style="bold")
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Mean", justify="right")
    perf_table.add_column("Std Dev", justify="right")

    perf_table.add_row(
        "Prompt TPS",
        f"{result.prompt_tps_mean:.1f} tok/s",
        f"± {result.prompt_tps_std:.1f}",
    )
    perf_table.add_row(
        "Generation TPS",
        f"{result.gen_tps_mean:.1f} tok/s",
        f"± {result.gen_tps_std:.1f}",
    )

    out.print(perf_table)
    out.print()

    # Comparison against catalog
    if result.classifications:
        comp_table = Table(
            title="Catalog Comparison",
            show_header=True,
            header_style="bold",
        )
        comp_table.add_column("Metric", style="cyan")
        comp_table.add_column("Measured", justify="right")
        comp_table.add_column("Catalog", justify="right")
        comp_table.add_column("Delta", justify="right")
        comp_table.add_column("Result", justify="center")

        for cls in result.classifications:
            # Color the classification
            if cls.classification == CLASSIFICATION_PASS:
                result_style = "[bold green]PASS[/bold green]"
            elif cls.classification == CLASSIFICATION_WARN:
                result_style = "[bold yellow]WARN[/bold yellow]"
            else:
                result_style = "[bold red]FAIL[/bold red]"

            # Format delta — positive means below catalog, negative means above
            delta_str = f"{cls.delta_pct:+.1f}%"

            metric_name = cls.metric.replace("_", " ").title().replace("Tps", "TPS")

            comp_table.add_row(
                metric_name,
                f"{cls.measured:.1f}",
                f"{cls.catalog:.1f}",
                delta_str,
                result_style,
            )

        out.print(comp_table)
        out.print()
    elif not result.catalog_data_available:
        out.print(
            "[yellow]No catalog benchmark data available for your hardware profile.[/yellow]\n"
            "Use [bold]--save[/bold] to record this benchmark for future comparisons."
        )
        out.print()

    # Tool-calling result
    if result.tool_call_result is not None:
        out.print(Text("Tool Calling", style="bold cyan"))
        tc = result.tool_call_result
        if tc.success:
            out.print(
                f"  [green]✓ Valid tool call[/green] — "
                f"round-trip: {tc.round_trip_time:.2f}s"
            )
        else:
            out.print(
                f"  [red]✗ Tool call failed[/red] — {tc.error}"
            )
        out.print()
    elif not result.tool_call_result:
        # Check if model supports tool calling from entry
        if not result.catalog_data_available:
            pass  # Skip silently if no catalog data
        else:
            out.print(
                "[dim]Tool calling: skipped (model does not support tool calling)[/dim]"
            )
            out.print()

    # Iteration details
    if result.iterations:
        iter_table = Table(
            title="Iteration Details",
            show_header=True,
            header_style="bold",
        )
        iter_table.add_column("#", style="dim", justify="right")
        iter_table.add_column("Prompt TPS", justify="right")
        iter_table.add_column("Gen TPS", justify="right")
        iter_table.add_column("Time", justify="right")

        for i, it in enumerate(result.iterations, 1):
            iter_table.add_row(
                str(i),
                f"{it.prompt_tps:.1f}",
                f"{it.gen_tps:.1f}",
                f"{it.total_time:.1f}s",
            )

        out.print(iter_table)
        out.print()

    # Save confirmation
    if save:
        out.print("[green]✓ Results saved.[/green] "
                  "These will be used by 'recommend' and 'init' for scoring.")
        out.print()
