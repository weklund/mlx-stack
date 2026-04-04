"""CLI command for model download — `mlx-stack pull`.

Downloads models from the catalog with source resolution, disk space
checking, progress display, duplicate detection, inventory tracking,
and optional post-download benchmark.

Supports --quant for quantization override, --bench for post-download
smoke test, and --force for re-downloading existing models.
"""

from __future__ import annotations

import click
from rich.console import Console

from mlx_stack.core.catalog import CatalogError
from mlx_stack.core.pull import (
    ConversionError,
    DiskSpaceError,
    DownloadError,
    GatedModelError,
    InvalidModelError,
    PullError,
    pull_model,
)

console = Console(stderr=True)


@click.command()
@click.argument("model", required=True)
@click.option(
    "--quant",
    type=str,
    default=None,
    help="Quantization level (int4, int8, bf16). For HF repos, stored as metadata only.",
)
@click.option(
    "--bench",
    is_flag=True,
    default=False,
    help="Run a quick benchmark after download.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-download even if model already exists.",
)
def pull(model: str, quant: str | None, bench: bool, force: bool) -> None:
    """Download a model by catalog ID or HuggingFace repo.

    MODEL is a catalog model ID (e.g., qwen3.5-8b) or a HuggingFace repo
    string (e.g., mlx-community/Phi-5-Mini-4bit).

    \b
    Catalog IDs are resolved via the built-in model catalog:
      mlx-stack pull qwen3.5-8b
    \b
    HuggingFace repo strings (containing '/') download directly:
      mlx-stack pull mlx-community/Phi-5-Mini-4bit

    Without --quant, uses the default quantization from config (default: int4).
    For HF repo pulls, --quant is stored as metadata only and does NOT change
    the download target. Invalid quantization values are rejected with a clear error.

    Downloads are checked against available disk space before starting.
    Already-downloaded models are detected and skipped unless --force is used.

    With --bench, runs a quick benchmark after download completes. This
    auto-installs vllm-mlx if needed.
    """
    out = Console()

    # Route based on whether MODEL contains '/' (HF repo) or not (catalog ID)
    is_hf_repo = "/" in model

    try:
        if is_hf_repo:
            result = pull_model(
                model_id=model,
                quant=quant,
                force=force,
                console=out,
                hf_repo_override=model,
            )
        else:
            result = pull_model(
                model_id=model,
                quant=quant,
                force=force,
                console=out,
            )

        if bench:
            _run_post_download_bench(model, result.quant, out)

    except InvalidModelError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None
    except DiskSpaceError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None
    except GatedModelError as exc:
        console.print(f"[bold red]Authentication required:[/bold red] {exc}")
        raise SystemExit(1) from None
    except DownloadError as exc:
        console.print(f"[bold red]Download error:[/bold red] {exc}")
        raise SystemExit(1) from None
    except ConversionError as exc:
        console.print(f"[bold red]Conversion error:[/bold red] {exc}")
        raise SystemExit(1) from None
    except PullError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None
    except CatalogError as exc:
        console.print(f"[bold red]Catalog error:[/bold red] {exc}")
        raise SystemExit(1) from None


def _run_post_download_bench(model_id: str, quant: str, out: Console) -> None:
    """Run a quick benchmark after downloading a model.

    Auto-installs vllm-mlx if needed.

    Args:
        model_id: The model ID that was pulled.
        quant: The quantization level.
        out: Rich console for output.
    """
    out.print()
    out.print("[bold cyan]Running post-download benchmark...[/bold cyan]")

    try:
        from mlx_stack.core.benchmark import BenchmarkError, run_benchmark

        result = run_benchmark(target=model_id, save=True)
        out.print(f"  Prompt TPS: {result.prompt_tps_mean:.1f} ± {result.prompt_tps_std:.1f} tok/s")
        out.print(f"  Gen TPS:    {result.gen_tps_mean:.1f} ± {result.gen_tps_std:.1f} tok/s")
        out.print()
        out.print("[dim]Results saved for use by 'recommend' and 'init' scoring.[/dim]")
    except BenchmarkError as exc:
        out.print(
            f"[yellow]Benchmark failed: {exc}[/yellow]\nRun 'mlx-stack bench {model_id}' to retry."
        )
    except Exception as exc:
        out.print(
            f"[yellow]Could not run benchmark: {exc}[/yellow]\n"
            "Skipping benchmark. Install vllm-mlx manually and run "
            f"'mlx-stack bench {model_id}'."
        )
