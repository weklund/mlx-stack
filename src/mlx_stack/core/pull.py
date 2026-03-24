"""Model download and inventory management for mlx-stack.

Resolves catalog ID + quant to HuggingFace source, prefers mlx-community
pre-converted weights with fallback to mlx_lm conversion. Checks disk
space before download, shows progress, tracks inventory in models.json,
detects duplicates, handles network errors with automatic retry, and
cleans up partial downloads on failure.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console

from mlx_stack.core.catalog import CatalogEntry, QuantSource, get_entry_by_id, load_catalog
from mlx_stack.core.config import ConfigCorruptError, get_value
from mlx_stack.core.paths import ensure_data_home, get_data_home

# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class PullError(Exception):
    """Raised when model pull operations fail."""


class DiskSpaceError(PullError):
    """Raised when insufficient disk space is available."""


class DownloadError(PullError):
    """Raised when model download fails."""


class ConversionError(PullError):
    """Raised when mlx_lm conversion fails."""


class InvalidModelError(PullError):
    """Raised when the model ID is not found in the catalog."""


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelInventoryEntry:
    """An entry in the local model inventory (models.json)."""

    model_id: str
    name: str
    quant: str
    source_type: str  # "mlx-community" or "converted"
    hf_repo: str
    local_path: str
    disk_size_gb: float
    downloaded_at: str  # ISO 8601 timestamp


@dataclass(frozen=True)
class PullResult:
    """Result of a model pull operation."""

    model_id: str
    name: str
    quant: str
    source_type: str
    local_path: Path
    already_existed: bool
    disk_size_gb: float


# --------------------------------------------------------------------------- #
# Models directory resolution
# --------------------------------------------------------------------------- #


def get_models_directory() -> Path:
    """Resolve the models directory from config.

    Respects custom model-dir from config, falling back to the default
    ~/.mlx-stack/models/.

    Returns:
        Path to the models directory.
    """
    try:
        model_dir = str(get_value("model-dir"))
        return Path(model_dir).expanduser()
    except (ConfigCorruptError, Exception):
        return get_data_home() / "models"


# --------------------------------------------------------------------------- #
# Inventory management (models.json)
# --------------------------------------------------------------------------- #


def _get_inventory_path() -> Path:
    """Return the path to the models inventory file."""
    return get_data_home() / "models.json"


def load_inventory() -> list[dict[str, Any]]:
    """Load the model inventory from models.json.

    Returns:
        List of model inventory entries as dicts.
    """
    path = _get_inventory_path()
    if not path.exists():
        return []

    try:
        content = path.read_text(encoding="utf-8")
        data = json.loads(content)
        if isinstance(data, list):
            return data
    except (OSError, json.JSONDecodeError):
        pass

    return []


def save_inventory(entries: list[dict[str, Any]]) -> None:
    """Save the model inventory to models.json.

    Args:
        entries: List of model inventory entries as dicts.
    """
    ensure_data_home()
    path = _get_inventory_path()
    content = json.dumps(entries, indent=2, sort_keys=False)
    path.write_text(content + "\n", encoding="utf-8")


def add_to_inventory(entry: ModelInventoryEntry) -> None:
    """Add a model entry to the inventory.

    Replaces any existing entry with the same model_id and quant.

    Args:
        entry: The inventory entry to add.
    """
    entries = load_inventory()

    # Remove existing entry for same model_id + quant
    entries = [
        e for e in entries
        if not (e.get("model_id") == entry.model_id and e.get("quant") == entry.quant)
    ]

    entries.append(asdict(entry))
    save_inventory(entries)


def find_in_inventory(model_id: str, quant: str) -> dict[str, Any] | None:
    """Check if a model is already in the inventory.

    Args:
        model_id: The catalog model ID.
        quant: The quantization level.

    Returns:
        The inventory entry dict if found, None otherwise.
    """
    entries = load_inventory()
    for entry in entries:
        if entry.get("model_id") == model_id and entry.get("quant") == quant:
            return entry
    return None


# --------------------------------------------------------------------------- #
# Source resolution
# --------------------------------------------------------------------------- #


def resolve_source(
    entry: CatalogEntry,
    quant: str,
) -> tuple[QuantSource, str]:
    """Resolve the download source for a model + quant combination.

    Prefers mlx-community pre-converted weights. Falls back to the
    convert_from source if the quant source has convert_from=True.

    Args:
        entry: The catalog entry.
        quant: The quantization level.

    Returns:
        Tuple of (QuantSource, source_type) where source_type is
        "mlx-community" or "converted".

    Raises:
        PullError: If the quant is not available for the model.
    """
    if quant not in entry.sources:
        available = ", ".join(sorted(entry.sources.keys()))
        msg = (
            f"Quantization '{quant}' is not available for {entry.name}. "
            f"Available: {available}"
        )
        raise PullError(msg)

    source = entry.sources[quant]

    if source.convert_from:
        return source, "converted"
    else:
        return source, "mlx-community"


# --------------------------------------------------------------------------- #
# Disk space check
# --------------------------------------------------------------------------- #


def check_disk_space(
    models_dir: Path,
    required_gb: float,
) -> tuple[bool, float]:
    """Check if there is enough disk space for the download.

    Args:
        models_dir: The directory where the model will be stored.
        required_gb: Required disk space in GB.

    Returns:
        Tuple of (has_space, available_gb).
    """
    # Ensure parent dir exists for statvfs
    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        stat = shutil.disk_usage(models_dir)
        available_gb = stat.free / (1024**3)
        # Add 20% buffer for safety
        return available_gb >= required_gb * 1.2, round(available_gb, 1)
    except OSError:
        # If we can't check, allow the download
        return True, 0.0


# --------------------------------------------------------------------------- #
# Model local path determination
# --------------------------------------------------------------------------- #


def get_model_local_path(models_dir: Path, hf_repo: str) -> Path:
    """Determine the local path for a model based on its HF repo name.

    Args:
        models_dir: The models directory.
        hf_repo: The HuggingFace repo name (e.g., "mlx-community/Qwen3.5-0.6B-4bit").

    Returns:
        The local path for the model directory.
    """
    # Use the repo name (last part) as the directory name
    repo_name = hf_repo.rsplit("/", 1)[-1] if "/" in hf_repo else hf_repo
    return models_dir / repo_name


def is_model_downloaded(model_path: Path) -> bool:
    """Check if a model directory already exists and has content.

    Args:
        model_path: Path to the model directory.

    Returns:
        True if the directory exists and contains files.
    """
    if not model_path.exists() or not model_path.is_dir():
        return False
    # Check for at least one file
    try:
        return any(model_path.iterdir())
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# Download with retry
# --------------------------------------------------------------------------- #


def _run_download(
    hf_repo: str,
    local_dir: Path,
    console: Console,
) -> None:
    """Run the huggingface-cli download command with real-time output.

    Uses subprocess.Popen to stream stdout line-by-line so the user can
    see download progress in real-time, while still capturing stderr for
    error reporting on failure.

    Args:
        hf_repo: The HuggingFace repo to download.
        local_dir: The local directory to download to.
        console: Rich console for output.

    Raises:
        DownloadError: If the download fails.
    """
    # Use huggingface-cli download via subprocess
    # The huggingface_hub library is available as a dependency of mlx_lm
    # Use Python -m approach to ensure we use the right environment
    cmd = [
        "huggingface-cli",
        "download",
        hf_repo,
        "--local-dir",
        str(local_dir),
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        msg = (
            "huggingface-cli not found. Install huggingface_hub:\n"
            "  pip install huggingface_hub\n"
            "Or: uv pip install huggingface_hub"
        )
        raise DownloadError(msg) from None
    except OSError as exc:
        msg = f"Failed to start download: {exc}"
        raise DownloadError(msg) from None

    # Stream stdout line-by-line to show download progress to the user
    assert proc.stdout is not None
    assert proc.stderr is not None
    try:
        for line in proc.stdout:
            stripped = line.rstrip("\n")
            if stripped:
                console.print(f"  {stripped}")

        # Wait for process to complete and capture stderr
        proc.wait(timeout=3600)
        stderr_output = proc.stderr.read()
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        msg = "Download timed out after 1 hour."
        raise DownloadError(msg) from None

    if proc.returncode != 0:
        stderr = stderr_output.strip() if stderr_output else ""
        msg = f"Download failed for {hf_repo}:\n{stderr}"
        raise DownloadError(msg)


def download_model(
    hf_repo: str,
    local_dir: Path,
    console: Console,
    max_retries: int = 2,
) -> None:
    """Download a model from HuggingFace with automatic retry.

    Args:
        hf_repo: The HuggingFace repo to download.
        local_dir: The local directory to download to.
        console: Rich console for output.
        max_retries: Maximum number of attempts (default 2 = 1 retry).

    Raises:
        DownloadError: If all download attempts fail.
    """
    local_dir.mkdir(parents=True, exist_ok=True)

    last_error: DownloadError | None = None
    for attempt in range(1, max_retries + 1):
        try:
            console.print(
                f"[cyan]Downloading {hf_repo}...[/cyan]"
                + (f" (attempt {attempt}/{max_retries})" if attempt > 1 else "")
            )
            _run_download(hf_repo, local_dir, console)
            console.print("[green]✓ Download complete.[/green]")
            return
        except DownloadError as exc:
            last_error = exc
            if attempt < max_retries:
                console.print(
                    f"[yellow]Download attempt {attempt} failed. "
                    f"Retrying...[/yellow]"
                )
                time.sleep(2)  # Brief pause before retry
            else:
                break

    # All attempts failed — clean up partial download
    _cleanup_partial(local_dir)

    assert last_error is not None
    msg = (
        f"{last_error}\n\n"
        "Check your network connection and HuggingFace authentication.\n"
        "Set HF_TOKEN environment variable if the model requires authentication."
    )
    raise DownloadError(msg)


# --------------------------------------------------------------------------- #
# MLX conversion
# --------------------------------------------------------------------------- #


def convert_model(
    hf_repo: str,
    local_dir: Path,
    quant: str,
    console: Console,
) -> None:
    """Convert a model using mlx_lm.

    Downloads the base model and converts it to the specified quantization.

    Args:
        hf_repo: The HuggingFace repo of the base model (e.g., "Qwen/Qwen3.5-8B").
        local_dir: The directory to write converted model to.
        quant: The quantization level (int4, int8).
        console: Rich console for output.

    Raises:
        ConversionError: If conversion fails.
    """
    # Map our quant names to mlx_lm quant names
    quant_map = {
        "int4": "4",
        "int8": "8",
    }
    mlx_quant = quant_map.get(quant)

    console.print(f"[cyan]Converting {hf_repo} to {quant}...[/cyan]")
    console.print("[dim]This may take several minutes.[/dim]")

    local_dir.mkdir(parents=True, exist_ok=True)

    if mlx_quant:
        # Quantized conversion
        cmd = [
            "python3",
            "-m",
            "mlx_lm.convert",
            "--hf-path",
            hf_repo,
            "--mlx-path",
            str(local_dir),
            "-q",
            "--q-bits",
            mlx_quant,
        ]
    else:
        # bf16 — just download, no quant
        cmd = [
            "python3",
            "-m",
            "mlx_lm.convert",
            "--hf-path",
            hf_repo,
            "--mlx-path",
            str(local_dir),
        ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout for large conversions
        )
    except FileNotFoundError:
        _cleanup_partial(local_dir)
        msg = (
            "mlx_lm not found. Install it with:\n"
            "  pip install mlx_lm\n"
            "Or: uv pip install mlx_lm"
        )
        raise ConversionError(msg) from None
    except subprocess.TimeoutExpired:
        _cleanup_partial(local_dir)
        msg = "Conversion timed out after 2 hours."
        raise ConversionError(msg) from None
    except OSError as exc:
        _cleanup_partial(local_dir)
        msg = f"Failed to start conversion: {exc}"
        raise ConversionError(msg) from None

    if result.returncode != 0:
        stderr = result.stderr.strip()
        _cleanup_partial(local_dir)
        msg = f"Conversion failed for {hf_repo}:\n{stderr}"
        raise ConversionError(msg)

    console.print("[green]✓ Conversion complete.[/green]")


# --------------------------------------------------------------------------- #
# Cleanup
# --------------------------------------------------------------------------- #


def _cleanup_partial(local_dir: Path) -> None:
    """Remove a partial/failed download directory.

    Args:
        local_dir: The directory to remove.
    """
    if local_dir.exists():
        try:
            shutil.rmtree(local_dir)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Quant validation
# --------------------------------------------------------------------------- #

VALID_QUANTS = {"int4", "int8", "bf16"}


def validate_quant(quant: str) -> str:
    """Validate a quantization value.

    Args:
        quant: The quantization string to validate.

    Returns:
        The validated quant string.

    Raises:
        PullError: If the quant is not valid.
    """
    if quant not in VALID_QUANTS:
        valid = ", ".join(sorted(VALID_QUANTS))
        msg = f"Invalid quantization '{quant}'. Valid values: {valid}"
        raise PullError(msg)
    return quant


# --------------------------------------------------------------------------- #
# Main pull orchestrator
# --------------------------------------------------------------------------- #


def pull_model(
    model_id: str,
    quant: str | None = None,
    force: bool = False,
    console: Console | None = None,
    catalog: list[CatalogEntry] | None = None,
) -> PullResult:
    """Pull (download) a model from the catalog.

    Orchestrates the full pull workflow:
    1. Resolve model from catalog
    2. Determine quant (from flag or config default)
    3. Resolve source (mlx-community or convert_from)
    4. Check disk space
    5. Check for existing download (duplicate detection)
    6. Download or convert
    7. Update inventory

    Args:
        model_id: The catalog model ID (e.g., "qwen3.5-8b").
        quant: Quantization override (None uses config default).
        force: If True, re-download even if model exists.
        console: Rich console for output (creates one if None).
        catalog: Pre-loaded catalog (loads from package if None).

    Returns:
        PullResult with details of the completed pull.

    Raises:
        InvalidModelError: If the model ID is not in the catalog.
        PullError: If the quant is invalid or unavailable.
        DiskSpaceError: If insufficient disk space.
        DownloadError: If download fails after retries.
        ConversionError: If mlx_lm conversion fails.
    """
    if console is None:
        console = Console()

    # 1. Load catalog and resolve model
    if catalog is None:
        catalog = load_catalog()

    entry = get_entry_by_id(catalog, model_id)
    if entry is None:
        msg = (
            f"Model '{model_id}' not found in catalog.\n"
            "Run 'mlx-stack models --catalog' to see available models."
        )
        raise InvalidModelError(msg)

    # 2. Determine quantization
    if quant is None:
        try:
            quant = str(get_value("default-quant"))
        except Exception:
            quant = "int4"

    quant = validate_quant(quant)

    # 3. Resolve source
    source, source_type = resolve_source(entry, quant)

    # 4. Get models directory and local path
    models_dir = get_models_directory()
    local_path = get_model_local_path(models_dir, source.hf_repo)

    # 5. Check for existing download (duplicate detection)
    if not force and is_model_downloaded(local_path):
        # Check inventory too
        inv_entry = find_in_inventory(model_id, quant)
        if inv_entry is not None or is_model_downloaded(local_path):
            console.print(
                f"[yellow]Model '{entry.name}' ({quant}) already exists at "
                f"{local_path}.[/yellow]\n"
                "Use --force to re-download."
            )
            return PullResult(
                model_id=model_id,
                name=entry.name,
                quant=quant,
                source_type=source_type,
                local_path=local_path,
                already_existed=True,
                disk_size_gb=source.disk_size_gb,
            )

    # 6. Check disk space
    has_space, available_gb = check_disk_space(models_dir, source.disk_size_gb)
    if not has_space:
        msg = (
            f"Insufficient disk space for {entry.name} ({quant}).\n"
            f"Required: {source.disk_size_gb:.1f} GB (+ 20% buffer)\n"
            f"Available: {available_gb:.1f} GB"
        )
        raise DiskSpaceError(msg)

    # 7. Display info
    console.print()
    console.print(f"[bold cyan]Pulling {entry.name}[/bold cyan]")
    console.print(f"  Quantization: {quant}")
    console.print(f"  Source: {source.hf_repo}")
    console.print(f"  Type: {source_type}")
    console.print(f"  Estimated size: {source.disk_size_gb:.1f} GB")
    console.print(f"  Destination: {local_path}")
    console.print()

    # 8. Download or convert
    if force and local_path.exists():
        console.print("[yellow]Removing existing download (--force)...[/yellow]")
        _cleanup_partial(local_path)

    if source_type == "mlx-community":
        download_model(source.hf_repo, local_path, console)
    else:
        # convert_from — need mlx_lm conversion
        convert_model(source.hf_repo, local_path, quant, console)

    # 9. Update inventory
    inv = ModelInventoryEntry(
        model_id=model_id,
        name=entry.name,
        quant=quant,
        source_type=source_type,
        hf_repo=source.hf_repo,
        local_path=str(local_path),
        disk_size_gb=source.disk_size_gb,
        downloaded_at=datetime.now(timezone.utc).isoformat(),
    )
    add_to_inventory(inv)

    console.print()
    console.print(
        f"[bold green]✓ {entry.name} ({quant}) is ready.[/bold green]"
    )
    console.print(f"  Location: {local_path}")

    return PullResult(
        model_id=model_id,
        name=entry.name,
        quant=quant,
        source_type=source_type,
        local_path=local_path,
        already_existed=False,
        disk_size_gb=source.disk_size_gb,
    )
