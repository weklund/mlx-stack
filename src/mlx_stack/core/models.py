"""Local model scanning and catalog listing for mlx-stack.

Scans the configured model directory for locally downloaded models,
determines disk size, quantization, and source type. Identifies active
stack models. Provides catalog listing with hardware-specific benchmark
data for the --catalog view.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mlx_stack.core.catalog import CatalogEntry, load_catalog
from mlx_stack.core.config import ConfigCorruptError, get_value
from mlx_stack.core.hardware import HardwareProfile, load_profile
from mlx_stack.core.paths import get_data_home, get_stacks_dir

# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class ModelsError(Exception):
    """Raised when model listing operations fail."""


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LocalModel:
    """A locally downloaded model discovered on disk."""

    name: str
    path: Path
    disk_size_bytes: int
    quant: str
    source_type: str  # "mlx-community", "converted", "unknown"
    is_active: bool  # True if referenced by the active stack
    catalog_name: str | None  # Human-readable name from catalog, if matched


@dataclass(frozen=True)
class CatalogModel:
    """A catalog model entry formatted for display in --catalog mode."""

    id: str
    name: str
    family: str
    params_b: float
    quants: list[str]
    gen_tps: float | None
    memory_gb: float | None
    prompt_tps: float | None
    is_estimated: bool
    is_local: bool
    tags: list[str]


# --------------------------------------------------------------------------- #
# Model directory resolution
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
# Active stack model detection
# --------------------------------------------------------------------------- #


def _load_active_stack() -> dict[str, Any] | None:
    """Load the active stack definition, if it exists.

    Returns:
        The parsed stack YAML dict, or None if no stack is configured.
    """
    stack_path = get_stacks_dir() / "default.yaml"
    if not stack_path.exists():
        return None

    try:
        content = stack_path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        if isinstance(data, dict):
            return data
    except (OSError, yaml.YAMLError):
        pass

    return None


def _get_active_stack_models(stack: dict[str, Any] | None) -> list[dict[str, str]]:
    """Extract model info from the active stack definition.

    Returns a list (not a dict) because the same model_id could appear
    with different quants in different tiers. Each entry contains
    model_id, quant, source, and tier name.

    Returns:
        List of dicts with keys: model_id, quant, source, tier.
    """
    if stack is None:
        return []

    tiers = stack.get("tiers", [])
    result: list[dict[str, str]] = []
    for tier in tiers:
        model_id = tier.get("model", "")
        if model_id:
            result.append({
                "model_id": model_id,
                "quant": tier.get("quant", ""),
                "source": tier.get("source", ""),
                "tier": tier.get("name", ""),
            })
    return result


# --------------------------------------------------------------------------- #
# Local model scanning
# --------------------------------------------------------------------------- #


def _compute_dir_size(path: Path) -> int:
    """Compute the total size of all files in a directory recursively.

    Args:
        path: Directory path.

    Returns:
        Total size in bytes.
    """
    total = 0
    try:
        for dirpath, _dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = Path(dirpath) / f
                try:
                    total += fp.stat().st_size
                except OSError:
                    continue
    except OSError:
        pass
    return total


def _detect_quant(model_dir: Path, dirname: str) -> str:
    """Detect quantization level from directory name or contents.

    Heuristic: looks for common patterns in directory names like
    '4bit', '8bit', 'int4', 'int8', 'bf16'.

    Args:
        model_dir: Path to the model directory.
        dirname: Directory name.

    Returns:
        Detected quantization string, or "unknown".
    """
    lower = dirname.lower()
    if "4bit" in lower or "int4" in lower or "-4bit" in lower:
        return "int4"
    if "8bit" in lower or "int8" in lower or "-8bit" in lower:
        return "int8"
    if "bf16" in lower or "bfloat16" in lower:
        return "bf16"
    if "fp16" in lower or "f16" in lower:
        return "bf16"  # Approximate — fp16 treated as bf16 for display

    # Check for config.json with quantization info
    config_path = model_dir / "config.json"
    if config_path.exists():
        try:
            import json

            config_data = json.loads(config_path.read_text(encoding="utf-8"))
            quant_config = config_data.get("quantization_config", {})
            bits = quant_config.get("bits")
            if bits == 4:
                return "int4"
            if bits == 8:
                return "int8"
        except (OSError, ValueError, KeyError):
            pass

    return "unknown"


def _detect_source_type(dirname: str) -> str:
    """Detect the source type from directory name.

    Args:
        dirname: Directory name.

    Returns:
        Source type string: "mlx-community", "converted", or "unknown".
    """
    lower = dirname.lower()
    if "mlx-community" in lower or lower.startswith("mlx-community"):
        return "mlx-community"
    # Directories from HF usually have org/repo pattern replaced with --
    if "--" in dirname:
        parts = dirname.split("--", 1)
        if parts[0].lower() == "mlx-community":
            return "mlx-community"
    return "unknown"


def _match_to_catalog(dirname: str, catalog: list[CatalogEntry]) -> CatalogEntry | None:
    """Try to match a local directory name to a catalog entry.

    Matches by checking if the HF repo name (last part) is in the dirname.

    Args:
        dirname: Local directory name.
        catalog: Loaded catalog entries.

    Returns:
        Matching CatalogEntry or None.
    """
    lower = dirname.lower()
    for entry in catalog:
        for _quant, source in entry.sources.items():
            # Extract repo name from hf_repo (after the /)
            repo_name = (
                source.hf_repo.rsplit("/", 1)[-1] if "/" in source.hf_repo else source.hf_repo
            )
            if repo_name.lower() == lower:
                return entry
    return None


def scan_local_models(
    models_dir: Path | None = None,
    catalog: list[CatalogEntry] | None = None,
    stack: dict[str, Any] | None = None,
) -> list[LocalModel]:
    """Scan the models directory for locally downloaded models.

    Args:
        models_dir: Override models directory (uses config if None).
        catalog: Pre-loaded catalog (loads from package if None).
        stack: Pre-loaded stack definition (loads from file if None).

    Returns:
        List of LocalModel entries found on disk, sorted by name.
    """
    if models_dir is None:
        models_dir = get_models_directory()

    if not models_dir.exists() or not models_dir.is_dir():
        return []

    if catalog is None:
        try:
            catalog = load_catalog()
        except Exception:
            catalog = []

    if stack is None:
        stack = _load_active_stack()

    active_stack_entries = _get_active_stack_models(stack)

    results: list[LocalModel] = []
    try:
        entries = sorted(models_dir.iterdir())
    except OSError:
        return []

    for item in entries:
        if not item.is_dir():
            continue

        dirname = item.name
        disk_size = _compute_dir_size(item)
        quant = _detect_quant(item, dirname)
        source_type = _detect_source_type(dirname)

        # Try to match to catalog entry for human-readable name
        catalog_entry = _match_to_catalog(dirname, catalog)
        catalog_name = catalog_entry.name if catalog_entry else None

        # Check if this model is active in the stack using
        # source+quant-aware identity matching.
        is_active = False
        for stack_entry in active_stack_entries:
            stack_source = stack_entry.get("source", "")
            stack_quant = stack_entry.get("quant", "")
            stack_model_id = stack_entry.get("model_id", "")
            source_dir = (
                stack_source.rsplit("/", 1)[-1]
                if "/" in stack_source
                else stack_source
            )

            # Primary match: source directory name matches AND quant matches
            if source_dir and source_dir == dirname:
                # Source dir match found — verify quant compatibility
                if not stack_quant or quant == "unknown" or stack_quant == quant:
                    is_active = True
                    break

            # Secondary match: model_id matches dirname AND quant matches
            if stack_model_id == dirname:
                if not stack_quant or quant == "unknown" or stack_quant == quant:
                    is_active = True
                    break

            # Tertiary match: catalog entry ID matches stack model ID,
            # AND the local model's source matches the catalog source for
            # the stack's quant
            if catalog_entry and catalog_entry.id == stack_model_id:
                # Verify quant-aware source match
                if stack_quant in catalog_entry.sources:
                    expected_source = catalog_entry.sources[stack_quant]
                    expected_dir = (
                        expected_source.hf_repo.rsplit("/", 1)[-1]
                        if "/" in expected_source.hf_repo
                        else expected_source.hf_repo
                    )
                    if expected_dir == dirname:
                        is_active = True
                        break
                elif not stack_quant or quant == "unknown" or stack_quant == quant:
                    # Fallback: catalog match with compatible quant
                    is_active = True
                    break

        results.append(
            LocalModel(
                name=dirname,
                path=item,
                disk_size_bytes=disk_size,
                quant=quant,
                source_type=source_type,
                is_active=is_active,
                catalog_name=catalog_name,
            )
        )

    return results


# --------------------------------------------------------------------------- #
# Remote-only stack models (in stack but not downloaded)
# --------------------------------------------------------------------------- #


def get_remote_stack_models(
    local_models: list[LocalModel],
    stack: dict[str, Any] | None = None,
    catalog: list[CatalogEntry] | None = None,
) -> list[dict[str, Any]]:
    """Find models referenced in the active stack but not downloaded locally.

    Uses source+quant-aware matching to determine whether a stack model
    is available locally.

    Args:
        local_models: Already-scanned local models.
        stack: Pre-loaded stack definition (loads from file if None).
        catalog: Pre-loaded catalog (loads from package if None).

    Returns:
        List of dicts with model info for remote-only stack models.
    """
    if stack is None:
        stack = _load_active_stack()

    if stack is None:
        return []

    if catalog is None:
        try:
            catalog = load_catalog()
        except Exception:
            catalog = []

    active_stack_entries = _get_active_stack_models(stack)

    remote_models: list[dict[str, Any]] = []
    for stack_entry in active_stack_entries:
        model_id = stack_entry["model_id"]
        stack_source = stack_entry.get("source", "")
        stack_quant = stack_entry.get("quant", "int4")
        source_dir = (
            stack_source.rsplit("/", 1)[-1]
            if "/" in stack_source
            else stack_source
        )

        # Check if locally available using source+quant-aware matching
        is_local = False

        for lm in local_models:
            # Match by source directory name + quant
            if source_dir and source_dir == lm.name:
                if not stack_quant or lm.quant == "unknown" or stack_quant == lm.quant:
                    is_local = True
                    break

            # Match by model_id as dirname + quant
            if model_id == lm.name:
                if not stack_quant or lm.quant == "unknown" or stack_quant == lm.quant:
                    is_local = True
                    break

            # Match by catalog entry with quant-aware source matching
            cat_entry = _find_catalog_entry(catalog, model_id)
            if cat_entry and lm.catalog_name and cat_entry.name == lm.catalog_name:
                # Verify quant-aware source match
                if stack_quant in cat_entry.sources:
                    expected_source = cat_entry.sources[stack_quant]
                    expected_dir = (
                        expected_source.hf_repo.rsplit("/", 1)[-1]
                        if "/" in expected_source.hf_repo
                        else expected_source.hf_repo
                    )
                    if expected_dir == lm.name:
                        is_local = True
                        break
                elif not stack_quant or lm.quant == "unknown" or stack_quant == lm.quant:
                    is_local = True
                    break

        if not is_local:
            # Find estimated download size from catalog
            entry = _find_catalog_entry(catalog, model_id)
            est_size_gb: float | None = None
            catalog_name: str | None = None
            if entry:
                catalog_name = entry.name
                if stack_quant in entry.sources:
                    est_size_gb = entry.sources[stack_quant].disk_size_gb

            remote_models.append(
                {
                    "model_id": model_id,
                    "tier": stack_entry.get("tier", ""),
                    "quant": stack_quant,
                    "source": "remote",
                    "catalog_name": catalog_name or model_id,
                    "est_size_gb": est_size_gb,
                }
            )

    return remote_models


def _find_catalog_entry(catalog: list[CatalogEntry], model_id: str) -> CatalogEntry | None:
    """Find a catalog entry by model ID."""
    for entry in catalog:
        if entry.id == model_id:
            return entry
    return None


# --------------------------------------------------------------------------- #
# Catalog listing with hardware-specific data
# --------------------------------------------------------------------------- #


def list_catalog_models(
    catalog: list[CatalogEntry] | None = None,
    profile: HardwareProfile | None = None,
    local_models: list[LocalModel] | None = None,
) -> list[CatalogModel]:
    """Build a catalog listing with hardware-specific benchmark data.

    Args:
        catalog: Pre-loaded catalog (loads from package if None).
        profile: Hardware profile for benchmark lookup (loads from file if None).
        local_models: Pre-scanned local models for local availability check.

    Returns:
        List of CatalogModel entries for display.
    """
    if catalog is None:
        catalog = load_catalog()

    if profile is None:
        profile = load_profile()

    if local_models is None:
        local_models = scan_local_models(catalog=catalog)

    # Build set of locally available model IDs (via catalog match)
    local_catalog_ids: set[str] = set()
    for lm in local_models:
        if lm.catalog_name:
            # Find ID from catalog by name
            for entry in catalog:
                if entry.name == lm.catalog_name:
                    local_catalog_ids.add(entry.id)
                    break
        # Also check if dirname matches any catalog entry's source repo
        for entry in catalog:
            for _quant, source in entry.sources.items():
                repo_name = (
                    source.hf_repo.rsplit("/", 1)[-1] if "/" in source.hf_repo else source.hf_repo
                )
                if repo_name == lm.name:
                    local_catalog_ids.add(entry.id)

    results: list[CatalogModel] = []
    for entry in catalog:
        quants = sorted(entry.sources.keys())

        gen_tps: float | None = None
        memory_gb: float | None = None
        prompt_tps: float | None = None
        is_estimated = False

        if profile is not None:
            profile_id = profile.profile_id
            if profile_id in entry.benchmarks:
                bench = entry.benchmarks[profile_id]
                gen_tps = bench.gen_tps
                memory_gb = bench.memory_gb
                prompt_tps = bench.prompt_tps
            else:
                # Bandwidth-ratio estimation
                estimated = _estimate_benchmark(entry, profile)
                if estimated:
                    gen_tps, memory_gb, prompt_tps = estimated
                    is_estimated = True

        is_local = entry.id in local_catalog_ids

        results.append(
            CatalogModel(
                id=entry.id,
                name=entry.name,
                family=entry.family,
                params_b=entry.params_b,
                quants=quants,
                gen_tps=gen_tps,
                memory_gb=memory_gb,
                prompt_tps=prompt_tps,
                is_estimated=is_estimated,
                is_local=is_local,
                tags=entry.tags,
            )
        )

    return results


def _estimate_benchmark(
    entry: CatalogEntry,
    profile: HardwareProfile,
) -> tuple[float, float, float] | None:
    """Estimate benchmark data using bandwidth ratio.

    Args:
        entry: Catalog entry with benchmark data.
        profile: Hardware profile with bandwidth info.

    Returns:
        Tuple of (gen_tps, memory_gb, prompt_tps) or None if no reference data.
    """
    from mlx_stack.core.scoring import _REFERENCE_PROFILES

    if not entry.benchmarks:
        return None

    # Find a reference benchmark
    ref_profile_id: str | None = None
    ref_bench = None

    for pid, bench in entry.benchmarks.items():
        if pid in _REFERENCE_PROFILES:
            ref_profile_id = pid
            ref_bench = bench
            break

    if ref_profile_id is None or ref_bench is None:
        # Use first available benchmark
        ref_profile_id = next(iter(entry.benchmarks))
        ref_bench = entry.benchmarks[ref_profile_id]

    ref_bw = _REFERENCE_PROFILES.get(ref_profile_id)
    if ref_bw is None:
        # Can't estimate without reference bandwidth
        # Use benchmark data as-is
        return ref_bench.gen_tps, ref_bench.memory_gb, ref_bench.prompt_tps

    ratio = profile.bandwidth_gbps / ref_bw
    return (
        round(ref_bench.gen_tps * ratio, 1),
        ref_bench.memory_gb,  # Memory usage is hardware-independent
        round(ref_bench.prompt_tps * ratio, 1),
    )


# --------------------------------------------------------------------------- #
# Human-readable size formatting
# --------------------------------------------------------------------------- #


def format_size(size_bytes: int) -> str:
    """Format a byte count into a human-readable size string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Formatted string like '5.2 GB', '850 MB', '120 KB'.
    """
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1_000_000_000:.1f} GB"
    if size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.0f} MB"
    if size_bytes >= 1_000:
        return f"{size_bytes / 1_000:.0f} KB"
    return f"{size_bytes} B"
