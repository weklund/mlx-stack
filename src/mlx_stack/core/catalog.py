"""Model catalog system for mlx-stack.

Loads, validates, and queries the curated catalog of MLX-compatible models
shipped as YAML data files with the package. Each catalog entry describes
a model's identity, architecture, quantization sources, capabilities,
quality scores, hardware benchmarks, and tags.

Uses importlib.resources for accessing package data files.
"""

from __future__ import annotations

import importlib.resources
from dataclasses import dataclass, field
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class CatalogError(Exception):
    """Raised when catalog loading or validation fails."""


# --------------------------------------------------------------------------- #
# Schema — required fields and their expected types
# --------------------------------------------------------------------------- #

# Top-level required fields and their types
_REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "id": str,
    "name": str,
    "family": str,
    "params_b": (int, float),
    "architecture": str,
    "min_mlx_lm_version": str,
    "sources": dict,
    "capabilities": dict,
    "quality": dict,
    "benchmarks": dict,
    "tags": list,
}

# Required capability fields
_REQUIRED_CAPABILITIES: set[str] = {
    "tool_calling",
    "tool_call_parser",
    "thinking",
    "reasoning_parser",
    "vision",
}

# Required quality fields
_REQUIRED_QUALITY_FIELDS: set[str] = {
    "overall",
    "coding",
    "reasoning",
    "instruction_following",
}

# Valid quantization levels
_VALID_QUANTS: set[str] = {"int4", "int8", "bf16"}

# Required source fields per quant
_REQUIRED_SOURCE_FIELDS: set[str] = {"hf_repo", "disk_size_gb"}


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class QuantSource:
    """A quantization source for a model."""

    hf_repo: str
    disk_size_gb: float
    convert_from: bool = False


@dataclass(frozen=True)
class Capabilities:
    """Model capabilities."""

    tool_calling: bool
    tool_call_parser: str | None
    thinking: bool
    reasoning_parser: str | None
    vision: bool


@dataclass(frozen=True)
class QualityScores:
    """Model quality scores (0-100 scale)."""

    overall: int
    coding: int
    reasoning: int
    instruction_following: int


@dataclass(frozen=True)
class BenchmarkResult:
    """Benchmark data for a specific hardware profile."""

    prompt_tps: float
    gen_tps: float
    memory_gb: float


@dataclass(frozen=True)
class CatalogEntry:
    """A single model entry in the catalog."""

    id: str
    name: str
    family: str
    params_b: float
    architecture: str
    min_mlx_lm_version: str
    sources: dict[str, QuantSource]
    capabilities: Capabilities
    quality: QualityScores
    benchmarks: dict[str, BenchmarkResult]
    tags: list[str] = field(default_factory=list)
    gated: bool = False


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _validate_entry(data: dict[str, Any], filename: str) -> None:
    """Validate a single catalog entry against the expected schema.

    Args:
        data: Parsed YAML data for a catalog entry.
        filename: The filename, used in error messages.

    Raises:
        CatalogError: If validation fails.
    """
    # Check required top-level fields
    for field_name, expected_type in _REQUIRED_FIELDS.items():
        if field_name not in data:
            msg = f"Catalog file '{filename}': missing required field '{field_name}'"
            raise CatalogError(msg)
        if not isinstance(data[field_name], expected_type):
            msg = (
                f"Catalog file '{filename}': field '{field_name}' has wrong type "
                f"(expected {expected_type}, got {type(data[field_name]).__name__})"
            )
            raise CatalogError(msg)

    # Validate sources — each quant must have required fields
    sources = data["sources"]
    if not sources:
        msg = f"Catalog file '{filename}': 'sources' must not be empty"
        raise CatalogError(msg)
    for quant, source_data in sources.items():
        if quant not in _VALID_QUANTS:
            msg = (
                f"Catalog file '{filename}': invalid quantization '{quant}' "
                f"(valid: {', '.join(sorted(_VALID_QUANTS))})"
            )
            raise CatalogError(msg)
        if not isinstance(source_data, dict):
            msg = f"Catalog file '{filename}': source for quant '{quant}' must be a mapping"
            raise CatalogError(msg)
        for req_field in _REQUIRED_SOURCE_FIELDS:
            if req_field not in source_data:
                msg = (
                    f"Catalog file '{filename}': source '{quant}' missing "
                    f"required field '{req_field}'"
                )
                raise CatalogError(msg)
        # Validate disk_size_gb is numeric
        disk_size = source_data["disk_size_gb"]
        if not isinstance(disk_size, (int, float)):
            msg = (
                f"Catalog file '{filename}': source '{quant}' field 'disk_size_gb' "
                f"must be numeric, got {type(disk_size).__name__}"
            )
            raise CatalogError(msg)

    # Validate capabilities
    caps = data["capabilities"]
    for cap_field in _REQUIRED_CAPABILITIES:
        if cap_field not in caps:
            msg = f"Catalog file '{filename}': capabilities missing required field '{cap_field}'"
            raise CatalogError(msg)

    # Validate quality scores
    quality = data["quality"]
    for q_field in _REQUIRED_QUALITY_FIELDS:
        if q_field not in quality:
            msg = f"Catalog file '{filename}': quality missing required field '{q_field}'"
            raise CatalogError(msg)
        q_value = quality[q_field]
        if not isinstance(q_value, (int, float)):
            msg = (
                f"Catalog file '{filename}': quality field '{q_field}' "
                f"must be numeric, got {type(q_value).__name__}"
            )
            raise CatalogError(msg)

    # Validate benchmarks — each entry must have prompt_tps, gen_tps, memory_gb
    benchmarks = data["benchmarks"]
    for hw_key, bench_data in benchmarks.items():
        if not isinstance(bench_data, dict):
            msg = f"Catalog file '{filename}': benchmark entry '{hw_key}' must be a mapping"
            raise CatalogError(msg)
        for req_field in ("prompt_tps", "gen_tps", "memory_gb"):
            if req_field not in bench_data:
                msg = (
                    f"Catalog file '{filename}': benchmark '{hw_key}' missing "
                    f"required field '{req_field}'"
                )
                raise CatalogError(msg)
            bench_value = bench_data[req_field]
            if not isinstance(bench_value, (int, float)):
                msg = (
                    f"Catalog file '{filename}': benchmark '{hw_key}' field "
                    f"'{req_field}' must be numeric, got {type(bench_value).__name__}"
                )
                raise CatalogError(msg)

    # Validate tags is a list of strings
    for tag in data["tags"]:
        if not isinstance(tag, str):
            msg = f"Catalog file '{filename}': tags must be strings, got {type(tag).__name__}"
            raise CatalogError(msg)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def _parse_entry(data: dict[str, Any]) -> CatalogEntry:
    """Parse a validated dictionary into a CatalogEntry.

    Args:
        data: Validated YAML data.

    Returns:
        A CatalogEntry instance.

    Raises:
        CatalogError: If type coercion fails for any nested field value.
    """
    model_id = data.get("id", "<unknown>")

    # Parse sources
    sources: dict[str, QuantSource] = {}
    for quant, source_data in data["sources"].items():
        try:
            sources[quant] = QuantSource(
                hf_repo=source_data["hf_repo"],
                disk_size_gb=float(source_data["disk_size_gb"]),
                convert_from=bool(source_data.get("convert_from", False)),
            )
        except (ValueError, TypeError) as exc:
            msg = f"Catalog entry '{model_id}': invalid value in source '{quant}': {exc}"
            raise CatalogError(msg) from None

    # Parse capabilities
    caps_data = data["capabilities"]
    try:
        capabilities = Capabilities(
            tool_calling=bool(caps_data["tool_calling"]),
            tool_call_parser=caps_data.get("tool_call_parser") or None,
            thinking=bool(caps_data["thinking"]),
            reasoning_parser=caps_data.get("reasoning_parser") or None,
            vision=bool(caps_data["vision"]),
        )
    except (ValueError, TypeError) as exc:
        msg = f"Catalog entry '{model_id}': invalid value in capabilities: {exc}"
        raise CatalogError(msg) from None

    # Parse quality scores
    q_data = data["quality"]
    try:
        quality = QualityScores(
            overall=int(q_data["overall"]),
            coding=int(q_data["coding"]),
            reasoning=int(q_data["reasoning"]),
            instruction_following=int(q_data["instruction_following"]),
        )
    except (ValueError, TypeError) as exc:
        msg = f"Catalog entry '{model_id}': invalid value in quality scores: {exc}"
        raise CatalogError(msg) from None

    # Parse benchmarks
    benchmarks: dict[str, BenchmarkResult] = {}
    for hw_key, bench_data in data["benchmarks"].items():
        try:
            benchmarks[hw_key] = BenchmarkResult(
                prompt_tps=float(bench_data["prompt_tps"]),
                gen_tps=float(bench_data["gen_tps"]),
                memory_gb=float(bench_data["memory_gb"]),
            )
        except (ValueError, TypeError) as exc:
            msg = f"Catalog entry '{model_id}': invalid value in benchmark '{hw_key}': {exc}"
            raise CatalogError(msg) from None

    try:
        return CatalogEntry(
            id=str(data["id"]),
            name=str(data["name"]),
            family=str(data["family"]),
            params_b=float(data["params_b"]),
            architecture=str(data["architecture"]),
            min_mlx_lm_version=str(data["min_mlx_lm_version"]),
            sources=sources,
            capabilities=capabilities,
            quality=quality,
            benchmarks=benchmarks,
            tags=list(data.get("tags", [])),
            gated=bool(data.get("gated", False)),
        )
    except (ValueError, TypeError) as exc:
        msg = f"Catalog entry '{model_id}': invalid top-level field value: {exc}"
        raise CatalogError(msg) from None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_catalog() -> list[CatalogEntry]:
    """Load all catalog entries from the shipped YAML data files.

    Uses importlib.resources to locate the catalog directory within the
    installed package. Each .yaml file in the catalog directory is loaded,
    validated, and parsed into a CatalogEntry.

    Returns:
        A list of CatalogEntry instances, sorted by family then params_b.

    Raises:
        CatalogError: If any catalog file is missing, corrupt, or invalid.
    """
    entries: list[CatalogEntry] = []

    try:
        catalog_pkg = importlib.resources.files("mlx_stack.data.catalog")
    except (ModuleNotFoundError, TypeError) as exc:
        msg = f"Could not locate catalog data directory: {exc}"
        raise CatalogError(msg) from None

    yaml_files: list[Any] = []
    try:
        yaml_files.extend(
            item
            for item in catalog_pkg.iterdir()
            if hasattr(item, "name") and item.name.endswith(".yaml")
        )
    except (OSError, TypeError) as exc:
        msg = f"Could not read catalog directory: {exc}"
        raise CatalogError(msg) from None

    if not yaml_files:
        msg = "No catalog YAML files found — catalog directory is empty"
        raise CatalogError(msg)

    for yaml_file in sorted(yaml_files, key=lambda f: f.name):
        filename = yaml_file.name
        try:
            content = yaml_file.read_text(encoding="utf-8")
        except OSError as exc:
            msg = f"Could not read catalog file '{filename}': {exc}"
            raise CatalogError(msg) from None

        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            msg = f"Catalog file '{filename}' contains invalid YAML: {exc}"
            raise CatalogError(msg) from None

        if not isinstance(data, dict):
            actual_type = type(data).__name__
            msg = f"Catalog file '{filename}' must contain a YAML mapping, got {actual_type}"
            raise CatalogError(msg) from None

        _validate_entry(data, filename)
        entries.append(_parse_entry(data))

    # Sort by family, then by params_b ascending
    entries.sort(key=lambda e: (e.family, e.params_b))

    return entries


def load_catalog_from_directory(directory: str) -> list[CatalogEntry]:
    """Load catalog entries from an arbitrary directory.

    This is useful for testing with custom catalog files.

    Args:
        directory: Path to a directory containing YAML catalog files.

    Returns:
        A list of CatalogEntry instances, sorted by family then params_b.

    Raises:
        CatalogError: If any catalog file is missing, corrupt, or invalid.
    """
    from pathlib import Path

    catalog_dir = Path(directory)

    if not catalog_dir.is_dir():
        msg = f"Catalog directory not found: {directory}"
        raise CatalogError(msg)

    yaml_files = sorted(catalog_dir.glob("*.yaml"))

    if not yaml_files:
        msg = f"No catalog YAML files found in '{directory}'"
        raise CatalogError(msg)

    entries: list[CatalogEntry] = []

    for yaml_file in yaml_files:
        filename = yaml_file.name
        try:
            content = yaml_file.read_text(encoding="utf-8")
        except OSError as exc:
            msg = f"Could not read catalog file '{filename}': {exc}"
            raise CatalogError(msg) from None

        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            msg = f"Catalog file '{filename}' contains invalid YAML: {exc}"
            raise CatalogError(msg) from None

        if not isinstance(data, dict):
            actual_type = type(data).__name__
            msg = f"Catalog file '{filename}' must contain a YAML mapping, got {actual_type}"
            raise CatalogError(msg) from None

        _validate_entry(data, filename)
        entries.append(_parse_entry(data))

    entries.sort(key=lambda e: (e.family, e.params_b))

    return entries


# --------------------------------------------------------------------------- #
# Querying
# --------------------------------------------------------------------------- #


def get_entry_by_id(catalog: list[CatalogEntry], model_id: str) -> CatalogEntry | None:
    """Look up a catalog entry by its model ID.

    Args:
        catalog: The loaded catalog.
        model_id: The model ID to look up.

    Returns:
        The matching CatalogEntry, or None if not found.
    """
    for entry in catalog:
        if entry.id == model_id:
            return entry
    return None


def query_by_family(catalog: list[CatalogEntry], family: str) -> list[CatalogEntry]:
    """Filter catalog entries by model family (case-insensitive).

    Args:
        catalog: The loaded catalog.
        family: The family name to filter by.

    Returns:
        A list of matching CatalogEntry instances.
    """
    family_lower = family.lower()
    return [e for e in catalog if e.family.lower() == family_lower]


def query_by_tag(catalog: list[CatalogEntry], tag: str) -> list[CatalogEntry]:
    """Filter catalog entries by tag (case-insensitive).

    Args:
        catalog: The loaded catalog.
        tag: The tag to filter by.

    Returns:
        A list of matching CatalogEntry instances.
    """
    tag_lower = tag.lower()
    return [e for e in catalog if tag_lower in [t.lower() for t in e.tags]]


def query_by_capability(
    catalog: list[CatalogEntry],
    **capabilities: bool,
) -> list[CatalogEntry]:
    """Filter catalog entries by capability flags.

    Supports filtering by any combination of: tool_calling, thinking, vision.

    Args:
        catalog: The loaded catalog.
        **capabilities: Capability flags to filter by (e.g., tool_calling=True).

    Returns:
        A list of matching CatalogEntry instances.

    Raises:
        ValueError: If an invalid capability name is given.
    """
    valid_caps = {"tool_calling", "thinking", "vision"}
    for cap_name in capabilities:
        if cap_name not in valid_caps:
            msg = f"Invalid capability filter '{cap_name}' (valid: {', '.join(sorted(valid_caps))})"
            raise ValueError(msg)

    results: list[CatalogEntry] = []
    for entry in catalog:
        match = True
        for cap_name, cap_value in capabilities.items():
            actual = getattr(entry.capabilities, cap_name)
            if actual != cap_value:
                match = False
                break
        if match:
            results.append(entry)
    return results
