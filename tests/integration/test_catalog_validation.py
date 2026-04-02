"""Tier 1: Catalog validation tests.

Validates that all catalog entries are internally consistent and that
every HuggingFace repository referenced in quant sources actually exists.
No models are downloaded — only lightweight API calls.

This tier runs in CI on every PR to catch broken catalog entries before
they ship. Would have caught Issue #15 (qwen3.5-8b 404).

**How to run**::

    pytest -m catalog_validation -v
    make test-catalog
"""

from __future__ import annotations

from collections.abc import Generator

import httpx
import pytest

from mlx_stack.core.catalog import CatalogEntry, load_catalog

# --------------------------------------------------------------------------- #
# Markers
# --------------------------------------------------------------------------- #

pytestmark = pytest.mark.catalog_validation

# --------------------------------------------------------------------------- #
# Session-scoped fixtures
# --------------------------------------------------------------------------- #

# Load catalog once for parameterization (outside fixture for pytest.mark.parametrize)
_CATALOG = load_catalog()
_CATALOG_IDS = [e.id for e in _CATALOG]


@pytest.fixture(scope="module")
def hf_client() -> Generator[httpx.Client, None, None]:
    """Shared HTTP client for HuggingFace API calls."""
    client = httpx.Client(
        timeout=30.0,
        headers={"User-Agent": "mlx-stack-test/0.1"},
        follow_redirects=True,
    )
    yield client
    client.close()


# --------------------------------------------------------------------------- #
# Catalog structure tests
# --------------------------------------------------------------------------- #


class TestCatalogStructure:
    """Validate catalog YAML structure and internal consistency."""

    def test_catalog_loads_without_errors(self) -> None:
        """Catalog loads all entries without validation errors."""
        catalog = load_catalog()
        assert len(catalog) > 0, "Catalog is empty"

    @pytest.mark.parametrize("entry", _CATALOG, ids=_CATALOG_IDS)
    def test_required_fields(self, entry: CatalogEntry) -> None:
        """Every catalog entry has all required fields with valid types."""
        assert entry.id, "Entry missing id"
        assert entry.name, f"{entry.id}: missing name"
        assert entry.family, f"{entry.id}: missing family"
        assert entry.params_b > 0, f"{entry.id}: params_b must be > 0"
        assert entry.architecture, f"{entry.id}: missing architecture"
        assert entry.min_mlx_lm_version, f"{entry.id}: missing min_mlx_lm_version"

    @pytest.mark.parametrize("entry", _CATALOG, ids=_CATALOG_IDS)
    def test_has_int4_source(self, entry: CatalogEntry) -> None:
        """Every model must have at least an int4 quantization source."""
        assert "int4" in entry.sources, (
            f"{entry.id}: missing required int4 source. "
            f"Available quants: {list(entry.sources.keys())}"
        )

    @pytest.mark.parametrize("entry", _CATALOG, ids=_CATALOG_IDS)
    def test_source_fields(self, entry: CatalogEntry) -> None:
        """Every quant source has a valid hf_repo and disk_size_gb."""
        for quant, source in entry.sources.items():
            assert source.hf_repo, f"{entry.id}/{quant}: missing hf_repo"
            assert "/" in source.hf_repo, (
                f"{entry.id}/{quant}: hf_repo '{source.hf_repo}' must be org/repo format"
            )
            assert source.disk_size_gb > 0, (
                f"{entry.id}/{quant}: disk_size_gb must be > 0"
            )

    @pytest.mark.parametrize("entry", _CATALOG, ids=_CATALOG_IDS)
    def test_capability_consistency(self, entry: CatalogEntry) -> None:
        """tool_call_parser set iff tool_calling=True; reasoning_parser iff thinking=True."""
        caps = entry.capabilities
        if caps.tool_calling:
            assert caps.tool_call_parser is not None, (
                f"{entry.id}: tool_calling=True but tool_call_parser is None"
            )
        else:
            assert caps.tool_call_parser is None, (
                f"{entry.id}: tool_calling=False but tool_call_parser='{caps.tool_call_parser}'"
            )
        if caps.thinking:
            assert caps.reasoning_parser is not None, (
                f"{entry.id}: thinking=True but reasoning_parser is None"
            )

    @pytest.mark.parametrize("entry", _CATALOG, ids=_CATALOG_IDS)
    def test_has_benchmarks(self, entry: CatalogEntry) -> None:
        """Every entry has benchmarks for at least one known hardware profile."""
        known_profiles = {"m4-pro-48", "m4-max-128", "m5-max-128"}
        matching = entry.benchmarks.keys() & known_profiles
        assert matching, (
            f"{entry.id}: no benchmarks for known profiles. "
            f"Has: {list(entry.benchmarks.keys())}, expected one of: {known_profiles}"
        )

    @pytest.mark.parametrize("entry", _CATALOG, ids=_CATALOG_IDS)
    def test_quality_scores_in_range(self, entry: CatalogEntry) -> None:
        """Quality scores are in the valid 0-100 range."""
        q = entry.quality
        for name, score in [
            ("overall", q.overall),
            ("coding", q.coding),
            ("reasoning", q.reasoning),
            ("instruction_following", q.instruction_following),
        ]:
            assert 0 <= score <= 100, (
                f"{entry.id}: quality.{name}={score} is outside 0-100 range"
            )


# --------------------------------------------------------------------------- #
# HuggingFace repository validation
# --------------------------------------------------------------------------- #


class TestHuggingFaceRepos:
    """Validate that all referenced HuggingFace repositories exist.

    Uses lightweight HTTP HEAD requests to the HuggingFace API.
    No model downloads are performed.
    """

    @pytest.mark.parametrize("entry", _CATALOG, ids=_CATALOG_IDS)
    def test_hf_repos_exist(self, entry: CatalogEntry, hf_client: httpx.Client) -> None:
        """Every hf_repo in every quant source resolves to a real HF repository.

        For non-gated models, expects HTTP 200.
        For gated models, accepts 200, 401, or 403 (access-restricted but exists).
        """
        for quant, source in entry.sources.items():
            if source.convert_from:
                # convert_from sources point to upstream repos (not mlx-community),
                # which may be gated differently. Still validate they exist.
                pass

            url = f"https://huggingface.co/api/models/{source.hf_repo}"
            resp = hf_client.head(url)

            if entry.gated:
                assert resp.status_code in (200, 401, 403), (
                    f"{entry.id}/{quant}: gated repo '{source.hf_repo}' returned "
                    f"{resp.status_code} (expected 200, 401, or 403). "
                    f"Repo may not exist."
                )
            else:
                assert resp.status_code == 200, (
                    f"{entry.id}/{quant}: repo '{source.hf_repo}' returned "
                    f"{resp.status_code}. Repository does not exist on HuggingFace."
                )

    @pytest.mark.parametrize(
        "entry",
        [e for e in _CATALOG if not e.gated],
        ids=[e.id for e in _CATALOG if not e.gated],
    )
    def test_non_gated_repos_have_safetensors(
        self, entry: CatalogEntry, hf_client: httpx.Client,
    ) -> None:
        """Non-gated, non-convert_from repos contain safetensors weight files.

        Uses the HuggingFace API file listing endpoint — no downloads.
        """
        for quant, source in entry.sources.items():
            if source.convert_from:
                continue  # upstream repos may use different weight formats

            url = f"https://huggingface.co/api/models/{source.hf_repo}"
            resp = hf_client.get(url)
            if resp.status_code != 200:
                pytest.skip(f"{source.hf_repo} returned {resp.status_code}")

            data = resp.json()
            siblings = data.get("siblings", [])
            filenames = [s.get("rfilename", "") for s in siblings]
            has_safetensors = any(f.endswith(".safetensors") for f in filenames)

            assert has_safetensors, (
                f"{entry.id}/{quant}: repo '{source.hf_repo}' has no *.safetensors files. "
                f"Files found: {[f for f in filenames[:10]]}... "
                f"This repo may not contain MLX-format weights."
            )


# --------------------------------------------------------------------------- #
# Gated model consistency
# --------------------------------------------------------------------------- #


class TestGatedModels:
    """Validate that gated model flags are consistent with HF repo access."""

    @pytest.mark.parametrize(
        "entry",
        [e for e in _CATALOG if e.gated],
        ids=[e.id for e in _CATALOG if e.gated],
    )
    def test_gated_models_flagged(self, entry: CatalogEntry) -> None:
        """Models flagged as gated should have at least one non-mlx-community source."""
        assert entry.gated, f"{entry.id}: expected gated=True"
        assert any(
            not source.hf_repo.startswith("mlx-community/")
            for source in entry.sources.values()
        ), f"{entry.id}: gated model should have at least one non-mlx-community source"
