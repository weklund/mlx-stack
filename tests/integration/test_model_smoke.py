"""Tier 2: Per-model smoke tests.

Downloads each catalog model, starts vllm-mlx, and validates that basic
inference works. For models with tool_calling or thinking capabilities,
also validates those features.

Produces a compatibility matrix showing which models pass/fail.

Would have caught Issue #15 (qwen3.5-0.8b ArraysCache crash on inference).

**Requirements:**
  - macOS (Apple Silicon)
  - vllm-mlx installed
  - Sufficient memory for each model (checked at runtime)
  - Run explicitly: ``pytest -m smoke -v``

**How to run a single model**::

    pytest -m smoke -k qwen3.5-0.8b -v
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import httpx
import psutil
import pytest

from mlx_stack.core.catalog import CatalogEntry, load_catalog
from mlx_stack.core.process import HealthCheckError
from mlx_stack.core.pull import pull_model

from .conftest import (
    PortAllocator,
    ServiceManager,
    check_memory_or_skip,
    model_timeout,
    skip_no_vllm,
    skip_not_darwin,
)
from .report import CompatibilityMatrix, ModelTestResult

# --------------------------------------------------------------------------- #
# Markers
# --------------------------------------------------------------------------- #

pytestmark = [pytest.mark.smoke, skip_not_darwin, skip_no_vllm]

# --------------------------------------------------------------------------- #
# Catalog loading for parameterization
# --------------------------------------------------------------------------- #

_CATALOG = load_catalog()
_CATALOG_IDS = [e.id for e in _CATALOG]
_TOOL_CALLING_ENTRIES = [e for e in _CATALOG if e.capabilities.tool_calling]
_TOOL_CALLING_IDS = [e.id for e in _TOOL_CALLING_ENTRIES]
_THINKING_ENTRIES = [e for e in _CATALOG if e.capabilities.thinking]
_THINKING_IDS = [e.id for e in _THINKING_ENTRIES]

# Default quant for smoke tests
SMOKE_QUANT = "int4"

# --------------------------------------------------------------------------- #
# Module-scoped compatibility matrix
# --------------------------------------------------------------------------- #

_matrix = CompatibilityMatrix()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write compatibility matrix at end of smoke test session."""
    if _matrix.results:
        path = _matrix.write()
        print(f"\nCompatibility matrix written to: {path}")
        summary = _matrix.summary()
        print(f"Results: {summary['pass']} pass, {summary['fail']} fail, {summary['skip']} skip")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _min_memory_for_entry(entry: CatalogEntry) -> float:
    """Estimate minimum memory needed to load a model (GB)."""
    if entry.benchmarks:
        return min(b.memory_gb for b in entry.benchmarks.values())
    # Fallback: rough estimate from params
    return entry.params_b * 0.6  # int4 ≈ 0.5 bytes/param + overhead


def _build_vllm_flags(entry: CatalogEntry) -> dict:
    """Build vllm flags based on model capabilities."""
    flags: dict = {
        "continuous_batching": True,
        "use_paged_cache": True,
    }
    if entry.capabilities.tool_calling:
        flags["enable_auto_tool_choice"] = True
        if entry.capabilities.tool_call_parser:
            flags["tool_call_parser"] = entry.capabilities.tool_call_parser
    return flags


# --------------------------------------------------------------------------- #
# Smoke tests
# --------------------------------------------------------------------------- #


class TestModelSmoke:
    """Per-model smoke tests parameterized over the catalog."""

    @pytest.mark.parametrize("entry", _CATALOG, ids=_CATALOG_IDS)
    def test_basic_inference(
        self,
        entry: CatalogEntry,
        integration_home: Path,
        port_allocator: PortAllocator,
        model_cache_dir: Path,
    ) -> None:
        """Model loads in vllm-mlx and responds to a trivial prompt.

        Steps:
            1. Check memory requirements (skip if insufficient)
            2. Pull model (uses persistent cache)
            3. Start vllm-mlx on a dynamic port
            4. Send a minimal chat completion request
            5. Assert response is non-empty
            6. Record result in compatibility matrix
        """
        result = ModelTestResult(model_id=entry.id, quant=SMOKE_QUANT)

        # Skip gated models without HF_TOKEN
        if entry.gated and not os.environ.get("HF_TOKEN"):
            result.inference = "skip"
            result.error = "Gated model, HF_TOKEN not set"
            _matrix.add_result(result)
            pytest.skip(f"{entry.id} is gated and HF_TOKEN not set")

        # Skip if insufficient memory
        min_mem = _min_memory_for_entry(entry)
        try:
            check_memory_or_skip(min_mem * 1.5)
        except pytest.skip.Exception:
            result.inference = "skip"
            result.error = f"Insufficient memory (need {min_mem * 1.5:.1f}GB)"
            _matrix.add_result(result)
            raise

        port = port_allocator.allocate()
        source = entry.sources[SMOKE_QUANT]

        try:
            # Pull model (cached across runs)
            pull_result = pull_model(model_id=entry.id, quant=SMOKE_QUANT)
            assert pull_result.local_path.exists(), (
                f"Model not found at {pull_result.local_path}"
            )

            # Start vllm-mlx
            start_time = time.monotonic()
            with ServiceManager(integration_home) as svc:
                try:
                    managed = svc.start_vllm(
                        tier_name=f"smoke-{entry.id}",
                        model_source=source.hf_repo,
                        port=port,
                        vllm_flags=_build_vllm_flags(entry),
                        timeout=model_timeout(entry.params_b),
                    )
                except HealthCheckError:
                    result.inference = "fail"
                    result.error = "Health check timed out — model failed to start"
                    _matrix.add_result(result)
                    pytest.fail(
                        f"{entry.id}: vllm-mlx failed health check after "
                        f"{model_timeout(entry.params_b)}s"
                    )

                result.startup_time_s = time.monotonic() - start_time

                # Record memory usage
                if managed.pid:
                    try:
                        proc = psutil.Process(managed.pid)
                        result.memory_actual_gb = proc.memory_info().rss / (1024**3)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                result.memory_catalog_gb = min_mem

                # Send inference request
                inference_start = time.monotonic()
                response = httpx.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={
                        "model": source.hf_repo,
                        "messages": [{"role": "user", "content": "Say hello."}],
                        "max_tokens": 5,
                    },
                    timeout=120.0,
                )
                result.inference_time_s = time.monotonic() - inference_start

                assert response.status_code == 200, (
                    f"{entry.id}: inference returned {response.status_code}: "
                    f"{response.text[:300]}"
                )

                data = response.json()
                content = data["choices"][0]["message"]["content"]
                assert content and len(content.strip()) > 0, (
                    f"{entry.id}: inference returned empty content: {data}"
                )

                result.inference = "pass"

        except Exception as e:
            if result.inference != "fail":
                result.inference = "fail"
                result.error = str(e)[:200]
            _matrix.add_result(result)
            raise
        else:
            _matrix.add_result(result)
        finally:
            port_allocator.release(port)

    @pytest.mark.parametrize("entry", _TOOL_CALLING_ENTRIES, ids=_TOOL_CALLING_IDS)
    def test_tool_calling(
        self,
        entry: CatalogEntry,
        integration_home: Path,
        port_allocator: PortAllocator,
        model_cache_dir: Path,
    ) -> None:
        """Models with tool_calling=True can handle function-calling requests.

        Sends a chat completion with a simple tool definition and validates
        the response contains a tool_calls array.
        """
        if entry.gated and not os.environ.get("HF_TOKEN"):
            pytest.skip(f"{entry.id} is gated and HF_TOKEN not set")

        min_mem = _min_memory_for_entry(entry)
        check_memory_or_skip(min_mem * 1.5)

        port = port_allocator.allocate()
        source = entry.sources[SMOKE_QUANT]

        tool_definition = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name",
                        },
                    },
                    "required": ["location"],
                },
            },
        }

        try:
            pull_model(model_id=entry.id, quant=SMOKE_QUANT)

            with ServiceManager(integration_home) as svc:
                svc.start_vllm(
                    tier_name=f"smoke-tool-{entry.id}",
                    model_source=source.hf_repo,
                    port=port,
                    vllm_flags=_build_vllm_flags(entry),
                    timeout=model_timeout(entry.params_b),
                )

                response = httpx.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={
                        "model": source.hf_repo,
                        "messages": [
                            {"role": "user", "content": "What is the weather in San Francisco?"},
                        ],
                        "tools": [tool_definition],
                        "max_tokens": 100,
                    },
                    timeout=120.0,
                )

                assert response.status_code == 200, (
                    f"{entry.id}: tool calling returned {response.status_code}: "
                    f"{response.text[:300]}"
                )

                data = response.json()
                msg = data["choices"][0]["message"]
                tool_calls = msg.get("tool_calls")
                assert tool_calls is not None and len(tool_calls) > 0, (
                    f"{entry.id}: expected tool_calls in response but got: {msg}"
                )

                # Validate tool call structure
                tc = tool_calls[0]
                assert tc.get("function", {}).get("name") == "get_weather", (
                    f"{entry.id}: expected get_weather tool call, got: {tc}"
                )

                # Update matrix for this model
                for r in _matrix.results:
                    if r.model_id == entry.id:
                        r.tool_calling = "pass"
                        break

        except Exception:
            for r in _matrix.results:
                if r.model_id == entry.id:
                    r.tool_calling = "fail"
                    break
            raise
        finally:
            port_allocator.release(port)

    @pytest.mark.parametrize("entry", _THINKING_ENTRIES, ids=_THINKING_IDS)
    def test_thinking(
        self,
        entry: CatalogEntry,
        integration_home: Path,
        port_allocator: PortAllocator,
        model_cache_dir: Path,
    ) -> None:
        """Models with thinking=True produce reasoning in their response."""
        if entry.gated and not os.environ.get("HF_TOKEN"):
            pytest.skip(f"{entry.id} is gated and HF_TOKEN not set")

        min_mem = _min_memory_for_entry(entry)
        check_memory_or_skip(min_mem * 1.5)

        port = port_allocator.allocate()
        source = entry.sources[SMOKE_QUANT]

        try:
            pull_model(model_id=entry.id, quant=SMOKE_QUANT)

            with ServiceManager(integration_home) as svc:
                svc.start_vllm(
                    tier_name=f"smoke-think-{entry.id}",
                    model_source=source.hf_repo,
                    port=port,
                    vllm_flags=_build_vllm_flags(entry),
                    timeout=model_timeout(entry.params_b),
                )

                response = httpx.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={
                        "model": source.hf_repo,
                        "messages": [
                            {"role": "user", "content": "What is 23 * 47? Think step by step."},
                        ],
                        "max_tokens": 200,
                    },
                    timeout=120.0,
                )

                assert response.status_code == 200, (
                    f"{entry.id}: thinking request returned {response.status_code}"
                )

                data = response.json()
                content = data["choices"][0]["message"]["content"]
                assert content and len(content.strip()) > 0, (
                    f"{entry.id}: thinking model returned empty response"
                )

                for r in _matrix.results:
                    if r.model_id == entry.id:
                        r.thinking = "pass"
                        break

        except Exception:
            for r in _matrix.results:
                if r.model_id == entry.id:
                    r.thinking = "fail"
                    break
            raise
        finally:
            port_allocator.release(port)
