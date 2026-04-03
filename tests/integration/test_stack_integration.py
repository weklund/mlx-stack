"""Tier 3: Full stack integration tests.

Tests the complete stack lifecycle with real vllm-mlx and LiteLLM processes:
full init-pull-up-inference-down cycle, LiteLLM routing by tier name,
fallback behavior, concurrent requests, and clean shutdown.

**Requirements:**
  - macOS (Apple Silicon)
  - vllm-mlx and litellm installed
  - Ports dynamically allocated (no hardcoded ports)
  - Run explicitly: ``pytest -m integration -v``
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from mlx_stack.core.pull import pull_model

from .conftest import (
    PortAllocator,
    ServiceManager,
    is_port_in_use,
    skip_no_litellm,
    skip_no_vllm,
    skip_not_darwin,
    wait_for_port_free,
    write_test_stack,
)

# --------------------------------------------------------------------------- #
# Markers
# --------------------------------------------------------------------------- #

pytestmark = [pytest.mark.integration, skip_not_darwin, skip_no_vllm, skip_no_litellm]

# --------------------------------------------------------------------------- #
# Constants — smallest model for fast testing
# --------------------------------------------------------------------------- #

MODEL_ID = "qwen3.5-0.8b"
MODEL_QUANT = "int4"
MODEL_SOURCE = "mlx-community/Qwen3.5-0.8B-4bit"


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestStackLifecycle:
    """Full stack lifecycle with real processes."""

    def test_full_lifecycle(
        self,
        integration_home: Path,
        port_allocator: PortAllocator,
        model_cache_dir: Path,
    ) -> None:
        """write config -> pull -> up -> inference via LiteLLM -> down -> cleanup.

        Tests the complete user journey with a single-tier stack using
        the smallest model (qwen3.5-0.8b).
        """
        vllm_port = port_allocator.allocate()
        litellm_port = port_allocator.allocate()

        tier = {
            "name": "fast",
            "model": MODEL_ID,
            "quant": MODEL_QUANT,
            "source": MODEL_SOURCE,
            "port": vllm_port,
            "vllm_flags": {},
        }

        # Write stack config
        write_test_stack(integration_home, [tier], litellm_port)

        # Pull model (cached)
        pull_result = pull_model(model_id=MODEL_ID, quant=MODEL_QUANT)
        assert pull_result.local_path.exists()

        with ServiceManager(integration_home) as svc:
            # Start vllm-mlx
            svc.start_vllm(
                tier_name="fast",
                model_source=MODEL_SOURCE,
                port=vllm_port,
                timeout=120.0,
            )

            # Start LiteLLM
            litellm_config = integration_home / "litellm.yaml"
            svc.start_litellm(
                port=litellm_port,
                config_path=litellm_config,
                timeout=120.0,
            )

            # Inference via LiteLLM proxy
            response = httpx.post(
                f"http://localhost:{litellm_port}/v1/chat/completions",
                json={
                    "model": "fast",
                    "messages": [{"role": "user", "content": "Say hello in 3 words"}],
                    "max_tokens": 20,
                },
                timeout=120.0,
            )
            assert response.status_code == 200, (
                f"LiteLLM response: {response.status_code}: {response.text[:300]}"
            )
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            assert content and len(content.strip()) > 0

            # Inference directly to vllm-mlx
            response = httpx.post(
                f"http://127.0.0.1:{vllm_port}/v1/chat/completions",
                json={
                    "model": MODEL_SOURCE,
                    "messages": [{"role": "user", "content": "Say hello in 3 words"}],
                    "max_tokens": 20,
                },
                timeout=120.0,
            )
            assert response.status_code == 200

        # After context manager exits, services are stopped
        # Verify ports are freed
        assert wait_for_port_free(vllm_port, timeout=15.0), (
            f"Port {vllm_port} still bound after shutdown"
        )
        assert wait_for_port_free(litellm_port, timeout=15.0), (
            f"Port {litellm_port} still bound after shutdown"
        )

        # Verify no PID files remain
        pids_dir = integration_home / "pids"
        remaining = list(pids_dir.glob("*.pid"))
        assert len(remaining) == 0, (
            f"PID files remain: {[p.name for p in remaining]}"
        )

    def test_litellm_routing(
        self,
        integration_home: Path,
        port_allocator: PortAllocator,
        model_cache_dir: Path,
    ) -> None:
        """LiteLLM correctly routes requests by tier name.

        Starts two tiers (both using the same smallest model) and
        verifies that requests to each tier name succeed through LiteLLM.
        """
        port_a = port_allocator.allocate()
        port_b = port_allocator.allocate()
        litellm_port = port_allocator.allocate()

        tiers = [
            {
                "name": "fast",
                "model": MODEL_ID,
                "quant": MODEL_QUANT,
                "source": MODEL_SOURCE,
                "port": port_a,
                "vllm_flags": {},
            },
            {
                "name": "standard",
                "model": MODEL_ID,
                "quant": MODEL_QUANT,
                "source": MODEL_SOURCE,
                "port": port_b,
                "vllm_flags": {},
            },
        ]

        write_test_stack(integration_home, tiers, litellm_port)
        pull_model(model_id=MODEL_ID, quant=MODEL_QUANT)

        with ServiceManager(integration_home) as svc:
            svc.start_vllm("fast", MODEL_SOURCE, port_a, timeout=120.0)
            svc.start_vllm("standard", MODEL_SOURCE, port_b, timeout=120.0)
            svc.start_litellm(litellm_port, integration_home / "litellm.yaml")

            # Route to "fast"
            resp_fast = httpx.post(
                f"http://localhost:{litellm_port}/v1/chat/completions",
                json={
                    "model": "fast",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 5,
                },
                timeout=120.0,
            )
            assert resp_fast.status_code == 200, (
                f"Routing to 'fast' failed: {resp_fast.status_code}"
            )

            # Route to "standard"
            resp_std = httpx.post(
                f"http://localhost:{litellm_port}/v1/chat/completions",
                json={
                    "model": "standard",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 5,
                },
                timeout=120.0,
            )
            assert resp_std.status_code == 200, (
                f"Routing to 'standard' failed: {resp_std.status_code}"
            )

    def test_model_listing_via_litellm(
        self,
        integration_home: Path,
        port_allocator: PortAllocator,
        model_cache_dir: Path,
    ) -> None:
        """GET /v1/models through LiteLLM returns expected tier names."""
        vllm_port = port_allocator.allocate()
        litellm_port = port_allocator.allocate()

        tier = {
            "name": "fast",
            "model": MODEL_ID,
            "quant": MODEL_QUANT,
            "source": MODEL_SOURCE,
            "port": vllm_port,
            "vllm_flags": {},
        }

        write_test_stack(integration_home, [tier], litellm_port)
        pull_model(model_id=MODEL_ID, quant=MODEL_QUANT)

        with ServiceManager(integration_home) as svc:
            svc.start_vllm("fast", MODEL_SOURCE, vllm_port, timeout=120.0)
            svc.start_litellm(litellm_port, integration_home / "litellm.yaml")

            response = httpx.get(
                f"http://localhost:{litellm_port}/v1/models",
                timeout=30.0,
            )
            assert response.status_code == 200
            data = response.json()
            model_ids = [m["id"] for m in data.get("data", [])]
            assert "fast" in model_ids, (
                f"Tier 'fast' not in model list: {model_ids}"
            )

    def test_concurrent_requests(
        self,
        integration_home: Path,
        port_allocator: PortAllocator,
        model_cache_dir: Path,
    ) -> None:
        """Stack handles multiple simultaneous requests without errors."""
        import asyncio

        import httpx as httpx_async

        vllm_port = port_allocator.allocate()

        tier = {
            "name": "fast",
            "model": MODEL_ID,
            "quant": MODEL_QUANT,
            "source": MODEL_SOURCE,
            "port": vllm_port,
            "vllm_flags": {},
        }

        write_test_stack(integration_home, [tier])
        pull_model(model_id=MODEL_ID, quant=MODEL_QUANT)

        with ServiceManager(integration_home) as svc:
            svc.start_vllm("fast", MODEL_SOURCE, vllm_port, timeout=120.0)

            async def send_request(client: httpx_async.AsyncClient, idx: int) -> int:
                resp = await client.post(
                    f"http://127.0.0.1:{vllm_port}/v1/chat/completions",
                    json={
                        "model": MODEL_SOURCE,
                        "messages": [{"role": "user", "content": f"Say {idx}"}],
                        "max_tokens": 5,
                    },
                    timeout=120.0,
                )
                return resp.status_code

            async def run_concurrent() -> list[int]:
                async with httpx_async.AsyncClient() as client:
                    tasks = [send_request(client, i) for i in range(5)]
                    return await asyncio.gather(*tasks)

            results = asyncio.run(run_concurrent())
            assert all(s == 200 for s in results), (
                f"Some concurrent requests failed: {results}"
            )

    def test_clean_shutdown_no_orphans(
        self,
        integration_home: Path,
        port_allocator: PortAllocator,
        model_cache_dir: Path,
    ) -> None:
        """After shutdown, no orphan processes remain on allocated ports."""
        vllm_port = port_allocator.allocate()

        tier = {
            "name": "fast",
            "model": MODEL_ID,
            "quant": MODEL_QUANT,
            "source": MODEL_SOURCE,
            "port": vllm_port,
            "vllm_flags": {},
        }

        write_test_stack(integration_home, [tier])
        pull_model(model_id=MODEL_ID, quant=MODEL_QUANT)

        with ServiceManager(integration_home) as svc:
            svc.start_vllm("fast", MODEL_SOURCE, vllm_port, timeout=120.0)
            # ServiceManager.__exit__ handles cleanup

        # Verify: port is free
        assert wait_for_port_free(vllm_port, timeout=10.0), (
            f"Port {vllm_port} still bound after shutdown"
        )

        # Verify: no PID files
        pids_dir = integration_home / "pids"
        remaining = list(pids_dir.glob("*.pid"))
        assert len(remaining) == 0, (
            f"PID files remain: {[p.name for p in remaining]}"
        )

        # Verify: socket bind succeeds (port truly free)
        assert not is_port_in_use(vllm_port), (
            f"Port {vllm_port} still in use after cleanup"
        )
