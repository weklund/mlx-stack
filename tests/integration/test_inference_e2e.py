"""End-to-end integration test for the full inference pipeline.

Tests the complete user journey with REAL processes:
  init → pull → up → inference → down → cleanup

**Requirements:**
  - macOS (darwin) — skipped on other platforms
  - Ports 8000 and 4000 must be free (skipped if occupied)
  - First run downloads ~500MB model; subsequent runs use cache
  - Run explicitly with: ``pytest -m integration``
  - Generous 5-minute timeout for model download + startup

**What it tests:**
  1. ``run_init()`` generates a valid stack config
  2. ``pull_model()`` downloads qwen3.5-0.8b (cached across runs)
  3. ``run_up()`` starts real vllm-mlx + LiteLLM with health checks
  4. LiteLLM proxy responds to chat completion requests
  5. Direct vllm-mlx responds to chat completion requests
  6. ``run_down()`` cleanly stops all services
  7. No PID files remain after shutdown
  8. Ports 8000 and 4000 are freed after shutdown

The test uses a persistent model cache at ~/.mlx-stack-test-cache/models/
so models are not re-downloaded every run. A try/finally block guarantees
cleanup even on assertion failures.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from mlx_stack.core.pull import pull_model
from mlx_stack.core.stack_down import run_down
from mlx_stack.core.stack_init import run_init
from mlx_stack.core.stack_up import run_up

# ---------------------------------------------------------------------------
# Markers and skip conditions
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.integration


def _is_port_in_use(port: int) -> bool:
    """Check if a TCP port is already bound."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            sock.bind(("127.0.0.1", port))
            return False
    except OSError:
        return True


_skip_not_darwin = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Inference e2e tests require macOS (Apple Silicon)",
)

_skip_port_8000_in_use = pytest.mark.skipif(
    _is_port_in_use(8000),
    reason="Port 8000 is already in use — skipping to avoid conflicts",
)

_skip_port_4000_in_use = pytest.mark.skipif(
    _is_port_in_use(4000),
    reason="Port 4000 is already in use — skipping to avoid conflicts",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODEL_CACHE_DIR = Path.home() / ".mlx-stack-test-cache" / "models"


@pytest.fixture(scope="session")
def model_cache_dir() -> Path:
    """Provide a persistent model cache directory across test runs.

    Uses ~/.mlx-stack-test-cache/models/ so that models downloaded
    in one test run are reused in subsequent runs without re-downloading.
    """
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_CACHE_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kill_processes_on_port(port: int) -> None:
    """Best-effort kill any process bound to the given port via lsof."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            for pid_str in pids:
                pid_str = pid_str.strip()
                if pid_str.isdigit():
                    try:
                        os.kill(int(pid_str), signal.SIGKILL)
                    except OSError:
                        pass
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass


def _wait_for_port_free(port: int, timeout: float = 10.0) -> bool:
    """Wait until a port is no longer bound, with timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_port_in_use(port):
            return True
        time.sleep(0.5)
    return False


def _read_stack_yaml(mlx_home: Path) -> dict[str, Any]:
    """Read and parse the generated stack YAML."""
    stack_path = mlx_home / "stacks" / "default.yaml"
    content = stack_path.read_text(encoding="utf-8")
    stack = yaml.safe_load(content)
    assert isinstance(stack, dict), "Stack YAML should be a mapping"
    return stack


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@_skip_not_darwin
@_skip_port_8000_in_use
@_skip_port_4000_in_use
class TestInferenceE2E:
    """End-to-end inference test: init → pull → up → inference → down.

    Uses a tmp_path-based MLX_STACK_HOME with a persistent model cache
    so models are not re-downloaded every run. The test exercises the
    full user journey with real vllm-mlx and LiteLLM processes.

    **How to run**::

        pytest -m integration tests/integration/test_inference_e2e.py -v

    **Note:** First run downloads ~500MB model. Subsequent runs use cache.
    """

    def test_full_inference_lifecycle(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        model_cache_dir: Path,
    ) -> None:
        """Test init → pull → up → inference → down with real processes.

        Steps:
            1. Set up an isolated MLX_STACK_HOME with persistent model cache
            2. Run init to generate stack config
            3. Pull the smallest model (qwen3.5-0.8b int4, ~500MB)
            4. Run up to start vllm-mlx + LiteLLM
            5. Send inference request to LiteLLM proxy
            6. Send inference request directly to vllm-mlx
            7. Run down to stop all services
            8. Verify cleanup (no PID files, ports freed)

        Cleanup:
            Always calls run_down() and kills port processes in finally.
        """
        # ---- Setup: isolated MLX_STACK_HOME ----
        mlx_home = tmp_path / ".mlx-stack"
        mlx_home.mkdir(parents=True, exist_ok=True)

        monkeypatch.setenv("MLX_STACK_HOME", str(mlx_home))

        # Configure model-dir to use the persistent cache
        config_path = mlx_home / "config.yaml"
        config_path.write_text(
            yaml.dump({"model-dir": str(model_cache_dir)}),
            encoding="utf-8",
        )

        try:
            # ---- Step 1: Init — generate stack config ----
            init_result = run_init(intent="balanced", force=True)
            assert init_result["stack_path"].exists(), "Stack YAML not generated"
            assert init_result["litellm_path"].exists(), "LiteLLM config not generated"

            # ---- Step 2: Read stack YAML dynamically ----
            stack = _read_stack_yaml(mlx_home)
            tiers = stack["tiers"]
            assert len(tiers) > 0, "Stack should have at least one tier"

            # Find the tier for qwen3.5-0.8b (smallest model) and its port
            # The init may assign it to any tier name; discover dynamically
            vllm_tier = None
            for tier in tiers:
                if tier.get("model") == "qwen3.5-0.8b":
                    vllm_tier = tier
                    break

            # If qwen3.5-0.8b is not in the stack (different hardware
            # profile may select different models), use the first tier
            if vllm_tier is None:
                vllm_tier = tiers[0]

            tier_name = vllm_tier["name"]
            vllm_port = vllm_tier["port"]
            model_id = vllm_tier["model"]
            hf_source = vllm_tier["source"]
            litellm_port = 4000  # Default; read from config if changed

            # ---- Step 3: Pull model ----
            pull_result = pull_model(
                model_id=model_id,
                quant="int4",
            )
            assert pull_result.local_path.exists(), (
                f"Model not found at {pull_result.local_path}"
            )

            # ---- Step 4: Up — start real services ----
            up_result = run_up()

            # Verify at least one tier is healthy
            healthy_tiers = [
                t for t in up_result.tiers
                if t.status in ("healthy", "already-running")
            ]
            assert len(healthy_tiers) > 0, (
                f"No healthy tiers after up. "
                f"Tier statuses: {[(t.name, t.status, t.error) for t in up_result.tiers]}"
            )

            # Verify LiteLLM is healthy
            assert up_result.litellm is not None, "LiteLLM result missing"
            assert up_result.litellm.status in ("healthy", "already-running"), (
                f"LiteLLM not healthy: {up_result.litellm.status} "
                f"({up_result.litellm.error})"
            )

            # ---- Step 5: Inference via LiteLLM proxy ----
            # Use the tier name as the model identifier for LiteLLM
            litellm_response = httpx.post(
                f"http://localhost:{litellm_port}/v1/chat/completions",
                json={
                    "model": tier_name,
                    "messages": [
                        {"role": "user", "content": "Say hello in exactly 3 words"},
                    ],
                    "max_tokens": 50,
                },
                timeout=60.0,
            )
            assert litellm_response.status_code == 200, (
                f"LiteLLM response status: {litellm_response.status_code}, "
                f"body: {litellm_response.text[:500]}"
            )
            litellm_data = litellm_response.json()
            litellm_content = litellm_data["choices"][0]["message"]["content"]
            assert litellm_content is not None and len(litellm_content.strip()) > 0, (
                f"LiteLLM returned empty content: {litellm_data}"
            )

            # ---- Step 6: Inference directly to vllm-mlx ----
            # Use the HF repo name as the model identifier for vllm-mlx
            vllm_response = httpx.post(
                f"http://127.0.0.1:{vllm_port}/v1/chat/completions",
                json={
                    "model": hf_source,
                    "messages": [
                        {"role": "user", "content": "Say hello in exactly 3 words"},
                    ],
                    "max_tokens": 50,
                },
                timeout=60.0,
            )
            assert vllm_response.status_code == 200, (
                f"vllm-mlx response status: {vllm_response.status_code}, "
                f"body: {vllm_response.text[:500]}"
            )
            vllm_data = vllm_response.json()
            vllm_content = vllm_data["choices"][0]["message"]["content"]
            assert vllm_content is not None and len(vllm_content.strip()) > 0, (
                f"vllm-mlx returned empty content: {vllm_data}"
            )

            # ---- Step 7: Down — stop all services ----
            down_result = run_down()

            # Verify services were stopped (not "nothing to stop")
            assert not down_result.nothing_to_stop, (
                "run_down reported nothing to stop — services may have crashed"
            )

            # ---- Step 8: Verify no PID files remain ----
            pids_dir = mlx_home / "pids"
            if pids_dir.exists():
                remaining_pids = list(pids_dir.glob("*.pid"))
                assert len(remaining_pids) == 0, (
                    f"PID files remain after down: {[p.name for p in remaining_pids]}"
                )

            # ---- Step 9: Verify ports are freed ----
            assert _wait_for_port_free(vllm_port, timeout=10.0), (
                f"Port {vllm_port} still bound after down"
            )
            assert _wait_for_port_free(litellm_port, timeout=10.0), (
                f"Port {litellm_port} still bound after down"
            )

        finally:
            # ---- Cleanup: belt-and-suspenders ----
            # Always attempt run_down() to clean up services
            try:
                run_down()
            except Exception:  # noqa: BLE001 — cleanup must not raise
                pass

            # Kill any remaining processes on ports 8000 and 4000
            _kill_processes_on_port(8000)
            _kill_processes_on_port(4000)

            # Also kill processes on any dynamically assigned ports
            # The stack may use ports 8001, 8002, etc.
            for port in range(8001, 8003):
                _kill_processes_on_port(port)

            # Wait briefly for processes to die
            time.sleep(1)
