"""Shared fixtures for mlx-stack integration tests.

Provides reusable infrastructure for all integration test tiers:
  - Dynamic port allocation (no hardcoded ports)
  - Persistent model cache across test runs
  - Isolated MLX_STACK_HOME with model cache
  - Service lifecycle management with guaranteed cleanup
  - Skip decorators for platform, memory, and dependency checks
  - Stack/LiteLLM config builders for arbitrary models and ports
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import psutil
import pytest
import yaml

from mlx_stack.core.catalog import CatalogEntry, load_catalog
from mlx_stack.core.process import wait_for_healthy

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

MODEL_CACHE_BASE = Path.home() / ".mlx-stack-test-cache"
REPORT_DIR = MODEL_CACHE_BASE / "reports"

# --------------------------------------------------------------------------- #
# Skip decorators
# --------------------------------------------------------------------------- #

skip_not_darwin = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Requires macOS (Apple Silicon)",
)

skip_no_vllm = pytest.mark.skipif(
    shutil.which("vllm-mlx") is None,
    reason="vllm-mlx not installed",
)

skip_no_litellm = pytest.mark.skipif(
    shutil.which("litellm") is None,
    reason="litellm not installed",
)


def skip_insufficient_memory(min_gb: float) -> pytest.MarkDecorator:
    """Skip if system has less than min_gb available memory."""
    try:
        available = psutil.virtual_memory().available / (1024**3)
        return pytest.mark.skipif(
            available < min_gb,
            reason=f"Requires {min_gb:.1f}GB free memory, have {available:.1f}GB",
        )
    except Exception:  # noqa: BLE001
        return pytest.mark.skip(reason="Could not determine available memory")


def check_memory_or_skip(min_gb: float) -> None:
    """Call inside a test to skip if insufficient memory at runtime."""
    try:
        available = psutil.virtual_memory().available / (1024**3)
        if available < min_gb:
            pytest.skip(f"Requires {min_gb:.1f}GB free memory, have {available:.1f}GB")
    except Exception:  # noqa: BLE001
        pytest.skip("Could not determine available memory")


# --------------------------------------------------------------------------- #
# Port allocation
# --------------------------------------------------------------------------- #


def _allocate_free_port() -> int:
    """Allocate an ephemeral port by binding to 0, reading assignment, closing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def is_port_in_use(port: int) -> bool:
    """Check if a TCP port is already bound."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            sock.bind(("127.0.0.1", port))
            return False
    except OSError:
        return True


def wait_for_port_free(port: int, timeout: float = 10.0) -> bool:
    """Wait until a port is no longer bound, with timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_port_in_use(port):
            return True
        time.sleep(0.5)
    return False


def kill_processes_on_port(port: int) -> None:
    """Best-effort kill any process bound to the given port via lsof.

    Excludes the current process to avoid killing pytest itself when it
    has client connections to the port being cleaned up.
    """
    my_pid = os.getpid()
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            for pid_str in result.stdout.strip().split("\n"):
                pid_str = pid_str.strip()
                if pid_str.isdigit() and int(pid_str) != my_pid:
                    try:
                        os.kill(int(pid_str), signal.SIGKILL)
                    except OSError:
                        pass
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass


@dataclass
class PortAllocator:
    """Allocates ephemeral ports and tracks them to prevent collisions."""

    _allocated: set[int] = field(default_factory=set)

    def allocate(self) -> int:
        """Allocate a free port, retrying to avoid collisions."""
        for _ in range(100):
            port = _allocate_free_port()
            if port not in self._allocated:
                self._allocated.add(port)
                return port
        msg = "Could not allocate free port after 100 attempts"
        raise RuntimeError(msg)

    def release(self, port: int) -> None:
        """Release a previously allocated port."""
        self._allocated.discard(port)

    def release_all(self) -> None:
        """Release all allocated ports."""
        self._allocated.clear()


@pytest.fixture(scope="session")
def port_allocator() -> PortAllocator:
    """Session-scoped port allocator to prevent collisions across tests."""
    return PortAllocator()


# --------------------------------------------------------------------------- #
# Model cache
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def model_cache_dir() -> Path:
    """Persistent model cache at ~/.mlx-stack-test-cache/models/.

    Models downloaded in one test run are reused in subsequent runs
    without re-downloading.
    """
    cache = MODEL_CACHE_BASE / "models"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def catalog() -> list[CatalogEntry]:
    """Load the full model catalog once per session."""
    return load_catalog()


# --------------------------------------------------------------------------- #
# Isolated MLX_STACK_HOME
# --------------------------------------------------------------------------- #


@pytest.fixture()
def integration_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_cache_dir: Path,
) -> Path:
    """Isolated MLX_STACK_HOME with model cache pointing to persistent cache.

    Pre-creates required subdirectories (stacks, logs, pids) and configures
    model-dir to use the session-scoped persistent model cache.
    """
    home = tmp_path / ".mlx-stack"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MLX_STACK_HOME", str(home))

    # Point model-dir to persistent cache
    config_path = home / "config.yaml"
    config_path.write_text(
        yaml.dump({"model-dir": str(model_cache_dir)}),
        encoding="utf-8",
    )

    # Pre-create required subdirectories
    (home / "stacks").mkdir()
    (home / "logs").mkdir()
    (home / "pids").mkdir()

    return home


# --------------------------------------------------------------------------- #
# Service lifecycle manager
# --------------------------------------------------------------------------- #


@dataclass
class ManagedService:
    """A service tracked by the ServiceManager for cleanup."""

    name: str
    port: int
    pid: int | None = None
    log_path: Path | None = None


class ServiceManager:
    """Manages vllm-mlx and LiteLLM service lifecycles with guaranteed cleanup.

    Use as a context manager to ensure all started services are stopped
    even on assertion failures::

        with ServiceManager(mlx_home) as svc:
            svc.start_vllm("fast", "mlx-community/Model-4bit", port=9000)
            # ... run assertions ...
        # services are automatically stopped
    """

    def __init__(self, mlx_home: Path) -> None:
        self._mlx_home = mlx_home
        self._services: list[ManagedService] = []

    def start_vllm(
        self,
        tier_name: str,
        model_source: str,
        port: int,
        vllm_flags: dict[str, Any] | None = None,
        timeout: float = 300.0,
    ) -> ManagedService:
        """Start a vllm-mlx instance and wait for health.

        Args:
            tier_name: Service name for PID/log files.
            model_source: HuggingFace repo or local path for vllm-mlx.
            port: Port to serve on.
            vllm_flags: Additional vllm-mlx flags (e.g. tool_call_parser).
            timeout: Max seconds to wait for health check.

        Returns:
            ManagedService with PID and log path.
        """
        flags = vllm_flags or {}
        cmd: list[str] = [
            "vllm-mlx", "serve", model_source,
            "--port", str(port),
            "--host", "127.0.0.1",
        ]
        if flags.get("continuous_batching"):
            cmd.append("--continuous-batching")
        if flags.get("use_paged_cache"):
            cmd.append("--use-paged-cache")
        if flags.get("enable_auto_tool_choice"):
            cmd.append("--enable-auto-tool-choice")
        if flags.get("tool_call_parser"):
            cmd.extend(["--tool-call-parser", flags["tool_call_parser"]])

        logs_dir = self._mlx_home / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{tier_name}.log"

        pids_dir = self._mlx_home / "pids"
        pids_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_path.open("w")
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        pid_path = pids_dir / f"{tier_name}.pid"
        pid_path.write_text(str(proc.pid), encoding="utf-8")

        managed = ManagedService(
            name=tier_name,
            port=port,
            pid=proc.pid,
            log_path=log_path,
        )
        self._services.append(managed)

        # Wait for health
        wait_for_healthy(
            port=port,
            path="/v1/models",
            total_timeout=timeout,
        )

        # Warmup: the health check only proves the HTTP server is up.
        # The first inference triggers MLX weight loading and JIT compilation,
        # which can take significantly longer. Send a throwaway request so
        # that callers' inference timeouts measure generation, not cold start.
        try:
            httpx.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={
                    "model": model_source,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 1,
                },
                timeout=timeout,
            )
        except (httpx.TimeoutException, httpx.HTTPError):
            pass  # warmup is best-effort; the real test will catch failures

        return managed

    def start_litellm(
        self,
        port: int,
        config_path: Path,
        timeout: float = 120.0,
    ) -> ManagedService:
        """Start a LiteLLM proxy and wait for health.

        Args:
            port: Port to serve on.
            config_path: Path to litellm.yaml config file.
            timeout: Max seconds to wait for health check.

        Returns:
            ManagedService with PID and log path.
        """
        cmd: list[str] = [
            "litellm",
            "--config", str(config_path),
            "--port", str(port),
            "--host", "127.0.0.1",
        ]

        logs_dir = self._mlx_home / "logs"
        log_path = logs_dir / "litellm.log"
        pids_dir = self._mlx_home / "pids"

        log_file = log_path.open("w")
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        pid_path = pids_dir / "litellm.pid"
        pid_path.write_text(str(proc.pid), encoding="utf-8")

        managed = ManagedService(
            name="litellm",
            port=port,
            pid=proc.pid,
            log_path=log_path,
        )
        self._services.append(managed)

        wait_for_healthy(
            port=port,
            path="/health/liveliness",
            total_timeout=timeout,
        )

        return managed

    def stop_all(self) -> None:
        """Stop all managed services. Always safe to call multiple times."""
        for svc in reversed(self._services):
            if svc.pid is not None:
                try:
                    os.kill(svc.pid, signal.SIGTERM)
                except OSError:
                    pass

        # Give services time to shut down gracefully
        time.sleep(2)

        # Force-kill anything still alive
        for svc in self._services:
            if svc.pid is not None:
                try:
                    os.kill(svc.pid, signal.SIGKILL)
                except OSError:
                    pass

        time.sleep(1)

        # Kill any orphaned child processes still holding the port
        for svc in self._services:
            kill_processes_on_port(svc.port)

        # Wait for ports to be freed
        for svc in self._services:
            wait_for_port_free(svc.port, timeout=15.0)

        # Clean up PID files
        pids_dir = self._mlx_home / "pids"
        if pids_dir.exists():
            for pid_file in pids_dir.glob("*.pid"):
                pid_file.unlink(missing_ok=True)

        self._services.clear()

    def __enter__(self) -> ServiceManager:
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop_all()


# --------------------------------------------------------------------------- #
# Stack config builders
# --------------------------------------------------------------------------- #


def build_test_stack_yaml(
    tiers: list[dict[str, Any]],
    hardware_profile: str = "test",
    intent: str = "balanced",
) -> dict[str, Any]:
    """Build a stack definition YAML dict for testing.

    Args:
        tiers: List of tier dicts, each with keys:
            name, model, quant, source, port, and optional vllm_flags.
        hardware_profile: Hardware profile name.
        intent: Recommendation intent.

    Returns:
        Stack definition dict ready for yaml.dump().
    """
    return {
        "schema_version": 1,
        "name": "default",
        "hardware_profile": hardware_profile,
        "intent": intent,
        "created": datetime.now(timezone.utc).isoformat(),
        "tiers": tiers,
    }


def build_test_litellm_yaml(
    tiers: list[dict[str, Any]],
    litellm_port: int | None = None,
) -> dict[str, Any]:
    """Build a LiteLLM config YAML dict for testing.

    Args:
        tiers: List of tier dicts with name, model, source, port keys.
        litellm_port: LiteLLM proxy port (for reference only, not in config).

    Returns:
        LiteLLM config dict ready for yaml.dump().
    """
    model_list = []
    for tier in tiers:
        model_list.append({
            "model_name": tier["name"],
            "litellm_params": {
                "model": f"openai/{tier['model']}",
                "api_base": f"http://localhost:{tier['port']}/v1",
                "api_key": "dummy",
            },
        })

    return {
        "model_list": model_list,
        "router_settings": {
            "routing_strategy": "simple-shuffle",
            "num_retries": 2,
            "timeout": 120,
        },
    }


def write_test_stack(
    mlx_home: Path,
    tiers: list[dict[str, Any]],
    litellm_port: int | None = None,
) -> tuple[Path, Path]:
    """Write stack + litellm config files to the given MLX_STACK_HOME.

    Returns:
        (stack_yaml_path, litellm_yaml_path)
    """
    stack_yaml = build_test_stack_yaml(tiers)
    stack_path = mlx_home / "stacks" / "default.yaml"
    stack_path.parent.mkdir(parents=True, exist_ok=True)
    stack_path.write_text(
        yaml.dump(stack_yaml, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    litellm_yaml = build_test_litellm_yaml(tiers, litellm_port)
    litellm_path = mlx_home / "litellm.yaml"
    litellm_path.write_text(
        yaml.dump(litellm_yaml, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    return stack_path, litellm_path


# --------------------------------------------------------------------------- #
# Timeout helpers
# --------------------------------------------------------------------------- #


def model_timeout(params_b: float) -> float:
    """Return appropriate health check timeout based on model size."""
    if params_b <= 3.0:
        return 120.0
    if params_b <= 14.0:
        return 300.0
    return 600.0
