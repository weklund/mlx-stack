"""Benchmark execution engine for mlx-stack.

Runs 3 iterations of 1024-token prompt + 100-token generation against
a vllm-mlx instance, reports mean ± std dev for prompt_tps and gen_tps.
Compares against catalog thresholds: PASS (<15%), WARN (15-30%),
FAIL (>30%). Handles tool-calling benchmarks for capable models.

Supports benchmarking running tier instances and local models via
temporary vllm-mlx instances with full cleanup (including Ctrl+C).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from mlx_stack.core.catalog import (
    CatalogEntry,
    get_entry_by_id,
    load_catalog,
)
from mlx_stack.core.config import ConfigCorruptError, get_value
from mlx_stack.core.deps import ensure_dependency
from mlx_stack.core.hardware import HardwareProfile, detect_hardware, load_profile
from mlx_stack.core.paths import (
    ensure_data_home,
    get_benchmarks_dir,
    get_data_home,
    get_stacks_dir,
)
from mlx_stack.core.process import (
    HealthCheckError,
    ProcessError,
    check_port_conflict,
    is_process_alive,
    read_pid_file,
    remove_pid_file,
    start_service,
    stop_service,
    wait_for_healthy,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Benchmark parameters
NUM_ITERATIONS = 3
PROMPT_TOKEN_COUNT = 1024
MAX_GENERATION_TOKENS = 100

# Health check timeout for temporary instances (seconds)
TEMP_INSTANCE_HEALTH_TIMEOUT = 120.0

# Port range for temporary instances (avoid 4000, 8000-8002)
TEMP_PORT_START = 8100
TEMP_PORT_END = 8200

# Threshold percentages for comparison
PASS_THRESHOLD = 0.15  # within 15%
WARN_THRESHOLD = 0.30  # within 30%

# Classification labels
CLASSIFICATION_PASS = "PASS"
CLASSIFICATION_WARN = "WARN"
CLASSIFICATION_FAIL = "FAIL"

# Temporary service name prefix
TEMP_SERVICE_PREFIX = "bench-temp"

# Tool-calling benchmark tool definition
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use.",
                },
            },
            "required": ["location"],
        },
    },
}


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class BenchmarkError(Exception):
    """Raised when a benchmark operation fails."""


class BenchmarkTargetError(BenchmarkError):
    """Raised when the benchmark target is not found."""


class BenchmarkRunError(BenchmarkError):
    """Raised when a benchmark iteration fails."""


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IterationResult:
    """Result of a single benchmark iteration."""

    prompt_tps: float
    gen_tps: float
    prompt_tokens: int
    completion_tokens: int
    total_time: float


@dataclass(frozen=True)
class MetricClassification:
    """Classification for a single metric against catalog threshold."""

    metric: str
    measured: float
    catalog: float
    delta_pct: float
    classification: str  # PASS, WARN, FAIL


@dataclass(frozen=True)
class ToolCallResult:
    """Result of a tool-calling benchmark."""

    success: bool
    round_trip_time: float
    error: str | None = None


@dataclass
class BenchmarkResult_:
    """Complete benchmark result for a model."""

    model_id: str
    quant: str
    iterations: list[IterationResult] = field(default_factory=list)
    prompt_tps_mean: float = 0.0
    prompt_tps_std: float = 0.0
    gen_tps_mean: float = 0.0
    gen_tps_std: float = 0.0
    classifications: list[MetricClassification] = field(default_factory=list)
    tool_call_result: ToolCallResult | None = None
    used_temporary_instance: bool = False
    catalog_data_available: bool = False

    def to_save_dict(self) -> dict[str, Any]:
        """Convert to dict for saving to benchmark JSON file."""
        return {
            "model_id": self.model_id,
            "quant": self.quant,
            "prompt_tps": self.prompt_tps_mean,
            "gen_tps": self.gen_tps_mean,
            "memory_gb": 0.0,  # placeholder, updated if available
        }


# --------------------------------------------------------------------------- #
# Prompt generation
# --------------------------------------------------------------------------- #


def _generate_prompt(token_count: int) -> str:
    """Generate a prompt with approximately the specified token count.

    Uses repeated text to approximate the desired token count. Each word
    is roughly 1-2 tokens, so we generate enough words to cover the count.

    Args:
        token_count: Target number of tokens.

    Returns:
        A text string approximately token_count tokens long.
    """
    # Each word is roughly 1.3 tokens on average for English text.
    # We use a repeating pattern to keep it simple and predictable.
    base_phrase = (
        "The quick brown fox jumps over the lazy dog near the river bank "
        "where flowers bloom in the warm sunshine of a summer afternoon "
    )
    words_needed = int(token_count / 1.3) + 10
    words = base_phrase.split()
    repeated = []
    for i in range(words_needed):
        repeated.append(words[i % len(words)])
    return " ".join(repeated)


# --------------------------------------------------------------------------- #
# Running tier resolution
# --------------------------------------------------------------------------- #


def _load_stack_tiers() -> list[dict[str, Any]]:
    """Load tier definitions from the active stack.

    Returns:
        List of tier dicts, or empty list if no stack is configured.
    """
    stack_path = get_stacks_dir() / "default.yaml"
    if not stack_path.exists():
        return []
    try:
        import yaml

        content = stack_path.read_text(encoding="utf-8")
        stack = yaml.safe_load(content)
        if isinstance(stack, dict):
            tiers = stack.get("tiers", [])
            if isinstance(tiers, list):
                return tiers
    except Exception:
        pass
    return []


def _find_running_tier(target: str) -> dict[str, Any] | None:
    """Find a running tier by name.

    Checks if the tier's PID file exists and the process is alive.

    Args:
        target: The tier name to look for.

    Returns:
        The tier dict if running, or None.
    """
    tiers = _load_stack_tiers()
    for tier in tiers:
        tier_name = tier.get("name", "")
        if tier_name == target:
            # Check if running
            try:
                pid = read_pid_file(tier_name)
                if pid is not None and is_process_alive(pid):
                    return tier
            except ProcessError:
                pass
    return None


def _get_running_tier_names() -> list[str]:
    """Get names of all running tiers from the active stack."""
    tiers = _load_stack_tiers()
    running = []
    for tier in tiers:
        name = tier.get("name", "")
        try:
            pid = read_pid_file(name)
            if pid is not None and is_process_alive(pid):
                running.append(name)
        except ProcessError:
            pass
    return running


def _get_all_tier_names() -> list[str]:
    """Get names of all tiers in the active stack."""
    tiers = _load_stack_tiers()
    return [t.get("name", "") for t in tiers if t.get("name")]


def _get_used_ports() -> set[int]:
    """Get all ports used by running stack services and LiteLLM."""
    ports: set[int] = set()
    tiers = _load_stack_tiers()
    for tier in tiers:
        port = tier.get("port")
        if port is not None:
            ports.add(int(port))
    # Add LiteLLM port
    try:
        litellm_port = int(get_value("litellm-port"))
        ports.add(litellm_port)
    except (ConfigCorruptError, ValueError, TypeError):
        ports.add(4000)  # default
    return ports


def _find_temp_port(used_ports: set[int]) -> int:
    """Find an available port for a temporary vllm-mlx instance.

    Avoids ports used by running stack and LiteLLM.

    Args:
        used_ports: Set of ports currently in use.

    Returns:
        An available port number.

    Raises:
        BenchmarkError: If no free port can be found.
    """

    for port in range(TEMP_PORT_START, TEMP_PORT_END):
        if port in used_ports:
            continue
        # Verify the port is actually free
        conflict = check_port_conflict(port)
        if conflict is None:
            return port

    msg = (
        f"Could not find an available port in range "
        f"{TEMP_PORT_START}-{TEMP_PORT_END} for temporary benchmark instance."
    )
    raise BenchmarkError(msg)


# --------------------------------------------------------------------------- #
# Benchmark iterations
# --------------------------------------------------------------------------- #


def _run_single_iteration(
    port: int,
    model_name: str,
    prompt: str,
    max_tokens: int = MAX_GENERATION_TOKENS,
    host: str = "127.0.0.1",
) -> IterationResult:
    """Run a single benchmark iteration against a vllm-mlx instance.

    Uses streaming to separately measure prompt processing time (time to
    first token) and generation time (first token to last token). This
    gives distinct prompt_tps and gen_tps measurements.

    Args:
        port: Port of the vllm-mlx instance.
        model_name: The model identifier for the API request.
        prompt: The prompt text.
        max_tokens: Maximum tokens to generate.
        host: The host to connect to.

    Returns:
        An IterationResult with timing data.

    Raises:
        BenchmarkRunError: If the API request fails.
    """
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    try:
        start_time = time.monotonic()
        first_token_time: float | None = None
        completion_tokens = 0
        prompt_tokens = 0
        chunk_count = 0

        with httpx.stream("POST", url, json=payload, timeout=300.0) as response:
            if response.status_code != 200:
                body = response.read().decode("utf-8", errors="replace")[:200]
                msg = (
                    f"API request failed with status {response.status_code}: "
                    f"{body}"
                )
                raise BenchmarkRunError(msg)

            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Check for usage in the final chunk
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)

                # Track first content token for TTFT measurement.
                # Thinking models (e.g. Qwen3) emit reasoning_content for
                # <think> tokens instead of content — check both fields so
                # first_token_time is set and TPS is calculated correctly.
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content") or delta.get("reasoning_content")
                    if content and first_token_time is None:
                        first_token_time = time.monotonic()
                    if content:
                        chunk_count += 1

        end_time = time.monotonic()
        total_time = end_time - start_time

        # Calculate distinct prompt_tps and gen_tps from timing phases:
        # - prompt_time = time from request start to first generated token
        # - gen_time = time from first generated token to last token
        if first_token_time is not None and prompt_tokens > 0:
            prompt_time = first_token_time - start_time
            gen_time = end_time - first_token_time

            prompt_tps = prompt_tokens / prompt_time if prompt_time > 0 else 0.0
            gen_tps = completion_tokens / gen_time if gen_time > 0 else 0.0
        else:
            prompt_tps = 0.0
            gen_tps = 0.0

        return IterationResult(
            prompt_tps=prompt_tps,
            gen_tps=gen_tps,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_time=total_time,
        )

    except BenchmarkRunError:
        raise
    except httpx.TimeoutException:
        msg = "Benchmark request timed out after 300s"
        raise BenchmarkRunError(msg) from None
    except httpx.HTTPError as exc:
        msg = f"HTTP error during benchmark: {exc}"
        raise BenchmarkRunError(msg) from None
    except (json.JSONDecodeError, KeyError) as exc:
        msg = f"Could not parse benchmark response: {exc}"
        raise BenchmarkRunError(msg) from None


def _run_iterations(
    port: int,
    model_name: str,
    num_iterations: int = NUM_ITERATIONS,
) -> list[IterationResult]:
    """Run multiple benchmark iterations.

    Args:
        port: Port of the vllm-mlx instance.
        model_name: Model identifier for API requests.
        num_iterations: Number of iterations to run.

    Returns:
        List of IterationResult from each iteration.

    Raises:
        BenchmarkRunError: If any iteration fails.
    """
    prompt = _generate_prompt(PROMPT_TOKEN_COUNT)
    results: list[IterationResult] = []

    for _i in range(num_iterations):
        result = _run_single_iteration(
            port=port,
            model_name=model_name,
            prompt=prompt,
            max_tokens=MAX_GENERATION_TOKENS,
        )
        results.append(result)

    return results


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def _compute_stats(values: list[float]) -> tuple[float, float]:
    """Compute mean and standard deviation.

    Args:
        values: List of numeric values.

    Returns:
        Tuple of (mean, std_dev).
    """
    if not values:
        return 0.0, 0.0

    n = len(values)
    mean = sum(values) / n

    if n < 2:
        return mean, 0.0

    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std_dev = math.sqrt(variance)
    return mean, std_dev


# --------------------------------------------------------------------------- #
# Catalog comparison
# --------------------------------------------------------------------------- #


def _classify_metric(
    metric_name: str,
    measured: float,
    catalog_value: float,
) -> MetricClassification:
    """Classify a metric against the catalog threshold.

    PASS: within 15% of catalog value
    WARN: 15-30% below catalog value
    FAIL: more than 30% below catalog value

    Args:
        metric_name: Name of the metric (e.g. "prompt_tps").
        measured: Measured value.
        catalog_value: Catalog reference value.

    Returns:
        A MetricClassification instance.
    """
    if catalog_value <= 0:
        return MetricClassification(
            metric=metric_name,
            measured=measured,
            catalog=catalog_value,
            delta_pct=0.0,
            classification=CLASSIFICATION_PASS,
        )

    delta_pct = (catalog_value - measured) / catalog_value

    if delta_pct <= PASS_THRESHOLD:
        classification = CLASSIFICATION_PASS
    elif delta_pct <= WARN_THRESHOLD:
        classification = CLASSIFICATION_WARN
    else:
        classification = CLASSIFICATION_FAIL

    return MetricClassification(
        metric=metric_name,
        measured=measured,
        catalog=catalog_value,
        delta_pct=delta_pct * 100,  # Store as percentage
        classification=classification,
    )


def _compare_against_catalog(
    prompt_tps_mean: float,
    gen_tps_mean: float,
    entry: CatalogEntry,
    profile: HardwareProfile,
) -> list[MetricClassification]:
    """Compare benchmark results against catalog thresholds.

    Args:
        prompt_tps_mean: Mean prompt tokens/sec.
        gen_tps_mean: Mean generation tokens/sec.
        entry: The catalog entry for the model.
        profile: The hardware profile for benchmark lookup.

    Returns:
        List of MetricClassification, empty if no catalog data.
    """
    profile_id = profile.profile_id
    if profile_id not in entry.benchmarks:
        return []

    bench = entry.benchmarks[profile_id]
    classifications: list[MetricClassification] = []

    classifications.append(
        _classify_metric("prompt_tps", prompt_tps_mean, bench.prompt_tps)
    )
    classifications.append(
        _classify_metric("gen_tps", gen_tps_mean, bench.gen_tps)
    )

    return classifications


# --------------------------------------------------------------------------- #
# Tool-calling benchmark
# --------------------------------------------------------------------------- #


def _run_tool_call_benchmark(
    port: int,
    model_name: str,
    host: str = "127.0.0.1",
) -> ToolCallResult:
    """Run a tool-calling benchmark.

    Sends a function-calling request with a tool definition and verifies
    the response contains a valid tool call structure.

    Args:
        port: Port of the vllm-mlx instance.
        model_name: Model identifier.
        host: Host to connect to.

    Returns:
        A ToolCallResult with timing and validity information.
    """
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": "What is the current weather in San Francisco, CA?",
            }
        ],
        "tools": [TOOL_DEFINITION],
        "tool_choice": "auto",
        "max_tokens": 1024,
        "temperature": 0.0,
    }

    try:
        start_time = time.monotonic()
        response = httpx.post(url, json=payload, timeout=60.0)
        round_trip_time = time.monotonic() - start_time

        if response.status_code != 200:
            return ToolCallResult(
                success=False,
                round_trip_time=round_trip_time,
                error=f"API returned status {response.status_code}",
            )

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            return ToolCallResult(
                success=False,
                round_trip_time=round_trip_time,
                error="No choices in response",
            )

        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls", [])
        if not tool_calls:
            return ToolCallResult(
                success=False,
                round_trip_time=round_trip_time,
                error="No tool calls in response",
            )

        # Validate tool call structure
        tool_call = tool_calls[0]
        fn = tool_call.get("function", {})
        fn_name = fn.get("name", "")
        fn_args_str = fn.get("arguments", "")

        if fn_name != "get_weather":
            return ToolCallResult(
                success=False,
                round_trip_time=round_trip_time,
                error=f"Wrong function name: {fn_name}",
            )

        # Try to parse arguments as JSON
        try:
            fn_args = json.loads(fn_args_str) if isinstance(fn_args_str, str) else fn_args_str
            if "location" not in fn_args:
                return ToolCallResult(
                    success=False,
                    round_trip_time=round_trip_time,
                    error="Missing 'location' in tool call arguments",
                )
        except (json.JSONDecodeError, TypeError):
            return ToolCallResult(
                success=False,
                round_trip_time=round_trip_time,
                error="Could not parse tool call arguments as JSON",
            )

        return ToolCallResult(
            success=True,
            round_trip_time=round_trip_time,
        )

    except httpx.TimeoutException:
        return ToolCallResult(
            success=False,
            round_trip_time=60.0,
            error="Tool call request timed out",
        )
    except (httpx.HTTPError, Exception) as exc:
        return ToolCallResult(
            success=False,
            round_trip_time=0.0,
            error=f"Tool call request failed: {exc}",
        )


# --------------------------------------------------------------------------- #
# Temporary instance management
# --------------------------------------------------------------------------- #


def _resolve_model_source(model_id: str, quant: str) -> str:
    """Resolve the model source path for vllm-mlx.

    Checks local models directory first, then falls back to the catalog
    HuggingFace repo.

    Args:
        model_id: Catalog model ID.
        quant: Quantization level.

    Returns:
        The model source path or HF repo identifier.

    Raises:
        BenchmarkError: If the model cannot be resolved.
    """
    catalog = load_catalog()
    entry = get_entry_by_id(catalog, model_id)
    if entry is None:
        msg = f"Model '{model_id}' not found in catalog."
        raise BenchmarkTargetError(msg)

    if quant not in entry.sources:
        available_quants = ", ".join(sorted(entry.sources.keys()))
        msg = (
            f"Quantization '{quant}' not available for model '{model_id}'. "
            f"Available: {available_quants}"
        )
        raise BenchmarkError(msg)

    source = entry.sources[quant]

    # Check local models directory
    try:
        model_dir = str(get_value("model-dir"))
        models_path = Path(model_dir).expanduser()
    except (ConfigCorruptError, Exception):
        models_path = get_data_home() / "models"

    # Check by repo directory name
    repo_dir_name = source.hf_repo.rsplit("/", 1)[-1] if "/" in source.hf_repo else source.hf_repo
    local_path = models_path / repo_dir_name
    if local_path.exists():
        return str(local_path)

    # Check by model ID
    model_path = models_path / model_id
    if model_path.exists():
        return str(model_path)

    # Use HuggingFace repo directly
    return source.hf_repo


def _start_temp_instance(
    model_source: str,
    port: int,
    entry: CatalogEntry,
    quant: str,
) -> str:
    """Start a temporary vllm-mlx instance for benchmarking.

    Args:
        model_source: Model source path or HF repo.
        port: Port to bind the instance to.
        entry: Catalog entry for the model.
        quant: Quantization level.

    Returns:
        The service name used for PID management.

    Raises:
        BenchmarkError: If the instance cannot be started.
    """
    service_name = f"{TEMP_SERVICE_PREFIX}-{entry.id}"

    # Ensure vllm-mlx is installed
    ensure_dependency("vllm-mlx")

    vllm_binary = shutil.which("vllm-mlx")
    if vllm_binary is None:
        msg = (
            "vllm-mlx binary not found on PATH after installation. "
            "Try: uv tool install vllm-mlx"
        )
        raise BenchmarkError(msg)

    cmd = [
        vllm_binary,
        "serve", model_source,
        "--port", str(port),
        "--host", "127.0.0.1",
    ]

    # Add tool-calling flags if the model supports it
    if entry.capabilities.tool_calling:
        cmd.append("--enable-auto-tool-choice")
        if entry.capabilities.tool_call_parser:
            cmd.extend(["--tool-call-parser", entry.capabilities.tool_call_parser])

    # Add reasoning parser for thinking models (e.g., Qwen3).
    # Without this flag, thinking models emit <think> tags that break
    # the tool-call parser (e.g., hermes).
    if entry.capabilities.thinking and entry.capabilities.reasoning_parser:
        cmd.extend(["--reasoning-parser", entry.capabilities.reasoning_parser])

    try:
        start_service(
            service_name=service_name,
            cmd=cmd,
            port=port,
        )
    except ProcessError as exc:
        msg = f"Could not start temporary vllm-mlx instance: {exc}"
        raise BenchmarkError(msg) from None

    # Wait for health
    try:
        wait_for_healthy(
            port=port,
            path="/v1/models",
            total_timeout=TEMP_INSTANCE_HEALTH_TIMEOUT,
        )
    except HealthCheckError:
        # Clean up the failed instance
        _cleanup_temp_instance(service_name)
        msg = (
            f"Temporary vllm-mlx instance did not become healthy "
            f"within {TEMP_INSTANCE_HEALTH_TIMEOUT}s."
        )
        raise BenchmarkError(msg) from None

    return service_name


def _cleanup_temp_instance(service_name: str) -> None:
    """Clean up a temporary vllm-mlx instance.

    Sends SIGTERM then SIGKILL to ensure full cleanup.

    Args:
        service_name: The service name from PID management.
    """
    try:
        stop_service(service_name, grace_period=5.0)
    except Exception:
        pass

    # Double-check: try reading PID and kill directly if still alive
    try:
        pid = read_pid_file(service_name)
        if pid is not None and is_process_alive(pid):
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.5)
    except (ProcessError, OSError):
        pass

    # Remove PID file
    remove_pid_file(service_name)


# --------------------------------------------------------------------------- #
# Save benchmark results
# --------------------------------------------------------------------------- #


def save_benchmark_results(
    result: BenchmarkResult_,
    profile: HardwareProfile,
) -> Path:
    """Save benchmark results to ~/.mlx-stack/benchmarks/<profile_id>.json.

    Loads existing data (if any) and merges/updates with the new result.

    Args:
        result: The benchmark result to save.
        profile: The hardware profile for path determination.

    Returns:
        Path to the saved file.
    """
    ensure_data_home()
    benchmarks_dir = get_benchmarks_dir()
    benchmarks_dir.mkdir(parents=True, exist_ok=True)

    benchmark_file = benchmarks_dir / f"{profile.profile_id}.json"

    # Load existing data
    existing: dict[str, Any] = {}
    if benchmark_file.exists():
        try:
            existing = json.loads(benchmark_file.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {}
        except (json.JSONDecodeError, OSError):
            existing = {}

    # Merge new result
    existing[result.model_id] = result.to_save_dict()

    # Write back
    benchmark_file.write_text(
        json.dumps(existing, indent=2) + "\n",
        encoding="utf-8",
    )

    return benchmark_file


# --------------------------------------------------------------------------- #
# Resolve benchmark target
# --------------------------------------------------------------------------- #


@dataclass
class BenchmarkTarget:
    """Resolved target for benchmarking."""

    model_id: str
    quant: str
    port: int
    model_name: str  # The model name used in API calls
    entry: CatalogEntry
    is_running_tier: bool
    temp_service_name: str | None = None  # Set if using a temp instance


def resolve_target(target: str) -> BenchmarkTarget:
    """Resolve a benchmark target to a specific model and port.

    Tries in order:
    1. Running tier by name
    2. Catalog model by ID (starts temp instance)

    Args:
        target: Tier name or model ID.

    Returns:
        A BenchmarkTarget with all needed info.

    Raises:
        BenchmarkTargetError: If the target cannot be resolved.
    """
    # 1. Try as a running tier
    tier = _find_running_tier(target)
    if tier is not None:
        port = tier["port"]
        model_id = tier.get("model", target)
        quant = tier.get("quant", "int4")
        source = tier.get("source", model_id)

        catalog = load_catalog()
        entry = get_entry_by_id(catalog, model_id)
        if entry is None:
            # Still benchmark it even without catalog data
            from mlx_stack.core.catalog import (
                Capabilities,
                CatalogEntry,
                QualityScores,
            )

            entry = CatalogEntry(
                id=model_id,
                name=model_id,
                family="unknown",
                params_b=0.0,
                architecture="unknown",
                min_mlx_lm_version="0.0.0",
                sources={},
                capabilities=Capabilities(
                    tool_calling=False,
                    tool_call_parser=None,
                    thinking=False,
                    reasoning_parser=None,
                    vision=False,
                ),
                quality=QualityScores(overall=0, coding=0, reasoning=0, instruction_following=0),
                benchmarks={},
                tags=[],
            )

        return BenchmarkTarget(
            model_id=model_id,
            quant=quant,
            port=port,
            model_name=source,
            entry=entry,
            is_running_tier=True,
        )

    # 2. Try as a catalog model
    catalog = load_catalog()
    entry = get_entry_by_id(catalog, target)
    if entry is not None:
        # Determine quant
        try:
            quant = str(get_value("default-quant"))
        except (ConfigCorruptError, Exception):
            quant = "int4"

        if quant not in entry.sources:
            available = sorted(entry.sources.keys())
            quant = available[0] if available else "int4"

        # Resolve model source
        model_source = _resolve_model_source(target, quant)

        # Find a free port
        used_ports = _get_used_ports()
        port = _find_temp_port(used_ports)

        # Start temp instance
        service_name = _start_temp_instance(model_source, port, entry, quant)

        return BenchmarkTarget(
            model_id=entry.id,
            quant=quant,
            port=port,
            model_name=model_source,
            entry=entry,
            is_running_tier=False,
            temp_service_name=service_name,
        )

    # Neither tier nor model
    tier_names = _get_all_tier_names()
    running_tiers = _get_running_tier_names()

    parts: list[str] = [f"'{target}' is not a running tier or known model."]

    if tier_names:
        parts.append(f"\nValid tier names: {', '.join(tier_names)}")
        if running_tiers:
            parts.append(f"Running tiers: {', '.join(running_tiers)}")

    parts.append("\nUse 'mlx-stack models --catalog' to see available models.")

    raise BenchmarkTargetError("\n".join(parts))


# --------------------------------------------------------------------------- #
# Main benchmark function
# --------------------------------------------------------------------------- #


def run_benchmark(
    target: str,
    save: bool = False,
) -> BenchmarkResult_:
    """Run a complete benchmark against a tier or model.

    This is the main entry point for the benchmark engine.

    Args:
        target: Tier name or model ID.
        save: Whether to persist results to disk.

    Returns:
        A BenchmarkResult_ with all collected data.

    Raises:
        BenchmarkError: If the benchmark fails.
        BenchmarkTargetError: If the target is not found.
    """
    # Auto-install vllm-mlx (needed for API calls even to running tiers)
    ensure_dependency("vllm-mlx")

    resolved = resolve_target(target)
    temp_service = resolved.temp_service_name

    try:
        result = _execute_benchmark(resolved, save)
        return result
    except Exception:
        # Ensure cleanup on any failure
        if temp_service:
            _cleanup_temp_instance(temp_service)
        raise
    finally:
        # Always clean up temp instance
        if temp_service:
            _cleanup_temp_instance(temp_service)


def _execute_benchmark(
    target: BenchmarkTarget,
    save: bool,
) -> BenchmarkResult_:
    """Execute the benchmark iterations and comparison.

    Args:
        target: The resolved benchmark target.
        save: Whether to persist results.

    Returns:
        A BenchmarkResult_.
    """
    # Run iterations
    iterations = _run_iterations(
        port=target.port,
        model_name=target.model_name,
        num_iterations=NUM_ITERATIONS,
    )

    # Compute stats
    prompt_tps_values = [it.prompt_tps for it in iterations]
    gen_tps_values = [it.gen_tps for it in iterations]

    prompt_tps_mean, prompt_tps_std = _compute_stats(prompt_tps_values)
    gen_tps_mean, gen_tps_std = _compute_stats(gen_tps_values)

    # Compare against catalog
    profile = load_profile()
    if profile is None:
        try:
            profile = detect_hardware()
        except Exception:
            profile = None

    classifications: list[MetricClassification] = []
    catalog_data_available = False
    if profile is not None:
        classifications = _compare_against_catalog(
            prompt_tps_mean, gen_tps_mean, target.entry, profile
        )
        catalog_data_available = len(classifications) > 0

    # Tool-calling benchmark
    tool_call_result: ToolCallResult | None = None
    if target.entry.capabilities.tool_calling:
        tool_call_result = _run_tool_call_benchmark(
            port=target.port,
            model_name=target.model_name,
        )

    bench_result = BenchmarkResult_(
        model_id=target.model_id,
        quant=target.quant,
        iterations=iterations,
        prompt_tps_mean=prompt_tps_mean,
        prompt_tps_std=prompt_tps_std,
        gen_tps_mean=gen_tps_mean,
        gen_tps_std=gen_tps_std,
        classifications=classifications,
        tool_call_result=tool_call_result,
        used_temporary_instance=not target.is_running_tier,
        catalog_data_available=catalog_data_available,
    )

    # Save results if requested
    if save and profile is not None:
        save_benchmark_results(bench_result, profile)

    return bench_result
