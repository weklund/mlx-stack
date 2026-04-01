# mlx-stack

CLI control plane for local LLM inference infrastructure on Apple Silicon.

mlx-stack orchestrates [vllm-mlx](https://github.com/vllm-project/vllm) model servers and a [LiteLLM](https://github.com/BerriAI/litellm) API gateway to run large language models locally on Apple Silicon Macs. It handles hardware profiling, model recommendation, downloading, configuration, process lifecycle, and benchmarking through a single CLI.

## Requirements

- macOS on Apple Silicon (M-series)
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended package manager)

## Installation

```
uv pip install .
```

For development:

```
uv sync
```

## Quick Start

```
# 1. Detect your hardware
mlx-stack profile

# 2. Get model recommendations for your hardware
mlx-stack recommend

# 3. Generate stack configuration (tier assignments + LiteLLM config)
mlx-stack init --accept-defaults

# 4. Download required models
mlx-stack pull qwen3.5-8b

# 5. Start all services
mlx-stack up

# 6. Check service health
mlx-stack status

# 7. Stop all services
mlx-stack down
```

Once the stack is running, the OpenAI-compatible API is available at `http://localhost:4000/v1`.

## Commands

### Setup and Configuration

**`mlx-stack profile`** -- Detect Apple Silicon hardware (chip, GPU cores, unified memory, memory bandwidth) and save the profile to `~/.mlx-stack/profile.json`.

**`mlx-stack config set <key> <value>`** -- Set a configuration value.

**`mlx-stack config get <key>`** -- Get a configuration value.

**`mlx-stack config list`** -- List all configuration values with defaults and sources.

**`mlx-stack config reset --yes`** -- Reset all configuration to defaults.

### Model Management

**`mlx-stack recommend`** -- Recommend an optimal model stack based on your hardware profile. Display-only; no files are written.

Options:
- `--budget <value>` -- Memory budget override (e.g., `30gb`, `16`). Defaults to 40% of unified memory.
- `--intent <balanced|agent-fleet>` -- Optimization strategy.
- `--show-all` -- Show all budget-fitting models ranked by score.

**`mlx-stack models`** -- List locally downloaded models with disk size, quantization, and active stack status.

Options:
- `--catalog` -- Show all catalog models with hardware-specific benchmark data.
- `--family <name>` -- Filter by model family (e.g., `qwen3.5`).
- `--tag <name>` -- Filter by tag (e.g., `agent-ready`).
- `--tool-calling` -- Filter to tool-calling-capable models only.

**`mlx-stack pull <model>`** -- Download a model from the catalog (e.g., `mlx-stack pull qwen3.5-8b`).

Options:
- `--quant <int4|int8|bf16>` -- Quantization level. Defaults to `int4`.
- `--bench` -- Run a quick benchmark after download.
- `--force` -- Re-download even if the model already exists.

**`mlx-stack init`** -- Generate stack definition (`~/.mlx-stack/stacks/default.yaml`) and LiteLLM proxy configuration (`~/.mlx-stack/litellm.yaml`) from hardware profile and recommendation.

Options:
- `--accept-defaults` -- Use defaults without prompting.
- `--intent <balanced|agent-fleet>` -- Optimization strategy.
- `--add <model>` -- Add a model to the stack (repeatable).
- `--remove <tier>` -- Remove a tier from the stack (repeatable).
- `--force` -- Overwrite existing stack configuration.

### Stack Lifecycle

**`mlx-stack up`** -- Start all services: one vllm-mlx process per tier plus the LiteLLM proxy.

Options:
- `--dry-run` -- Show the commands that would be executed without starting anything.
- `--tier <name>` -- Start only the specified tier.

**`mlx-stack down`** -- Stop all managed services. Sends SIGTERM with a 10-second grace period, then SIGKILL.

Options:
- `--tier <name>` -- Stop only the specified tier.

**`mlx-stack status`** -- Show health and status of all running services (healthy, degraded, down, crashed, stopped).

Options:
- `--json` -- Output in JSON format.

### Diagnostics

**`mlx-stack bench <target>`** -- Benchmark a running tier or catalog model. Runs 3 iterations of 1024-token prompt + 100-token generation and reports mean/std-dev for prompt and generation tokens per second. Compares against catalog thresholds (PASS/WARN/FAIL).

Options:
- `--save` -- Persist results for use by `recommend` and `init` scoring.

### Ops and Reliability

**`mlx-stack logs [service]`** -- View and manage service logs. Without arguments, lists all available log files with sizes and modification times.

Options:
- `--follow` / `-f` -- Follow log output in real-time.
- `--tail <N>` -- Show last N lines (default 50).
- `--service <name>` -- Filter to a specific service's log.
- `--rotate` -- Rotate eligible log files.
- `--all` -- Show archived and current logs in chronological order.

**`mlx-stack watch`** -- Health monitor that polls service status and auto-restarts crashed services. Includes flap detection, exponential backoff, and log rotation.

Options:
- `--interval <seconds>` -- Seconds between health polls (default 30).
- `--max-restarts <N>` -- Maximum restarts before marking a service as flapping (default 5).
- `--restart-delay <seconds>` -- Base delay before restart with exponential backoff (default 5).
- `--daemon` -- Run in background as a daemon.

**`mlx-stack install`** -- Install the watchdog as a macOS LaunchAgent for automatic startup on login.

Options:
- `--status` -- Show current launchd agent status without installing.

**`mlx-stack uninstall`** -- Remove the watchdog LaunchAgent. Running services are not affected.

## Configuration

Configuration is stored in `~/.mlx-stack/config.yaml`. Available keys:

| Key | Default | Description |
|-----|---------|-------------|
| `openrouter-key` | (not set) | OpenRouter API key for cloud fallback |
| `default-quant` | `int4` | Default quantization level (`int4`, `int8`, `bf16`) |
| `memory-budget-pct` | `40` | Percentage of unified memory to budget for models (1-100) |
| `litellm-port` | `4000` | LiteLLM proxy port |
| `model-dir` | `~/.mlx-stack/models` | Model storage directory |
| `auto-health-check` | `true` | Run health checks automatically on startup |
| `log-max-size-mb` | `50` | Maximum log file size in MB before rotation |
| `log-max-files` | `3` | Number of rotated log files to retain |

## 24/7 Operation

mlx-stack can run unattended as a persistent local inference service.

### Quick setup

```
mlx-stack init --accept-defaults
mlx-stack install
```

This installs a macOS LaunchAgent that starts the watchdog automatically on login. The watchdog monitors service health every 30 seconds, auto-restarts crashed processes with exponential backoff, detects flapping services, and rotates logs to prevent unbounded disk usage.

### Manual monitoring

Run the watchdog in the foreground for interactive monitoring:

```
mlx-stack watch
```

This displays a Rich-formatted status table each poll cycle and prints restart events as they happen. Use `--interval 60` to poll less frequently or `--daemon` to run in the background without a LaunchAgent.

### Log management

View recent output from any service:

```
mlx-stack logs                      # List all log files
mlx-stack logs fast                 # Last 50 lines of fast tier
mlx-stack logs fast --follow        # Stream in real-time
mlx-stack logs --rotate             # Rotate all eligible logs
```

Log rotation happens automatically during watchdog polls. Configure rotation thresholds with `log-max-size-mb` (default 50 MB) and `log-max-files` (default 3 retained archives).

### Removing the agent

```
mlx-stack uninstall
```

This stops the watchdog and removes the LaunchAgent plist. Running services are not affected.

## Model Catalog

The built-in catalog includes 15 models across multiple families:

- Qwen 3.5 (0.8B, 3B, 8B, 14B, 32B, 72B)
- Qwen 3 (8B)
- Gemma 3 (4B, 12B, 27B)
- Llama 3.3 (8B)
- DeepSeek R1 (8B, 32B)
- Nemotron (8B, 49B)

Each model includes catalog benchmark data for common Apple Silicon configurations, quality scores, and capability metadata (tool calling, thinking/reasoning, vision).

## Architecture

mlx-stack manages a tiered local inference stack:

- **vllm-mlx** -- One instance per tier (e.g., `standard`, `fast`, `longctx`), each serving a model on a dedicated port.
- **LiteLLM** -- API gateway that routes requests across tiers, providing an OpenAI-compatible `/v1` endpoint on a single port (default 4000).
- **Cloud fallback** -- Optional OpenRouter integration for a premium tier when a local model is insufficient.

The recommendation engine scores models against your hardware profile and memory budget, then assigns them to tiers optimized for different use cases. Saved benchmark data from `mlx-stack bench --save` is used to refine scoring with real measurements instead of catalog estimates.

## Development

```
# Install dev dependencies
uv sync

# Run tests
pytest

# Type checking
pyright

# Linting
ruff check src tests
```

## License

MIT
