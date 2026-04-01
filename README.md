# mlx-stack

**CLI control plane for local LLM inference infrastructure on Apple Silicon.**

[![CI](https://github.com/weklund/mlx-stack/actions/workflows/ci.yml/badge.svg)](https://github.com/weklund/mlx-stack/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mlx-stack.svg)](https://pypi.org/project/mlx-stack/)
[![Python 3.13+](https://img.shields.io/badge/python-≥3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform: macOS Apple Silicon](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg)](https://support.apple.com/en-us/116943)

---

## Table of Contents

- [Architecture](#architecture)
- [Feature Highlights](#feature-highlights)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [24/7 Operation](#247-operation)
- [Model Catalog](#model-catalog)
- [Architecture Details](#architecture-details)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Architecture

```
                        ┌──────────────────────────────────────────────────┐
                        │                  mlx-stack CLI                   │
                        │  hardware detection · recommendation · lifecycle │
                        └──────────────┬───────────────────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        ▼                        ▼
   ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
   │   vllm-mlx :8000  │  │   vllm-mlx :8001  │  │   vllm-mlx :8002  │
   │  ── standard ──   │  │    ── fast ──      │  │  ── longctx ──    │
   │  Qwen 3.5 14B     │  │  Qwen 3.5 3B      │  │  DeepSeek R1 8B   │
   └────────┬──────────┘  └────────┬──────────┘  └────────┬──────────┘
            │                      │                       │
            └──────────────────────┼───────────────────────┘
                                   │
                        ┌──────────▼──────────┐
                        │  LiteLLM Proxy :4000│
                        │  routing · fallback  │
                        │  load balancing      │
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │  OpenAI-compatible   │
                        │  /v1 endpoint        │
                        │                      │
                        │  ← Your app / agent  │
                        └─────────────────────┘
```

mlx-stack orchestrates [vllm-mlx](https://github.com/vllm-project/vllm) model servers and a [LiteLLM](https://github.com/BerriAI/litellm) API gateway to serve large language models locally on Apple Silicon Macs. Each tier runs a dedicated model optimized for a specific workload — quality, speed, or long-context — and LiteLLM routes requests through a single OpenAI-compatible endpoint with automatic fallback.

## Feature Highlights

- **Hardware-Aware Recommendations** — Detects your Apple Silicon chip (M1–M5, Pro/Max/Ultra), measures memory bandwidth, and recommends an optimal model stack tuned to your exact hardware.
- **Tiered Model Serving** — Assigns models to `standard`, `fast`, and `longctx` tiers so agents and apps can target the right balance of quality and speed per request.
- **24/7 Unattended Operation** — Built-in watchdog with auto-restart, flap detection, exponential backoff, and macOS LaunchAgent integration for always-on inference on headless Mac Minis.
- **One-Command Setup** — `mlx-stack init --accept-defaults` profiles your hardware, picks models, generates configs, and gets you from zero to a running OpenAI-compatible endpoint in minutes.
- **15-Model Curated Catalog** — Ships with benchmark data for Qwen 3.5, Gemma 3, DeepSeek R1, Nemotron, and Llama 3.3 families — with quality scores, tool-calling metadata, and per-hardware performance data.

## Installation

The recommended way to install mlx-stack is with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install mlx-stack
```

This installs `mlx-stack` globally as an isolated tool — no need to manage virtual environments.

Alternatively, you can use [pipx](https://pipx.pypa.io/):

```bash
pipx install mlx-stack
```

Or try it without installing:

```bash
uvx mlx-stack profile
```

> **Note:** `uvx` runs in an ephemeral environment, which works great for one-off commands. For the watchdog and LaunchAgent features (`mlx-stack watch`, `mlx-stack install`), use `uv tool install` so the binary has a stable path.

## Quick Start

```bash
# 1. Detect your hardware
mlx-stack profile

# 2. Generate stack configuration
mlx-stack init --accept-defaults

# 3. Download required models
mlx-stack pull qwen3.5-8b

# 4. Start all services
mlx-stack up

# 5. Verify
mlx-stack status
```

The OpenAI-compatible API is now available at `http://localhost:4000/v1`.

```bash
# Stop everything when done
mlx-stack down
```

## CLI Reference

### Setup & Configuration

| Command | Description |
|---------|-------------|
| `mlx-stack profile` | Detect Apple Silicon hardware and save profile to `~/.mlx-stack/profile.json` |
| `mlx-stack config set <key> <value>` | Set a configuration value |
| `mlx-stack config get <key>` | Get a configuration value |
| `mlx-stack config list` | List all configuration values with defaults and sources |
| `mlx-stack config reset --yes` | Reset all configuration to defaults |

### Model Management

**`mlx-stack recommend`** — Recommend an optimal model stack based on your hardware profile.

| Option | Description |
|--------|-------------|
| `--budget <value>` | Memory budget override (e.g., `30gb`). Defaults to 40% of unified memory |
| `--intent <balanced\|agent-fleet>` | Optimization strategy |
| `--show-all` | Show all budget-fitting models ranked by score |

**`mlx-stack models`** — List locally downloaded models with disk size, quantization, and active stack status.

| Option | Description |
|--------|-------------|
| `--catalog` | Show all catalog models with hardware-specific benchmark data |
| `--family <name>` | Filter by model family (e.g., `qwen3.5`) |
| `--tag <name>` | Filter by tag (e.g., `agent-ready`) |
| `--tool-calling` | Filter to tool-calling-capable models only |

**`mlx-stack pull <model>`** — Download a model from the catalog.

| Option | Description |
|--------|-------------|
| `--quant <int4\|int8\|bf16>` | Quantization level (default: `int4`) |
| `--bench` | Run a quick benchmark after download |
| `--force` | Re-download even if the model already exists |

**`mlx-stack init`** — Generate stack definition and LiteLLM proxy configuration.

| Option | Description |
|--------|-------------|
| `--accept-defaults` | Use defaults without prompting |
| `--intent <balanced\|agent-fleet>` | Optimization strategy |
| `--add <model>` | Add a model to the stack (repeatable) |
| `--remove <tier>` | Remove a tier from the stack (repeatable) |
| `--force` | Overwrite existing stack configuration |

### Stack Lifecycle

**`mlx-stack up`** — Start all services: one vllm-mlx process per tier plus the LiteLLM proxy.

| Option | Description |
|--------|-------------|
| `--dry-run` | Show exact commands without starting anything |
| `--tier <name>` | Start only the specified tier |

**`mlx-stack down`** — Stop all managed services (SIGTERM → 10s grace → SIGKILL).

| Option | Description |
|--------|-------------|
| `--tier <name>` | Stop only the specified tier |

**`mlx-stack status`** — Show health and status of all services (healthy, degraded, down, crashed, stopped).

| Option | Description |
|--------|-------------|
| `--json` | Output in JSON format |

### Diagnostics

**`mlx-stack bench <target>`** — Benchmark a running tier or catalog model. Runs 3 iterations and compares against catalog thresholds (PASS/WARN/FAIL).

| Option | Description |
|--------|-------------|
| `--save` | Persist results for use by `recommend` and `init` scoring |

### Ops & Reliability

**`mlx-stack logs [service]`** — View and manage service logs. Without arguments, lists all log files.

| Option | Description |
|--------|-------------|
| `--follow` / `-f` | Follow log output in real-time |
| `--tail <N>` | Show last N lines (default: 50) |
| `--service <name>` | Filter to a specific service |
| `--rotate` | Rotate eligible log files |
| `--all` | Show archived and current logs chronologically |

**`mlx-stack watch`** — Health monitor with auto-restart, flap detection, and log rotation.

| Option | Description |
|--------|-------------|
| `--interval <seconds>` | Polling interval (default: 30) |
| `--max-restarts <N>` | Restarts before marking as flapping (default: 5) |
| `--restart-delay <seconds>` | Base restart delay with exponential backoff (default: 5) |
| `--daemon` | Run in background as a daemon |

**`mlx-stack install`** — Install the watchdog as a macOS LaunchAgent.

| Option | Description |
|--------|-------------|
| `--status` | Show current LaunchAgent status |

**`mlx-stack uninstall`** — Remove the watchdog LaunchAgent. Running services are not affected.

## Configuration

Configuration is stored in `~/.mlx-stack/config.yaml`. Available keys:

| Key | Default | Description |
|-----|---------|-------------|
| `openrouter-key` | *(not set)* | OpenRouter API key for cloud fallback |
| `default-quant` | `int4` | Default quantization level (`int4`, `int8`, `bf16`) |
| `memory-budget-pct` | `40` | Percentage of unified memory to budget for models (1–100) |
| `litellm-port` | `4000` | LiteLLM proxy port |
| `model-dir` | `~/.mlx-stack/models` | Model storage directory |
| `auto-health-check` | `true` | Run health checks automatically on startup |
| `log-max-size-mb` | `50` | Maximum log file size in MB before rotation |
| `log-max-files` | `3` | Number of rotated log files to retain |

## 24/7 Operation

mlx-stack is designed to run unattended on always-on hardware like a Mac Mini.

### Quick setup

```bash
mlx-stack init --accept-defaults
mlx-stack install
```

This installs a macOS LaunchAgent that starts the watchdog on login. The watchdog:

- Monitors service health every 30 seconds
- Auto-restarts crashed processes with exponential backoff
- Detects flapping services and stops restart loops
- Rotates logs automatically to prevent unbounded disk usage

### Manual monitoring

```bash
mlx-stack watch                  # Foreground with Rich status table
mlx-stack watch --interval 60   # Less frequent polling
mlx-stack watch --daemon         # Background without LaunchAgent
```

### Log management

```bash
mlx-stack logs                   # List all log files
mlx-stack logs fast              # Last 50 lines of fast tier
mlx-stack logs fast --follow     # Stream in real-time
mlx-stack logs --rotate          # Rotate all eligible logs now
```

### Removing the agent

```bash
mlx-stack uninstall
```

This stops the watchdog and removes the LaunchAgent plist. Running services are not affected.

## Model Catalog

The built-in catalog includes 15 models across 5 families:

| Family | Models | Parameters |
|--------|--------|------------|
| Qwen 3.5 | 6 variants | 0.8B, 3B, 8B, 14B, 32B, 72B |
| Gemma 3 | 3 variants | 4B, 12B, 27B |
| DeepSeek R1 | 2 variants | 8B, 32B |
| Nemotron | 2 variants | 8B, 49B |
| Qwen 3 / Llama 3.3 | 2 variants | 8B each |

Each entry includes benchmark data for common Apple Silicon configurations, quality scores, and capability metadata (tool calling, thinking/reasoning, vision).

## Architecture Details

mlx-stack manages a **tiered local inference stack** with three layers:

### Model Servers (vllm-mlx)

One [vllm-mlx](https://github.com/vllm-project/vllm) instance per tier, each serving a single model on a dedicated port:

- **standard** (port 8000) — Highest-quality model that fits your memory budget. Optimized for accuracy-sensitive tasks.
- **fast** (port 8001) — Fastest model for latency-sensitive workloads like autocomplete and quick tool calls.
- **longctx** (port 8002) — Architecturally diverse model (e.g., Mamba2 hybrid) for extended context windows.

Each server runs with continuous batching, paged KV cache, and automatic tool-call parsing enabled.

### API Gateway (LiteLLM)

[LiteLLM](https://github.com/BerriAI/litellm) acts as the unified entry point on port 4000, providing:

- **OpenAI-compatible `/v1` API** — Drop-in replacement for `api.openai.com` in any client or agent framework.
- **Tier-based routing** — Requests target specific tiers by model name, or fall through a configurable chain.
- **Automatic fallback** — If the primary tier is unavailable, requests cascade to the next healthy tier.

### Cloud Fallback (Optional)

With an OpenRouter API key configured, a `premium` cloud tier is available as a last-resort fallback, giving you access to frontier models when local capacity is insufficient.

### Recommendation Engine

The recommendation engine scores all catalog models against your hardware profile:

1. **Hardware profiling** — Detects chip variant, GPU cores, unified memory, and memory bandwidth.
2. **Memory budgeting** — Filters models to those fitting within your configured memory budget (default: 40% of unified memory).
3. **Composite scoring** — Weights speed, quality, tool-calling capability, and memory efficiency based on your chosen intent (`balanced` or `agent-fleet`).
4. **Tier assignment** — Assigns top-scoring models to `standard`, `fast`, and `longctx` tiers.
5. **Local calibration** — Saved benchmark data from `mlx-stack bench --save` overrides catalog estimates for precise scoring.

### Process Management

- **PID tracking** — Each service writes its PID to `~/.mlx-stack/pids/` for reliable lifecycle management.
- **Lockfile** — Prevents concurrent `up`/`down` operations via `fcntl.flock`.
- **Health checks** — HTTP polling with exponential backoff and 120-second timeout per service.
- **5-state model** — Services are reported as `healthy`, `degraded`, `down`, `crashed`, or `stopped`.
- **Graceful shutdown** — SIGTERM with 10-second grace period, escalating to SIGKILL.

## Development

See [DEVELOPING.md](DEVELOPING.md) for the full developer guide, including project architecture, testing strategy, and how to add new models or commands.

```bash
# Install dev dependencies
uv sync

# Run tests
uv run pytest

# Type checking
uv run python -m pyright

# Linting
uv run ruff check src/ tests/
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting bugs, suggesting features, and submitting pull requests.

## License

[MIT](LICENSE)
