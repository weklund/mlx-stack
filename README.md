# mlx-stack

**Run multiple LLMs simultaneously on Apple Silicon. One endpoint. Automatic routing. Always on.**

[![CI](https://github.com/weklund/mlx-stack/actions/workflows/ci.yml/badge.svg)](https://github.com/weklund/mlx-stack/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mlx-stack.svg)](https://pypi.org/project/mlx-stack/)
[![Python 3.13+](https://img.shields.io/badge/python-≥3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform: macOS Apple Silicon](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg)](https://support.apple.com/en-us/116943)

---

Most local LLM tools serve **one model at a time** and leave you to figure out which model to run on your hardware. mlx-stack serves **three models simultaneously** — each optimized for a different workload — behind a single OpenAI-compatible endpoint that routes requests automatically. It turns your Mac into an always-on inference server that agents and apps can hit like a cloud API.

```bash
uv tool install mlx-stack
mlx-stack setup                    # detects hardware, picks models, pulls, starts — one command
# → OpenAI-compatible API at http://localhost:4000/v1
```

## Why mlx-stack?

### Other tools give you a model. mlx-stack gives you infrastructure.

Ollama, LM Studio, and llama.cpp are great at running a single model. But if you're building agents, serving multiple workloads, or running local inference 24/7, you need more than a model runner — you need a **control plane**.

|  | mlx-stack | Ollama | LM Studio | llama.cpp |
|--|-----------|--------|-----------|-----------|
| Simultaneous models | 3 tiers + cloud fallback | 1 at a time | 1 at a time | 1 at a time |
| API routing & fallback | Automatic tier-based routing, cascade fallback | Single endpoint | Single endpoint | No API layer |
| Hardware-aware model selection | Scores models against your exact chip (M1–M5 Pro/Max/Ultra) | Manual selection | Manual selection | Manual selection |
| 24/7 headless operation | Watchdog, auto-restart, flap detection, LaunchAgent | Manual monitoring | GUI required | Manual monitoring |
| Agent-optimized | `agent-fleet` intent, tool-call parser routing | General-purpose | General-purpose | General-purpose |
| Apple Silicon optimization | Native MLX, per-chip bandwidth profiling | Generic backend | Generic backend | Generic GGUF |
| Cloud escape hatch | OpenRouter fallback when local capacity is exceeded | None | None | None |

### Built for agents, not just chat

Most local LLM tools are designed for interactive chat. mlx-stack is designed for **agentic workloads** where different requests need different models:

- **Fast tier** — Low-latency model for tool calls, autocomplete, quick decisions
- **Standard tier** — High-quality model for reasoning, code generation, complex instructions
- **Long-context tier** — Extended context model for document processing, large codebases

Your agent framework hits one endpoint (`localhost:4000/v1`) and targets tiers by model name. If a tier goes down, requests automatically cascade to the next healthy tier — or to cloud models via OpenRouter as a last resort.

### Turn a Mac Mini into an inference server

mlx-stack is built for unattended operation. Install the LaunchAgent and walk away:

```bash
mlx-stack install   # starts on login, restarts on crash, runs forever
```

The watchdog monitors every service, auto-restarts crashed processes with exponential backoff, detects flapping services to prevent restart loops, and rotates logs to prevent unbounded disk usage. Your Mac Mini serves local inference like a cloud endpoint — no babysitting required.

### Your hardware, your stack — automatically

Instead of googling "what model should I run on M4 Max with 128GB," mlx-stack profiles your chip, measures bandwidth, and scores every model in its catalog against your exact hardware:

```bash
mlx-stack recommend --intent agent-fleet
```

The recommendation engine filters models to your memory budget, scores them across speed, quality, tool-calling capability, and memory efficiency, then assigns the optimal model to each tier. Saved benchmarks from `mlx-stack bench --save` override catalog estimates for even more precise scoring.

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

## Table of Contents

- [Why mlx-stack?](#why-mlx-stack)
- [Architecture](#architecture)
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

The fastest way to get running is the interactive setup command:

```bash
mlx-stack setup
```

This walks you through hardware detection, model selection, downloading, and starting all services in one guided flow. For CI or scripting, pass `--accept-defaults` to skip all prompts:

```bash
mlx-stack setup --accept-defaults
```

The OpenAI-compatible API is now available at `http://localhost:4000/v1`.

```bash
# Check service health
mlx-stack status

# Stop everything when done
mlx-stack down
```

<details>
<summary>Manual step-by-step setup</summary>

If you prefer full control over each step:

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

</details>

## CLI Reference

### Setup & Configuration

**`mlx-stack setup`** — Interactive guided setup: detects hardware, selects models, pulls weights, and starts the stack in one command.

| Option | Description |
|--------|-------------|
| `--accept-defaults` | Skip all prompts and use recommended defaults |
| `--intent <balanced\|agent-fleet>` | Use case intent (prompted if not provided) |
| `--budget-pct <10-90>` | Memory budget as percentage of unified memory (default: 40) |

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
mlx-stack setup --accept-defaults
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

Some models (Gemma 3, Llama 3.3) are **gated** on HuggingFace and require accepting a license before download. `mlx-stack init --accept-defaults` automatically selects non-gated models so the zero-config path works without authentication. To use gated models:

```bash
# 1. Accept the model license on huggingface.co
# 2. Set your token
export HF_TOKEN=hf_...

# 3. Pull the gated model
mlx-stack pull gemma3-12b
```

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

# Run all checks (lint + typecheck + tests) — same as CI
make check

# Or individually
make lint    # ruff + pyright
make test    # pytest with coverage
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting bugs, suggesting features, and submitting pull requests.

## License

[MIT](LICENSE)
