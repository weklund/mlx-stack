# Architecture

How the mlx-stack system works at a high level.

## Overview

mlx-stack is a CLI tool that manages local LLM infrastructure on Apple Silicon. It orchestrates vllm-mlx model servers behind a LiteLLM proxy, providing a unified OpenAI-compatible API endpoint.

## Layers

```
CLI Layer (src/mlx_stack/cli/)
  ├── Commands: setup, up, down, status, models, pull, bench, logs, config, watch, install, uninstall
  └── Each command is a Click command registered in main.py

Core Layer (src/mlx_stack/core/)
  ├── hardware.py      — Apple Silicon detection (chip, GPU cores, memory, bandwidth)
  ├── catalog.py       — YAML catalog loading, validation, querying (15 curated models)
  ├── discovery.py     — Live HuggingFace API query for mlx-community models
  ├── scoring.py       — Hardware-aware model recommendation engine
  ├── onboarding.py    — Setup wizard orchestration (scoring variant for DiscoveredModel)
  ├── stack_init.py    — Stack definition generation (stack.yaml + litellm.yaml)
  ├── litellm_gen.py   — LiteLLM proxy config generation
  ├── stack_up.py      — Process management (start/stop vllm-mlx + LiteLLM)
  ├── pull.py          — Model download (HuggingFace snapshot_download)
  ├── benchmark.py     — Performance benchmarking
  ├── watchdog.py      — Health monitoring + auto-restart
  ├── launchd.py       — macOS LaunchAgent management
  ├── config.py        — User config (~/.mlx-stack/config.yaml)
  ├── paths.py         — Path resolution for data/config/stacks
  └── process.py       — Low-level process management

Data Layer (src/mlx_stack/data/)
  ├── catalog/*.yaml   — Curated model entries (15 files)
  └── benchmark_data.json — Static performance overlay from mlx_transformers_benchmark
```

## Data Flow

1. **Hardware detection** → `HardwareProfile` (chip, memory, bandwidth, GPU cores)
2. **Model discovery** → `CatalogEntry` (from YAML catalog) or `DiscoveredModel` (from HF API)
3. **Scoring** → `ScoredModel` / `ScoredDiscoveredModel` with composite scores
4. **Tier assignment** → `TierAssignment` (model → tier name mapping)
5. **Config generation** → `stack.yaml` (tier definitions) + `litellm.yaml` (proxy config)
6. **Process management** → vllm-mlx subprocesses + LiteLLM proxy process

## Key Files for This Mission

- `cli/main.py` — Command registration, `_COMMAND_CATEGORIES`, welcome screen, help formatting
- `cli/pull.py` — Pull command (being ungated to accept HF repos)
- `cli/status.py` — Status command (absorbing hardware display from profile)
- `cli/models.py` — Models command (absorbing recommend functionality)
- `cli/setup.py` — Setup command (gaining modification flags)
- `cli/profile.py` — Being DELETED
- `cli/recommend.py` — Being DELETED
- `cli/init.py` — Being DELETED
- `core/pull.py` — Download infrastructure (already accepts arbitrary HF repos)
- `core/stack_init.py` — Config generation (preserved for internal use by setup)
- `core/onboarding.py` — Setup wizard orchestration

## Testing Patterns

- All CLI tests use Click's `CliRunner().invoke(cli, ["command", ...])`
- Core functions mocked via `@patch("mlx_stack.core.module.function")` or `monkeypatch.setattr`
- `FakeServiceLayer` test double for stack_up/watchdog tests
- Test factories in `tests/factories.py` for creating test data
- No real HF downloads, no real hardware detection in unit tests
