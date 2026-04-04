# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.5](https://github.com/weklund/mlx-stack/compare/v0.3.4...v0.3.5) (2026-04-04)


### Features

* expand ruff lint rules with tier 1+2 quality rulesets ([#22](https://github.com/weklund/mlx-stack/issues/22)) ([75490f6](https://github.com/weklund/mlx-stack/commit/75490f6817a87a6b63818fa1f7c1660e59766ba3))

## [0.3.4](https://github.com/weklund/mlx-stack/compare/v0.3.3...v0.3.4) (2026-04-03)


### Features

* 4-tier integration testing framework ([#16](https://github.com/weklund/mlx-stack/issues/16)) ([e3dcf9a](https://github.com/weklund/mlx-stack/commit/e3dcf9ae4eaa55ee8ca522819d3dc16846d3e1ea))

## [0.3.3](https://github.com/weklund/mlx-stack/compare/v0.3.2...v0.3.3) (2026-04-02)


### Bug Fixes

* replace raw status strings with ServiceHealth enum and assert behavior in tests ([#13](https://github.com/weklund/mlx-stack/issues/13)) ([ef0161c](https://github.com/weklund/mlx-stack/commit/ef0161c9f0b64096917fd06045786aad49fe8c93))

## [0.3.2](https://github.com/weklund/mlx-stack/compare/v0.3.1...v0.3.2) (2026-04-02)


### Bug Fixes

* correct inverted sign on benchmark delta percentage display ([ae7d209](https://github.com/weklund/mlx-stack/commit/ae7d20981e48b6033de9ed05115a2754576fd2a7))
* pass explicit token to release-please action ([96c9c68](https://github.com/weklund/mlx-stack/commit/96c9c686f21873a2b1a3b3596bcf87005e08843e))

## [0.1.0] - 2025-03-31

Initial release of mlx-stack — a CLI control plane for local LLM inference infrastructure on Apple Silicon.

### Added

#### Foundation
- **Hardware detection** (`mlx-stack profile`) — Detects Apple Silicon chip model (M1–M5, including Pro/Max/Ultra variants), GPU core count, unified memory, and memory bandwidth from a lookup table of 17 known variants. Writes profile to `~/.mlx-stack/profile.json`. Handles unknown chips with bandwidth estimation and rejects non-Apple-Silicon hardware gracefully.
- **Model catalog** — 15 curated model entries across 5 families (Qwen 3.5, Gemma 3, DeepSeek R1, Nemotron, Qwen 3 / Llama 3.3) with per-hardware benchmark data, quality scores, tool-calling metadata, and disk size information.
- **Configuration management** (`mlx-stack config`) — Persistent key-value configuration with `set`, `get`, `list`, and `reset` subcommands. Supports 8 config keys with type validation, default values, and API key masking. Stored in `~/.mlx-stack/config.yaml`.
- **Dependency management** — Auto-detection and installation of vllm-mlx and litellm as pinned uv tools. Version mismatch warnings. Triggered only by commands that need these tools.
- **Data directory auto-creation** — `~/.mlx-stack/` created automatically on first use with appropriate subdirectories.

#### Recommendation
- **Scoring engine** — Weighted composite scoring across speed, quality, tool-calling, and memory efficiency dimensions. Two intents: `balanced` and `agent-fleet`. Log-scaled gen_tps normalization. Memory budget filtering (default 40% of unified memory).
- **Recommendation command** (`mlx-stack recommend`) — Recommends an optimal model stack based on hardware profile and intent. Supports `--budget`, `--intent`, and `--show-all` flags. Cloud fallback tier shown when OpenRouter key is configured. Display-only — no files written.
- **Config generation** (`mlx-stack init`) — Generates stack definition (`~/.mlx-stack/stacks/default.yaml`) and LiteLLM proxy config (`~/.mlx-stack/litellm.yaml`). Supports `--accept-defaults`, `--intent`, `--add`/`--remove`, and `--force`. Includes port collision detection, overwrite protection, and missing model warnings.
- **Model listing** (`mlx-stack models`) — Lists locally downloaded models with disk size, quantization, and active stack indicator. `--catalog` shows full catalog with hardware-specific benchmark data. Supports `--family`, `--tag`, and `--tool-calling` filters.

#### Lifecycle
- **Process management** — PID file tracking, fcntl-based lockfile for concurrent invocation prevention, HTTP health checks with exponential backoff, and SIGTERM/SIGKILL graceful shutdown.
- **Start services** (`mlx-stack up`) — Starts vllm-mlx instances and LiteLLM proxy from stack definition. Sequential startup (largest model first). Supports `--dry-run` and `--tier`. Handles port conflicts, missing models, partial failures, stale PID cleanup, and auto-installs dependencies.
- **Stop services** (`mlx-stack down`) — Stops all managed processes (LiteLLM first, then model servers in reverse order). SIGTERM with 10s grace period, SIGKILL escalation. Supports `--tier` for selective stop.
- **Health monitoring** (`mlx-stack status`) — 5-state health reporting: healthy, degraded, down, crashed, stopped. Formatted table with uptime. `--json` for machine-parseable output. Read-only — does not modify files.

#### Tooling
- **Model download** (`mlx-stack pull`) — Downloads models from HuggingFace, preferring mlx-community pre-converted weights with fallback to mlx_lm conversion. Disk space checking, progress display, duplicate detection, inventory tracking. Supports `--quant`, `--bench`, and `--force`.
- **Benchmarking** (`mlx-stack bench`) — Runs 3-iteration benchmarks (1024-token prompt, 100-token generation). Compares against catalog thresholds (PASS/WARN/FAIL). Supports running tiers and local models (temporary vllm-mlx instance). `--save` persists results for scoring. Tool-calling benchmark for capable models.

#### Ops
- **Log rotation** — Copytruncate-based rotation for service logs. Configurable size threshold (`log-max-size-mb`, default 50MB) and retention count (`log-max-files`, default 5). Gzip compression with sequential numbering.
- **Logs command** (`mlx-stack logs`) — View service logs with `--follow`, `--tail`, `--service` filtering. `--rotate` for on-demand rotation. `--all` for current and archived log viewing.
- **Watchdog** (`mlx-stack watch`) — Health monitor with auto-restart of crashed services. Configurable polling interval, flap detection, exponential backoff on restart delay. `--daemon` for background operation. Triggers log rotation during polling.
- **launchd integration** (`mlx-stack install` / `mlx-stack uninstall`) — Generates macOS LaunchAgent plist for the watchdog. RunAtLoad + KeepAlive. `--status` flag for agent state reporting.

#### CLI
- **Rich-formatted output** — All commands use Rich tables and styled output.
- **Typo suggestions** — Near-miss command names show "Did you mean ...?" suggestions.
- **Grouped help** — `mlx-stack --help` organizes commands by category (Setup, Model Management, Lifecycle, Diagnostics, Ops).
- **Version flag** — `mlx-stack --version` prints the current version.

#### Infrastructure
- **GitHub Actions CI** — macOS runner with lint (ruff), typecheck (pyright), and test (pytest) on push and PR.
- **Comprehensive test suite** — 1300+ unit tests covering core modules and CLI commands with mocked external calls.

### Changed

N/A — initial release.

### Fixed

N/A — initial release.

[0.1.0]: https://github.com/weklund/mlx-stack/releases/tag/v0.1.0
