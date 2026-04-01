# Architecture

Architectural decisions, patterns discovered, and conventions.

**What belongs here:** Architecture decisions, module patterns, code conventions.

---

## Project Structure
- `src/mlx_stack/` — main package (src layout)
- `src/mlx_stack/cli/` — Click CLI package
  - `cli/__init__.py` — package init
  - `cli/main.py` — CLI entry point with Click command group
  - `cli/profile.py` — `mlx-stack profile` command
  - `cli/config.py` — `mlx-stack config` commands
  - `cli/init.py` — `mlx-stack init` command (stack + LiteLLM config generation)
  - `cli/recommend.py` — `mlx-stack recommend` command
  - `cli/models.py` — `mlx-stack models` command (local model listing + catalog browsing)
- `src/mlx_stack/core/` — shared business logic modules
  - `core/hardware.py` — hardware detection (Apple Silicon profiling)
  - `core/config.py` — configuration management (YAML-based)
  - `core/catalog.py` — model catalog system (query API over YAML entries)
  - `core/deps.py` — dependency management (auto-installing uv tools)
  - `core/paths.py` — path utilities (`~/.mlx-stack/` and friends)
  - `core/scoring.py` — recommendation scoring engine (intent-weighted composite scoring)
  - `core/litellm_gen.py` — LiteLLM proxy config generation (model_list, router_settings, fallbacks)
  - `core/stack_init.py` — stack initialization logic (port allocation, vllm_flags, overwrite protection)
  - `core/models.py` — local model scanning, catalog listing, size formatting
- `src/mlx_stack/data/` — static data files
  - `data/catalog/` — shipped YAML catalog files (15 models)
- `src/mlx_stack/utils/` — utility modules
- `tests/` — pytest tests
- `tests/fixtures/` — mock data (profiles, catalogs, etc.)

## Conventions
- Click for CLI, Rich for terminal output
- PyYAML for all YAML operations
- httpx for HTTP requests (async not needed — use sync client)
- psutil for process management
- All state lives in `~/.mlx-stack/` (configurable via `model-dir` for models)
- Tests use `tmp_path` pytest fixture — NEVER touch real `~/.mlx-stack/`
- External commands (sysctl, system_profiler, subprocess) are always mocked in unit tests
- Click eager options (`--help`, `--version`) may exit before the group callback runs, so callback-based setup hooks should not be relied on for those code paths
- Note: The config module currently sends success output to stderr. Future features should use stdout for successful output and stderr only for errors/warnings.

## Key Design Decisions
- One vllm-mlx process per model (ADR-003)
- vllm-mlx and litellm managed as pinned uv tools, auto-installed on first use
- Catalog schema: no int6, disk_size_gb per quant source, min_mlx_lm_version top-level, verified_on in separate data/verification.yaml
- 2 intents for MVP: balanced, agent-fleet (architecture supports more)
- 40% default memory budget of total unified memory
- Recommendation/init budget behavior: budget filtering is per-model eligibility (`model.memory_gb <= budget`); the combined memory of selected tiers can exceed the budget

## Ops Layer (Milestone 5)

### New Modules
- `core/log_rotation.py` — Copytruncate-based log rotation (copy → gzip → truncate)
- `core/log_viewer.py` — Log viewing/following/listing logic
- `core/watchdog.py` — Health polling loop, auto-restart, flap detection, daemon mode
- `core/launchd.py` — Plist generation/loading/unloading via plistlib + launchctl
- `cli/logs.py` — `mlx-stack logs` command
- `cli/watch.py` — `mlx-stack watch` command
- `cli/install.py` — `mlx-stack install` / `mlx-stack uninstall` commands

### Key Integration Points
- `process.py:start_service` — Log file open mode changed from "w" to "a" for rotation compatibility
- `core/config.py` — 2 new keys: log-max-size-mb (int, default 50), log-max-files (int, default 5)
- `process.py:acquire_lock` — Watchdog uses per-restart lock, not held during polling
- `paths.py` — Watchdog PID at get_pids_dir()/watchdog.pid
- `stack_status.py:run_status` — Used by watchdog for health polling
- `process.py:start_service` / `stop_service` — Used by watchdog for restart
- `cli/main.py` — 3 new commands registered: logs (Diagnostics), watch (Lifecycle), install/uninstall (Lifecycle)

### Log Rotation Strategy
- Copytruncate: copy log to archive, gzip compress, truncate original in-place
- Service FDs remain valid (point to same inode, just at offset 0 after truncation)
- Naming: service.log.1.gz (most recent) → service.log.N.gz (oldest)
- Archives shifted up before new rotation
- No cooperation needed from child processes (vllm-mlx, litellm)

### Log Follow Caveat
- `core/log_viewer.py:follow_log` detects truncation when `current_size < position`.
- Edge case: truncate + immediate rewrite back to exactly the previous byte length may not trigger truncation detection (`current_size == position`), so the stream can miss lines until new writes advance file size.

### Watchdog Architecture
- Single foreground loop (or daemonized with --daemon)
- Polls get_service_status for all services each interval
- Restart trigger: crashed state only (PID file exists, process dead)
- NOT restarted: stopped (no PID file), healthy, degraded
- Flap detection: rolling window of restart timestamps per service
- Lock: acquire_lock only during actual restart, released immediately
- Log rotation: triggered as side-effect of each poll cycle
