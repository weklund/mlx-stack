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
- `src/mlx_stack/core/` — shared business logic modules
  - `core/hardware.py` — hardware detection (Apple Silicon profiling)
  - `core/config.py` — configuration management (YAML-based)
  - `core/catalog.py` — model catalog system (query API over YAML entries)
  - `core/deps.py` — dependency management (auto-installing uv tools)
  - `core/paths.py` — path utilities (`~/.mlx-stack/` and friends)
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
- Note: The config module currently sends success output to stderr. Future features should use stdout for successful output and stderr only for errors/warnings.

## Key Design Decisions
- One vllm-mlx process per model (ADR-003)
- vllm-mlx and litellm managed as pinned uv tools, auto-installed on first use
- Catalog schema: no int6, disk_size_gb per quant source, min_mlx_lm_version top-level, verified_on in separate data/verification.yaml
- 2 intents for MVP: balanced, agent-fleet (architecture supports more)
- 40% default memory budget of total unified memory
