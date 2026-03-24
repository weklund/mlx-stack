# Architecture

Architectural decisions, patterns discovered, and conventions.

**What belongs here:** Architecture decisions, module patterns, code conventions.

---

## Project Structure
- `src/mlx_stack/` — main package (src layout)
- `src/mlx_stack/cli.py` — Click CLI entry point with command group
- `src/mlx_stack/commands/` — one module per CLI command
- `src/mlx_stack/core/` — shared business logic modules
- `src/mlx_stack/catalog/` — shipped YAML catalog files (15 models)
- `src/mlx_stack/data/` — static data files (chip_specs.yaml, benchmarks.json, verification.yaml)
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

## Key Design Decisions
- One vllm-mlx process per model (ADR-003)
- vllm-mlx and litellm managed as pinned uv tools, auto-installed on first use
- Catalog schema: no int6, disk_size_gb per quant source, min_mlx_lm_version top-level, verified_on in separate data/verification.yaml
- 2 intents for MVP: balanced, agent-fleet (architecture supports more)
- 40% default memory budget of total unified memory
