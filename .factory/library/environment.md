# Environment

Environment variables, external dependencies, and setup notes.

**What belongs here:** Required env vars, external API keys/services, dependency quirks, platform-specific notes.
**What does NOT belong here:** Service ports/commands (use `.factory/services.yaml`).

---

## Python Environment

- Python 3.14+ via `uv`
- All dependencies managed by `uv sync --dev`
- Virtual environment at `.venv/` (created by uv)

## Key Dependencies

- `click` — CLI framework
- `rich` — Terminal UI (tables, colors, progress)
- `pyyaml` — YAML parsing
- `huggingface_hub` — HF API + model downloads
- `pytest` + `pytest-cov` — Testing
- `ruff` — Linting
- `pyright` — Type checking

## Environment Variables

- `MLX_STACK_HOME` — Override data directory (default: `~/.mlx-stack/`). Used extensively in tests via `mlx_stack_home` fixture.

## Data Directories

- `~/.mlx-stack/` — User data home
- `~/.mlx-stack/stacks/default.yaml` — Stack definition
- `~/.mlx-stack/litellm.yaml` — LiteLLM proxy config
- `~/.mlx-stack/profile.json` — Hardware profile
- `~/.mlx-stack/config.yaml` — User configuration
- `~/.mlx-stack/models/` — Downloaded model files
- `~/.mlx-stack/benchmarks/` — Saved benchmark results
