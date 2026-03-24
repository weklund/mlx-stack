# User Testing

Testing surface, required testing skills/tools, resource cost classification per surface.

**What belongs here:** How to test the user-facing surface, tools needed, concurrency limits.

---

## Validation Surface

**Surface:** CLI commands executed in terminal
**Tool:** Direct shell command execution (subprocess or Click CliRunner)
**Required tools:**
- Python 3.13+ with uv
- vllm-mlx v0.2.6 (installed as uv tool)
- litellm (installed as uv tool)
- curl (for HTTP endpoint verification)

**Setup needed for validation:**
- A downloaded model (small, e.g., qwen3.5-0.8b int4) for lifecycle testing
- `mlx-stack init --accept-defaults` to generate configs
- No browser or GUI tools needed

**Gaps:**
- Full integration testing of `up`/`down`/`status` requires downloaded models and sufficient memory
- Benchmark validation requires a running model server
- Tool-call benchmark requires a model that supports tool calling
- Foundation milestone user-testing run (2026-03-24) observed placeholder CLI surfaces for `models --catalog`, `up`, and `bench`; related catalog/dependency assertions were blocked until those commands are implemented.

## Validation Concurrency

**Machine:** M5 Max 128GB, 18 cores, ~97GB free at baseline
**CLI surface:** Lightweight Python process execution (~100-200MB per validator)
**Max concurrent validators:** 5
**Rationale:** Each validator runs a CLI command (Python process ~200MB). 5 concurrent = ~1GB. Even with model servers running during lifecycle tests (~10-20GB per model), the machine has ample headroom. Using 70% of available headroom: 67.9GB available * 0.7 = 47.5GB budget. Each lifecycle validator with a model server: ~12GB worst case. Max concurrent lifecycle validators: 3. For non-lifecycle tests: 5.

## Flow Validator Guidance: CLI

- Use only terminal-based validation commands (`uv run mlx-stack ...`) and shell inspection commands.
- Enforce isolation with a unique `MLX_STACK_HOME` per validator (example: `/tmp/mlx-stack-user-testing/<group-id>`). Never reuse another validator's home.
- Do not read from or write to real `~/.mlx-stack/`; keep all generated files under each validator's assigned `MLX_STACK_HOME`.
- Keep evidence in the assigned mission evidence directory only.
- Stay within assigned assertion scope and avoid commands that mutate global/shared system state.

## Recommendation milestone run notes (2026-03-24)

- `recommend` is currently display-only and does **not** persist `profile.json` when auto-detecting hardware.
- `models --catalog` currently does not expose filter flags for family/tag/capability on the CLI surface.
- `pull` and `bench` remain placeholder commands in this build, which blocks benchmark-save recommendation validation flows.
- For validator fixture scripting, prefer `uv run python` over system `python3` so project dependencies (e.g., PyYAML) are available.

## Lifecycle milestone rerun notes (2026-03-24)

- In isolated lifecycle rerun flow `r2-g1-fixes`, macOS denied `psutil.net_connections(kind='inet')` with `AccessDenied`; port conflict output fell back to `PID 0 (<unknown>)` even though preflight conflict skipping worked. Treat owner-resolution checks as potentially permission-sensitive on this host.
