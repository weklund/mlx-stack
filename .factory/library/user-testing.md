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

## Validation Concurrency

**Machine:** M5 Max 128GB, 18 cores, ~97GB free at baseline
**CLI surface:** Lightweight Python process execution (~100-200MB per validator)
**Max concurrent validators:** 5
**Rationale:** Each validator runs a CLI command (Python process ~200MB). 5 concurrent = ~1GB. Even with model servers running during lifecycle tests (~10-20GB per model), the machine has ample headroom. Using 70% of available headroom: 67.9GB available * 0.7 = 47.5GB budget. Each lifecycle validator with a model server: ~12GB worst case. Max concurrent lifecycle validators: 3. For non-lifecycle tests: 5.
