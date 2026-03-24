# Environment

Environment variables, external dependencies, and setup notes.

**What belongs here:** Required env vars, external API keys/services, dependency quirks, platform-specific notes.
**What does NOT belong here:** Service ports/commands (use `.factory/services.yaml`).

---

## Machine
- Apple MacBook Pro M5 Max, 128 GB unified memory, 18 CPU cores, 40 GPU cores
- macOS 26.x
- Python 3.14.3 (targeting 3.13+ compatibility)

## Tools
- uv 0.10.12 (package manager)
- vllm-mlx v0.2.6 (installed as uv tool at ~/.local/bin/vllm-mlx)
- litellm (installed as uv tool at ~/.local/bin/litellm)
- For robust `uv tool list` parsing, set `NO_COLOR=1` when invoking uv to avoid ANSI escape sequences in output

## External Dependencies
- HuggingFace Hub (for model downloads — optional HF_TOKEN for rate limiting)
- OpenRouter API (optional, for cloud fallback — key stored in ~/.mlx-stack/config.yaml)
