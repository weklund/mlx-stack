# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in mlx-stack, please report it responsibly via **private disclosure**. Do **not** open a public GitHub issue for security vulnerabilities.

### How to Report

Email **<mlx-stack-security@proton.me>** with:

1. **Description** — A clear description of the vulnerability.
2. **Reproduction steps** — How to trigger the issue.
3. **Impact assessment** — What an attacker could achieve.
4. **Affected versions** — Which version(s) of mlx-stack are affected.
5. **Suggested fix** — If you have one (optional).

### What to Expect

| Stage | Timeline |
|-------|----------|
| **Acknowledgment** | Within 48 hours of your report |
| **Initial assessment** | Within 1 week |
| **Fix or mitigation** | Within 30 days for confirmed vulnerabilities |
| **Public disclosure** | After a fix is released, coordinated with reporter |

We will keep you informed of progress throughout the process.

---

## Scope

mlx-stack is a **local CLI tool** that manages LLM inference processes on a single machine. Its security surface is different from a networked service. The following areas are considered in scope for security reports:

### In Scope

- **API key exposure** — Accidental logging, display, or leaking of the OpenRouter API key (stored in `~/.mlx-stack/config.yaml`) through CLI output, log files, process arguments visible in `ps`, or error messages.
- **Process management vulnerabilities** — Issues where mlx-stack could be tricked into starting, stopping, or signaling processes it should not control (e.g., PID file manipulation, symlink attacks on PID/log directories).
- **File permission issues** — Insecure permissions on config files, PID files, or log files that could expose sensitive data or allow unauthorized modification.
- **Command injection** — Any path where user-supplied input (model names, config values, file paths) could lead to arbitrary command execution.
- **Denial of service** — Issues that could cause resource exhaustion, orphaned processes, or lockfile deadlocks that prevent normal operation.
- **Dependency vulnerabilities** — Security issues in pinned dependencies (vllm-mlx, litellm) that affect mlx-stack users.

### Out of Scope

- **Network exposure by design** — vllm-mlx and LiteLLM bind to `127.0.0.1` (localhost only). Accessing these services requires local machine access, which is by design.
- **Physical access attacks** — If an attacker has local access to the machine, they can already read `~/.mlx-stack/` directly.
- **Model content safety** — The safety or alignment of the LLM models themselves is outside mlx-stack's scope.
- **Upstream vulnerabilities** — Issues in Python, macOS, or Apple Silicon hardware should be reported to the respective vendors.

---

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x | ✅ Current release |
| < 0.1 | ❌ Not supported |

Security fixes are applied to the latest release only.

---

## Security Best Practices for Users

- **Protect your config file** — `~/.mlx-stack/config.yaml` may contain your OpenRouter API key. Ensure it has appropriate file permissions (`chmod 600`).
- **Review log files** — Log files in `~/.mlx-stack/logs/` may contain request data. Rotate and clean logs regularly using `mlx-stack logs --rotate`.
- **Keep dependencies updated** — Run `uv tool upgrade vllm-mlx litellm` periodically to get security patches for managed tools.
