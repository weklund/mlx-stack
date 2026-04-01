# Contributing to mlx-stack

Thank you for your interest in contributing to mlx-stack! This document provides guidelines and information to help you contribute effectively.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Features](#suggesting-features)
  - [Submitting Pull Requests](#submitting-pull-requests)
- [PR Checklist](#pr-checklist)
- [Commit Message Convention](#commit-message-convention)
- [Development Setup](#development-setup)

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior as described in the Code of Conduct.

---

## Getting Started

mlx-stack is a CLI tool for managing local LLM infrastructure on Apple Silicon Macs. Before contributing, please review:

- **[README.md](README.md)** — Project overview, architecture, and CLI reference
- **[DEVELOPING.md](DEVELOPING.md)** — Full developer guide including project architecture, testing strategy, how to add models and commands, and code quality standards

---

## How to Contribute

### Reporting Bugs

If you find a bug, please [open an issue](https://github.com/weklund/mlx-stack/issues/new) with:

1. **A clear, descriptive title** — Summarize the problem in a few words.
2. **Steps to reproduce** — Exact commands you ran and the order you ran them.
3. **Expected behavior** — What you expected to happen.
4. **Actual behavior** — What actually happened. Include the full terminal output.
5. **Environment details:**
   - macOS version and Apple Silicon chip (e.g., M4 Pro)
   - Python version (`python3 --version`)
   - mlx-stack version (`mlx-stack --version`)
   - Unified memory size
6. **Log files** — If relevant, include contents from `~/.mlx-stack/logs/`.

> **Tip:** Run `mlx-stack profile` and include the output — it captures your hardware details in one step.

### Suggesting Features

Feature suggestions are welcome! [Open an issue](https://github.com/weklund/mlx-stack/issues/new) with:

1. **Problem statement** — What problem does this solve? What's the use case?
2. **Proposed solution** — How should it work from the user's perspective?
3. **Alternatives considered** — Other approaches you thought about and why you prefer your suggestion.
4. **Scope** — Is this a new command, an enhancement to an existing one, a catalog addition, or something else?

### Submitting Pull Requests

1. **Fork the repository** and create a feature branch from `main`:

   ```bash
   git checkout -b feat/my-feature
   ```

2. **Make your changes** following the patterns described in [DEVELOPING.md](DEVELOPING.md):
   - CLI commands go in `src/mlx_stack/cli/` (thin wrappers)
   - Business logic goes in `src/mlx_stack/core/`
   - Tests go in `tests/unit/`

3. **Write tests** for all new functionality:
   - Unit tests for `core/` modules
   - CLI tests via Click `CliRunner`
   - Use `mlx_stack_home` fixture for filesystem isolation — never touch `~/.mlx-stack/`

4. **Run all checks** before submitting (see [PR Checklist](#pr-checklist)).

5. **Open a pull request** against `main` with a clear description of what you changed and why.

---

## PR Checklist

Before submitting your pull request, ensure **all** of the following pass:

- [ ] **Tests pass** — `uv run pytest`
- [ ] **Type checking is clean** — `uv run python -m pyright`
- [ ] **Linting is clean** — `uv run ruff check src/ tests/`
- [ ] **Code is formatted** — `uv run ruff format --check src/ tests/`
- [ ] **Commit messages follow [conventional format](#commit-message-convention)**
- [ ] **New code has test coverage**
- [ ] **No Python tracebacks reach the user** for expected error scenarios

---

## Commit Message Convention

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short description>
```

### Types

| Type | When to use |
|------|-------------|
| `feat:` | New feature or command |
| `fix:` | Bug fix |
| `test:` | Adding or updating tests |
| `docs:` | Documentation changes |
| `chore:` | Tooling, CI, dependency updates |
| `refactor:` | Code restructuring without behavior change |

### Examples

```
feat: implement mlx-stack pull command with HuggingFace download
fix: clamp normalized speed scores to [0,1] range
test: add regression tests for high-bandwidth hardware scoring
docs: add 24/7 ops section to README
chore: add GitHub Actions CI workflow for macOS
```

Keep the subject line under 72 characters. Use the imperative mood ("add", "fix", "implement" — not "added", "fixed").

---

## Development Setup

For complete setup instructions, project architecture, testing patterns, and coding conventions, see **[DEVELOPING.md](DEVELOPING.md)**.

Quick start:

```bash
git clone https://github.com/weklund/mlx-stack.git
cd mlx-stack
uv sync
uv run pytest
```

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.13+ |
| uv | 0.10+ |
| macOS | Apple Silicon (M1+) |
