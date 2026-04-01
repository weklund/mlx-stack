# Developing mlx-stack

A comprehensive guide for contributors working on the mlx-stack codebase.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Getting Started](#getting-started)
3. [Project Architecture](#project-architecture)
4. [Key Concepts](#key-concepts)
5. [Adding a New Model](#adding-a-new-model)
6. [Adding a New CLI Command](#adding-a-new-cli-command)
7. [Testing](#testing)
8. [Code Quality](#code-quality)
9. [Configuration System](#configuration-system)
10. [Process Management](#process-management)
11. [Commit Conventions](#commit-conventions)
12. [PR Process](#pr-process)

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.13+ | Apple Silicon native build recommended |
| **uv** | 0.10+ | Fast Python package manager — [install guide](https://docs.astral.sh/uv/getting-started/installation/) |
| **macOS** | Apple Silicon (M1+) | Intel Macs and Linux are not supported |
| **Git** | 2.30+ | For version control |

mlx-stack is designed exclusively for Apple Silicon Macs. Hardware detection, model serving (vllm-mlx), and performance benchmarks all depend on the Metal GPU and unified memory architecture.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/weklund/mlx-stack.git
cd mlx-stack
```

### 2. Install dependencies

```bash
uv sync
```

This creates a virtual environment in `.venv/` and installs all runtime and dev dependencies (pytest, pyright, ruff, etc.) from `pyproject.toml`.

### 3. Verify the installation

```bash
uv run mlx-stack --help
```

You should see the Rich-formatted help output listing all commands grouped by category (Setup & Configuration, Model Management, Stack Lifecycle, Diagnostics).

### 4. Run the tests

```bash
uv run pytest
```

All unit tests should pass. Integration tests are excluded by default (see [Testing](#testing) for details).

---

## Project Architecture

```
src/mlx_stack/
├── __init__.py              # Package root — exports __version__
├── py.typed                 # PEP 561 marker for type checking
├── cli/                     # CLI layer — thin Click wrappers
│   ├── __init__.py          # Exports the cli() entry point
│   ├── main.py              # Click group, --help/--version, typo suggestions
│   ├── profile.py           # mlx-stack profile
│   ├── config.py            # mlx-stack config (set/get/list/reset)
│   ├── recommend.py         # mlx-stack recommend
│   ├── init.py              # mlx-stack init
│   ├── models.py            # mlx-stack models
│   ├── pull.py              # mlx-stack pull
│   ├── up.py                # mlx-stack up
│   ├── down.py              # mlx-stack down
│   ├── status.py            # mlx-stack status
│   ├── bench.py             # mlx-stack bench
│   ├── logs.py              # mlx-stack logs
│   ├── watch.py             # mlx-stack watch
│   └── install.py           # mlx-stack install / uninstall
├── core/                    # Business logic — importable without CLI
│   ├── __init__.py
│   ├── paths.py             # Path management (MLX_STACK_HOME, data dirs)
│   ├── hardware.py          # Apple Silicon detection, bandwidth lookup
│   ├── catalog.py           # YAML catalog loading, validation, querying
│   ├── config.py            # ConfigKeyDef system, get/set/validate/persist
│   ├── deps.py              # Dependency management (vllm-mlx, litellm)
│   ├── scoring.py           # Recommendation engine, tier assignment
│   ├── stack_init.py        # Stack definition + LiteLLM config generation
│   ├── litellm_gen.py       # LiteLLM YAML config builder
│   ├── models.py            # Local model inventory management
│   ├── process.py           # PID files, lockfile, health checks, start/stop
│   ├── stack_up.py          # Orchestrates starting all services
│   ├── stack_down.py        # Orchestrates stopping all services
│   ├── stack_status.py      # 5-state health reporting
│   ├── pull.py              # Model download with HuggingFace Hub
│   ├── benchmark.py         # Benchmarking engine (prompt/gen TPS)
│   ├── log_rotation.py      # Copytruncate log rotation
│   ├── log_viewer.py        # Log viewing, following, archive reading
│   ├── watchdog.py          # Health monitor with auto-restart
│   └── launchd.py           # macOS LaunchAgent integration
├── data/                    # Static data shipped with the package
│   ├── __init__.py
│   └── catalog/             # 15 model YAML files
│       ├── qwen3.5-0.8b.yaml
│       ├── qwen3.5-3b.yaml
│       ├── qwen3.5-8b.yaml
│       ├── qwen3.5-14b.yaml
│       ├── qwen3.5-32b.yaml
│       ├── qwen3.5-72b.yaml
│       ├── gemma3-4b.yaml
│       ├── gemma3-12b.yaml
│       ├── gemma3-27b.yaml
│       ├── deepseek-r1-8b.yaml
│       ├── deepseek-r1-32b.yaml
│       ├── nemotron-8b.yaml
│       ├── nemotron-49b.yaml
│       ├── qwen3-8b.yaml
│       └── llama3.3-8b.yaml
└── utils/                   # Shared utility modules
    └── __init__.py

tests/
├── conftest.py              # Shared fixtures (mlx_stack_home, etc.)
├── unit/                    # Unit tests — mocked external calls
│   ├── test_hardware.py
│   ├── test_catalog.py
│   ├── test_config.py
│   ├── test_deps.py
│   ├── test_scoring.py
│   ├── test_process.py
│   ├── test_cli_profile.py
│   ├── test_cli_config.py
│   ├── test_cli_recommend.py
│   ├── test_cli_init.py
│   ├── test_cli_models.py
│   ├── test_cli_up.py
│   ├── test_cli_down.py
│   ├── test_cli_status.py
│   ├── test_cli_pull.py
│   ├── test_cli_bench.py
│   ├── test_cli_logs.py
│   ├── test_cli_watch.py
│   ├── test_cli_install.py
│   └── ...                  # Cross-area, robustness, lifecycle tests
├── integration/             # Real-system integration tests
│   ├── test_inference_e2e.py
│   └── test_launchd_e2e.py
└── fixtures/                # Shared test data
```

### Design Principles

- **CLI modules are thin wrappers.** Each file in `cli/` defines a Click command that parses arguments, calls into `core/`, and formats output with Rich. No business logic lives in `cli/`.

- **`core/` has all business logic.** Every module in `core/` is importable and testable independently of the CLI layer. This makes unit testing straightforward — you test `core/` functions directly.

- **`data/catalog/` holds curated model YAMLs.** These are loaded at runtime via `importlib.resources` so they work correctly whether installed as a package or run from source.

- **All state lives in `~/.mlx-stack/`.** The data directory (overridable via `MLX_STACK_HOME` env var) contains `profile.json`, `config.yaml`, `stacks/`, `models/`, `pids/`, `logs/`, and `benchmarks/`.

---

## Key Concepts

### Tiers

mlx-stack assigns models to three **tiers**, each optimized for a different workload:

| Tier | Port | Purpose |
|------|------|---------|
| `standard` | 8000 | Highest-quality model within memory budget |
| `fast` | 8001 | Fastest model for latency-sensitive tasks |
| `longctx` | 8002 | Architecturally diverse model (e.g., Mamba2 hybrid) |

Tier assignment is performed by the scoring engine in `core/scoring.py`. The `standard` tier gets the model with the highest composite score weighted toward quality, `fast` gets the highest speed, and `longctx` prefers architecturally different models when available.

### Catalog Entries

Each model in `data/catalog/` is a YAML file describing:

- **Identity:** `id`, `name`, `family`, `params_b`, `architecture`
- **Sources:** Per-quantization HuggingFace repos with `disk_size_gb`
- **Capabilities:** `tool_calling`, `thinking`, `vision`, parser names
- **Quality scores:** `overall`, `coding`, `reasoning`, `instruction_following` (0–100)
- **Benchmarks:** Per-hardware `prompt_tps`, `gen_tps`, `memory_gb`
- **Tags:** Searchable labels like `balanced`, `agent-ready`, `thinking`

### Stack Definitions

A stack definition (`~/.mlx-stack/stacks/default.yaml`) is the output of `mlx-stack init`. It specifies:

- `schema_version: 1` — for forward compatibility
- `hardware_profile` — the profile ID (e.g., `m4-pro-48`)
- `intent` — the optimization strategy used (`balanced` or `agent-fleet`)
- `tiers` — list of tier entries, each with `name`, `model`, `quant`, `source`, `port`, and `vllm_flags`

A companion `~/.mlx-stack/litellm.yaml` is generated alongside it for the LiteLLM proxy.

### Process Management

mlx-stack manages vllm-mlx and LiteLLM as subprocesses:

- **PID files** in `~/.mlx-stack/pids/` track each running service (e.g., `fast.pid`, `litellm.pid`). Each file contains a single integer PID.
- **Lockfile** at `~/.mlx-stack/lock` uses `fcntl.flock` to prevent concurrent `up`/`down` operations.
- **Health checks** poll each service's HTTP endpoint with exponential backoff (0.5s → 10s, 120s total timeout).

### The 5-State Model

Every service reported by `mlx-stack status` is in one of five states:

| State | Condition |
|-------|-----------|
| **healthy** | PID alive, HTTP 200 within 2 seconds |
| **degraded** | PID alive, HTTP 200 but response > 2 seconds |
| **down** | PID alive, no HTTP response within 5 seconds |
| **crashed** | PID file exists, but the process is dead |
| **stopped** | No PID file exists |

---

## Adding a New Model

To add a new model to the curated catalog:

### Step 1: Create the YAML file

Create a new file in `src/mlx_stack/data/catalog/` following the naming convention `<family>-<size>.yaml`:

```yaml
id: my-model-8b
name: My Model 8B
family: My Model
params_b: 8.0
architecture: transformer
min_mlx_lm_version: "0.22.0"
sources:
  int4:
    hf_repo: mlx-community/My-Model-8B-4bit
    disk_size_gb: 4.5
  int8:
    hf_repo: mlx-community/My-Model-8B-8bit
    disk_size_gb: 8.5
  bf16:
    hf_repo: MyOrg/My-Model-8B
    disk_size_gb: 16.0
    convert_from: true
capabilities:
  tool_calling: true
  tool_call_parser: hermes
  thinking: false
  reasoning_parser: ""
  vision: false
quality:
  overall: 65
  coding: 60
  reasoning: 58
  instruction_following: 70
benchmarks:
  m4-pro-48:
    prompt_tps: 90.0
    gen_tps: 50.0
    memory_gb: 5.0
  m4-max-128:
    prompt_tps: 130.0
    gen_tps: 70.0
    memory_gb: 5.0
tags:
  - balanced
  - agent-ready
```

### Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | ✓ | Unique identifier (used in CLI commands) |
| `name` | string | ✓ | Human-readable display name |
| `family` | string | ✓ | Model family for grouping |
| `params_b` | float | ✓ | Parameter count in billions |
| `architecture` | string | ✓ | Architecture type (`transformer`, `mamba2-hybrid`, etc.) |
| `min_mlx_lm_version` | string | ✓ | Minimum mlx_lm version required |
| `sources` | dict | ✓ | Per-quant sources (`int4`, `int8`, `bf16`) |
| `sources.<quant>.hf_repo` | string | ✓ | HuggingFace repository path |
| `sources.<quant>.disk_size_gb` | float | ✓ | On-disk size in GB |
| `sources.<quant>.convert_from` | bool | — | If `true`, requires local conversion via mlx_lm |
| `capabilities.tool_calling` | bool | ✓ | Supports function/tool calling |
| `capabilities.tool_call_parser` | string | ✓ | Parser name (e.g., `hermes`) or empty string |
| `capabilities.thinking` | bool | ✓ | Supports thinking/reasoning mode |
| `capabilities.reasoning_parser` | string | ✓ | Parser name (e.g., `qwen3`) or empty string |
| `capabilities.vision` | bool | ✓ | Supports vision/image input |
| `quality.overall` | int | ✓ | Overall quality score (0–100) |
| `quality.coding` | int | ✓ | Coding quality score (0–100) |
| `quality.reasoning` | int | ✓ | Reasoning quality score (0–100) |
| `quality.instruction_following` | int | ✓ | Instruction following score (0–100) |
| `benchmarks` | dict | ✓ | Per-hardware-profile benchmark data |
| `benchmarks.<profile>.prompt_tps` | float | ✓ | Prompt tokens per second |
| `benchmarks.<profile>.gen_tps` | float | ✓ | Generation tokens per second |
| `benchmarks.<profile>.memory_gb` | float | ✓ | Runtime memory usage in GB |
| `tags` | list[str] | ✓ | Searchable labels |

### Step 2: Validate the catalog loads

```bash
uv run python -c "from mlx_stack.core.catalog import load_catalog; c = load_catalog(); print(f'{len(c)} models loaded')"
```

### Step 3: Add tests

Add test cases in `tests/unit/test_catalog.py` to verify your new entry loads correctly and is queryable by family, tags, and capabilities.

### Step 4: Run the full suite

```bash
uv run pytest
```

---

## Adding a New CLI Command

To add a new command (e.g., `mlx-stack export`):

### Step 1: Create the core module

Create `src/mlx_stack/core/export.py` with the business logic:

```python
"""Export module for mlx-stack."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExportResult:
    """Result of an export operation."""

    path: str
    format: str


def export_stack(stack_path: str, output_format: str = "json") -> ExportResult:
    """Export a stack definition to the given format.

    Args:
        stack_path: Path to the stack YAML file.
        output_format: Output format ('json' or 'toml').

    Returns:
        ExportResult with the output path and format.
    """
    output_path = stack_path.rsplit(".", 1)[0] + f".{output_format}"
    # Read the stack YAML and convert to the requested format
    return ExportResult(path=output_path, format=output_format)
```

### Step 2: Create the CLI command

Create `src/mlx_stack/cli/export.py`:

```python
"""CLI command for mlx-stack export."""

from __future__ import annotations

import click
from rich.console import Console

from mlx_stack.core.export import export_stack

console = Console(stderr=True)


@click.command()
@click.argument("output_path", required=False)
@click.option("--format", "output_format", default="json", help="Output format.")
def export(output_path: str | None, output_format: str) -> None:
    """Export the active stack definition."""
    try:
        result = export_stack(output_path or "stack.json", output_format)
        console.print(f"[green]Exported to {result.path}[/green]")
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise SystemExit(1) from None
```

### Step 3: Register in `cli/main.py`

Add the import and registration in `src/mlx_stack/cli/main.py`:

```python
from mlx_stack.cli.export import export as export_command

# In the command registration section:
cli.add_command(export_command, "export")
```

Also add the command to the `command_categories` dict inside `RichGroup.format_help()` so it appears in the right help category.

### Step 4: Add tests

Create both core and CLI tests:

- `tests/unit/test_export.py` — test the core logic directly
- `tests/unit/test_cli_export.py` — test the Click command via `CliRunner`

Example CLI test:

```python
"""Tests for the export CLI command."""

from click.testing import CliRunner

from mlx_stack.cli.main import cli


def test_export_help():
    """Export command has help text."""
    runner = CliRunner()
    result = runner.invoke(cli, ["export", "--help"])
    assert result.exit_code == 0
    assert "Export" in result.output
```

### Step 5: Run tests and checks

```bash
uv run pytest
uv run python -m pyright src/mlx_stack/cli/export.py src/mlx_stack/core/export.py
uv run ruff check src/ tests/
```

---

## Testing

### Strategy

mlx-stack uses a layered testing strategy:

- **Unit tests** (primary layer, 80%+ coverage on `core/`) — All external calls (`sysctl`, `system_profiler`, `subprocess.Popen`, network requests) are mocked. Tests run fast and are deterministic.
- **CLI tests** — Use Click's `CliRunner` to invoke commands in-process, capturing output and exit codes without spawning subprocesses.
- **Integration tests** — Real system interaction (hardware detection, model downloads, launchctl). Excluded by default, run explicitly with `-m integration`.

### Test isolation

Every test that touches the filesystem uses **`tmp_path`** fixtures to avoid modifying the real `~/.mlx-stack/` directory:

```python
# tests/conftest.py provides these fixtures:

@pytest.fixture()
def mlx_stack_home(tmp_path, monkeypatch):
    """Isolated MLX_STACK_HOME that already exists."""
    home = tmp_path / ".mlx-stack"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MLX_STACK_HOME", str(home))
    return home

@pytest.fixture()
def clean_mlx_stack_home(tmp_path, monkeypatch):
    """MLX_STACK_HOME that does NOT exist yet (for auto-creation tests)."""
    home = tmp_path / ".mlx-stack"
    monkeypatch.setenv("MLX_STACK_HOME", str(home))
    return home
```

Use `mlx_stack_home` for most tests. Use `clean_mlx_stack_home` when testing directory auto-creation.

### Running tests

```bash
# Run all unit tests (default — integration tests are excluded)
uv run pytest

# Run a specific test module
uv run pytest tests/unit/test_catalog.py -v

# Run tests matching a pattern
uv run pytest -k "test_scoring" -v

# Run with coverage report
uv run pytest --cov=src/mlx_stack --cov-report=term-missing

# Run integration tests only (requires Apple Silicon hardware)
uv run pytest -m integration -v

# Run everything including integration tests
uv run pytest -m "" -v
```

### Mocking patterns

External system calls are always mocked in unit tests. Common patterns:

```python
# Mocking subprocess calls (e.g., for process management)
def test_start_service(monkeypatch, mlx_stack_home):
    mock_popen = MagicMock()
    mock_popen.pid = 12345
    monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: mock_popen)

# Mocking hardware detection (sysctl, system_profiler)
def test_detect_hardware(monkeypatch):
    monkeypatch.setattr(
        "mlx_stack.core.hardware._run_sysctl",
        lambda key: "Apple M4 Pro"
    )

# CLI tests with CliRunner
def test_profile_command(mlx_stack_home):
    runner = CliRunner()
    result = runner.invoke(cli, ["profile"])
    assert result.exit_code == 0
```

---

## Code Quality

### Linting with Ruff

[Ruff](https://docs.astral.sh/ruff/) handles both linting and formatting:

```bash
# Check for lint issues
uv run ruff check src/ tests/

# Auto-fix lint issues
uv run ruff check --fix src/ tests/

# Format code
uv run ruff format src/ tests/

# Check formatting without modifying files
uv run ruff format --check src/ tests/
```

Configuration in `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py313"
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "W"]   # Errors, pyflakes, isort, warnings
```

### Type checking with Pyright

[Pyright](https://github.com/microsoft/pyright) is used for static type analysis:

```bash
uv run python -m pyright
```

Configuration in `pyproject.toml`:

```toml
[tool.pyright]
pythonVersion = "3.13"
pythonPlatform = "Darwin"
venvPath = "."
venv = ".venv"
typeCheckingMode = "basic"
```

The project uses pyright in **basic mode**, which enforces type annotations on function signatures and catches common type errors without requiring full strict-mode annotations everywhere. Contributors should aim for **zero pyright errors** — the CI pipeline enforces this. All public functions in `core/` should have complete type annotations.

### Pre-commit checklist

Before pushing, run all three checks:

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run python -m pyright
uv run pytest
```

---

## Configuration System

The configuration system is built around the `ConfigKeyDef` dataclass in `core/config.py`.

### How ConfigKeyDef works

Each config key is defined as a frozen dataclass with validation metadata:

```python
@dataclass(frozen=True)
class ConfigKeyDef:
    name: str           # Key name (e.g., "default-quant")
    description: str    # Human-readable description
    default: Any        # Default value
    value_type: str     # "string", "int", "bool", or "path"
    validator: str | None = None  # Named validator function
```

All keys are registered in the `CONFIG_KEYS` dict. The `validate_key()` function checks that a key exists, `parse_value()` handles type coercion and runs the appropriate validator, and `mask_value()` masks sensitive values (like `openrouter-key`) in display output.

### Current config keys

| Key | Type | Default | Validator |
|-----|------|---------|-----------|
| `openrouter-key` | string | `""` | — |
| `default-quant` | string | `int4` | Must be `int4`, `int8`, or `bf16` |
| `memory-budget-pct` | int | `40` | Must be 1–100 |
| `litellm-port` | int | `4000` | Must be 1–65535 |
| `model-dir` | path | `~/.mlx-stack/models` | — |
| `auto-health-check` | bool | `true` | — |
| `log-max-size-mb` | int | `50` | Must be ≥ 1 |
| `log-max-files` | int | `5` | Must be ≥ 1 |

### Adding a new config key

1. **Define the key** in the `CONFIG_KEYS` dict in `src/mlx_stack/core/config.py`:

```python
CONFIG_KEYS: dict[str, ConfigKeyDef] = {
    # ... existing keys ...
    "my-new-key": ConfigKeyDef(
        name="my-new-key",
        description="Description of what this controls",
        default=42,
        value_type="int",
        validator="positive_int",  # or None, "quant", "memory_pct", "port"
    ),
}
```

2. **Add a custom validator** if needed by creating a `_validate_<name>()` function and adding a branch in `parse_value()`.

3. **Add tests** in `tests/unit/test_config.py` covering:
   - Default value is returned when unset
   - Valid values are accepted and round-tripped
   - Invalid values raise `ConfigValidationError`
   - The key appears in `config list` output

4. **Add CLI tests** in `tests/unit/test_cli_config.py` verifying the key works through `config set`/`get`/`list`.

---

## Process Management

The process lifecycle is managed by `core/process.py`, orchestrated by `core/stack_up.py` and `core/stack_down.py`.

### Starting a service

`start_service()` in `core/process.py`:

1. Ensures the `pids/` and `logs/` directories exist
2. Opens a log file in **append mode** (`"a"`) at `~/.mlx-stack/logs/<name>.log`
3. Launches the process via `subprocess.Popen` with stdout/stderr redirected to the log file
4. Writes the PID to `~/.mlx-stack/pids/<name>.pid`
5. If the PID file write fails, the process is killed to prevent leaking unmanaged processes

### Waiting for healthy

`wait_for_healthy()` polls the service's HTTP endpoint:

- Initial delay: 0.5 seconds
- Backoff factor: 2× per retry (0.5s → 1s → 2s → 4s → 8s → 10s cap)
- Maximum per-retry delay: 10 seconds
- Total timeout: 120 seconds
- Checks `GET /v1/models` and expects HTTP 200

### Stopping a service

`stop_service()` implements graceful shutdown:

1. Sends **SIGTERM** to the process
2. Waits up to 10 seconds for the process to exit
3. If still alive, sends **SIGKILL** (forced termination)
4. Verifies the process is actually dead after SIGKILL
5. Removes the PID file only after confirmed termination
6. Returns a `ShutdownResult` indicating whether shutdown was graceful or forced

### PID files

Each running service has a PID file at `~/.mlx-stack/pids/<service>.pid`:

- Contains exactly one integer (the process ID)
- Created by `start_service()`, removed by `stop_service()`
- `status` reads PID files to determine service state without modifying them
- Stale PIDs (file exists but process is dead) are detected and cleaned up by `up` and `down`

### Lockfile

The lockfile at `~/.mlx-stack/lock` prevents concurrent `up`/`down` operations:

```python
from mlx_stack.core.process import acquire_lock

with acquire_lock():
    # Only one process can hold the lock at a time
    start_services()
```

Uses `fcntl.flock` with `LOCK_EX | LOCK_NB` for non-blocking exclusive locking. The lock is automatically released when the file descriptor is closed (including on crash).

The `status` command and the watchdog's polling loop do **not** acquire the lock — they are read-only. The watchdog acquires the lock only during restart operations.

---

## Commit Conventions

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

## PR Process

### 1. Fork and branch

```bash
# Fork the repo on GitHub, then:
git clone https://github.com/<your-username>/mlx-stack.git
cd mlx-stack
git remote add upstream https://github.com/weklund/mlx-stack.git
git checkout -b feat/my-feature
```

### 2. Make your changes

Follow the patterns in this guide — thin CLI wrappers, business logic in `core/`, tests for both layers.

### 3. Write tests

- Unit tests for all new `core/` functions
- CLI tests via `CliRunner` for new commands
- Use `mlx_stack_home` fixture for filesystem isolation

### 4. Run all checks

```bash
# Tests
uv run pytest

# Type checking
uv run python -m pyright

# Linting
uv run ruff check src/ tests/

# Formatting
uv run ruff format --check src/ tests/
```

All four must pass before submitting.

### 5. Commit with conventional format

```bash
git add .
git commit -m "feat: add export command for stack definitions"
```

### 6. Push and open a PR

```bash
git push origin feat/my-feature
```

Open a pull request against the `main` branch on GitHub.

### PR checklist

- [ ] All tests pass (`uv run pytest`)
- [ ] Type checking is clean (`uv run python -m pyright`)
- [ ] Linting is clean (`uv run ruff check src/ tests/`)
- [ ] Code is formatted (`uv run ruff format --check src/ tests/`)
- [ ] Commit messages follow conventional format
- [ ] New code has test coverage
- [ ] No Python tracebacks reach the user for expected error scenarios
