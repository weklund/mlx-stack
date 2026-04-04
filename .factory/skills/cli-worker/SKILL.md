---
name: cli-worker
description: Implements CLI command changes, module refactoring, and test updates for mlx-stack
---

# CLI Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Use for features that involve:
- Adding, removing, or modifying Click CLI commands
- Updating command registration in `main.py`
- Modifying core modules called by CLI commands
- Writing or rewriting pytest unit tests for CLI commands
- Updating help text, command categories, error messages

## Required Skills

None — all work uses standard file editing and shell commands (pytest, pyright, ruff).

## Work Procedure

### Step 1: Understand the Feature

Read the feature description, preconditions, expectedBehavior, and verificationSteps carefully. Read AGENTS.md for conventions and boundaries. Read `.factory/library/architecture.md` for system structure.

### Step 2: Read Affected Files

Before writing any code, read ALL files that will be affected:
- The CLI command file(s) being changed
- The core module(s) being called
- The test file(s) being updated
- `cli/main.py` if command registration changes
- Any test files that import from affected modules

Understand the existing patterns, mock strategies, and test structure.

### Step 3: Write Tests First (TDD)

Write failing tests BEFORE implementing changes:
1. Create or update the test file with new test cases
2. Run `uv run pytest tests/unit/<test_file> -x -q --tb=short` to confirm tests fail (red)
3. Each test should test ONE specific behavior from the feature's expectedBehavior

Test patterns to follow:
```python
from click.testing import CliRunner
from mlx_stack.cli.main import cli

def test_example(mlx_stack_home):
    runner = CliRunner()
    with patch("mlx_stack.core.module.function") as mock_fn:
        result = runner.invoke(cli, ["command", "--flag", "arg"])
    assert result.exit_code == 0
    assert "expected output" in result.output
    mock_fn.assert_called_once_with(...)
```

### Step 4: Implement Changes

Make the minimum changes needed to make all tests pass:
1. Modify CLI command files
2. Modify core modules if needed
3. Update `cli/main.py` command registration if needed

Follow existing patterns:
- Use `console = Console(stderr=True)` for errors, `out = Console()` for output
- Catch domain exceptions, print user-friendly errors, `raise SystemExit(1)`
- Use absolute imports: `from mlx_stack.core.module import Class`

### Step 5: Run Tests (Green)

1. Run the specific test file: `uv run pytest tests/unit/<test_file> -x -q --tb=short`
2. Run the FULL test suite: `uv run pytest --cov=src/mlx_stack -x -q --tb=short`
3. Fix any failures in other test files caused by your changes

### Step 6: Run Validators

1. Type check: `uv run python -m pyright`
2. Lint: `uv run ruff check src/ tests/`
3. Fix any issues

### Step 7: Verify Manually

For each changed command, run a quick manual check:
```bash
uv run mlx-stack --help                    # verify help output
uv run mlx-stack <command> --help          # verify command help
```

If the feature removes a command, verify it's gone:
```bash
uv run mlx-stack <removed-command>          # should show error
```

### Step 8: Clean Up

- Remove any deleted test files from disk
- Remove any deleted CLI command files from disk
- Ensure no orphaned imports remain
- Run the full test suite one final time

## Example Handoff

```json
{
  "salientSummary": "Ungated pull command to accept HF repo strings. Added slash-based routing (contains '/' = HF repo, no '/' = catalog ID). Wrote 12 new tests in test_cli_pull.py covering HF repo acceptance, error handling, and flag combinations. All 1400+ tests pass, pyright clean, ruff clean.",
  "whatWasImplemented": "Modified cli/pull.py to detect HF repo strings (containing '/') and bypass catalog lookup, routing directly to download_model(). Updated core/pull.py pull_model() to accept hf_repo_override parameter. Updated help text to document both input types. Added 12 new test cases and updated 3 existing tests.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      { "command": "uv run pytest tests/unit/test_cli_pull.py -x -q --tb=short", "exitCode": 0, "observation": "77 passed (12 new + 65 existing)" },
      { "command": "uv run pytest --cov=src/mlx_stack -x -q --tb=short", "exitCode": 0, "observation": "1412 passed, 0 failed" },
      { "command": "uv run python -m pyright", "exitCode": 0, "observation": "0 errors, 0 warnings" },
      { "command": "uv run ruff check src/ tests/", "exitCode": 0, "observation": "All checks passed" },
      { "command": "uv run mlx-stack pull --help", "exitCode": 0, "observation": "Help text mentions HF repo and catalog ID" }
    ],
    "interactiveChecks": [
      { "action": "Ran 'uv run mlx-stack pull --help'", "observed": "Help text now says 'MODEL is a catalog model ID (e.g., qwen3.5-8b) or HuggingFace repo (e.g., mlx-community/Phi-5-Mini-4bit)'" }
    ]
  },
  "tests": {
    "added": [
      {
        "file": "tests/unit/test_cli_pull.py",
        "cases": [
          { "name": "test_pull_hf_repo_downloads_directly", "verifies": "HF repo string bypasses catalog lookup" },
          { "name": "test_pull_hf_repo_with_quant_stores_metadata", "verifies": "--quant flag stores metadata for HF repo" },
          { "name": "test_pull_hf_repo_nonexistent_shows_error", "verifies": "Invalid HF repo shows user-friendly error" }
        ]
      }
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Feature depends on changes that haven't been made yet (e.g., needs a core module that another feature creates)
- Test failures in unrelated areas that can't be resolved without understanding broader context
- Ambiguity in feature requirements that can't be resolved from AGENTS.md or feature description
- A boundary violation would be needed to complete the feature (e.g., need to change scoring.py)
