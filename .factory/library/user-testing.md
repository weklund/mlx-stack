# User Testing

Testing surface, required tools, and validation approach.

## Validation Surface

**Primary surface:** CLI commands via pytest CliRunner (unit-level) and shell invocation (smoke-level).

This is a CLI-only mission with no browser UI, no running services, and no external API dependencies during testing. All HuggingFace API calls and model downloads are mocked in tests.

### Tools

- **pytest** with Click's `CliRunner` — primary test executor
- **Shell invocation** — for smoke tests that verify real subprocess CLI behavior
- **pyright** — type checking gate
- **ruff** — linting gate

### Test Commands

```bash
uv run pytest --cov=src/mlx_stack -x -q --tb=short   # unit tests
uv run python -m pyright                                # type check
uv run ruff check src/ tests/                           # lint
```

## Validation Concurrency

**Max concurrent validators: 5**

Rationale: CLI tests are lightweight (no browser, no services). Each pytest invocation uses ~100MB RAM. Machine has 128GB RAM and 18 CPU cores. Even 5 concurrent test runs would use <1GB total. No infrastructure contention.

## Testing Patterns

- CLI commands tested via `CliRunner().invoke(cli, ["command", "--flag", "arg"])`
- Exit codes checked: 0 for success, non-zero for errors
- Output checked via `result.output` string matching
- Side effects verified via mock assertions (`mock_download.assert_called_once()`, etc.)
- File system effects checked via `tmp_path` fixtures
- Test factories in `tests/factories.py` for creating test data consistently
