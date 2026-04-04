.PHONY: install lint typecheck test check test-catalog test-smoke test-integration test-harness

## Install dev dependencies
install:
	uv sync --dev

## Lint source and tests (ruff + pyright)
lint:
	uv run ruff check src/ tests/
	uv run python -m pyright

## Run type checker only (alias kept for CI compatibility)
typecheck:
	uv run python -m pyright

## Run tests with coverage
test:
	uv run pytest --cov=src/mlx_stack -x -q --tb=short

## Run all checks (same as CI)
check: lint typecheck test

## Run catalog validation (requires network, no models)
test-catalog:
	uv run pytest -m catalog_validation -v --tb=short

## Run per-model smoke tests (requires macOS + vllm-mlx)
test-smoke:
	uv run pytest -m smoke -v --tb=long

## Run full stack integration tests (requires macOS + vllm-mlx + litellm)
test-integration:
	uv run pytest -m integration -v --tb=long

## Run harness compatibility tests (requires above + openai)
test-harness:
	uv run pytest -m harness -v --tb=long
