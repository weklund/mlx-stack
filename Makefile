.PHONY: install lint typecheck test check

## Install dev dependencies
install:
	uv sync --dev

## Lint source and tests
lint:
	uv run ruff check src/ tests/

## Run type checker across the full project
typecheck:
	uv run python -m pyright

## Run tests with coverage
test:
	uv run pytest --cov=src/mlx_stack -x -q --tb=short

## Run all checks (same as CI)
check: lint typecheck test
