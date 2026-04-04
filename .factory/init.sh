#!/bin/bash
set -e

cd /Users/weae1504/Projects/mlx-stack

# Install dev dependencies (idempotent)
uv sync --dev
