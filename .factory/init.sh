#!/usr/bin/env bash
set -euo pipefail

# Verify Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required="3.13"
if [ "$(printf '%s\n' "$required" "$python_version" | sort -V | head -n1)" != "$required" ]; then
    echo "ERROR: Python >= 3.13 required (found $python_version)"
    exit 1
fi

# Install dependencies if pyproject.toml exists
if [ -f pyproject.toml ]; then
    uv sync
fi
