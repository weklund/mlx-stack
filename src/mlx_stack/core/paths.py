"""Path management for mlx-stack data directory.

Handles auto-creation of the ~/.mlx-stack/ data directory and provides
canonical paths for all data files used by the application.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_data_home() -> Path:
    """Return the mlx-stack data directory path.

    Uses the MLX_STACK_HOME environment variable if set (for testing),
    otherwise defaults to ~/.mlx-stack/.

    Returns:
        Path to the mlx-stack data directory.
    """
    env_home = os.environ.get("MLX_STACK_HOME")
    if env_home:
        return Path(env_home)
    return Path.home() / ".mlx-stack"


def ensure_data_home() -> Path:
    """Ensure the mlx-stack data directory exists and return its path.

    Creates the directory with standard user permissions (0o755) if it
    does not already exist. Also creates standard subdirectories.

    Returns:
        Path to the mlx-stack data directory.
    """
    data_home = get_data_home()
    data_home.mkdir(parents=True, exist_ok=True)
    return data_home


def get_profile_path() -> Path:
    """Return the path to the hardware profile JSON file."""
    return get_data_home() / "profile.json"


def get_config_path() -> Path:
    """Return the path to the user config YAML file."""
    return get_data_home() / "config.yaml"


def get_stacks_dir() -> Path:
    """Return the path to the stacks directory."""
    return get_data_home() / "stacks"


def get_models_dir() -> Path:
    """Return the default models directory path."""
    return get_data_home() / "models"


def get_logs_dir() -> Path:
    """Return the path to the logs directory."""
    return get_data_home() / "logs"


def get_pids_dir() -> Path:
    """Return the path to the PID files directory."""
    return get_data_home() / "pids"


def get_benchmarks_dir() -> Path:
    """Return the path to the benchmarks directory."""
    return get_data_home() / "benchmarks"


def get_lock_path() -> Path:
    """Return the path to the lockfile."""
    return get_data_home() / "lock"
