"""Configuration management module for mlx-stack.

Handles persistent configuration in ~/.mlx-stack/config.yaml with
type validation, default values, and masking of sensitive data.

Supports 8 configuration keys:
- openrouter-key: OpenRouter API key (masked in display)
- default-quant: Default quantization level (int4, int8, bf16)
- memory-budget-pct: Memory budget percentage (1-100)
- litellm-port: LiteLLM proxy port (1-65535)
- model-dir: Model storage directory path
- auto-health-check: Auto health check on startup (true/false)
- log-max-size-mb: Maximum log file size in MB before rotation (1+)
- log-max-files: Maximum number of rotated log archives to keep (1+)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml

from mlx_stack.core.paths import ensure_data_home, get_config_path, get_models_dir

# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class ConfigError(Exception):
    """Raised when configuration operations fail."""


class ConfigValidationError(ConfigError):
    """Raised when a configuration value fails validation."""


class ConfigCorruptError(ConfigError):
    """Raised when the config file is corrupt or unreadable."""


# --------------------------------------------------------------------------- #
# Config key definitions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ConfigKeyDef:
    """Definition of a configuration key with its type and validation."""

    name: str
    description: str
    default: Any
    value_type: str  # "string", "int", "bool", "path"
    validator: str | None = None  # name of validation function


# The 6 supported configuration keys with their defaults
CONFIG_KEYS: dict[str, ConfigKeyDef] = {
    "openrouter-key": ConfigKeyDef(
        name="openrouter-key",
        description="OpenRouter API key for cloud fallback",
        default="",
        value_type="string",
    ),
    "default-quant": ConfigKeyDef(
        name="default-quant",
        description="Default quantization level",
        default="int4",
        value_type="string",
        validator="quant",
    ),
    "memory-budget-pct": ConfigKeyDef(
        name="memory-budget-pct",
        description="Memory budget percentage (1-100)",
        default=40,
        value_type="int",
        validator="memory_pct",
    ),
    "litellm-port": ConfigKeyDef(
        name="litellm-port",
        description="LiteLLM proxy port",
        default=4000,
        value_type="int",
        validator="port",
    ),
    "model-dir": ConfigKeyDef(
        name="model-dir",
        description="Model storage directory",
        default="~/.mlx-stack/models",
        value_type="path",
    ),
    "auto-health-check": ConfigKeyDef(
        name="auto-health-check",
        description="Auto health check on startup",
        default=True,
        value_type="bool",
    ),
    "log-max-size-mb": ConfigKeyDef(
        name="log-max-size-mb",
        description="Maximum log file size in MB before rotation",
        default=50,
        value_type="int",
        validator="positive_int",
    ),
    "log-max-files": ConfigKeyDef(
        name="log-max-files",
        description="Maximum number of rotated log archives to keep",
        default=5,
        value_type="int",
        validator="positive_int",
    ),
}

VALID_KEYS: list[str] = sorted(CONFIG_KEYS.keys())

# Valid quantization values
_VALID_QUANTS: set[str] = {"int4", "int8", "bf16"}


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_key(key: str) -> ConfigKeyDef:
    """Validate that a key is a known configuration key.

    Args:
        key: The configuration key name.

    Returns:
        The ConfigKeyDef for the key.

    Raises:
        ConfigError: If the key is not recognized.
    """
    if key not in CONFIG_KEYS:
        valid_list = ", ".join(VALID_KEYS)
        msg = f"Unknown config key '{key}'. Valid keys: {valid_list}"
        raise ConfigError(msg)
    return CONFIG_KEYS[key]


def _validate_quant(value: str) -> str:
    """Validate a quantization value.

    Raises:
        ConfigValidationError: If the value is not a valid quantization.
    """
    if value not in _VALID_QUANTS:
        valid = ", ".join(sorted(_VALID_QUANTS))
        msg = f"Invalid quantization '{value}'. Valid values: {valid}"
        raise ConfigValidationError(msg)
    return value


def _validate_memory_pct(value: int) -> int:
    """Validate a memory budget percentage.

    Raises:
        ConfigValidationError: If the value is not in 1-100 range.
    """
    if not (1 <= value <= 100):
        msg = f"Invalid memory budget '{value}'. Must be between 1 and 100."
        raise ConfigValidationError(msg)
    return value


def _validate_port(value: int) -> int:
    """Validate a TCP port number.

    Raises:
        ConfigValidationError: If the value is not in 1-65535 range.
    """
    if not (1 <= value <= 65535):
        msg = f"Invalid port '{value}'. Must be between 1 and 65535."
        raise ConfigValidationError(msg)
    return value


def _validate_positive_int(key_name: str, value: int) -> int:
    """Validate that a value is a positive integer (>= 1).

    Raises:
        ConfigValidationError: If the value is less than 1.
    """
    if value < 1:
        msg = f"Invalid value '{value}' for '{key_name}'. Must be at least 1."
        raise ConfigValidationError(msg)
    return value


def parse_value(key_def: ConfigKeyDef, raw_value: str) -> Any:
    """Parse a raw string value into the appropriate type for a config key.

    Args:
        key_def: The configuration key definition.
        raw_value: The raw string value from CLI input.

    Returns:
        The parsed and validated value.

    Raises:
        ConfigValidationError: If the value fails type conversion or validation.
    """
    value: Any

    if key_def.value_type == "int":
        try:
            value = int(raw_value)
        except ValueError:
            msg = f"Invalid value '{raw_value}' for '{key_def.name}'. Expected an integer."
            raise ConfigValidationError(msg) from None

    elif key_def.value_type == "bool":
        lower = raw_value.lower()
        if lower in ("true", "1", "yes", "on"):
            value = True
        elif lower in ("false", "0", "no", "off"):
            value = False
        else:
            msg = (
                f"Invalid value '{raw_value}' for '{key_def.name}'. "
                f"Expected true/false, yes/no, 1/0, or on/off."
            )
            raise ConfigValidationError(msg)

    elif key_def.value_type == "path":
        value = raw_value

    else:
        # string type
        value = raw_value

    # Run specific validators
    if key_def.validator == "quant":
        _validate_quant(str(value))
    elif key_def.validator == "memory_pct":
        _validate_memory_pct(int(value))
    elif key_def.validator == "port":
        _validate_port(int(value))
    elif key_def.validator == "positive_int":
        _validate_positive_int(key_def.name, int(value))

    return value


# --------------------------------------------------------------------------- #
# Masking
# --------------------------------------------------------------------------- #


def mask_value(key: str, value: Any) -> str:
    """Mask sensitive values for display.

    The openrouter-key is masked to show only the first 3 and last 3
    characters. All other values are returned as-is.

    Args:
        key: The configuration key name.
        value: The value to potentially mask.

    Returns:
        The masked or unmasked string representation.
    """
    str_value = str(value)
    if key == "openrouter-key" and str_value and str_value != "":
        if len(str_value) <= 6:
            return "****"
        return f"{str_value[:3]}****{str_value[-3:]}"
    return str_value


# --------------------------------------------------------------------------- #
# File I/O
# --------------------------------------------------------------------------- #


def _load_raw_config() -> dict[str, Any]:
    """Load the raw config dictionary from disk.

    Returns:
        A dictionary of config key-value pairs. Empty dict if file
        is missing or empty.

    Raises:
        ConfigCorruptError: If the config file contains invalid YAML.
    """
    config_path = get_config_path()

    if not config_path.exists():
        return {}

    try:
        content = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"Could not read config file: {exc}"
        raise ConfigCorruptError(msg) from None

    if not content.strip():
        return {}

    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        msg = (
            f"Config file is corrupt ({config_path}): {exc}\n"
            f"Run 'mlx-stack config reset --yes' to restore defaults."
        )
        raise ConfigCorruptError(msg) from None

    if data is None:
        return {}

    if not isinstance(data, dict):
        msg = (
            f"Config file has invalid format ({config_path}): "
            f"expected a mapping, got {type(data).__name__}.\n"
            f"Run 'mlx-stack config reset --yes' to restore defaults."
        )
        raise ConfigCorruptError(msg) from None

    return data


def _save_raw_config(data: dict[str, Any]) -> None:
    """Save the raw config dictionary to disk.

    Creates the data directory if it doesn't exist.

    Args:
        data: The config dictionary to write.

    Raises:
        ConfigError: If the file cannot be written.
    """
    ensure_data_home()
    config_path = get_config_path()

    try:
        content = yaml.dump(data, default_flow_style=False, sort_keys=True)
        config_path.write_text(content, encoding="utf-8")
    except OSError as exc:
        msg = f"Could not write config file: {exc}"
        raise ConfigError(msg) from None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def get_default_value(key: str) -> Any:
    """Return the default value for a config key.

    For model-dir, expands ~ to the actual models directory path.

    Args:
        key: The configuration key name.

    Returns:
        The default value.

    Raises:
        ConfigError: If the key is not recognized.
    """
    key_def = validate_key(key)
    if key == "model-dir":
        return str(get_models_dir())
    return key_def.default


def get_value(key: str) -> Any:
    """Get the current value of a config key.

    Returns the user-set value if present, otherwise the default.

    Args:
        key: The configuration key name.

    Returns:
        The current value (user-set or default).

    Raises:
        ConfigError: If the key is not recognized.
        ConfigCorruptError: If the config file is corrupt.
    """
    validate_key(key)
    data = _load_raw_config()

    if key in data:
        return data[key]

    return get_default_value(key)


def set_value(key: str, raw_value: str) -> Any:
    """Set a config key to a new value.

    Validates the key and value, then persists to disk.

    Args:
        key: The configuration key name.
        raw_value: The raw string value from CLI input.

    Returns:
        The parsed and validated value that was stored.

    Raises:
        ConfigError: If the key is not recognized.
        ConfigValidationError: If the value fails validation.
        ConfigCorruptError: If the config file is corrupt.
    """
    key_def = validate_key(key)
    value = parse_value(key_def, raw_value)

    data = _load_raw_config()
    data[key] = value
    _save_raw_config(data)

    return value


def get_all_config() -> list[dict[str, Any]]:
    """Get all configuration keys with their current values and metadata.

    Returns:
        A list of dicts, each with keys: name, value, default, is_default,
        description, masked_value.

    Raises:
        ConfigCorruptError: If the config file is corrupt.
    """
    data = _load_raw_config()
    result: list[dict[str, Any]] = []

    for key, key_def in CONFIG_KEYS.items():
        default = get_default_value(key)
        is_default = key not in data
        value = data.get(key, default)

        result.append({
            "name": key,
            "value": value,
            "default": default,
            "is_default": is_default,
            "description": key_def.description,
            "masked_value": mask_value(key, value),
        })

    return result


def reset_config() -> None:
    """Reset all user-set config values by removing the config file.

    After reset, all keys return their default values.

    Raises:
        ConfigError: If the config file cannot be removed.
    """
    config_path = get_config_path()

    if config_path.exists():
        try:
            config_path.unlink()
        except OSError as exc:
            msg = f"Could not remove config file: {exc}"
            raise ConfigError(msg) from None
