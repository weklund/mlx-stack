"""Tests for the configuration management core module (core/config.py).

Covers key validation, value parsing, type coercion, validation rules,
masking, file I/O, and config operations (get, set, list, reset).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mlx_stack.core.config import (
    CONFIG_KEYS,
    VALID_KEYS,
    ConfigCorruptError,
    ConfigError,
    ConfigValidationError,
    get_all_config,
    get_default_value,
    get_value,
    mask_value,
    parse_value,
    reset_config,
    set_value,
    validate_key,
)
from mlx_stack.core.paths import get_config_path

# --------------------------------------------------------------------------- #
# Key validation
# --------------------------------------------------------------------------- #


class TestValidateKey:
    """Tests for validate_key()."""

    def test_valid_keys_accepted(self) -> None:
        for key in VALID_KEYS:
            key_def = validate_key(key)
            assert key_def.name == key

    def test_invalid_key_rejected(self) -> None:
        with pytest.raises(ConfigError, match="Unknown config key 'bad-key'"):
            validate_key("bad-key")

    def test_invalid_key_lists_valid_keys(self) -> None:
        with pytest.raises(ConfigError) as exc_info:
            validate_key("nonexistent")
        error_msg = str(exc_info.value)
        for key in VALID_KEYS:
            assert key in error_msg

    def test_all_six_keys_defined(self) -> None:
        expected_keys = {
            "openrouter-key",
            "default-quant",
            "memory-budget-pct",
            "litellm-port",
            "model-dir",
            "auto-health-check",
        }
        assert set(CONFIG_KEYS.keys()) == expected_keys


# --------------------------------------------------------------------------- #
# Value parsing
# --------------------------------------------------------------------------- #


class TestParseValue:
    """Tests for parse_value() — type conversion and validation."""

    # String types
    def test_parse_string_value(self) -> None:
        key_def = CONFIG_KEYS["openrouter-key"]
        assert parse_value(key_def, "sk-test-key-123") == "sk-test-key-123"

    def test_parse_empty_string(self) -> None:
        key_def = CONFIG_KEYS["openrouter-key"]
        assert parse_value(key_def, "") == ""

    # Integer types
    def test_parse_int_value(self) -> None:
        key_def = CONFIG_KEYS["memory-budget-pct"]
        assert parse_value(key_def, "60") == 60

    def test_parse_int_invalid(self) -> None:
        key_def = CONFIG_KEYS["memory-budget-pct"]
        with pytest.raises(ConfigValidationError, match="Expected an integer"):
            parse_value(key_def, "abc")

    def test_parse_int_float_rejected(self) -> None:
        key_def = CONFIG_KEYS["litellm-port"]
        with pytest.raises(ConfigValidationError, match="Expected an integer"):
            parse_value(key_def, "3.14")

    # Boolean types
    def test_parse_bool_true_variants(self) -> None:
        key_def = CONFIG_KEYS["auto-health-check"]
        for v in ("true", "True", "TRUE", "1", "yes", "YES", "on", "ON"):
            assert parse_value(key_def, v) is True

    def test_parse_bool_false_variants(self) -> None:
        key_def = CONFIG_KEYS["auto-health-check"]
        for v in ("false", "False", "FALSE", "0", "no", "NO", "off", "OFF"):
            assert parse_value(key_def, v) is False

    def test_parse_bool_invalid(self) -> None:
        key_def = CONFIG_KEYS["auto-health-check"]
        with pytest.raises(ConfigValidationError, match="Expected true/false"):
            parse_value(key_def, "maybe")

    # Path types
    def test_parse_path_value(self) -> None:
        key_def = CONFIG_KEYS["model-dir"]
        assert parse_value(key_def, "/tmp/my-models") == "/tmp/my-models"

    def test_parse_path_with_tilde(self) -> None:
        key_def = CONFIG_KEYS["model-dir"]
        assert parse_value(key_def, "~/models") == "~/models"


# --------------------------------------------------------------------------- #
# Specific validators
# --------------------------------------------------------------------------- #


class TestQuantValidation:
    """Tests for default-quant validation."""

    def test_valid_quants(self) -> None:
        key_def = CONFIG_KEYS["default-quant"]
        for quant in ("int4", "int8", "bf16"):
            assert parse_value(key_def, quant) == quant

    def test_invalid_quant_int6(self) -> None:
        key_def = CONFIG_KEYS["default-quant"]
        with pytest.raises(ConfigValidationError, match="Invalid quantization 'int6'"):
            parse_value(key_def, "int6")

    def test_invalid_quant_fp32(self) -> None:
        key_def = CONFIG_KEYS["default-quant"]
        with pytest.raises(ConfigValidationError, match="Invalid quantization 'fp32'"):
            parse_value(key_def, "fp32")

    def test_invalid_quant_lists_valid(self) -> None:
        key_def = CONFIG_KEYS["default-quant"]
        with pytest.raises(ConfigValidationError) as exc_info:
            parse_value(key_def, "bad")
        error_msg = str(exc_info.value)
        assert "bf16" in error_msg
        assert "int4" in error_msg
        assert "int8" in error_msg


class TestMemoryPctValidation:
    """Tests for memory-budget-pct validation."""

    def test_valid_range(self) -> None:
        key_def = CONFIG_KEYS["memory-budget-pct"]
        assert parse_value(key_def, "1") == 1
        assert parse_value(key_def, "50") == 50
        assert parse_value(key_def, "100") == 100

    def test_zero_rejected(self) -> None:
        key_def = CONFIG_KEYS["memory-budget-pct"]
        with pytest.raises(ConfigValidationError, match="Must be between 1 and 100"):
            parse_value(key_def, "0")

    def test_over_100_rejected(self) -> None:
        key_def = CONFIG_KEYS["memory-budget-pct"]
        with pytest.raises(ConfigValidationError, match="Must be between 1 and 100"):
            parse_value(key_def, "101")

    def test_negative_rejected(self) -> None:
        key_def = CONFIG_KEYS["memory-budget-pct"]
        with pytest.raises(ConfigValidationError, match="Must be between 1 and 100"):
            parse_value(key_def, "-5")

    def test_non_integer_rejected(self) -> None:
        key_def = CONFIG_KEYS["memory-budget-pct"]
        with pytest.raises(ConfigValidationError, match="Expected an integer"):
            parse_value(key_def, "abc")


class TestPortValidation:
    """Tests for litellm-port validation."""

    def test_valid_ports(self) -> None:
        key_def = CONFIG_KEYS["litellm-port"]
        assert parse_value(key_def, "1") == 1
        assert parse_value(key_def, "4000") == 4000
        assert parse_value(key_def, "65535") == 65535

    def test_zero_rejected(self) -> None:
        key_def = CONFIG_KEYS["litellm-port"]
        with pytest.raises(ConfigValidationError, match="Must be between 1 and 65535"):
            parse_value(key_def, "0")

    def test_over_65535_rejected(self) -> None:
        key_def = CONFIG_KEYS["litellm-port"]
        with pytest.raises(ConfigValidationError, match="Must be between 1 and 65535"):
            parse_value(key_def, "65536")

    def test_non_integer_rejected(self) -> None:
        key_def = CONFIG_KEYS["litellm-port"]
        with pytest.raises(ConfigValidationError, match="Expected an integer"):
            parse_value(key_def, "abc")


# --------------------------------------------------------------------------- #
# Masking
# --------------------------------------------------------------------------- #


class TestMasking:
    """Tests for mask_value()."""

    def test_api_key_masked(self) -> None:
        result = mask_value("openrouter-key", "sk-test-key-123456")
        assert result == "sk-****456"

    def test_short_api_key_fully_masked(self) -> None:
        result = mask_value("openrouter-key", "abcdef")
        assert result == "****"

    def test_very_short_api_key_masked(self) -> None:
        result = mask_value("openrouter-key", "ab")
        assert result == "****"

    def test_empty_api_key_not_masked(self) -> None:
        result = mask_value("openrouter-key", "")
        assert result == ""

    def test_non_sensitive_key_not_masked(self) -> None:
        result = mask_value("default-quant", "int4")
        assert result == "int4"

    def test_integer_not_masked(self) -> None:
        result = mask_value("litellm-port", 4000)
        assert result == "4000"

    def test_bool_not_masked(self) -> None:
        result = mask_value("auto-health-check", True)
        assert result == "True"


# --------------------------------------------------------------------------- #
# Default values
# --------------------------------------------------------------------------- #


class TestDefaultValues:
    """Tests for get_default_value()."""

    def test_default_quant(self) -> None:
        assert get_default_value("default-quant") == "int4"

    def test_default_memory_budget(self) -> None:
        assert get_default_value("memory-budget-pct") == 40

    def test_default_port(self) -> None:
        assert get_default_value("litellm-port") == 4000

    def test_default_model_dir(self, mlx_stack_home: Path) -> None:
        val = get_default_value("model-dir")
        assert val == str(mlx_stack_home / "models")

    def test_default_auto_health_check(self) -> None:
        assert get_default_value("auto-health-check") is True

    def test_default_openrouter_key(self) -> None:
        assert get_default_value("openrouter-key") == ""

    def test_invalid_key_raises(self) -> None:
        with pytest.raises(ConfigError):
            get_default_value("bad-key")


# --------------------------------------------------------------------------- #
# Get / Set round-trip
# --------------------------------------------------------------------------- #


class TestGetSet:
    """Tests for get_value() and set_value()."""

    def test_get_returns_default_when_unset(self, mlx_stack_home: Path) -> None:
        assert get_value("default-quant") == "int4"

    def test_set_and_get_string(self, mlx_stack_home: Path) -> None:
        set_value("openrouter-key", "sk-test-123")
        assert get_value("openrouter-key") == "sk-test-123"

    def test_set_and_get_integer(self, mlx_stack_home: Path) -> None:
        set_value("memory-budget-pct", "60")
        assert get_value("memory-budget-pct") == 60

    def test_set_and_get_boolean_true(self, mlx_stack_home: Path) -> None:
        set_value("auto-health-check", "false")
        assert get_value("auto-health-check") is False

    def test_set_and_get_boolean_false(self, mlx_stack_home: Path) -> None:
        set_value("auto-health-check", "true")
        assert get_value("auto-health-check") is True

    def test_set_and_get_path(self, mlx_stack_home: Path) -> None:
        set_value("model-dir", "/tmp/my-models")
        assert get_value("model-dir") == "/tmp/my-models"

    def test_set_and_get_port(self, mlx_stack_home: Path) -> None:
        set_value("litellm-port", "5000")
        assert get_value("litellm-port") == 5000

    def test_set_and_get_quant(self, mlx_stack_home: Path) -> None:
        set_value("default-quant", "int8")
        assert get_value("default-quant") == "int8"

    def test_set_overwrites_previous_value(self, mlx_stack_home: Path) -> None:
        set_value("litellm-port", "5000")
        set_value("litellm-port", "6000")
        assert get_value("litellm-port") == 6000

    def test_set_invalid_key_rejected(self, mlx_stack_home: Path) -> None:
        with pytest.raises(ConfigError, match="Unknown config key"):
            set_value("bad-key", "value")

    def test_set_invalid_value_rejected(self, mlx_stack_home: Path) -> None:
        with pytest.raises(ConfigValidationError):
            set_value("default-quant", "int6")

    def test_invalid_key_not_written_to_file(self, mlx_stack_home: Path) -> None:
        config_path = get_config_path()
        try:
            set_value("bad-key", "value")
        except ConfigError:
            pass
        if config_path.exists():
            content = config_path.read_text()
            assert "bad-key" not in content

    def test_invalid_value_not_written_to_file(self, mlx_stack_home: Path) -> None:
        config_path = get_config_path()
        try:
            set_value("default-quant", "int6")
        except ConfigValidationError:
            pass
        if config_path.exists():
            content = config_path.read_text()
            assert "int6" not in content


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


class TestPersistence:
    """Tests for config file persistence."""

    def test_config_written_as_yaml(self, mlx_stack_home: Path) -> None:
        set_value("litellm-port", "5000")
        config_path = get_config_path()
        assert config_path.exists()
        data = yaml.safe_load(config_path.read_text())
        assert data["litellm-port"] == 5000

    def test_api_key_stored_unmasked(self, mlx_stack_home: Path) -> None:
        set_value("openrouter-key", "sk-test-secret-key")
        config_path = get_config_path()
        data = yaml.safe_load(config_path.read_text())
        assert data["openrouter-key"] == "sk-test-secret-key"

    def test_multiple_values_persisted(self, mlx_stack_home: Path) -> None:
        set_value("litellm-port", "5000")
        set_value("default-quant", "int8")
        set_value("auto-health-check", "false")
        assert get_value("litellm-port") == 5000
        assert get_value("default-quant") == "int8"
        assert get_value("auto-health-check") is False

    def test_missing_file_returns_defaults(self, mlx_stack_home: Path) -> None:
        assert get_value("default-quant") == "int4"
        assert get_value("memory-budget-pct") == 40

    def test_empty_file_returns_defaults(self, mlx_stack_home: Path) -> None:
        config_path = get_config_path()
        config_path.write_text("")
        assert get_value("default-quant") == "int4"

    def test_data_directory_auto_created(self, clean_mlx_stack_home: Path) -> None:
        set_value("litellm-port", "5000")
        assert clean_mlx_stack_home.exists()
        assert get_value("litellm-port") == 5000


# --------------------------------------------------------------------------- #
# Corrupt file handling
# --------------------------------------------------------------------------- #


class TestCorruptConfig:
    """Tests for handling corrupt config files."""

    def test_invalid_yaml_raises_corrupt_error(self, mlx_stack_home: Path) -> None:
        config_path = get_config_path()
        config_path.write_text("{{{invalid yaml")
        with pytest.raises(ConfigCorruptError, match="corrupt"):
            get_value("default-quant")

    def test_corrupt_error_suggests_reset(self, mlx_stack_home: Path) -> None:
        config_path = get_config_path()
        config_path.write_text("{{{")
        with pytest.raises(ConfigCorruptError) as exc_info:
            get_value("default-quant")
        assert "config reset --yes" in str(exc_info.value)

    def test_non_mapping_yaml_raises_corrupt_error(self, mlx_stack_home: Path) -> None:
        config_path = get_config_path()
        config_path.write_text("- item1\n- item2\n")
        with pytest.raises(ConfigCorruptError, match="invalid format"):
            get_value("default-quant")

    def test_yaml_null_returns_defaults(self, mlx_stack_home: Path) -> None:
        config_path = get_config_path()
        config_path.write_text("---\n")  # YAML null document
        assert get_value("default-quant") == "int4"


# --------------------------------------------------------------------------- #
# get_all_config
# --------------------------------------------------------------------------- #


class TestGetAllConfig:
    """Tests for get_all_config()."""

    def test_returns_all_six_keys(self, mlx_stack_home: Path) -> None:
        entries = get_all_config()
        assert len(entries) == 6
        names = {e["name"] for e in entries}
        assert names == set(CONFIG_KEYS.keys())

    def test_default_values_marked(self, mlx_stack_home: Path) -> None:
        entries = get_all_config()
        for entry in entries:
            assert entry["is_default"] is True

    def test_user_set_values_marked(self, mlx_stack_home: Path) -> None:
        set_value("litellm-port", "5000")
        entries = get_all_config()
        port_entry = next(e for e in entries if e["name"] == "litellm-port")
        assert port_entry["is_default"] is False
        assert port_entry["value"] == 5000

    def test_api_key_masked_in_entries(self, mlx_stack_home: Path) -> None:
        set_value("openrouter-key", "sk-secret-key-12345")
        entries = get_all_config()
        key_entry = next(e for e in entries if e["name"] == "openrouter-key")
        assert "sk-secret-key-12345" != key_entry["masked_value"]
        assert "****" in key_entry["masked_value"]

    def test_corrupt_file_raises(self, mlx_stack_home: Path) -> None:
        config_path = get_config_path()
        config_path.write_text("{{{")
        with pytest.raises(ConfigCorruptError):
            get_all_config()


# --------------------------------------------------------------------------- #
# Reset
# --------------------------------------------------------------------------- #


class TestReset:
    """Tests for reset_config()."""

    def test_reset_clears_all_values(self, mlx_stack_home: Path) -> None:
        set_value("litellm-port", "5000")
        set_value("default-quant", "int8")
        reset_config()
        assert get_value("litellm-port") == 4000
        assert get_value("default-quant") == "int4"

    def test_reset_removes_config_file(self, mlx_stack_home: Path) -> None:
        set_value("litellm-port", "5000")
        config_path = get_config_path()
        assert config_path.exists()
        reset_config()
        assert not config_path.exists()

    def test_reset_when_no_file_succeeds(self, mlx_stack_home: Path) -> None:
        # Should not raise even if config file doesn't exist
        reset_config()

    def test_defaults_after_reset(self, mlx_stack_home: Path) -> None:
        set_value("memory-budget-pct", "80")
        set_value("auto-health-check", "false")
        reset_config()
        assert get_value("memory-budget-pct") == 40
        assert get_value("auto-health-check") is True
        assert get_value("default-quant") == "int4"
        assert get_value("litellm-port") == 4000
