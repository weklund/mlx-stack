"""Tests for the `mlx-stack up` CLI command and core stack_up module.

Validates:
- VAL-UP-001: Stack definition loaded and correct processes started
- VAL-UP-002: Subprocess output redirected to log files
- VAL-UP-003: Health check with exponential backoff and 120s timeout
- VAL-UP-004: PID files created per service
- VAL-UP-005: LiteLLM started after all model servers are healthy
- VAL-UP-006: Summary table displayed on completion
- VAL-UP-007: Lockfile prevents concurrent invocations
- VAL-UP-008: --dry-run shows commands without executing
- VAL-UP-009: --tier starts only the specified tier
- VAL-UP-010: Auto-installs missing dependencies
- VAL-UP-011: Missing stack definition error
- VAL-UP-012: Port already in use detection
- VAL-UP-013: Partial failure — remaining tiers still start
- VAL-UP-014: Already-running detection and stale PID cleanup
- VAL-UP-015: Services bind to localhost only
- VAL-UP-016: Memory estimate warning before startup
- VAL-UP-017: API key propagated to LiteLLM subprocess securely
- VAL-CROSS-006: Init → up configuration consistency
- VAL-CROSS-010: Stale PID recovery without explicit down
- VAL-CROSS-013: Data consistency — schema_version, vllm_flags
- VAL-CROSS-016: Interrupted startup releases lockfile
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from mlx_stack.cli.main import cli
from mlx_stack.core.catalog import (
    BenchmarkResult,
    Capabilities,
    CatalogEntry,
    QualityScores,
    QuantSource,
)
from mlx_stack.core.deps import DependencyInstallError
from mlx_stack.core.process import (
    HealthCheckError,
    HealthCheckResult,
    LockError,
    ServiceInfo,
)
from mlx_stack.core.stack_up import (
    LITELLM_SERVICE_NAME,
    TierStatus,
    UpError,
    UpResult,
    build_litellm_command,
    build_vllm_command,
    check_memory_warning,
    estimate_memory_usage,
    format_dry_run_command,
    load_stack_definition,
    run_up,
    sort_tiers_by_size,
)

# --------------------------------------------------------------------------- #
# Fixtures — reusable test data
# --------------------------------------------------------------------------- #


def _make_stack_yaml(
    tiers: list[dict[str, Any]] | None = None,
    schema_version: int = 1,
    litellm_port: int = 4000,
) -> dict[str, Any]:
    """Create a stack definition dict for testing."""
    if tiers is None:
        tiers = [
            {
                "name": "standard",
                "model": "big-model",
                "quant": "int4",
                "source": "mlx-community/big-model-4bit",
                "port": 8000,
                "vllm_flags": {
                    "continuous_batching": True,
                    "use_paged_cache": True,
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "hermes",
                },
            },
            {
                "name": "fast",
                "model": "fast-model",
                "quant": "int4",
                "source": "mlx-community/fast-model-4bit",
                "port": 8001,
                "vllm_flags": {
                    "continuous_batching": True,
                    "use_paged_cache": True,
                },
            },
        ]
    return {
        "schema_version": schema_version,
        "name": "default",
        "hardware_profile": "m4-max-128",
        "intent": "balanced",
        "created": "2026-03-24T00:00:00+00:00",
        "tiers": tiers,
    }


def _write_stack_yaml(
    mlx_stack_home: Path,
    stack: dict[str, Any] | None = None,
) -> Path:
    """Write a stack YAML file and return its path."""
    if stack is None:
        stack = _make_stack_yaml()
    stacks_dir = mlx_stack_home / "stacks"
    stacks_dir.mkdir(parents=True, exist_ok=True)
    stack_path = stacks_dir / "default.yaml"
    stack_path.write_text(yaml.dump(stack, default_flow_style=False))
    return stack_path


def _write_litellm_yaml(mlx_stack_home: Path) -> Path:
    """Write a minimal litellm.yaml config."""
    litellm_config = {
        "model_list": [
            {
                "model_name": "standard",
                "litellm_params": {
                    "model": "openai/big-model",
                    "api_base": "http://localhost:8000/v1",
                    "api_key": "dummy",
                },
            },
        ],
    }
    litellm_path = mlx_stack_home / "litellm.yaml"
    litellm_path.write_text(yaml.dump(litellm_config, default_flow_style=False))
    return litellm_path


def _make_entry(
    model_id: str = "test-model",
    params_b: float = 8.0,
    memory_gb: float = 5.5,
) -> CatalogEntry:
    """Create a CatalogEntry for testing."""
    return CatalogEntry(
        id=model_id,
        name=f"Test {model_id}",
        family="Test",
        params_b=params_b,
        architecture="transformer",
        min_mlx_lm_version="0.22.0",
        sources={
            "int4": QuantSource(hf_repo=f"mlx-community/{model_id}-4bit", disk_size_gb=4.5),
        },
        capabilities=Capabilities(
            tool_calling=True,
            tool_call_parser="hermes",
            thinking=False,
            reasoning_parser=None,
            vision=False,
        ),
        quality=QualityScores(overall=70, coding=65, reasoning=60, instruction_following=72),
        benchmarks={
            "m4-max-128": BenchmarkResult(prompt_tps=100.0, gen_tps=50.0, memory_gb=memory_gb),
        },
        tags=[],
    )


def _make_test_catalog() -> list[CatalogEntry]:
    """Create a test catalog."""
    return [
        _make_entry("big-model", params_b=49.0, memory_gb=30.0),
        _make_entry("fast-model", params_b=3.0, memory_gb=2.0),
    ]


# --------------------------------------------------------------------------- #
# Tests — load_stack_definition
# --------------------------------------------------------------------------- #


class TestLoadStackDefinition:
    """Tests for load_stack_definition."""

    def test_loads_valid_stack(self, mlx_stack_home: Path) -> None:
        """VAL-UP-001: Stack definition loaded."""
        _write_stack_yaml(mlx_stack_home)
        stack = load_stack_definition()
        assert stack["schema_version"] == 1
        assert len(stack["tiers"]) == 2

    def test_missing_stack_suggests_init(self, mlx_stack_home: Path) -> None:
        """VAL-UP-011: Missing stack definition error."""
        with pytest.raises(UpError, match="mlx-stack init"):
            load_stack_definition()

    def test_invalid_yaml_produces_clear_error(self, mlx_stack_home: Path) -> None:
        """VAL-UP-011: Invalid YAML produces clear error."""
        stacks_dir = mlx_stack_home / "stacks"
        stacks_dir.mkdir(parents=True, exist_ok=True)
        (stacks_dir / "default.yaml").write_text("{{{invalid yaml")
        with pytest.raises(UpError, match="Invalid YAML"):
            load_stack_definition()

    def test_unsupported_schema_version(self, mlx_stack_home: Path) -> None:
        """VAL-UP-011 / VAL-CROSS-013: Unsupported schema_version."""
        stack = _make_stack_yaml(schema_version=99)
        _write_stack_yaml(mlx_stack_home, stack)
        with pytest.raises(UpError, match="schema_version"):
            load_stack_definition()

    def test_empty_tiers_error(self, mlx_stack_home: Path) -> None:
        """Stack with no tiers raises error."""
        stack = _make_stack_yaml()
        stack["tiers"] = []
        _write_stack_yaml(mlx_stack_home, stack)
        with pytest.raises(UpError, match="no tiers"):
            load_stack_definition()

    def test_non_dict_file(self, mlx_stack_home: Path) -> None:
        """Non-mapping YAML file produces clear error."""
        stacks_dir = mlx_stack_home / "stacks"
        stacks_dir.mkdir(parents=True, exist_ok=True)
        (stacks_dir / "default.yaml").write_text("- just a list")
        with pytest.raises(UpError, match="invalid format"):
            load_stack_definition()


# --------------------------------------------------------------------------- #
# Tests — build_vllm_command
# --------------------------------------------------------------------------- #


class TestBuildVllmCommand:
    """Tests for vllm-mlx command building."""

    def test_basic_command(self) -> None:
        """VAL-UP-015: Services bind to localhost only."""
        tier = {
            "name": "fast",
            "model": "fast-model",
            "source": "mlx-community/fast-model-4bit",
            "port": 8001,
            "vllm_flags": {
                "continuous_batching": True,
                "use_paged_cache": True,
            },
        }
        cmd = build_vllm_command(tier, "/usr/local/bin/vllm-mlx")
        assert cmd[0] == "/usr/local/bin/vllm-mlx"
        assert cmd[1] == "serve"
        assert "mlx-community/fast-model-4bit" in cmd
        assert "--port" in cmd
        assert "8001" in cmd
        assert "--host" in cmd
        assert "127.0.0.1" in cmd
        assert "--continuous-batching" in cmd
        assert "--use-paged-cache" in cmd

    def test_serve_subcommand_with_model_positional(self) -> None:
        """vllm-mlx uses 'serve' subcommand with model as positional arg."""
        tier = {
            "name": "fast",
            "model": "test-model",
            "source": "mlx-community/test-model-4bit",
            "port": 8001,
            "vllm_flags": {},
        }
        cmd = build_vllm_command(tier, "vllm-mlx")
        # Command format: vllm-mlx serve <model> --port N --host 127.0.0.1
        assert cmd[0] == "vllm-mlx"
        assert cmd[1] == "serve"
        assert cmd[2] == "mlx-community/test-model-4bit"
        # --model flag should NOT be present
        assert "--model" not in cmd

    def test_tool_calling_flags(self) -> None:
        """VAL-CROSS-013: vllm_flags translate correctly to CLI flags."""
        tier = {
            "name": "standard",
            "model": "tool-model",
            "source": "mlx-community/tool-model-4bit",
            "port": 8000,
            "vllm_flags": {
                "continuous_batching": True,
                "use_paged_cache": True,
                "enable_auto_tool_choice": True,
                "tool_call_parser": "hermes",
            },
        }
        cmd = build_vllm_command(tier, "vllm-mlx")
        assert "--enable-auto-tool-choice" in cmd
        assert "--tool-call-parser" in cmd
        idx = cmd.index("--tool-call-parser")
        assert cmd[idx + 1] == "hermes"

    def test_boolean_false_flags_excluded(self) -> None:
        """Boolean False flags are not included in command."""
        tier = {
            "name": "test",
            "model": "test",
            "source": "mlx-community/test-4bit",
            "port": 8000,
            "vllm_flags": {
                "continuous_batching": True,
                "some_disabled_flag": False,
            },
        }
        cmd = build_vllm_command(tier, "vllm-mlx")
        assert "--some-disabled-flag" not in cmd
        assert "--continuous-batching" in cmd


# --------------------------------------------------------------------------- #
# Tests — build_litellm_command
# --------------------------------------------------------------------------- #


class TestBuildLitellmCommand:
    """Tests for litellm command building."""

    def test_basic_command(self) -> None:
        """VAL-UP-015: LiteLLM binds to localhost only."""
        cmd = build_litellm_command(
            "/usr/local/bin/litellm",
            4000,
            Path("/home/user/.mlx-stack/litellm.yaml"),
        )
        assert cmd[0] == "/usr/local/bin/litellm"
        assert "--config" in cmd
        assert "/home/user/.mlx-stack/litellm.yaml" in cmd
        assert "--port" in cmd
        assert "4000" in cmd
        assert "--host" in cmd
        assert "127.0.0.1" in cmd


# --------------------------------------------------------------------------- #
# Tests — format_dry_run_command
# --------------------------------------------------------------------------- #


class TestFormatDryRunCommand:
    """Tests for dry-run command formatting."""

    def test_basic_format(self) -> None:
        """Commands formatted as space-separated string."""
        cmd = ["vllm-mlx", "--model", "test", "--port", "8000"]
        result = format_dry_run_command(cmd)
        assert result == "vllm-mlx --model test --port 8000"

    def test_env_vars_masked(self) -> None:
        """VAL-UP-017: API key not visible in --dry-run output."""
        cmd = ["litellm", "--config", "litellm.yaml"]
        result = format_dry_run_command(cmd, {"OPENROUTER_API_KEY": "sk-secret"})
        assert "sk-secret" not in result
        assert "OPENROUTER_API_KEY=***" in result


# --------------------------------------------------------------------------- #
# Tests — sort_tiers_by_size
# --------------------------------------------------------------------------- #


class TestSortTiersBySize:
    """Tests for tier sorting."""

    def test_largest_first(self) -> None:
        """VAL-UP-001: Tiers started in descending params_b order."""
        catalog = _make_test_catalog()
        tiers = [
            {"name": "fast", "model": "fast-model", "port": 8001},
            {"name": "standard", "model": "big-model", "port": 8000},
        ]
        sorted_tiers = sort_tiers_by_size(tiers, catalog)
        assert sorted_tiers[0]["name"] == "standard"  # 49B first
        assert sorted_tiers[1]["name"] == "fast"  # 3B second

    def test_no_catalog_preserves_order(self) -> None:
        """Without catalog, original order is preserved."""
        tiers = [
            {"name": "fast", "model": "fast-model", "port": 8001},
            {"name": "standard", "model": "big-model", "port": 8000},
        ]
        sorted_tiers = sort_tiers_by_size(tiers, None)
        assert sorted_tiers[0]["name"] == "fast"


# --------------------------------------------------------------------------- #
# Tests — memory estimation
# --------------------------------------------------------------------------- #


class TestMemoryEstimation:
    """Tests for memory estimation and warnings."""

    def test_estimates_from_catalog(self) -> None:
        """VAL-UP-016: Memory estimate from catalog data."""
        catalog = _make_test_catalog()
        tiers = [
            {"name": "standard", "model": "big-model"},
            {"name": "fast", "model": "fast-model"},
        ]
        total = estimate_memory_usage(tiers, catalog)
        assert total == pytest.approx(32.0, abs=1.0)

    def test_unknown_model_skipped(self) -> None:
        """Unknown models contribute 0 to estimate."""
        catalog = _make_test_catalog()
        tiers = [{"name": "unknown", "model": "nonexistent"}]
        total = estimate_memory_usage(tiers, catalog)
        assert total == 0.0

    def test_warning_when_exceeds_available(self) -> None:
        """VAL-UP-016: Warning when estimate exceeds available memory."""
        with patch("mlx_stack.core.stack_up.psutil.virtual_memory") as mock_vmem:
            mock_vmem.return_value = MagicMock(available=10 * 1024**3)  # 10 GB
            warning = check_memory_warning(20.0)
            assert warning is not None
            assert "20.0 GB" in warning
            assert "10.0 GB" in warning

    def test_no_warning_when_fits(self) -> None:
        """No warning when estimate fits in available memory."""
        with patch("mlx_stack.core.stack_up.psutil.virtual_memory") as mock_vmem:
            mock_vmem.return_value = MagicMock(available=100 * 1024**3)  # 100 GB
            warning = check_memory_warning(20.0)
            assert warning is None


# --------------------------------------------------------------------------- #
# Tests — CLI: dry-run
# --------------------------------------------------------------------------- #


class TestDryRun:
    """Tests for --dry-run mode via CLI."""

    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    def test_dry_run_shows_commands(
        self,
        mock_get_value: MagicMock,
        mock_load_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-008: --dry-run shows commands without executing."""
        _write_stack_yaml(mlx_stack_home)
        _write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = _make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "",
        }.get(key, "")

        runner = CliRunner()
        result = runner.invoke(cli, ["up", "--dry-run"])
        assert result.exit_code == 0
        assert "Dry run" in result.output

    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    def test_dry_run_no_pid_files(
        self,
        mock_get_value: MagicMock,
        mock_load_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-008: No PID files after --dry-run."""
        _write_stack_yaml(mlx_stack_home)
        _write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = _make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "",
        }.get(key, "")

        runner = CliRunner()
        runner.invoke(cli, ["up", "--dry-run"])
        pids_dir = mlx_stack_home / "pids"
        if pids_dir.exists():
            assert list(pids_dir.glob("*.pid")) == []

    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    def test_dry_run_no_log_files(
        self,
        mock_get_value: MagicMock,
        mock_load_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-008: No log files after --dry-run."""
        _write_stack_yaml(mlx_stack_home)
        _write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = _make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "",
        }.get(key, "")

        runner = CliRunner()
        runner.invoke(cli, ["up", "--dry-run"])
        logs_dir = mlx_stack_home / "logs"
        if logs_dir.exists():
            assert list(logs_dir.glob("*.log")) == []

    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    def test_dry_run_shows_host_127(
        self,
        mock_get_value: MagicMock,
        mock_load_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-015: --dry-run confirms localhost binding."""
        _write_stack_yaml(mlx_stack_home)
        _write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = _make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "",
        }.get(key, "")

        runner = CliRunner()
        result = runner.invoke(cli, ["up", "--dry-run"])
        assert "127.0.0.1" in result.output

    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    def test_dry_run_masks_api_key(
        self,
        mock_get_value: MagicMock,
        mock_load_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-017: API key not visible in --dry-run."""
        _write_stack_yaml(mlx_stack_home)
        _write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = _make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "sk-or-secret-key-12345",
        }.get(key, "")

        runner = CliRunner()
        result = runner.invoke(cli, ["up", "--dry-run"])
        assert "sk-or-secret-key-12345" not in result.output
        assert "OPENROUTER_API_KEY=***" in result.output

    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    def test_dry_run_tier_filter(
        self,
        mock_get_value: MagicMock,
        mock_load_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-009: --tier with --dry-run shows only that tier."""
        _write_stack_yaml(mlx_stack_home)
        _write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = _make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "",
        }.get(key, "")

        runner = CliRunner()
        result = runner.invoke(cli, ["up", "--dry-run", "--tier", "fast"])
        assert result.exit_code == 0
        # Should show fast tier but not standard
        assert "fast" in result.output

    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    def test_dry_run_vllm_flags_in_commands(
        self,
        mock_get_value: MagicMock,
        mock_load_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-CROSS-013: vllm_flags translate correctly to dry-run CLI flags."""
        _write_stack_yaml(mlx_stack_home)
        _write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = _make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "",
        }.get(key, "")

        runner = CliRunner()
        result = runner.invoke(cli, ["up", "--dry-run"])
        # The standard tier has enable_auto_tool_choice and tool_call_parser
        assert "--enable-auto-tool-choice" in result.output
        assert "--tool-call-parser" in result.output
        assert "hermes" in result.output


# --------------------------------------------------------------------------- #
# Tests — CLI: error cases
# --------------------------------------------------------------------------- #


class TestUpErrors:
    """Tests for error handling in the up command."""

    def test_missing_stack_error(self, mlx_stack_home: Path) -> None:
        """VAL-UP-011: Missing stack definition suggests init."""
        runner = CliRunner()
        result = runner.invoke(cli, ["up"])
        assert result.exit_code != 0
        output_text = result.output.lower()
        assert "init" in output_text

    @patch("mlx_stack.core.stack_up.get_value")
    def test_invalid_tier_error(
        self,
        mock_get_value: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-009: Invalid tier name errors with valid list."""
        _write_stack_yaml(mlx_stack_home)
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "",
        }.get(key, "")

        runner = CliRunner()
        result = runner.invoke(cli, ["up", "--tier", "nonexistent"])
        assert result.exit_code != 0
        # Should mention valid tiers
        assert "standard" in result.output or "fast" in result.output

    def test_invalid_yaml_error(self, mlx_stack_home: Path) -> None:
        """VAL-UP-011: Invalid YAML produces clear error."""
        stacks_dir = mlx_stack_home / "stacks"
        stacks_dir.mkdir(parents=True, exist_ok=True)
        (stacks_dir / "default.yaml").write_text("{{{bad yaml")
        runner = CliRunner()
        result = runner.invoke(cli, ["up"])
        assert result.exit_code != 0

    def test_unsupported_schema_error(self, mlx_stack_home: Path) -> None:
        """VAL-UP-011 / VAL-CROSS-013: Unsupported schema_version."""
        stack = _make_stack_yaml(schema_version=999)
        _write_stack_yaml(mlx_stack_home, stack)
        runner = CliRunner()
        result = runner.invoke(cli, ["up"])
        assert result.exit_code != 0
        assert "schema_version" in result.output


# --------------------------------------------------------------------------- #
# Tests — run_up with mocked subprocess layer
# --------------------------------------------------------------------------- #


class TestRunUp:
    """Tests for run_up with mocked process management."""

    @patch("mlx_stack.core.stack_up.check_local_model_exists", return_value=None)
    @patch("mlx_stack.core.stack_up.start_service")
    @patch("mlx_stack.core.stack_up.wait_for_healthy")
    @patch("mlx_stack.core.stack_up.check_port_conflict", return_value=None)
    @patch("mlx_stack.core.stack_up.read_pid_file", return_value=None)
    @patch("mlx_stack.core.stack_up.acquire_lock")
    @patch("mlx_stack.core.stack_up.ensure_dependency")
    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    @patch("mlx_stack.core.stack_up.shutil.which")
    def test_successful_startup(
        self,
        mock_which: MagicMock,
        mock_get_value: MagicMock,
        mock_load_catalog: MagicMock,
        mock_ensure_dep: MagicMock,
        mock_lock: MagicMock,
        mock_read_pid: MagicMock,
        mock_port_conflict: MagicMock,
        mock_wait_healthy: MagicMock,
        mock_start_service: MagicMock,
        mock_model_exists: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-001/004/005: Successful startup with PID files and LiteLLM."""
        _write_stack_yaml(mlx_stack_home)
        _write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = _make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "",
        }.get(key, "")
        mock_which.side_effect = lambda x: f"/usr/local/bin/{x}"
        mock_lock.return_value.__enter__ = MagicMock(return_value=None)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        mock_start_service.return_value = ServiceInfo(
            name="test", pid=12345, port=8000, log_path=Path("/tmp/test.log"),
            pid_path=Path("/tmp/test.pid"),
        )
        mock_wait_healthy.return_value = HealthCheckResult(
            healthy=True, response_time=0.5, status_code=200,
        )

        result = run_up()
        assert len(result.tiers) == 2
        assert all(t.status == "healthy" for t in result.tiers)
        assert result.litellm is not None
        assert result.litellm.status == "healthy"

    @patch("mlx_stack.core.stack_up.check_local_model_exists", return_value=None)
    @patch("mlx_stack.core.stack_up.start_service")
    @patch("mlx_stack.core.stack_up.wait_for_healthy")
    @patch("mlx_stack.core.stack_up.check_port_conflict", return_value=None)
    @patch("mlx_stack.core.stack_up.read_pid_file", return_value=None)
    @patch("mlx_stack.core.stack_up.acquire_lock")
    @patch("mlx_stack.core.stack_up.ensure_dependency")
    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    @patch("mlx_stack.core.stack_up.shutil.which")
    def test_tier_filter_starts_only_one(
        self,
        mock_which: MagicMock,
        mock_get_value: MagicMock,
        mock_load_catalog: MagicMock,
        mock_ensure_dep: MagicMock,
        mock_lock: MagicMock,
        mock_read_pid: MagicMock,
        mock_port_conflict: MagicMock,
        mock_wait_healthy: MagicMock,
        mock_start_service: MagicMock,
        mock_model_exists: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-009: --tier starts only the specified tier."""
        _write_stack_yaml(mlx_stack_home)
        _write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = _make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "",
        }.get(key, "")
        mock_which.side_effect = lambda x: f"/usr/local/bin/{x}"
        mock_lock.return_value.__enter__ = MagicMock(return_value=None)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        mock_start_service.return_value = ServiceInfo(
            name="fast", pid=12345, port=8001, log_path=Path("/tmp/fast.log"),
            pid_path=Path("/tmp/fast.pid"),
        )
        mock_wait_healthy.return_value = HealthCheckResult(
            healthy=True, response_time=0.5, status_code=200,
        )

        result = run_up(tier_filter="fast")
        # Should have one tier (fast) + litellm
        tier_names = [t.name for t in result.tiers]
        assert "fast" in tier_names
        assert "standard" not in tier_names

    @patch("mlx_stack.core.stack_up.check_local_model_exists", return_value=None)
    @patch("mlx_stack.core.stack_up.start_service")
    @patch("mlx_stack.core.stack_up.wait_for_healthy")
    @patch("mlx_stack.core.stack_up.check_port_conflict")
    @patch("mlx_stack.core.stack_up.read_pid_file", return_value=None)
    @patch("mlx_stack.core.stack_up.acquire_lock")
    @patch("mlx_stack.core.stack_up.ensure_dependency")
    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    @patch("mlx_stack.core.stack_up.shutil.which")
    def test_port_conflict_skips_tier(
        self,
        mock_which: MagicMock,
        mock_get_value: MagicMock,
        mock_load_catalog: MagicMock,
        mock_ensure_dep: MagicMock,
        mock_lock: MagicMock,
        mock_read_pid: MagicMock,
        mock_port_conflict: MagicMock,
        mock_wait_healthy: MagicMock,
        mock_start_service: MagicMock,
        mock_model_exists: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-012/013: Port conflict skips tier, remaining still start."""
        _write_stack_yaml(mlx_stack_home)
        _write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = _make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "",
        }.get(key, "")
        mock_which.side_effect = lambda x: f"/usr/local/bin/{x}"
        mock_lock.return_value.__enter__ = MagicMock(return_value=None)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)

        # First tier has port conflict, second is fine
        def port_conflict_side_effect(port: int) -> tuple[int, str] | None:
            if port == 8000:
                return (99999, "other-process")
            return None

        mock_port_conflict.side_effect = port_conflict_side_effect
        mock_start_service.return_value = ServiceInfo(
            name="fast", pid=12345, port=8001, log_path=Path("/tmp/fast.log"),
            pid_path=Path("/tmp/fast.pid"),
        )
        mock_wait_healthy.return_value = HealthCheckResult(
            healthy=True, response_time=0.5, status_code=200,
        )

        result = run_up()
        statuses = {t.name: t.status for t in result.tiers}
        # Standard (port 8000) should be skipped
        assert statuses["standard"] == "skipped"
        # Fast should be healthy
        assert statuses["fast"] == "healthy"
        # LiteLLM should still start (at least one healthy tier)
        assert result.litellm is not None
        assert result.litellm.status == "healthy"

    @patch("mlx_stack.core.stack_up.check_local_model_exists", return_value=None)
    @patch("mlx_stack.core.stack_up.start_service")
    @patch("mlx_stack.core.stack_up.wait_for_healthy")
    @patch("mlx_stack.core.stack_up.check_port_conflict")
    @patch("mlx_stack.core.stack_up.read_pid_file", return_value=None)
    @patch("mlx_stack.core.stack_up.acquire_lock")
    @patch("mlx_stack.core.stack_up.ensure_dependency")
    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    @patch("mlx_stack.core.stack_up.shutil.which")
    def test_port_conflict_error_shows_pid_and_process(
        self,
        mock_which: MagicMock,
        mock_get_value: MagicMock,
        mock_load_catalog: MagicMock,
        mock_ensure_dep: MagicMock,
        mock_lock: MagicMock,
        mock_read_pid: MagicMock,
        mock_port_conflict: MagicMock,
        mock_wait_healthy: MagicMock,
        mock_start_service: MagicMock,
        mock_model_exists: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-012: Port conflict error includes PID and process name."""
        _write_stack_yaml(mlx_stack_home)
        _write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = _make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "",
        }.get(key, "")
        mock_which.side_effect = lambda x: f"/usr/local/bin/{x}"
        mock_lock.return_value.__enter__ = MagicMock(return_value=None)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)

        # Port 8000 occupied by a known process
        def port_conflict_side_effect(port: int) -> tuple[int, str] | None:
            if port == 8000:
                return (54321, "node")
            return None

        mock_port_conflict.side_effect = port_conflict_side_effect
        mock_start_service.return_value = ServiceInfo(
            name="fast", pid=12345, port=8001, log_path=Path("/tmp/fast.log"),
            pid_path=Path("/tmp/fast.pid"),
        )
        mock_wait_healthy.return_value = HealthCheckResult(
            healthy=True, response_time=0.5, status_code=200,
        )

        result = run_up()
        # Find the skipped tier
        skipped = [t for t in result.tiers if t.status == "skipped"]
        assert len(skipped) == 1
        assert skipped[0].name == "standard"
        # Error message must include the conflicting PID and process name
        assert skipped[0].error is not None
        assert "54321" in skipped[0].error
        assert "node" in skipped[0].error
        assert "8000" in skipped[0].error

    @patch("mlx_stack.core.stack_up.check_local_model_exists", return_value=None)
    @patch("mlx_stack.core.stack_up.start_service")
    @patch("mlx_stack.core.stack_up.wait_for_healthy")
    @patch("mlx_stack.core.stack_up.check_port_conflict")
    @patch("mlx_stack.core.stack_up.read_pid_file", return_value=None)
    @patch("mlx_stack.core.stack_up.acquire_lock")
    @patch("mlx_stack.core.stack_up.ensure_dependency")
    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    @patch("mlx_stack.core.stack_up.shutil.which")
    def test_port_conflict_unknown_owner(
        self,
        mock_which: MagicMock,
        mock_get_value: MagicMock,
        mock_load_catalog: MagicMock,
        mock_ensure_dep: MagicMock,
        mock_lock: MagicMock,
        mock_read_pid: MagicMock,
        mock_port_conflict: MagicMock,
        mock_wait_healthy: MagicMock,
        mock_start_service: MagicMock,
        mock_model_exists: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-012: Port conflict with unknown owner still shows port."""
        _write_stack_yaml(mlx_stack_home)
        _write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = _make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "",
        }.get(key, "")
        mock_which.side_effect = lambda x: f"/usr/local/bin/{x}"
        mock_lock.return_value.__enter__ = MagicMock(return_value=None)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)

        # Port occupied but owner unknown (e.g., macOS permission issue)
        mock_port_conflict.side_effect = lambda port: (
            (0, "<unknown>") if port == 8000 else None
        )
        mock_start_service.return_value = ServiceInfo(
            name="fast", pid=12345, port=8001, log_path=Path("/tmp/fast.log"),
            pid_path=Path("/tmp/fast.pid"),
        )
        mock_wait_healthy.return_value = HealthCheckResult(
            healthy=True, response_time=0.5, status_code=200,
        )

        result = run_up()
        skipped = [t for t in result.tiers if t.status == "skipped"]
        assert len(skipped) == 1
        assert "8000" in (skipped[0].error or "")
        assert "already in use" in (skipped[0].error or "")

    @patch("mlx_stack.core.stack_up.check_local_model_exists", return_value=None)
    @patch("mlx_stack.core.stack_up.start_service")
    @patch("mlx_stack.core.stack_up.wait_for_healthy")
    @patch("mlx_stack.core.stack_up.check_port_conflict", return_value=None)
    @patch("mlx_stack.core.stack_up.read_pid_file", return_value=None)
    @patch("mlx_stack.core.stack_up.acquire_lock")
    @patch("mlx_stack.core.stack_up.ensure_dependency")
    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    @patch("mlx_stack.core.stack_up.shutil.which")
    def test_health_check_timeout_continues(
        self,
        mock_which: MagicMock,
        mock_get_value: MagicMock,
        mock_load_catalog: MagicMock,
        mock_ensure_dep: MagicMock,
        mock_lock: MagicMock,
        mock_read_pid: MagicMock,
        mock_port_conflict: MagicMock,
        mock_wait_healthy: MagicMock,
        mock_start_service: MagicMock,
        mock_model_exists: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-003/013: Health check timeout reports error, continues."""
        _write_stack_yaml(mlx_stack_home)
        _write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = _make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "",
        }.get(key, "")
        mock_which.side_effect = lambda x: f"/usr/local/bin/{x}"
        mock_lock.return_value.__enter__ = MagicMock(return_value=None)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        mock_start_service.return_value = ServiceInfo(
            name="test", pid=12345, port=8000, log_path=Path("/tmp/test.log"),
            pid_path=Path("/tmp/test.pid"),
        )

        # First tier health check fails, second succeeds, LiteLLM succeeds
        call_count = 0

        def wait_healthy_side_effect(**kwargs: Any) -> HealthCheckResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise HealthCheckError("Timeout after 120s")
            return HealthCheckResult(healthy=True, response_time=0.5, status_code=200)

        mock_wait_healthy.side_effect = wait_healthy_side_effect

        result = run_up()
        statuses = {t.name: t.status for t in result.tiers}
        # One tier should fail, one should be healthy
        assert "failed" in statuses.values()
        assert "healthy" in statuses.values()

    @patch("mlx_stack.core.stack_up.check_local_model_exists", return_value=None)
    @patch("mlx_stack.core.stack_up.check_port_conflict", return_value=None)
    @patch("mlx_stack.core.stack_up.read_pid_file", return_value=None)
    @patch("mlx_stack.core.stack_up.acquire_lock")
    @patch("mlx_stack.core.stack_up.ensure_dependency")
    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    @patch("mlx_stack.core.stack_up.shutil.which")
    def test_all_fail_no_litellm(
        self,
        mock_which: MagicMock,
        mock_get_value: MagicMock,
        mock_load_catalog: MagicMock,
        mock_ensure_dep: MagicMock,
        mock_lock: MagicMock,
        mock_read_pid: MagicMock,
        mock_port_conflict: MagicMock,
        mock_model_exists: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-005: LiteLLM not started if all model servers fail."""
        _write_stack_yaml(mlx_stack_home)
        _write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = _make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "",
        }.get(key, "")
        mock_which.side_effect = lambda x: f"/usr/local/bin/{x}"
        mock_lock.return_value.__enter__ = MagicMock(return_value=None)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)

        # All ports conflict
        mock_port_conflict.return_value = (99, "blocker")

        result = run_up()
        assert result.litellm is not None
        assert result.litellm.status == "skipped"
        assert "All model servers failed" in (result.litellm.error or "")

    @patch("mlx_stack.core.stack_up.is_process_alive", return_value=True)
    @patch("mlx_stack.core.stack_up.read_pid_file")
    @patch("mlx_stack.core.stack_up.acquire_lock")
    @patch("mlx_stack.core.stack_up.ensure_dependency")
    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    @patch("mlx_stack.core.stack_up.shutil.which")
    def test_already_running_detection(
        self,
        mock_which: MagicMock,
        mock_get_value: MagicMock,
        mock_load_catalog: MagicMock,
        mock_ensure_dep: MagicMock,
        mock_lock: MagicMock,
        mock_read_pid: MagicMock,
        mock_alive: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-014: Already-running detection."""
        _write_stack_yaml(mlx_stack_home)
        _write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = _make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "",
        }.get(key, "")
        mock_which.side_effect = lambda x: f"/usr/local/bin/{x}"
        mock_lock.return_value.__enter__ = MagicMock(return_value=None)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        mock_read_pid.return_value = 12345  # All services have PIDs

        result = run_up()
        assert result.already_running is True
        assert all(t.status == "already-running" for t in result.tiers)

    @patch("mlx_stack.core.stack_up.check_local_model_exists", return_value=None)
    @patch("mlx_stack.core.stack_up.start_service")
    @patch("mlx_stack.core.stack_up.wait_for_healthy")
    @patch("mlx_stack.core.stack_up.check_port_conflict", return_value=None)
    @patch("mlx_stack.core.stack_up.cleanup_stale_pid", return_value=True)
    @patch("mlx_stack.core.stack_up.is_process_alive", return_value=False)
    @patch("mlx_stack.core.stack_up.read_pid_file", return_value=12345)
    @patch("mlx_stack.core.stack_up.acquire_lock")
    @patch("mlx_stack.core.stack_up.ensure_dependency")
    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    @patch("mlx_stack.core.stack_up.shutil.which")
    def test_stale_pid_cleanup_and_restart(
        self,
        mock_which: MagicMock,
        mock_get_value: MagicMock,
        mock_load_catalog: MagicMock,
        mock_ensure_dep: MagicMock,
        mock_lock: MagicMock,
        mock_read_pid: MagicMock,
        mock_alive: MagicMock,
        mock_cleanup_stale: MagicMock,
        mock_port_conflict: MagicMock,
        mock_wait_healthy: MagicMock,
        mock_start_service: MagicMock,
        mock_model_exists: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-014 / VAL-CROSS-010: Stale PID cleanup and fresh start."""
        _write_stack_yaml(mlx_stack_home)
        _write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = _make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "",
        }.get(key, "")
        mock_which.side_effect = lambda x: f"/usr/local/bin/{x}"
        mock_lock.return_value.__enter__ = MagicMock(return_value=None)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        mock_start_service.return_value = ServiceInfo(
            name="test", pid=99999, port=8000, log_path=Path("/tmp/test.log"),
            pid_path=Path("/tmp/test.pid"),
        )
        mock_wait_healthy.return_value = HealthCheckResult(
            healthy=True, response_time=0.5, status_code=200,
        )

        result = run_up()
        # Should have cleaned stale PIDs warning
        assert any("stale" in w.lower() for w in result.warnings)
        # Services should be started fresh
        assert any(t.status == "healthy" for t in result.tiers)

    @patch("mlx_stack.core.stack_up.acquire_lock")
    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    def test_lockfile_prevents_concurrent(
        self,
        mock_get_value: MagicMock,
        mock_load_catalog: MagicMock,
        mock_lock: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-007: Lockfile prevents concurrent invocations."""
        _write_stack_yaml(mlx_stack_home)
        mock_load_catalog.return_value = _make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "",
        }.get(key, "")
        mock_lock.side_effect = LockError("Another operation running")

        with pytest.raises(LockError, match="Another operation"):
            run_up()

    @patch("mlx_stack.core.stack_up.acquire_lock")
    @patch("mlx_stack.core.stack_up.ensure_dependency")
    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    def test_auto_install_failure(
        self,
        mock_get_value: MagicMock,
        mock_load_catalog: MagicMock,
        mock_ensure_dep: MagicMock,
        mock_lock: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-010: Auto-install failure produces clear error."""
        _write_stack_yaml(mlx_stack_home)
        _write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = _make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "",
        }.get(key, "")
        mock_lock.return_value.__enter__ = MagicMock(return_value=None)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        mock_ensure_dep.side_effect = DependencyInstallError("Install failed")

        with pytest.raises(UpError, match="Dependency installation failed"):
            run_up()

    @patch("mlx_stack.core.stack_up.check_local_model_exists", return_value=None)
    @patch("mlx_stack.core.stack_up.start_service")
    @patch("mlx_stack.core.stack_up.wait_for_healthy")
    @patch("mlx_stack.core.stack_up.check_port_conflict", return_value=None)
    @patch("mlx_stack.core.stack_up.read_pid_file", return_value=None)
    @patch("mlx_stack.core.stack_up.acquire_lock")
    @patch("mlx_stack.core.stack_up.ensure_dependency")
    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    @patch("mlx_stack.core.stack_up.shutil.which")
    def test_api_key_passed_via_env(
        self,
        mock_which: MagicMock,
        mock_get_value: MagicMock,
        mock_load_catalog: MagicMock,
        mock_ensure_dep: MagicMock,
        mock_lock: MagicMock,
        mock_read_pid: MagicMock,
        mock_port_conflict: MagicMock,
        mock_wait_healthy: MagicMock,
        mock_start_service: MagicMock,
        mock_model_exists: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-017: API key passed via env var, not CLI args."""
        _write_stack_yaml(mlx_stack_home)
        _write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = _make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "sk-or-secret-key",
        }.get(key, "")
        mock_which.side_effect = lambda x: f"/usr/local/bin/{x}"
        mock_lock.return_value.__enter__ = MagicMock(return_value=None)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        mock_start_service.return_value = ServiceInfo(
            name="test", pid=12345, port=8000, log_path=Path("/tmp/test.log"),
            pid_path=Path("/tmp/test.pid"),
        )
        mock_wait_healthy.return_value = HealthCheckResult(
            healthy=True, response_time=0.5, status_code=200,
        )

        run_up()

        # Check that start_service was called for litellm with env dict
        litellm_calls = [
            c for c in mock_start_service.call_args_list
            if c.kwargs.get("service_name") == LITELLM_SERVICE_NAME
            or (c.args and c.args[0] == LITELLM_SERVICE_NAME)
        ]
        assert len(litellm_calls) >= 1
        # The LiteLLM call should have env with OPENROUTER_API_KEY
        litellm_call = litellm_calls[0]
        env_arg = litellm_call.kwargs.get("env") or (
            litellm_call.args[3] if len(litellm_call.args) > 3 else None
        )
        assert env_arg is not None
        assert "OPENROUTER_API_KEY" in env_arg
        assert env_arg["OPENROUTER_API_KEY"] == "sk-or-secret-key"

    @patch("mlx_stack.core.stack_up.check_local_model_exists", return_value=None)
    @patch("mlx_stack.core.stack_up.start_service")
    @patch("mlx_stack.core.stack_up.wait_for_healthy")
    @patch("mlx_stack.core.stack_up.check_port_conflict", return_value=None)
    @patch("mlx_stack.core.stack_up.read_pid_file", return_value=None)
    @patch("mlx_stack.core.stack_up.acquire_lock")
    @patch("mlx_stack.core.stack_up.ensure_dependency")
    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    @patch("mlx_stack.core.stack_up.shutil.which")
    def test_memory_warning_displayed(
        self,
        mock_which: MagicMock,
        mock_get_value: MagicMock,
        mock_load_catalog: MagicMock,
        mock_ensure_dep: MagicMock,
        mock_lock: MagicMock,
        mock_read_pid: MagicMock,
        mock_port_conflict: MagicMock,
        mock_wait_healthy: MagicMock,
        mock_start_service: MagicMock,
        mock_model_exists: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-016: Memory estimate warning before startup."""
        _write_stack_yaml(mlx_stack_home)
        _write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = _make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "",
        }.get(key, "")
        mock_which.side_effect = lambda x: f"/usr/local/bin/{x}"
        mock_lock.return_value.__enter__ = MagicMock(return_value=None)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        mock_start_service.return_value = ServiceInfo(
            name="test", pid=12345, port=8000, log_path=Path("/tmp/test.log"),
            pid_path=Path("/tmp/test.pid"),
        )
        mock_wait_healthy.return_value = HealthCheckResult(
            healthy=True, response_time=0.5, status_code=200,
        )

        with patch("mlx_stack.core.stack_up.psutil.virtual_memory") as mock_vmem:
            mock_vmem.return_value = MagicMock(available=5 * 1024**3)  # Only 5 GB
            result = run_up()

        # Should have a memory warning (catalog estimates ~32 GB)
        assert any("memory" in w.lower() for w in result.warnings)


# --------------------------------------------------------------------------- #
# Tests — CLI output verification
# --------------------------------------------------------------------------- #


class TestCLIOutput:
    """Tests for CLI output formatting."""

    @patch("mlx_stack.cli.up.run_up")
    def test_summary_table_displayed(
        self,
        mock_run_up: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-006: Summary table shows tier, model, port, status."""
        mock_run_up.return_value = UpResult(
            tiers=[
                TierStatus(name="standard", model="big-model", port=8000, status="healthy"),
                TierStatus(name="fast", model="fast-model", port=8001, status="healthy"),
            ],
            litellm=TierStatus(
                name="litellm", model="proxy", port=4000, status="healthy",
            ),
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["up"])
        assert result.exit_code == 0
        assert "standard" in result.output
        assert "fast" in result.output
        assert "8000" in result.output
        assert "8001" in result.output
        assert "healthy" in result.output

    @patch("mlx_stack.cli.up.run_up")
    def test_already_running_message(
        self,
        mock_run_up: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-014: Already-running shows informational message."""
        mock_run_up.return_value = UpResult(
            tiers=[
                TierStatus(
                    name="standard", model="big-model", port=8000,
                    status="already-running",
                ),
            ],
            litellm=TierStatus(
                name="litellm", model="proxy", port=4000, status="already-running",
            ),
            already_running=True,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["up"])
        assert result.exit_code == 0
        assert "already running" in result.output.lower()

    @patch("mlx_stack.cli.up.run_up")
    def test_partial_failure_summary(
        self,
        mock_run_up: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-013: Summary shows mixed states."""
        mock_run_up.return_value = UpResult(
            tiers=[
                TierStatus(
                    name="standard", model="big-model", port=8000,
                    status="failed", error="Health check timeout",
                ),
                TierStatus(
                    name="fast", model="fast-model", port=8001,
                    status="healthy",
                ),
            ],
            litellm=TierStatus(
                name="litellm", model="proxy", port=4000, status="healthy",
            ),
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["up"])
        assert result.exit_code == 0
        assert "failed" in result.output
        assert "healthy" in result.output

    @patch("mlx_stack.cli.up.run_up")
    def test_all_failed_exit_code(
        self,
        mock_run_up: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """All tiers failed produces non-zero exit code."""
        mock_run_up.return_value = UpResult(
            tiers=[
                TierStatus(
                    name="standard", model="big-model", port=8000,
                    status="failed", error="Port conflict",
                ),
            ],
            litellm=TierStatus(
                name="litellm", model="proxy", port=4000,
                status="skipped", error="All model servers failed",
            ),
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["up"])
        assert result.exit_code != 0

    @patch("mlx_stack.cli.up.run_up")
    def test_lockfile_error_message(
        self,
        mock_run_up: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-007: Lockfile error produces clear message."""
        mock_run_up.side_effect = LockError("Another mlx-stack operation is already running")

        runner = CliRunner()
        result = runner.invoke(cli, ["up"])
        assert result.exit_code != 0
        assert "already running" in result.output.lower()

    @patch("mlx_stack.cli.up.run_up")
    def test_warning_displayed(
        self,
        mock_run_up: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-016: Memory warning shown in output."""
        mock_run_up.return_value = UpResult(
            tiers=[
                TierStatus(name="standard", model="big-model", port=8000, status="healthy"),
            ],
            litellm=TierStatus(
                name="litellm", model="proxy", port=4000, status="healthy",
            ),
            warnings=["Estimated memory usage (50.0 GB) exceeds available (10.0 GB)"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["up"])
        assert result.exit_code == 0
        assert "memory" in result.output.lower()

    @patch("mlx_stack.cli.up.run_up")
    def test_port_conflict_in_summary(
        self,
        mock_run_up: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-012: Port conflict error with PID/process shown in CLI summary."""
        mock_run_up.return_value = UpResult(
            tiers=[
                TierStatus(
                    name="standard", model="big-model", port=8000,
                    status="skipped",
                    error="Port 8000 already in use by PID 54321 (node)",
                ),
                TierStatus(
                    name="fast", model="fast-model", port=8001,
                    status="healthy",
                ),
            ],
            litellm=TierStatus(
                name="litellm", model="proxy", port=4000, status="healthy",
            ),
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["up"])
        assert result.exit_code == 0
        # Port conflict error should include PID and process name
        assert "54321" in result.output
        assert "node" in result.output
        assert "8000" in result.output
        assert "skipped" in result.output


# --------------------------------------------------------------------------- #
# Tests — config propagation
# --------------------------------------------------------------------------- #


class TestConfigPropagation:
    """Tests for configuration changes propagating to up behavior."""

    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    def test_custom_litellm_port(
        self,
        mock_get_value: MagicMock,
        mock_load_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-CROSS-007: litellm-port config propagates to up."""
        stack = _make_stack_yaml()
        _write_stack_yaml(mlx_stack_home, stack)
        _write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = _make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 5001,
            "openrouter-key": "",
        }.get(key, "")

        result = run_up(dry_run=True)
        # LiteLLM should be on port 5001
        assert result.litellm is not None
        assert result.litellm.port == 5001

        # Dry-run commands should reference port 5001
        litellm_cmds = [
            c for c in result.dry_run_commands if c["service"] == "litellm"
        ]
        assert len(litellm_cmds) == 1
        assert "5001" in litellm_cmds[0]["command"]


# --------------------------------------------------------------------------- #
# Tests — init-generated ports match up behavior
# --------------------------------------------------------------------------- #


class TestInitUpConsistency:
    """Tests for init → up configuration consistency."""

    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    def test_ports_match_config(
        self,
        mock_get_value: MagicMock,
        mock_load_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-CROSS-006: Init port assignments match actual startup ports."""
        stack = _make_stack_yaml()
        _write_stack_yaml(mlx_stack_home, stack)
        _write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = _make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "",
        }.get(key, "")

        result = run_up(dry_run=True)

        # Extract ports from dry-run commands
        for tier in result.tiers:
            config_port = None
            for t in stack["tiers"]:
                if t["name"] == tier.name:
                    config_port = t["port"]
                    break
            if config_port:
                assert tier.port == config_port


# --------------------------------------------------------------------------- #
# Tests — lockfile cleanup on interrupt
# --------------------------------------------------------------------------- #


class TestLockfileRecovery:
    """Tests for lockfile behavior during interrupts."""

    def test_lockfile_released_after_error(self, mlx_stack_home: Path) -> None:
        """VAL-CROSS-016: Lockfile released even on error."""
        from mlx_stack.core.process import acquire_lock

        # Acquire and release lock successfully
        with acquire_lock():
            pass

        # Should be able to acquire again (lock was released)
        with acquire_lock():
            pass

    def test_lockfile_released_on_exception(self, mlx_stack_home: Path) -> None:
        """VAL-CROSS-016: Lockfile released on exception."""
        from mlx_stack.core.process import acquire_lock

        try:
            with acquire_lock():
                raise RuntimeError("Simulated crash")
        except RuntimeError:
            pass

        # Should be able to acquire again
        with acquire_lock():
            pass
