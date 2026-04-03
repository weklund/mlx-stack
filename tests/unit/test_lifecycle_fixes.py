"""Tests for lifecycle preflight checks and read-only data-home gating.

Validates two scrutiny fixes:
1. up-command preflight local-model existence checks per tier before launch.
   Missing models emit a diagnostic with pull suggestion and skip the tier.
2. Read-only commands (status, recommend, models, config get/list, bench)
   do NOT create ~/.mlx-stack/ if it does not exist. State-writing commands
   (profile, config set, init, pull, up) still auto-create it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

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
from mlx_stack.core.process import HealthCheckResult, ServiceInfo
from mlx_stack.core.stack_up import (
    TierStatus,
    UpResult,
    check_local_model_exists,
    run_up,
)

# --------------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------------- #


def _make_stack_yaml(
    tiers: list[dict[str, Any]] | None = None,
    schema_version: int = 1,
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


# =========================================================================== #
# Issue 1: Preflight local-model existence checks
# =========================================================================== #


class TestCheckLocalModelExists:
    """Unit tests for the check_local_model_exists function."""

    def test_model_found_by_id(self, mlx_stack_home: Path) -> None:
        """Model found when directory matches model ID."""
        models_dir = mlx_stack_home / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "big-model").mkdir()

        tier = {"model": "big-model", "source": "mlx-community/big-model-4bit"}
        assert check_local_model_exists(tier) is None

    def test_model_found_by_source_dir(self, mlx_stack_home: Path) -> None:
        """Model found when directory matches source repo name."""
        models_dir = mlx_stack_home / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "big-model-4bit").mkdir()

        tier = {"model": "big-model", "source": "mlx-community/big-model-4bit"}
        assert check_local_model_exists(tier) is None

    def test_model_missing_returns_diagnostic(self, mlx_stack_home: Path) -> None:
        """Missing model returns error message with pull suggestion."""
        models_dir = mlx_stack_home / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        tier = {
            "model": "missing-model",
            "source": "mlx-community/missing-model-4bit",
        }
        error = check_local_model_exists(tier)
        assert error is not None
        assert "missing-model" in error
        assert "mlx-stack pull" in error

    def test_model_missing_no_models_dir(self, mlx_stack_home: Path) -> None:
        """Missing model when models directory doesn't exist."""
        tier = {"model": "any-model", "source": "mlx-community/any-model-4bit"}
        error = check_local_model_exists(tier)
        assert error is not None
        assert "mlx-stack pull" in error

    def test_model_found_empty_source(self, mlx_stack_home: Path) -> None:
        """Model found by ID even when source is empty."""
        models_dir = mlx_stack_home / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "my-model").mkdir()

        tier = {"model": "my-model", "source": ""}
        assert check_local_model_exists(tier) is None


class TestPreflightInStartup:
    """Tests that the preflight check is integrated into the startup flow."""

    @patch("mlx_stack.core.stack_up.start_service")
    @patch("mlx_stack.core.stack_up.wait_for_healthy")
    @patch("mlx_stack.core.stack_up.check_port_conflict", return_value=None)
    @patch("mlx_stack.core.stack_up.read_pid_file", return_value=None)
    @patch("mlx_stack.core.stack_up.acquire_lock")
    @patch("mlx_stack.core.stack_up.ensure_dependency")
    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    @patch("mlx_stack.core.stack_up.shutil.which")
    def test_missing_model_skips_tier(
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
        mlx_stack_home: Path,
    ) -> None:
        """Tier with missing model is skipped with pull suggestion."""
        _write_stack_yaml(mlx_stack_home)
        _write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = _make_test_catalog()
        models_dir = mlx_stack_home / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "",
            "model-dir": str(models_dir),
        }.get(key, "")
        mock_which.side_effect = lambda x: f"/usr/local/bin/{x}"
        mock_lock.return_value.__enter__ = MagicMock(return_value=None)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        mock_start_service.return_value = ServiceInfo(
            name="test",
            pid=12345,
            port=8000,
            log_path=Path("/tmp/test.log"),
            pid_path=Path("/tmp/test.pid"),
        )
        mock_wait_healthy.return_value = HealthCheckResult(
            healthy=True, response_time=0.5, status_code=200
        )

        # No models on disk → both tiers should be skipped
        result = run_up()

        # All tiers should be skipped with missing model message
        skipped_tiers = [t for t in result.tiers if t.status == "skipped"]
        assert len(skipped_tiers) == 2
        for tier in skipped_tiers:
            assert "not found locally" in (tier.error or "")
            assert "mlx-stack pull" in (tier.error or "")

        # LiteLLM should be skipped since no tiers are healthy
        assert result.litellm is not None
        assert result.litellm.status == "skipped"

    @patch("mlx_stack.core.stack_up.start_service")
    @patch("mlx_stack.core.stack_up.wait_for_healthy")
    @patch("mlx_stack.core.stack_up.check_port_conflict", return_value=None)
    @patch("mlx_stack.core.stack_up.read_pid_file", return_value=None)
    @patch("mlx_stack.core.stack_up.acquire_lock")
    @patch("mlx_stack.core.stack_up.ensure_dependency")
    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    @patch("mlx_stack.core.stack_up.shutil.which")
    def test_partial_models_present(
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
        mlx_stack_home: Path,
    ) -> None:
        """One model present, one missing → mixed results."""
        _write_stack_yaml(mlx_stack_home)
        _write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = _make_test_catalog()
        models_dir = mlx_stack_home / "models"
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "",
            "model-dir": str(models_dir),
        }.get(key, "")
        mock_which.side_effect = lambda x: f"/usr/local/bin/{x}"
        mock_lock.return_value.__enter__ = MagicMock(return_value=None)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        mock_start_service.return_value = ServiceInfo(
            name="test",
            pid=12345,
            port=8001,
            log_path=Path("/tmp/test.log"),
            pid_path=Path("/tmp/test.pid"),
        )
        mock_wait_healthy.return_value = HealthCheckResult(
            healthy=True, response_time=0.5, status_code=200
        )

        # Create model directory for "fast-model" source
        models_dir = mlx_stack_home / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "fast-model-4bit").mkdir()

        result = run_up()

        statuses = {t.name: t.status for t in result.tiers}
        # big-model (standard) should be skipped — not on disk
        assert statuses["standard"] == "skipped"
        message = next(t.error for t in result.tiers if t.name == "standard") or ""
        assert "not found locally" in message
        # fast-model (fast) should be healthy — on disk
        assert statuses["fast"] == "healthy"

        # LiteLLM should start since at least one tier is healthy
        assert result.litellm is not None
        assert result.litellm.status == "healthy"

    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    def test_dry_run_shows_missing_model_warning(
        self,
        mock_get_value: MagicMock,
        mock_load_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """Dry-run still shows commands even for missing models."""
        _write_stack_yaml(mlx_stack_home)
        _write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = _make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "",
        }.get(key, "")

        # No models on disk — dry-run still shows commands
        runner = CliRunner()
        result = runner.invoke(cli, ["up", "--dry-run"])
        # Dry-run should succeed — it doesn't check for model existence
        assert result.exit_code == 0
        assert "Dry run" in result.output

    def test_cli_missing_model_shows_pull_suggestion(
        self,
        mlx_stack_home: Path,
    ) -> None:
        """CLI output shows pull suggestion for missing models."""
        _write_stack_yaml(mlx_stack_home)
        _write_litellm_yaml(mlx_stack_home)

        runner = CliRunner()

        with (
            patch("mlx_stack.cli.up.run_up") as mock_run_up,
        ):
            mock_run_up.return_value = UpResult(
                tiers=[
                    TierStatus(
                        name="standard",
                        model="big-model",
                        port=8000,
                        status="skipped",
                        error="Model 'big-model' not found locally. "
                        "Run 'mlx-stack pull big-model' to download it.",
                    ),
                ],
                litellm=TierStatus(
                    name="litellm",
                    model="proxy",
                    port=4000,
                    status="skipped",
                    error="All model servers failed; LiteLLM not started.",
                ),
            )
            result = runner.invoke(cli, ["up"])

        assert result.exit_code != 0
        assert "pull" in result.output.lower()


# =========================================================================== #
# Issue 2: Read-only data-home gating
# =========================================================================== #


class TestReadOnlyNoDataHomeCreation:
    """Tests that read-only commands do NOT create ~/.mlx-stack/."""

    def test_status_no_create(self, clean_mlx_stack_home: Path) -> None:
        """status command does not create ~/.mlx-stack/."""
        assert not clean_mlx_stack_home.exists()
        runner = CliRunner()
        result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert not clean_mlx_stack_home.exists()

    def test_status_json_no_create(self, clean_mlx_stack_home: Path) -> None:
        """status --json does not create ~/.mlx-stack/."""
        assert not clean_mlx_stack_home.exists()
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--json"])
        assert result.exit_code == 0
        assert not clean_mlx_stack_home.exists()

    def test_recommend_no_create(self, clean_mlx_stack_home: Path) -> None:
        """recommend command does not create ~/.mlx-stack/.

        Note: recommend may fail (no profile), but the directory should
        still not be created.
        """
        assert not clean_mlx_stack_home.exists()
        runner = CliRunner()
        # This will likely fail with no hardware profile, but that's OK
        runner.invoke(cli, ["recommend"])
        # Regardless of success/failure, directory should not be created
        assert not clean_mlx_stack_home.exists()

    def test_models_no_create(self, clean_mlx_stack_home: Path) -> None:
        """models command does not create ~/.mlx-stack/."""
        assert not clean_mlx_stack_home.exists()
        runner = CliRunner()
        runner.invoke(cli, ["models"])
        assert not clean_mlx_stack_home.exists()

    def test_config_get_no_create(self, clean_mlx_stack_home: Path) -> None:
        """config get does not create ~/.mlx-stack/."""
        assert not clean_mlx_stack_home.exists()
        runner = CliRunner()
        runner.invoke(cli, ["config", "get", "default-quant"])
        assert not clean_mlx_stack_home.exists()

    def test_config_list_no_create(self, clean_mlx_stack_home: Path) -> None:
        """config list does not create ~/.mlx-stack/."""
        assert not clean_mlx_stack_home.exists()
        runner = CliRunner()
        runner.invoke(cli, ["config", "list"])
        assert not clean_mlx_stack_home.exists()

    def test_help_no_create(self, clean_mlx_stack_home: Path) -> None:
        """--help does not create ~/.mlx-stack/."""
        assert not clean_mlx_stack_home.exists()
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert not clean_mlx_stack_home.exists()

    def test_version_no_create(self, clean_mlx_stack_home: Path) -> None:
        """--version does not create ~/.mlx-stack/."""
        assert not clean_mlx_stack_home.exists()
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert not clean_mlx_stack_home.exists()

    def test_bench_no_create(self, clean_mlx_stack_home: Path) -> None:
        """bench (placeholder, no --save) does not create ~/.mlx-stack/."""
        assert not clean_mlx_stack_home.exists()
        runner = CliRunner()
        # bench is a placeholder returning exit code 1, but
        # it should still not create the directory
        runner.invoke(cli, ["bench"])
        assert not clean_mlx_stack_home.exists()


class TestStateWritingCommandsStillCreateDataHome:
    """Tests that state-writing commands still auto-create ~/.mlx-stack/."""

    def test_config_set_creates_dir(self, clean_mlx_stack_home: Path) -> None:
        """config set creates ~/.mlx-stack/ (needs it to store config)."""
        assert not clean_mlx_stack_home.exists()
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "set", "default-quant", "int4"])
        assert result.exit_code == 0
        assert clean_mlx_stack_home.exists()

    def test_profile_creates_dir(self, clean_mlx_stack_home: Path) -> None:
        """profile command creates ~/.mlx-stack/ (needs it to store profile).

        Profile calls save_profile which calls ensure_data_home internally.
        """
        assert not clean_mlx_stack_home.exists()
        runner = CliRunner()
        # Profile needs real hardware — mock detect, but let save run real
        with (
            patch("mlx_stack.cli.profile.detect_hardware") as mock_detect,
        ):
            from mlx_stack.core.hardware import HardwareProfile

            mock_detect.return_value = HardwareProfile(
                chip="Apple M5 Max",
                gpu_cores=40,
                memory_gb=128,
                bandwidth_gbps=546.0,
                is_estimate=False,
            )
            runner.invoke(cli, ["profile"])
        # Profile command should create the data directory
        assert clean_mlx_stack_home.exists()
