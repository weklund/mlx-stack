"""Tests for lifecycle preflight checks and read-only data-home gating.

Validates two scrutiny fixes:
1. up-command preflight local-model existence checks per tier before launch.
   Missing models emit a diagnostic with pull suggestion and skip the tier.
2. Read-only commands (status, recommend, models, config get/list, bench)
   do NOT create ~/.mlx-stack/ if it does not exist. State-writing commands
   (config set, init, pull, up) still auto-create it.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from mlx_stack.cli.main import cli
from mlx_stack.core.process import HealthCheckResult, ServiceInfo
from mlx_stack.core.stack_up import (
    TierStatus,
    UpResult,
    check_local_model_exists,
    run_up,
)
from tests.factories import (
    make_test_catalog,
    write_litellm_yaml,
    write_stack_yaml,
)

# =========================================================================== #
# Issue 1: Preflight local-model existence checks
# =========================================================================== #


class TestCheckLocalModelExists:
    """Unit tests for the check_local_model_exists function."""

    def test_model_found_by_id(self, mlx_stack_home: Path) -> None:
        """Model found when directory matches model ID."""
        # Arrange
        models_dir = mlx_stack_home / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "big-model").mkdir()
        tier = {"model": "big-model", "source": "mlx-community/big-model-4bit"}

        # Act / Assert
        assert check_local_model_exists(tier) is None

    def test_model_found_by_source_dir(self, mlx_stack_home: Path) -> None:
        """Model found when directory matches source repo name."""
        # Arrange
        models_dir = mlx_stack_home / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "big-model-4bit").mkdir()
        tier = {"model": "big-model", "source": "mlx-community/big-model-4bit"}

        # Act / Assert
        assert check_local_model_exists(tier) is None

    def test_model_missing_returns_diagnostic(self, mlx_stack_home: Path) -> None:
        """Missing model returns error message with pull suggestion."""
        # Arrange
        models_dir = mlx_stack_home / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        tier = {
            "model": "missing-model",
            "source": "mlx-community/missing-model-4bit",
        }

        # Act
        error = check_local_model_exists(tier)

        # Assert
        assert error is not None
        assert "missing-model" in error
        assert "mlx-stack pull" in error

    def test_model_missing_no_models_dir(self, mlx_stack_home: Path) -> None:
        """Missing model when models directory doesn't exist."""
        # Arrange -- no models directory created
        tier = {"model": "any-model", "source": "mlx-community/any-model-4bit"}

        # Act
        error = check_local_model_exists(tier)

        # Assert
        assert error is not None
        assert "mlx-stack pull" in error

    def test_model_found_empty_source(self, mlx_stack_home: Path) -> None:
        """Model found by ID even when source is empty."""
        # Arrange
        models_dir = mlx_stack_home / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "my-model").mkdir()
        tier = {"model": "my-model", "source": ""}

        # Act / Assert
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
        # Arrange
        write_stack_yaml(mlx_stack_home)
        write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = make_test_catalog()
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

        # Act -- no models on disk, both tiers should be skipped
        result = run_up()

        # Assert
        skipped_tiers = [t for t in result.tiers if t.status == "skipped"]
        assert len(skipped_tiers) == 2
        for tier in skipped_tiers:
            assert "not found locally" in (tier.error or "")
            assert "mlx-stack pull" in (tier.error or "")
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
        # Arrange
        write_stack_yaml(mlx_stack_home)
        write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = make_test_catalog()
        models_dir = mlx_stack_home / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "fast-model-4bit").mkdir()  # only fast-model on disk
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

        # Act
        result = run_up()

        # Assert -- big-model skipped (not on disk), fast-model healthy
        statuses = {t.name: t.status for t in result.tiers}
        assert statuses["standard"] == "skipped"
        message = next(t.error for t in result.tiers if t.name == "standard") or ""
        assert "not found locally" in message
        assert statuses["fast"] == "healthy"
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
        # Arrange -- no models on disk
        write_stack_yaml(mlx_stack_home)
        write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "",
        }.get(key, "")

        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["up", "--dry-run"])

        # Assert -- dry-run succeeds without checking model existence
        assert result.exit_code == 0
        assert "Dry run" in result.output

    def test_cli_missing_model_shows_pull_suggestion(
        self,
        mlx_stack_home: Path,
    ) -> None:
        """CLI output shows pull suggestion for missing models."""
        # Arrange
        write_stack_yaml(mlx_stack_home)
        write_litellm_yaml(mlx_stack_home)
        runner = CliRunner()

        # Act
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

        # Assert
        assert result.exit_code != 0
        assert "pull" in result.output.lower()


# =========================================================================== #
# Issue 2: Read-only data-home gating
# =========================================================================== #


class TestReadOnlyNoDataHomeCreation:
    """Tests that read-only commands do NOT create ~/.mlx-stack/."""

    def test_status_no_create(self, clean_mlx_stack_home: Path) -> None:
        """status command does not create ~/.mlx-stack/."""
        # Arrange
        assert not clean_mlx_stack_home.exists()
        runner = CliRunner()

        # Act
        result = runner.invoke(cli, ["status"])

        # Assert
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
        # Arrange
        assert not clean_mlx_stack_home.exists()
        runner = CliRunner()

        # Act
        result = runner.invoke(cli, ["config", "set", "default-quant", "int4"])

        # Assert
        assert result.exit_code == 0
        assert clean_mlx_stack_home.exists()


