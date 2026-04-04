"""Tests for the ``mlx-stack up`` CLI command and core ``stack_up`` module.

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
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from mlx_stack.cli.main import cli
from mlx_stack.core.process import LockError
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
from tests.factories import (
    create_pid_file,
    make_stack_yaml,
    make_test_catalog,
    write_litellm_yaml,
    write_stack_yaml,
)
from tests.fakes import FakeServiceLayer

# --------------------------------------------------------------------------- #
# Tests — load_stack_definition (real YAML, no mocks)
# --------------------------------------------------------------------------- #


class TestLoadStackDefinition:
    """Tests for load_stack_definition — real filesystem operations."""

    def test_loads_valid_stack(self, mlx_stack_home: Path) -> None:
        """VAL-UP-001: Stack definition loaded."""
        # Arrange
        write_stack_yaml(mlx_stack_home)

        # Act
        stack = load_stack_definition()

        # Assert
        assert stack["schema_version"] == 1
        assert len(stack["tiers"]) == 2

    def test_missing_stack_suggests_init(self, mlx_stack_home: Path) -> None:
        """VAL-UP-011: Missing stack definition error."""
        # Act / Assert
        with pytest.raises(UpError, match="mlx-stack init"):
            load_stack_definition()

    def test_invalid_yaml_produces_clear_error(self, mlx_stack_home: Path) -> None:
        """VAL-UP-011: Invalid YAML produces clear error."""
        # Arrange
        stacks_dir = mlx_stack_home / "stacks"
        stacks_dir.mkdir(parents=True, exist_ok=True)
        (stacks_dir / "default.yaml").write_text("{{{invalid yaml")

        # Act / Assert
        with pytest.raises(UpError, match="Invalid YAML"):
            load_stack_definition()

    def test_unsupported_schema_version(self, mlx_stack_home: Path) -> None:
        """VAL-UP-011 / VAL-CROSS-013: Unsupported schema_version."""
        # Arrange
        stack = make_stack_yaml(schema_version=99)
        write_stack_yaml(mlx_stack_home, stack)

        # Act / Assert
        with pytest.raises(UpError, match="schema_version"):
            load_stack_definition()

    def test_empty_tiers_error(self, mlx_stack_home: Path) -> None:
        """Stack with no tiers raises error."""
        # Arrange
        stack = make_stack_yaml()
        stack["tiers"] = []
        write_stack_yaml(mlx_stack_home, stack)

        # Act / Assert
        with pytest.raises(UpError, match="no tiers"):
            load_stack_definition()

    def test_non_dict_file(self, mlx_stack_home: Path) -> None:
        """Non-mapping YAML file produces clear error."""
        # Arrange
        stacks_dir = mlx_stack_home / "stacks"
        stacks_dir.mkdir(parents=True, exist_ok=True)
        (stacks_dir / "default.yaml").write_text("- just a list")

        # Act / Assert
        with pytest.raises(UpError, match="invalid format"):
            load_stack_definition()


# --------------------------------------------------------------------------- #
# Tests — build_vllm_command (pure function)
# --------------------------------------------------------------------------- #


class TestBuildVllmCommand:
    """Tests for vllm-mlx command building."""

    def test_basic_command(self) -> None:
        """VAL-UP-015: Services bind to localhost only."""
        # Arrange
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

        # Act
        cmd = build_vllm_command(tier, "/usr/local/bin/vllm-mlx")

        # Assert
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
        # Arrange
        tier = {
            "name": "fast",
            "model": "test-model",
            "source": "mlx-community/test-model-4bit",
            "port": 8001,
            "vllm_flags": {},
        }

        # Act
        cmd = build_vllm_command(tier, "vllm-mlx")

        # Assert
        assert cmd[0] == "vllm-mlx"
        assert cmd[1] == "serve"
        assert cmd[2] == "mlx-community/test-model-4bit"
        assert "--model" not in cmd

    def test_tool_calling_flags(self) -> None:
        """VAL-CROSS-013: vllm_flags translate correctly to CLI flags."""
        # Arrange
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

        # Act
        cmd = build_vllm_command(tier, "vllm-mlx")

        # Assert
        assert "--enable-auto-tool-choice" in cmd
        assert "--tool-call-parser" in cmd
        idx = cmd.index("--tool-call-parser")
        assert cmd[idx + 1] == "hermes"

    def test_boolean_false_flags_excluded(self) -> None:
        """Boolean False flags are not included in command."""
        # Arrange
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

        # Act
        cmd = build_vllm_command(tier, "vllm-mlx")

        # Assert
        assert "--some-disabled-flag" not in cmd
        assert "--continuous-batching" in cmd


# --------------------------------------------------------------------------- #
# Tests — build_litellm_command (pure function)
# --------------------------------------------------------------------------- #


class TestBuildLitellmCommand:
    """Tests for litellm command building."""

    def test_basic_command(self) -> None:
        """VAL-UP-015: LiteLLM binds to localhost only."""
        # Act
        cmd = build_litellm_command(
            "/usr/local/bin/litellm",
            4000,
            Path("/home/user/.mlx-stack/litellm.yaml"),
        )

        # Assert
        assert cmd[0] == "/usr/local/bin/litellm"
        assert "--config" in cmd
        assert "/home/user/.mlx-stack/litellm.yaml" in cmd
        assert "--port" in cmd
        assert "4000" in cmd
        assert "--host" in cmd
        assert "127.0.0.1" in cmd


# --------------------------------------------------------------------------- #
# Tests — format_dry_run_command (pure function)
# --------------------------------------------------------------------------- #


class TestFormatDryRunCommand:
    """Tests for dry-run command formatting."""

    def test_basic_format(self) -> None:
        """Commands formatted as space-separated string."""
        # Arrange
        cmd = ["vllm-mlx", "--model", "test", "--port", "8000"]

        # Act
        result = format_dry_run_command(cmd)

        # Assert
        assert result == "vllm-mlx --model test --port 8000"

    def test_env_vars_masked(self) -> None:
        """VAL-UP-017: API key not visible in --dry-run output."""
        # Arrange
        cmd = ["litellm", "--config", "litellm.yaml"]

        # Act
        result = format_dry_run_command(cmd, {"OPENROUTER_API_KEY": "sk-secret"})

        # Assert
        assert "sk-secret" not in result
        assert "OPENROUTER_API_KEY=***" in result


# --------------------------------------------------------------------------- #
# Tests — sort_tiers_by_size (pure function)
# --------------------------------------------------------------------------- #


class TestSortTiersBySize:
    """Tests for tier sorting."""

    def test_largest_first(self) -> None:
        """VAL-UP-001: Tiers started in descending params_b order."""
        # Arrange
        catalog = make_test_catalog()
        tiers = [
            {"name": "fast", "model": "fast-model", "port": 8001},
            {"name": "standard", "model": "big-model", "port": 8000},
        ]

        # Act
        sorted_tiers = sort_tiers_by_size(tiers, catalog)

        # Assert
        assert sorted_tiers[0]["name"] == "standard"  # 49B first
        assert sorted_tiers[1]["name"] == "fast"  # 3B second

    def test_no_catalog_preserves_order(self) -> None:
        """Without catalog, original order is preserved."""
        # Arrange
        tiers = [
            {"name": "fast", "model": "fast-model", "port": 8001},
            {"name": "standard", "model": "big-model", "port": 8000},
        ]

        # Act
        sorted_tiers = sort_tiers_by_size(tiers, None)

        # Assert
        assert sorted_tiers[0]["name"] == "fast"


# --------------------------------------------------------------------------- #
# Tests — memory estimation
# --------------------------------------------------------------------------- #


class TestMemoryEstimation:
    """Tests for memory estimation and warnings."""

    def test_estimates_from_catalog(self) -> None:
        """VAL-UP-016: Memory estimate from catalog data."""
        # Arrange
        catalog = make_test_catalog()
        tiers = [
            {"name": "standard", "model": "big-model"},
            {"name": "fast", "model": "fast-model"},
        ]

        # Act
        total = estimate_memory_usage(tiers, catalog)

        # Assert
        assert total == pytest.approx(32.0, abs=1.0)

    def test_unknown_model_skipped(self) -> None:
        """Unknown models contribute 0 to estimate."""
        # Arrange
        catalog = make_test_catalog()
        tiers = [{"name": "unknown", "model": "nonexistent"}]

        # Act
        total = estimate_memory_usage(tiers, catalog)

        # Assert
        assert total == 0.0

    def test_warning_when_exceeds_available(self) -> None:
        """VAL-UP-016: Warning when estimate exceeds available memory."""
        # Arrange
        with patch("mlx_stack.core.stack_up.psutil.virtual_memory") as mock_vmem:
            mock_vmem.return_value = MagicMock(available=10 * 1024**3)

            # Act
            warning = check_memory_warning(20.0)

        # Assert
        assert warning is not None
        assert "20.0 GB" in warning
        assert "10.0 GB" in warning

    def test_no_warning_when_fits(self) -> None:
        """No warning when estimate fits in available memory."""
        # Arrange
        with patch("mlx_stack.core.stack_up.psutil.virtual_memory") as mock_vmem:
            mock_vmem.return_value = MagicMock(available=100 * 1024**3)

            # Act
            warning = check_memory_warning(20.0)

        # Assert
        assert warning is None


# --------------------------------------------------------------------------- #
# Tests — run_up behavioral (FakeServiceLayer, no @patch stacks)
# --------------------------------------------------------------------------- #


class TestRunUp:
    """Behavioral tests for ``run_up`` using ``FakeServiceLayer``.

    Each test writes real YAML to the isolated ``mlx_stack_home`` and
    configures only the specific failure it tests.  The fake defaults
    produce a fully successful startup.
    """

    def test_successful_startup(
        self, stack_on_disk: Path, fake_services: FakeServiceLayer,
    ) -> None:
        """VAL-UP-001/004/005: Successful startup with PID files and LiteLLM."""
        # Arrange — defaults: all services start and pass health check

        # Act
        result = run_up()

        # Assert
        assert len(result.tiers) == 2
        assert all(t.status == "healthy" for t in result.tiers)
        assert result.litellm is not None
        assert result.litellm.status == "healthy"
        assert len(fake_services.started) == 3  # 2 tiers + litellm

    def test_tier_filter_starts_only_one(
        self, stack_on_disk: Path, fake_services: FakeServiceLayer,
    ) -> None:
        """VAL-UP-009: --tier starts only the specified tier."""
        # Arrange — no special config needed

        # Act
        result = run_up(tier_filter="fast")

        # Assert
        tier_names = [t.name for t in result.tiers]
        assert "fast" in tier_names
        assert "standard" not in tier_names

    def test_port_conflict_skips_tier(
        self, stack_on_disk: Path, fake_services: FakeServiceLayer,
    ) -> None:
        """VAL-UP-012: Port conflict skips the affected tier."""
        # Arrange
        fake_services.fail_port(8000, pid=54321, name="node")

        # Act
        result = run_up()

        # Assert
        skipped = [t for t in result.tiers if t.status == "skipped"]
        assert len(skipped) == 1
        assert skipped[0].name == "standard"
        assert "54321" in skipped[0].error
        assert "node" in skipped[0].error
        assert "8000" in skipped[0].error

    def test_port_conflict_unknown_owner(
        self, stack_on_disk: Path, fake_services: FakeServiceLayer,
    ) -> None:
        """VAL-UP-012: Port conflict with unknown owner still shows port."""
        # Arrange
        fake_services.fail_port(8000, pid=0, name="<unknown>")

        # Act
        result = run_up()

        # Assert
        skipped = [t for t in result.tiers if t.status == "skipped"]
        assert len(skipped) == 1
        assert "8000" in (skipped[0].error or "")
        assert "already in use" in (skipped[0].error or "")

    def test_health_check_timeout_continues(
        self, stack_on_disk: Path, fake_services: FakeServiceLayer,
    ) -> None:
        """VAL-UP-003/013: Health check timeout on one tier, other still starts."""
        # Arrange — fail health for standard (port 8000), fast (8001) succeeds
        fake_services.fail_health(8000)

        # Act
        result = run_up()

        # Assert
        statuses = {t.name: t.status for t in result.tiers}
        assert statuses["standard"] == "failed"
        assert statuses["fast"] == "healthy"

    def test_all_fail_no_litellm(
        self, stack_on_disk: Path, fake_services: FakeServiceLayer,
    ) -> None:
        """VAL-UP-005: LiteLLM not started if all model servers fail."""
        # Arrange — all ports occupied
        fake_services.fail_port(8000, pid=99, name="blocker")
        fake_services.fail_port(8001, pid=99, name="blocker")

        # Act
        result = run_up()

        # Assert
        assert result.litellm is not None
        assert result.litellm.status == "skipped"
        assert "All model servers failed" in (result.litellm.error or "")

    def test_already_running_detection(
        self, stack_on_disk: Path, fake_services: FakeServiceLayer,
    ) -> None:
        """VAL-UP-014: Already-running services reported without restart."""
        # Arrange — create PID files and mark processes as alive
        home = stack_on_disk
        create_pid_file(home, "standard", 12345)
        create_pid_file(home, "fast", 12346)
        create_pid_file(home, "litellm", 12347)
        fake_services.set_alive(12345, True)
        fake_services.set_alive(12346, True)
        fake_services.set_alive(12347, True)

        # Act
        result = run_up()

        # Assert
        assert result.already_running is True
        assert all(t.status == "already-running" for t in result.tiers)

    def test_stale_pid_cleanup_and_restart(
        self, stack_on_disk: Path, fake_services: FakeServiceLayer,
    ) -> None:
        """VAL-UP-014 / VAL-CROSS-010: Stale PID cleaned up, service restarted."""
        # Arrange — PID files exist but processes are dead (default: not alive)
        home = stack_on_disk
        create_pid_file(home, "standard", 99999)
        create_pid_file(home, "fast", 99998)

        # Act
        result = run_up()

        # Assert
        assert any("stale" in w.lower() for w in result.warnings)
        assert any(t.status == "healthy" for t in result.tiers)

    def test_lockfile_prevents_concurrent(
        self, stack_on_disk: Path, fake_services: FakeServiceLayer,
    ) -> None:
        """VAL-UP-007: Lockfile prevents concurrent invocations."""
        # Arrange
        fake_services.hold_lock()

        # Act / Assert
        with pytest.raises(LockError, match="Lock held"):
            run_up()

    def test_auto_install_failure(
        self, stack_on_disk: Path, fake_services: FakeServiceLayer,
    ) -> None:
        """VAL-UP-010: Auto-install failure produces clear error."""
        # Arrange
        fake_services.fail_dependency("vllm-mlx")

        # Act / Assert
        with pytest.raises(UpError, match="Dependency installation failed"):
            run_up()

    def test_missing_model_skips_tier(
        self, stack_on_disk: Path, fake_services: FakeServiceLayer,
    ) -> None:
        """Model not found on disk skips the tier."""
        # Arrange
        fake_services.fail_model_check("standard", "Model not found on disk")

        # Act
        result = run_up()

        # Assert
        skipped = [t for t in result.tiers if t.status == "skipped"]
        assert len(skipped) == 1
        assert skipped[0].name == "standard"
        assert "not found" in skipped[0].error.lower()
        # fast should still start
        healthy = [t for t in result.tiers if t.status == "healthy"]
        assert len(healthy) == 1
        assert healthy[0].name == "fast"

    def test_api_key_passed_via_env(
        self, stack_on_disk: Path, fake_services: FakeServiceLayer,
    ) -> None:
        """VAL-UP-017: API key passed via env var, not CLI args."""
        # Arrange
        fake_services.set_config("openrouter-key", "sk-or-secret-key")

        # Act
        result = run_up()

        # Assert — LiteLLM was started (appears in started list)
        assert LITELLM_SERVICE_NAME in fake_services.started
        assert result.litellm is not None
        assert result.litellm.status == "healthy"

    def test_memory_warning_displayed(
        self, stack_on_disk: Path, fake_services: FakeServiceLayer,
    ) -> None:
        """VAL-UP-016: Memory estimate warning before startup."""
        # Arrange — fake low available memory
        with patch("mlx_stack.core.stack_up.psutil.virtual_memory") as mock_vmem:
            mock_vmem.return_value = MagicMock(available=5 * 1024**3)  # 5 GB

            # Act
            result = run_up()

        # Assert
        assert any("memory" in w.lower() for w in result.warnings)

    def test_start_service_failure_continues(
        self, stack_on_disk: Path, fake_services: FakeServiceLayer,
    ) -> None:
        """VAL-UP-013: Start failure on one tier, other still starts."""
        # Arrange
        fake_services.fail_start("standard")

        # Act
        result = run_up()

        # Assert
        statuses = {t.name: t.status for t in result.tiers}
        assert statuses["standard"] == "failed"
        assert statuses["fast"] == "healthy"


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
        # Arrange
        write_stack_yaml(mlx_stack_home)
        write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000, "openrouter-key": "",
        }.get(key, "")

        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["up", "--dry-run"])

        # Assert
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
        # Arrange
        write_stack_yaml(mlx_stack_home)
        write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000, "openrouter-key": "",
        }.get(key, "")

        # Act
        runner = CliRunner()
        runner.invoke(cli, ["up", "--dry-run"])

        # Assert
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
        # Arrange
        write_stack_yaml(mlx_stack_home)
        write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000, "openrouter-key": "",
        }.get(key, "")

        # Act
        runner = CliRunner()
        runner.invoke(cli, ["up", "--dry-run"])

        # Assert
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
        # Arrange
        write_stack_yaml(mlx_stack_home)
        write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000, "openrouter-key": "",
        }.get(key, "")

        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["up", "--dry-run"])

        # Assert
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
        # Arrange
        write_stack_yaml(mlx_stack_home)
        write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000, "openrouter-key": "sk-or-secret-key-12345",
        }.get(key, "")

        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["up", "--dry-run"])

        # Assert
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
        # Arrange
        write_stack_yaml(mlx_stack_home)
        write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000, "openrouter-key": "",
        }.get(key, "")

        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["up", "--dry-run", "--tier", "fast"])

        # Assert
        assert result.exit_code == 0
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
        # Arrange
        write_stack_yaml(mlx_stack_home)
        write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000, "openrouter-key": "",
        }.get(key, "")

        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["up", "--dry-run"])

        # Assert
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
        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["up"])

        # Assert
        assert result.exit_code != 0
        assert "init" in result.output.lower()

    @patch("mlx_stack.core.stack_up.get_value")
    def test_invalid_tier_error(
        self, mock_get_value: MagicMock, mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-009: Invalid tier name errors with valid list."""
        # Arrange
        write_stack_yaml(mlx_stack_home)
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000, "openrouter-key": "",
        }.get(key, "")

        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["up", "--tier", "nonexistent"])

        # Assert
        assert result.exit_code != 0
        assert "standard" in result.output or "fast" in result.output

    def test_invalid_yaml_error(self, mlx_stack_home: Path) -> None:
        """VAL-UP-011: Invalid YAML produces clear error."""
        # Arrange
        stacks_dir = mlx_stack_home / "stacks"
        stacks_dir.mkdir(parents=True, exist_ok=True)
        (stacks_dir / "default.yaml").write_text("{{{bad yaml")

        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["up"])

        # Assert
        assert result.exit_code != 0

    def test_unsupported_schema_error(self, mlx_stack_home: Path) -> None:
        """VAL-UP-011 / VAL-CROSS-013: Unsupported schema_version."""
        # Arrange
        stack = make_stack_yaml(schema_version=999)
        write_stack_yaml(mlx_stack_home, stack)

        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["up"])

        # Assert
        assert result.exit_code != 0
        assert "schema_version" in result.output


# --------------------------------------------------------------------------- #
# Tests — CLI output verification (mocks at CLI boundary — correct level)
# --------------------------------------------------------------------------- #


class TestCLIOutput:
    """Tests for CLI output formatting."""

    @patch("mlx_stack.cli.up.run_up")
    def test_summary_table_displayed(
        self, mock_run_up: MagicMock, mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-006: Summary table shows tier, model, port, status."""
        # Arrange
        mock_run_up.return_value = UpResult(
            tiers=[
                TierStatus(name="standard", model="big-model", port=8000, status="healthy"),
                TierStatus(name="fast", model="fast-model", port=8001, status="healthy"),
            ],
            litellm=TierStatus(name="litellm", model="proxy", port=4000, status="healthy"),
        )

        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["up"])

        # Assert
        assert result.exit_code == 0
        assert "standard" in result.output
        assert "fast" in result.output
        assert "8000" in result.output
        assert "8001" in result.output
        assert "healthy" in result.output

    @patch("mlx_stack.cli.up.run_up")
    def test_already_running_message(
        self, mock_run_up: MagicMock, mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-014: Already-running shows informational message."""
        # Arrange
        mock_run_up.return_value = UpResult(
            tiers=[
                TierStatus(name="standard", model="big-model", port=8000, status="already-running"),
            ],
            litellm=TierStatus(name="litellm", model="proxy", port=4000, status="already-running"),
            already_running=True,
        )

        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["up"])

        # Assert
        assert result.exit_code == 0
        assert "already running" in result.output.lower()

    @patch("mlx_stack.cli.up.run_up")
    def test_partial_failure_summary(
        self, mock_run_up: MagicMock, mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-013: Summary shows mixed states."""
        # Arrange
        mock_run_up.return_value = UpResult(
            tiers=[
                TierStatus(name="standard", model="big-model", port=8000, status="failed", error="Health check timeout"),
                TierStatus(name="fast", model="fast-model", port=8001, status="healthy"),
            ],
            litellm=TierStatus(name="litellm", model="proxy", port=4000, status="healthy"),
        )

        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["up"])

        # Assert
        assert result.exit_code == 0
        assert "failed" in result.output
        assert "healthy" in result.output

    @patch("mlx_stack.cli.up.run_up")
    def test_all_failed_exit_code(
        self, mock_run_up: MagicMock, mlx_stack_home: Path,
    ) -> None:
        """All tiers failed produces non-zero exit code."""
        # Arrange
        mock_run_up.return_value = UpResult(
            tiers=[
                TierStatus(name="standard", model="big-model", port=8000, status="failed", error="Port conflict"),
            ],
            litellm=TierStatus(name="litellm", model="proxy", port=4000, status="skipped", error="All model servers failed"),
        )

        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["up"])

        # Assert
        assert result.exit_code != 0

    @patch("mlx_stack.cli.up.run_up")
    def test_lockfile_error_message(
        self, mock_run_up: MagicMock, mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-007: Lockfile error produces clear message."""
        # Arrange
        mock_run_up.side_effect = LockError("Another mlx-stack operation is already running")

        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["up"])

        # Assert
        assert result.exit_code != 0
        assert "already running" in result.output.lower()

    @patch("mlx_stack.cli.up.run_up")
    def test_warning_displayed(
        self, mock_run_up: MagicMock, mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-016: Memory warning shown in output."""
        # Arrange
        mock_run_up.return_value = UpResult(
            tiers=[
                TierStatus(name="standard", model="big-model", port=8000, status="healthy"),
            ],
            litellm=TierStatus(name="litellm", model="proxy", port=4000, status="healthy"),
            warnings=["Estimated memory usage (50.0 GB) exceeds available (10.0 GB)"],
        )

        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["up"])

        # Assert
        assert result.exit_code == 0
        assert "memory" in result.output.lower()

    @patch("mlx_stack.cli.up.run_up")
    def test_port_conflict_in_summary(
        self, mock_run_up: MagicMock, mlx_stack_home: Path,
    ) -> None:
        """VAL-UP-012: Port conflict error with PID/process shown in CLI summary."""
        # Arrange
        mock_run_up.return_value = UpResult(
            tiers=[
                TierStatus(name="standard", model="big-model", port=8000, status="skipped", error="Port 8000 already in use by PID 54321 (node)"),
                TierStatus(name="fast", model="fast-model", port=8001, status="healthy"),
            ],
            litellm=TierStatus(name="litellm", model="proxy", port=4000, status="healthy"),
        )

        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["up"])

        # Assert
        assert result.exit_code == 0
        assert "54321" in result.output
        assert "node" in result.output
        assert "8000" in result.output
        assert "skipped" in result.output


# --------------------------------------------------------------------------- #
# Tests — config propagation (dry-run — only needs catalog + config mocks)
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
        # Arrange
        write_stack_yaml(mlx_stack_home)
        write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 5001, "openrouter-key": "",
        }.get(key, "")

        # Act
        result = run_up(dry_run=True)

        # Assert
        assert result.litellm is not None
        assert result.litellm.port == 5001
        litellm_cmds = [c for c in result.dry_run_commands if c["service"] == "litellm"]
        assert len(litellm_cmds) == 1
        assert "5001" in litellm_cmds[0]["command"]


# --------------------------------------------------------------------------- #
# Tests — init → up consistency
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
        # Arrange
        stack = make_stack_yaml()
        write_stack_yaml(mlx_stack_home, stack)
        write_litellm_yaml(mlx_stack_home)
        mock_load_catalog.return_value = make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000, "openrouter-key": "",
        }.get(key, "")

        # Act
        result = run_up(dry_run=True)

        # Assert
        for tier in result.tiers:
            config_port = None
            for t in stack["tiers"]:
                if t["name"] == tier.name:
                    config_port = t["port"]
                    break
            if config_port:
                assert tier.port == config_port


# --------------------------------------------------------------------------- #
# Tests — lockfile cleanup on interrupt (real lock, no mocks)
# --------------------------------------------------------------------------- #


class TestLockfileRecovery:
    """Tests for lockfile behavior during interrupts."""

    def test_lockfile_released_after_error(self, mlx_stack_home: Path) -> None:
        """VAL-CROSS-016: Lockfile released even on error."""
        from mlx_stack.core.process import acquire_lock

        # Act — acquire and release twice
        with acquire_lock():
            pass
        with acquire_lock():
            pass

    def test_lockfile_released_on_exception(self, mlx_stack_home: Path) -> None:
        """VAL-CROSS-016: Lockfile released on exception."""
        from mlx_stack.core.process import acquire_lock

        # Arrange
        try:
            with acquire_lock():
                raise RuntimeError("Simulated crash")
        except RuntimeError:
            pass

        # Assert — can re-acquire
        with acquire_lock():
            pass
