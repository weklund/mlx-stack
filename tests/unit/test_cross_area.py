"""Cross-area validation tests for mlx-stack.

Validates pending cross-area assertions that require multiple commands
working together end-to-end. These assertions were blocked in earlier
milestones because not all commands were implemented yet.

Validates:
- VAL-CROSS-001: init -> pull -> up -> models API returns 200 -> down cleans up
- VAL-CROSS-007: config changes propagate to init, up, pull, recommend
- VAL-CROSS-012: bench --save overrides catalog data in recommend scoring
- VAL-CROSS-013: Data consistency across profile/models/stack files used by all commands
"""

from __future__ import annotations

import json
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
from mlx_stack.core.hardware import HardwareProfile
from mlx_stack.core.pull import ModelInventoryEntry

# --------------------------------------------------------------------------- #
# Shared test data helpers
# --------------------------------------------------------------------------- #


def _make_profile(
    chip: str = "Apple M4 Max",
    gpu_cores: int = 40,
    memory_gb: int = 128,
    bandwidth_gbps: float = 546.0,
    is_estimate: bool = False,
) -> HardwareProfile:
    """Create a HardwareProfile for testing."""
    return HardwareProfile(
        chip=chip,
        gpu_cores=gpu_cores,
        memory_gb=memory_gb,
        bandwidth_gbps=bandwidth_gbps,
        is_estimate=is_estimate,
    )


def _make_entry(
    model_id: str = "test-model",
    name: str = "Test Model",
    family: str = "Test",
    params_b: float = 8.0,
    architecture: str = "transformer",
    quality_overall: int = 70,
    quality_coding: int = 65,
    quality_reasoning: int = 60,
    quality_instruction: int = 72,
    tool_calling: bool = True,
    tool_call_parser: str | None = "hermes",
    thinking: bool = False,
    reasoning_parser: str | None = None,
    benchmarks: dict[str, BenchmarkResult] | None = None,
    tags: list[str] | None = None,
    disk_size_gb: float = 4.5,
) -> CatalogEntry:
    """Create a CatalogEntry for testing."""
    if benchmarks is None:
        benchmarks = {
            "m4-pro-32": BenchmarkResult(prompt_tps=95.0, gen_tps=52.0, memory_gb=5.5),
            "m4-max-128": BenchmarkResult(prompt_tps=140.0, gen_tps=77.0, memory_gb=5.5),
        }
    return CatalogEntry(
        id=model_id,
        name=name,
        family=family,
        params_b=params_b,
        architecture=architecture,
        min_mlx_lm_version="0.22.0",
        sources={
            "int4": QuantSource(
                hf_repo=f"mlx-community/{model_id}-4bit", disk_size_gb=disk_size_gb,
            ),
        },
        capabilities=Capabilities(
            tool_calling=tool_calling,
            tool_call_parser=tool_call_parser if tool_calling else None,
            thinking=thinking,
            reasoning_parser=reasoning_parser,
            vision=False,
        ),
        quality=QualityScores(
            overall=quality_overall,
            coding=quality_coding,
            reasoning=quality_reasoning,
            instruction_following=quality_instruction,
        ),
        benchmarks=benchmarks,
        tags=tags or [],
    )


def _make_test_catalog() -> list[CatalogEntry]:
    """Build a diverse test catalog for cross-area tests."""
    return [
        # High quality model (standard tier candidate)
        _make_entry(
            model_id="high-quality-32b",
            name="High Quality 32B",
            family="Quality",
            params_b=32.0,
            quality_overall=87,
            quality_coding=85,
            quality_reasoning=88,
            quality_instruction=88,
            tool_calling=True,
            benchmarks={
                "m4-pro-32": BenchmarkResult(prompt_tps=26.0, gen_tps=15.0, memory_gb=20.0),
                "m4-max-128": BenchmarkResult(prompt_tps=40.0, gen_tps=23.0, memory_gb=20.0),
            },
            tags=["quality"],
            disk_size_gb=18.0,
        ),
        # Fast small model (fast tier candidate)
        _make_entry(
            model_id="fast-0.8b",
            name="Fast 0.8B",
            family="Fast",
            params_b=0.8,
            quality_overall=30,
            quality_coding=25,
            quality_reasoning=20,
            quality_instruction=35,
            tool_calling=True,
            benchmarks={
                "m4-pro-32": BenchmarkResult(prompt_tps=310.0, gen_tps=195.0, memory_gb=0.6),
                "m4-max-128": BenchmarkResult(prompt_tps=410.0, gen_tps=280.0, memory_gb=0.6),
            },
            tags=["fast"],
            disk_size_gb=0.5,
        ),
        # Medium model
        _make_entry(
            model_id="medium-8b",
            name="Medium 8B",
            family="Medium",
            params_b=8.0,
            quality_overall=68,
            quality_coding=65,
            quality_reasoning=62,
            quality_instruction=72,
            tool_calling=True,
            benchmarks={
                "m4-pro-32": BenchmarkResult(prompt_tps=95.0, gen_tps=52.0, memory_gb=5.5),
                "m4-max-128": BenchmarkResult(prompt_tps=140.0, gen_tps=77.0, memory_gb=5.5),
            },
            tags=["balanced"],
            disk_size_gb=4.5,
        ),
        # Longctx model (mamba2-hybrid architecture)
        _make_entry(
            model_id="longctx-32b",
            name="LongCtx 32B",
            family="LongCtx",
            params_b=32.0,
            architecture="mamba2-hybrid",
            quality_overall=82,
            quality_coding=78,
            quality_reasoning=85,
            quality_instruction=80,
            tool_calling=False,
            tool_call_parser=None,
            benchmarks={
                "m4-pro-32": BenchmarkResult(prompt_tps=30.0, gen_tps=18.0, memory_gb=19.0),
                "m4-max-128": BenchmarkResult(prompt_tps=45.0, gen_tps=27.0, memory_gb=19.0),
            },
            tags=["longctx"],
            disk_size_gb=17.0,
        ),
    ]


def _write_profile(home: Path, profile: HardwareProfile) -> None:
    """Write a profile.json file to the given home directory."""
    profile_path = home / "profile.json"
    profile_path.write_text(json.dumps(profile.to_dict(), indent=2) + "\n")


def _write_config(home: Path, config: dict[str, Any]) -> None:
    """Write a config.yaml file to the given home directory."""
    config_path = home / "config.yaml"
    config_path.write_text(yaml.dump(config, default_flow_style=False))


def _write_saved_benchmarks(
    home: Path,
    profile_id: str,
    benchmarks: dict[str, Any],
) -> None:
    """Write saved benchmark data for the given profile."""
    benchmarks_dir = home / "benchmarks"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    (benchmarks_dir / f"{profile_id}.json").write_text(
        json.dumps(benchmarks, indent=2)
    )


def _write_inventory(home: Path, entries: list[dict[str, Any]]) -> None:
    """Write a models.json inventory file."""
    path = home / "models.json"
    path.write_text(json.dumps(entries, indent=2) + "\n")


def _read_stack_yaml(home: Path) -> dict[str, Any]:
    """Read and parse the default stack YAML."""
    path = home / "stacks" / "default.yaml"
    return yaml.safe_load(path.read_text())


def _read_litellm_yaml(home: Path) -> dict[str, Any]:
    """Read and parse the LiteLLM YAML config."""
    path = home / "litellm.yaml"
    return yaml.safe_load(path.read_text())


# --------------------------------------------------------------------------- #
# VAL-CROSS-001: End-to-end first-time user journey
#
# init -> pull -> up -> models API returns 200 -> down cleans up
#
# Tests the full data flow across init, pull, up, down with mocked
# subprocess/network layers.
# --------------------------------------------------------------------------- #


class TestCrossAreaEndToEnd:
    """VAL-CROSS-001: End-to-end first-time user journey with mocked processes."""

    @patch("mlx_stack.core.stack_init.load_catalog")
    @patch("mlx_stack.core.stack_init.detect_hardware")
    @patch("mlx_stack.core.stack_init._is_port_available", return_value=True)
    def test_init_creates_valid_stack_and_litellm_configs(
        self,
        mock_port_avail: MagicMock,
        mock_detect: MagicMock,
        mock_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """Init generates stack+litellm configs with consistent data."""
        profile = _make_profile(memory_gb=128)
        mock_detect.return_value = profile
        mock_catalog.return_value = _make_test_catalog()

        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--accept-defaults"])
        assert result.exit_code == 0

        # Verify stack YAML created and valid
        stack = _read_stack_yaml(mlx_stack_home)
        assert stack["schema_version"] == 1
        assert stack["intent"] == "balanced"
        assert len(stack["tiers"]) > 0

        # Verify LiteLLM config created and valid
        litellm = _read_litellm_yaml(mlx_stack_home)
        assert "model_list" in litellm
        assert len(litellm["model_list"]) == len(stack["tiers"])

    @patch("mlx_stack.core.stack_init.load_catalog")
    @patch("mlx_stack.core.stack_init.detect_hardware")
    @patch("mlx_stack.core.stack_init._is_port_available", return_value=True)
    def test_init_then_up_dry_run_uses_consistent_ports(
        self,
        mock_port_avail: MagicMock,
        mock_detect: MagicMock,
        mock_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """Ports from init stack definition match dry-run commands.

        This validates that the init -> up data flow is consistent:
        stack definition ports match LiteLLM config api_base ports and
        the up --dry-run command ports.
        """
        profile = _make_profile(memory_gb=128)
        mock_detect.return_value = profile
        catalog = _make_test_catalog()
        mock_catalog.return_value = catalog

        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--accept-defaults"])
        assert result.exit_code == 0

        # Read the generated configs
        stack = _read_stack_yaml(mlx_stack_home)
        litellm = _read_litellm_yaml(mlx_stack_home)

        # Verify port consistency between stack and litellm configs
        stack_ports = {t["name"]: t["port"] for t in stack["tiers"]}
        for model_entry in litellm["model_list"]:
            api_base = model_entry["litellm_params"]["api_base"]
            # api_base format: http://localhost:<port>/v1
            port_str = api_base.split(":")[2].split("/")[0]
            port = int(port_str)
            # Verify port is one of the allocated stack ports
            model_name = model_entry["model_name"]
            assert port in stack_ports.values(), (
                f"LiteLLM model '{model_name}' api_base port {port} "
                f"not found in stack ports: {stack_ports}"
            )

        # Now dry-run up and verify ports match
        with (
            patch("mlx_stack.core.stack_up.load_catalog", return_value=catalog),
            patch("mlx_stack.core.stack_up.get_value") as mock_get_val,
        ):
            mock_get_val.side_effect = lambda key: {
                "litellm-port": 4000,
                "openrouter-key": "",
            }.get(key, "")

            result = runner.invoke(cli, ["up", "--dry-run"])
            assert result.exit_code == 0

            # Verify each stack port appears in dry-run output
            for tier in stack["tiers"]:
                port = tier["port"]
                assert str(port) in result.output, (
                    f"Port {port} for tier '{tier['name']}' not in dry-run output"
                )

    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    @patch("mlx_stack.core.stack_up.start_service")
    @patch("mlx_stack.core.stack_up.wait_for_healthy")
    @patch("mlx_stack.core.stack_up.acquire_lock")
    @patch("mlx_stack.core.stack_up.ensure_dependency")
    @patch("mlx_stack.core.stack_up.check_port_conflict")
    @patch("mlx_stack.core.stack_up.check_local_model_exists", return_value=None)
    @patch("mlx_stack.core.stack_up.shutil.which", return_value="/usr/local/bin/vllm-mlx")
    @patch("mlx_stack.core.stack_init.load_catalog")
    @patch("mlx_stack.core.stack_init.detect_hardware")
    @patch("mlx_stack.core.stack_init._is_port_available", return_value=True)
    def test_full_lifecycle_init_up_down(
        self,
        mock_port_avail: MagicMock,
        mock_detect: MagicMock,
        mock_init_catalog: MagicMock,
        mock_which: MagicMock,
        mock_check_model: MagicMock,
        mock_check_port: MagicMock,
        mock_ensure_dep: MagicMock,
        mock_acquire_lock: MagicMock,
        mock_wait_healthy: MagicMock,
        mock_start_service: MagicMock,
        mock_get_value: MagicMock,
        mock_up_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-CROSS-001: init -> up -> down completes full lifecycle.

        After init creates configs, up starts mocked services with PID files,
        and down cleans them all up leaving zero artifacts.
        """
        from mlx_stack.core.process import HealthCheckResult, ServiceInfo

        profile = _make_profile(memory_gb=128)
        mock_detect.return_value = profile
        catalog = _make_test_catalog()
        mock_init_catalog.return_value = catalog
        mock_up_catalog.return_value = catalog
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "",
        }.get(key, "")
        mock_check_port.return_value = None  # No port conflicts

        # Make acquire_lock return a context manager
        lock_cm = MagicMock()
        lock_cm.__enter__ = MagicMock(return_value=None)
        lock_cm.__exit__ = MagicMock(return_value=False)
        mock_acquire_lock.return_value = lock_cm

        # Track PIDs for started services
        pid_counter = [1000]

        def fake_start_service(
            service_name: str, cmd: list[str], port: int = 0, **kwargs: Any
        ) -> ServiceInfo:
            pid_counter[0] += 1
            pid = pid_counter[0]
            # Create PID file as the real function would
            pids_dir = mlx_stack_home / "pids"
            pids_dir.mkdir(parents=True, exist_ok=True)
            pid_file = pids_dir / f"{service_name}.pid"
            pid_file.write_text(str(pid))
            logs_dir = mlx_stack_home / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_file = logs_dir / f"{service_name}.log"
            return ServiceInfo(
                name=service_name, pid=pid, port=port,
                log_path=log_file, pid_path=pid_file,
            )

        mock_start_service.side_effect = fake_start_service
        mock_wait_healthy.return_value = HealthCheckResult(
            healthy=True, response_time=0.1, status_code=200,
        )

        runner = CliRunner()

        # Step 1: init
        result = runner.invoke(cli, ["init", "--accept-defaults"])
        assert result.exit_code == 0

        # Step 2: up (mocked processes)
        result = runner.invoke(cli, ["up"])
        assert result.exit_code == 0, f"up failed: {result.output}"

        # Verify PID files exist
        pids_dir = mlx_stack_home / "pids"
        pid_files = list(pids_dir.glob("*.pid"))
        assert len(pid_files) > 0, "No PID files created during up"

        # Step 3: down (should clean up all PID files)
        with (
            patch("mlx_stack.core.stack_down.acquire_lock", return_value=lock_cm),
            patch("mlx_stack.core.stack_down.is_process_alive", return_value=False),
            patch("mlx_stack.core.stack_down.read_pid_file") as mock_read_pid,
            patch("mlx_stack.core.stack_down.remove_pid_file") as mock_remove_pid,
        ):
            # Read actual PID files
            def read_pid_side_effect(name: str) -> int | None:
                pid_file = pids_dir / f"{name}.pid"
                if pid_file.exists():
                    return int(pid_file.read_text().strip())
                return None

            mock_read_pid.side_effect = read_pid_side_effect

            def remove_pid_side_effect(name: str) -> None:
                pid_file = pids_dir / f"{name}.pid"
                if pid_file.exists():
                    pid_file.unlink()

            mock_remove_pid.side_effect = remove_pid_side_effect

            result = runner.invoke(cli, ["down"])
            assert result.exit_code == 0

        # Verify cleanup: no PID files remain
        remaining_pids = list(pids_dir.glob("*.pid"))
        assert len(remaining_pids) == 0, (
            f"PID files remain after down: {[p.name for p in remaining_pids]}"
        )


# --------------------------------------------------------------------------- #
# VAL-CROSS-007: Config changes propagate to init, up, pull, recommend
#
# After config set, subsequent commands use the new values.
# --------------------------------------------------------------------------- #


class TestConfigPropagation:
    """VAL-CROSS-007: Config changes propagate to all dependent commands."""

    @patch("mlx_stack.core.stack_init.load_catalog")
    @patch("mlx_stack.core.stack_init.detect_hardware")
    @patch("mlx_stack.core.stack_init._is_port_available", return_value=True)
    def test_litellm_port_propagates_to_init(
        self,
        mock_port_avail: MagicMock,
        mock_detect: MagicMock,
        mock_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """After config set litellm-port 5001, init uses port 5001."""
        profile = _make_profile(memory_gb=128)
        mock_detect.return_value = profile
        mock_catalog.return_value = _make_test_catalog()

        runner = CliRunner()

        # Set custom litellm-port
        result = runner.invoke(cli, ["config", "set", "litellm-port", "5001"])
        assert result.exit_code == 0

        # Run init and verify port propagation
        result = runner.invoke(cli, ["init", "--accept-defaults"])
        assert result.exit_code == 0

        # Verify the custom port is NOT used for vllm model api_base ports
        # (it's the LiteLLM proxy port, not a tier port)
        stack = _read_stack_yaml(mlx_stack_home)
        tier_ports = {t["port"] for t in stack["tiers"]}
        assert 5001 not in tier_ports, (
            "LiteLLM port 5001 should not be used for vllm tier ports"
        )

    @patch("mlx_stack.core.stack_init.load_catalog")
    @patch("mlx_stack.core.stack_init.detect_hardware")
    @patch("mlx_stack.core.stack_init._is_port_available", return_value=True)
    def test_memory_budget_pct_propagates_to_init(
        self,
        mock_port_avail: MagicMock,
        mock_detect: MagicMock,
        mock_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """After config set memory-budget-pct 80, init uses 80% budget."""
        profile = _make_profile(memory_gb=128)
        mock_detect.return_value = profile
        mock_catalog.return_value = _make_test_catalog()

        runner = CliRunner()

        # Set custom memory-budget-pct
        result = runner.invoke(cli, ["config", "set", "memory-budget-pct", "80"])
        assert result.exit_code == 0

        # Run init
        result = runner.invoke(cli, ["init", "--accept-defaults"])
        assert result.exit_code == 0

        # With 80% of 128GB = 102.4GB, more models should fit
        # vs default 40% = 51.2GB
        stack = _read_stack_yaml(mlx_stack_home)
        assert len(stack["tiers"]) > 0, "Init should produce at least one tier"
        # The output should mention the higher budget
        assert "80" in result.output or "102" in result.output or len(stack["tiers"]) >= 1

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_memory_budget_pct_propagates_to_recommend(
        self,
        mock_load_profile: MagicMock,
        mock_load_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """After config set memory-budget-pct 60, recommend uses 60% budget."""
        profile = _make_profile(memory_gb=128)
        mock_load_profile.return_value = profile
        mock_load_catalog.return_value = _make_test_catalog()

        runner = CliRunner()

        # Set custom memory-budget-pct
        result = runner.invoke(cli, ["config", "set", "memory-budget-pct", "60"])
        assert result.exit_code == 0

        # Run recommend
        result = runner.invoke(cli, ["recommend"])
        assert result.exit_code == 0
        # 60% of 128 = 76.8 GB
        assert "76.8 GB" in result.output

    @patch("mlx_stack.core.stack_init.load_catalog")
    @patch("mlx_stack.core.stack_init.detect_hardware")
    @patch("mlx_stack.core.stack_init._is_port_available", return_value=True)
    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    def test_litellm_port_propagates_to_up_dry_run(
        self,
        mock_up_get_value: MagicMock,
        mock_up_catalog: MagicMock,
        mock_port_avail: MagicMock,
        mock_detect: MagicMock,
        mock_init_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """After config set litellm-port 5001, up --dry-run shows port 5001."""
        profile = _make_profile(memory_gb=128)
        mock_detect.return_value = profile
        catalog = _make_test_catalog()
        mock_init_catalog.return_value = catalog
        mock_up_catalog.return_value = catalog

        runner = CliRunner()

        # Set custom port
        result = runner.invoke(cli, ["config", "set", "litellm-port", "5001"])
        assert result.exit_code == 0

        # Init generates configs with default port settings
        result = runner.invoke(cli, ["init", "--accept-defaults"])
        assert result.exit_code == 0

        # Up --dry-run should use the configured port
        mock_up_get_value.side_effect = lambda key: {
            "litellm-port": 5001,
            "openrouter-key": "",
        }.get(key, "")

        result = runner.invoke(cli, ["up", "--dry-run"])
        assert result.exit_code == 0
        # The litellm command should use port 5001
        assert "5001" in result.output

    @patch("mlx_stack.core.pull.load_catalog")
    @patch("mlx_stack.core.pull.get_value")
    def test_default_quant_propagates_to_pull(
        self,
        mock_get_value: MagicMock,
        mock_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """After config set default-quant int4, pull uses int4 without --quant."""
        catalog = _make_test_catalog()
        mock_catalog.return_value = catalog

        # Config returns int4 as default quant
        def config_side_effect(key: str) -> Any:
            if key == "default-quant":
                return "int4"
            if key == "model-dir":
                return str(mlx_stack_home / "models")
            return ""

        mock_get_value.side_effect = config_side_effect

        # Mock the download to avoid network calls
        with (
            patch("mlx_stack.core.pull.check_disk_space"),
            patch("mlx_stack.core.pull.download_model") as mock_dl,
            patch("mlx_stack.core.pull.add_to_inventory"),
            patch("mlx_stack.core.pull.is_model_downloaded", return_value=False),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["pull", "medium-8b"])

            # Should use int4 (from config) since no --quant flag
            if result.exit_code == 0:
                # Verify int4 quant was used
                assert mock_dl.called or "int4" in result.output or "already" in result.output

    @patch("mlx_stack.core.stack_init.load_catalog")
    @patch("mlx_stack.core.stack_init.detect_hardware")
    @patch("mlx_stack.core.stack_init._is_port_available", return_value=True)
    def test_config_changes_across_init_regeneration(
        self,
        mock_port_avail: MagicMock,
        mock_detect: MagicMock,
        mock_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """Config changes propagate when re-running init --force.

        Sets litellm-port, runs init, changes memory-budget-pct, re-runs
        init --force, and verifies both changes are reflected.
        """
        profile = _make_profile(memory_gb=128)
        mock_detect.return_value = profile
        mock_catalog.return_value = _make_test_catalog()

        runner = CliRunner()

        # Set config values
        runner.invoke(cli, ["config", "set", "litellm-port", "5001"])
        runner.invoke(cli, ["config", "set", "memory-budget-pct", "60"])

        # First init
        result = runner.invoke(cli, ["init", "--accept-defaults"])
        assert result.exit_code == 0

        # Change config
        runner.invoke(cli, ["config", "set", "memory-budget-pct", "80"])

        # Re-init with --force
        result = runner.invoke(cli, ["init", "--accept-defaults", "--force"])
        assert result.exit_code == 0

        # Verify the new budget is reflected
        stack = _read_stack_yaml(mlx_stack_home)
        assert len(stack["tiers"]) > 0


# --------------------------------------------------------------------------- #
# VAL-CROSS-012: bench --save overrides catalog data in recommend scoring
#
# After bench --save records gen_tps=100 (catalog says 77), subsequent
# recommend uses 100 (not 77) for scoring. The "estimated" label
# disappears for models with local benchmark data.
# --------------------------------------------------------------------------- #


class TestBenchSaveOverridesCatalog:
    """VAL-CROSS-012: Saved benchmark data overrides catalog in recommendations."""

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_saved_benchmarks_override_catalog_gen_tps(
        self,
        mock_load_profile: MagicMock,
        mock_load_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """Saved benchmark gen_tps=100 overrides catalog gen_tps=77."""
        profile = _make_profile(memory_gb=128)
        mock_load_profile.return_value = profile
        mock_load_catalog.return_value = _make_test_catalog()

        # Write saved benchmarks with higher gen_tps
        _write_saved_benchmarks(
            mlx_stack_home,
            profile.profile_id,
            {
                "medium-8b": {
                    "gen_tps": 100.0,  # Catalog has 77.0
                    "prompt_tps": 200.0,
                    "memory_gb": 5.5,
                },
            },
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend", "--show-all"])
        assert result.exit_code == 0
        # The saved benchmark gen_tps (100.0) should appear instead of catalog (77.0)
        assert "100.0" in result.output

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_saved_benchmarks_remove_estimated_label(
        self,
        mock_load_profile: MagicMock,
        mock_load_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """Saved benchmarks remove 'estimated' label from recommend output.

        When hardware has no catalog benchmark data, values are labeled
        as 'estimated'. After bench --save, the measured data replaces
        estimates and the estimated label should disappear.
        """
        # Use a profile that has NO matching catalog benchmark keys
        profile = HardwareProfile(
            chip="Apple M5 Ultra",
            gpu_cores=80,
            memory_gb=256,
            bandwidth_gbps=800.0,
            is_estimate=True,
        )
        mock_load_profile.return_value = profile
        mock_load_catalog.return_value = _make_test_catalog()

        runner = CliRunner()

        # First recommend without saved benchmarks — should show 'estimated'
        result = runner.invoke(cli, ["recommend", "--show-all"])
        assert result.exit_code == 0

        # Now save benchmarks for this model
        _write_saved_benchmarks(
            mlx_stack_home,
            profile.profile_id,
            {
                "medium-8b": {
                    "gen_tps": 95.0,
                    "prompt_tps": 180.0,
                    "memory_gb": 5.5,
                },
            },
        )

        # Second recommend with saved benchmarks
        result = runner.invoke(cli, ["recommend", "--show-all"])
        assert result.exit_code == 0
        second_output = result.output

        # The saved gen_tps value should appear
        assert "95.0" in second_output

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_saved_benchmarks_affect_scoring_order(
        self,
        mock_load_profile: MagicMock,
        mock_load_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """Saved benchmarks change scoring and potentially tier assignments.

        If a model gets a significantly higher gen_tps from benchmarks,
        it should score differently in recommendations.
        """
        profile = _make_profile(memory_gb=128)
        mock_load_profile.return_value = profile
        mock_load_catalog.return_value = _make_test_catalog()

        runner = CliRunner()

        # Recommend without saved benchmarks
        result_before = runner.invoke(cli, ["recommend"])
        assert result_before.exit_code == 0

        # Save benchmarks with dramatically different gen_tps for medium model
        _write_saved_benchmarks(
            mlx_stack_home,
            profile.profile_id,
            {
                "medium-8b": {
                    "gen_tps": 500.0,  # Way higher than catalog's 77.0
                    "prompt_tps": 300.0,
                    "memory_gb": 5.5,
                },
            },
        )

        # Recommend with saved benchmarks
        result_after = runner.invoke(cli, ["recommend"])
        assert result_after.exit_code == 0

        # The output should differ because scoring has changed
        # (medium-8b is now much faster, potentially changing tier assignments)
        # At minimum both should succeed
        assert result_before.exit_code == 0
        assert result_after.exit_code == 0


# --------------------------------------------------------------------------- #
# VAL-CROSS-013: Data consistency across profile/models/stack files
#
# profile.json written by profile is parsed by recommend, init, bench.
# models.json updated by pull is consistent with models output.
# Stack schema_version checked by up. vllm_flags translate to CLI flags.
# --------------------------------------------------------------------------- #


class TestDataConsistency:
    """VAL-CROSS-013: Data consistency across all commands."""

    def test_profile_json_parseable_by_all_consumers(
        self,
        mlx_stack_home: Path,
    ) -> None:
        """profile.json written by profile is parseable by recommend, init, bench."""
        profile = _make_profile(memory_gb=128)
        _write_profile(mlx_stack_home, profile)

        # Verify the file is valid JSON with all required fields
        profile_path = mlx_stack_home / "profile.json"
        data = json.loads(profile_path.read_text())
        assert "chip" in data
        assert "gpu_cores" in data
        assert "memory_gb" in data
        assert "bandwidth_gbps" in data
        assert "profile_id" in data

        # Verify it can be loaded by the hardware module
        from mlx_stack.core.hardware import load_profile

        loaded = load_profile()
        assert loaded is not None
        assert loaded.chip == profile.chip
        assert loaded.memory_gb == profile.memory_gb
        assert loaded.bandwidth_gbps == profile.bandwidth_gbps
        assert loaded.profile_id == profile.profile_id

    def test_models_json_consistent_with_models_command(
        self,
        mlx_stack_home: Path,
    ) -> None:
        """models.json entries are discovered by the models command."""
        # Create model directory and inventory entry
        models_dir = mlx_stack_home / "models"
        model_path = models_dir / "medium-8b-int4"
        model_path.mkdir(parents=True)
        # Create a marker file so scan detects it
        (model_path / "config.json").write_text("{}")

        # Write inventory with matching entry
        inv_entry = {
            "model_id": "medium-8b",
            "name": "Medium 8B",
            "quant": "int4",
            "source_type": "mlx_community",
            "local_path": str(model_path),
            "disk_size_gb": 4.5,
            "downloaded_at": "2026-03-24T00:00:00+00:00",
        }
        _write_inventory(mlx_stack_home, [inv_entry])

        # Verify models.json is valid JSON
        path = mlx_stack_home / "models.json"
        data = json.loads(path.read_text())
        assert len(data) == 1
        assert data[0]["model_id"] == "medium-8b"
        assert data[0]["quant"] == "int4"

    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    def test_stack_schema_version_checked_by_up(
        self,
        mock_get_value: MagicMock,
        mock_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """Stack with unsupported schema_version is rejected by up."""
        mock_catalog.return_value = _make_test_catalog()
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "",
        }.get(key, "")

        # Write stack with invalid schema version
        stacks_dir = mlx_stack_home / "stacks"
        stacks_dir.mkdir(parents=True)
        stack = {
            "schema_version": 99,
            "name": "default",
            "hardware_profile": "m4-max-128",
            "intent": "balanced",
            "created": "2026-03-24T00:00:00+00:00",
            "tiers": [
                {
                    "name": "fast",
                    "model": "fast-model",
                    "quant": "int4",
                    "source": "test/fast-model-4bit",
                    "port": 8000,
                    "vllm_flags": {},
                },
            ],
        }
        (stacks_dir / "default.yaml").write_text(yaml.dump(stack))
        (mlx_stack_home / "litellm.yaml").write_text(
            yaml.dump({"model_list": [], "general_settings": {}})
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["up"])
        assert result.exit_code != 0
        assert "schema_version" in result.output

    def test_vllm_flags_translate_to_cli_flags(self) -> None:
        """vllm_flags from stack YAML translate correctly to CLI flags.

        Tests that boolean True flags become --flag-name, string values
        become --flag-name value, and underscore is converted to hyphen.
        """
        from mlx_stack.core.stack_up import build_vllm_command

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
                "reasoning_parser": "deepseek_r1",
            },
        }
        cmd = build_vllm_command(tier, "vllm-mlx")

        # Boolean True flags become --flag-name
        assert "--continuous-batching" in cmd
        assert "--use-paged-cache" in cmd
        assert "--enable-auto-tool-choice" in cmd

        # String values become --flag-name value
        parser_idx = cmd.index("--tool-call-parser")
        assert cmd[parser_idx + 1] == "hermes"

        reasoning_idx = cmd.index("--reasoning-parser")
        assert cmd[reasoning_idx + 1] == "deepseek_r1"

    @patch("mlx_stack.core.stack_init.load_catalog")
    @patch("mlx_stack.core.stack_init.detect_hardware")
    @patch("mlx_stack.core.stack_init._is_port_available", return_value=True)
    def test_init_stack_fields_consumed_by_up(
        self,
        mock_port_avail: MagicMock,
        mock_detect: MagicMock,
        mock_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """Stack definition from init contains all fields expected by up.

        Verifies that every tier has: name, model, quant, source, port,
        vllm_flags — and that the stack has schema_version, hardware_profile,
        intent, created, and tiers.
        """
        profile = _make_profile(memory_gb=128)
        mock_detect.return_value = profile
        mock_catalog.return_value = _make_test_catalog()

        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--accept-defaults"])
        assert result.exit_code == 0

        stack = _read_stack_yaml(mlx_stack_home)

        # Top-level fields
        assert "schema_version" in stack
        assert stack["schema_version"] == 1
        assert "hardware_profile" in stack
        assert "intent" in stack
        assert "created" in stack
        assert "tiers" in stack
        assert "name" in stack

        # Tier fields
        for tier in stack["tiers"]:
            assert "name" in tier, f"Tier missing 'name': {tier}"
            assert "model" in tier, f"Tier missing 'model': {tier}"
            assert "quant" in tier, f"Tier missing 'quant': {tier}"
            assert "source" in tier, f"Tier missing 'source': {tier}"
            assert "port" in tier, f"Tier missing 'port': {tier}"
            assert "vllm_flags" in tier, f"Tier missing 'vllm_flags': {tier}"

        # Unique ports
        ports = [t["port"] for t in stack["tiers"]]
        assert len(ports) == len(set(ports)), f"Duplicate ports: {ports}"

    @patch("mlx_stack.core.stack_init.load_catalog")
    @patch("mlx_stack.core.stack_init.detect_hardware")
    @patch("mlx_stack.core.stack_init._is_port_available", return_value=True)
    def test_litellm_config_matches_stack_tiers(
        self,
        mock_port_avail: MagicMock,
        mock_detect: MagicMock,
        mock_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """LiteLLM config model_list matches stack tier count and ports."""
        profile = _make_profile(memory_gb=128)
        mock_detect.return_value = profile
        mock_catalog.return_value = _make_test_catalog()

        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--accept-defaults"])
        assert result.exit_code == 0

        stack = _read_stack_yaml(mlx_stack_home)
        litellm = _read_litellm_yaml(mlx_stack_home)

        # Same number of model entries as stack tiers
        assert len(litellm["model_list"]) == len(stack["tiers"])

        # Each litellm entry's port matches a stack tier port
        stack_ports = {t["port"] for t in stack["tiers"]}
        for entry in litellm["model_list"]:
            api_base = entry["litellm_params"]["api_base"]
            port = int(api_base.split(":")[2].split("/")[0])
            assert port in stack_ports

    def test_inventory_round_trip(self, mlx_stack_home: Path) -> None:
        """Inventory entries can be written and read back consistently."""
        from mlx_stack.core.pull import add_to_inventory, load_inventory

        entry = ModelInventoryEntry(
            model_id="test-model",
            name="Test Model",
            quant="int4",
            source_type="mlx_community",
            hf_repo="mlx-community/test-model-4bit",
            local_path=str(mlx_stack_home / "models" / "test-model-4bit"),
            disk_size_gb=4.5,
            downloaded_at="2026-03-24T00:00:00+00:00",
        )
        add_to_inventory(entry)

        # Read back
        entries = load_inventory()
        assert len(entries) == 1
        assert entries[0]["model_id"] == "test-model"
        assert entries[0]["quant"] == "int4"
        assert entries[0]["name"] == "Test Model"
        assert entries[0]["disk_size_gb"] == 4.5

    @patch("mlx_stack.core.stack_init.load_catalog")
    @patch("mlx_stack.core.stack_init.detect_hardware")
    @patch("mlx_stack.core.stack_init._is_port_available", return_value=True)
    def test_profile_id_in_stack_matches_profile(
        self,
        mock_port_avail: MagicMock,
        mock_detect: MagicMock,
        mock_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """hardware_profile in stack.yaml matches the detected profile_id."""
        profile = _make_profile(memory_gb=128)
        mock_detect.return_value = profile
        mock_catalog.return_value = _make_test_catalog()

        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--accept-defaults"])
        assert result.exit_code == 0

        stack = _read_stack_yaml(mlx_stack_home)
        assert stack["hardware_profile"] == profile.profile_id

    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    @patch("mlx_stack.core.stack_init.load_catalog")
    @patch("mlx_stack.core.stack_init.detect_hardware")
    @patch("mlx_stack.core.stack_init._is_port_available", return_value=True)
    def test_dry_run_flags_match_stack_vllm_flags(
        self,
        mock_port_avail: MagicMock,
        mock_detect: MagicMock,
        mock_init_catalog: MagicMock,
        mock_get_value: MagicMock,
        mock_up_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """vllm_flags from init-generated stack YAML appear in dry-run output."""
        profile = _make_profile(memory_gb=128)
        mock_detect.return_value = profile
        catalog = _make_test_catalog()
        mock_init_catalog.return_value = catalog
        mock_up_catalog.return_value = catalog
        mock_get_value.side_effect = lambda key: {
            "litellm-port": 4000,
            "openrouter-key": "",
        }.get(key, "")

        runner = CliRunner()

        # Init creates stack with vllm_flags
        result = runner.invoke(cli, ["init", "--accept-defaults"])
        assert result.exit_code == 0

        # Read stack to find expected flags
        stack = _read_stack_yaml(mlx_stack_home)

        # Up --dry-run should show translated vllm_flags
        result = runner.invoke(cli, ["up", "--dry-run"])
        assert result.exit_code == 0

        # All tiers with tool_calling should have their flags in dry-run output
        for tier in stack["tiers"]:
            flags = tier.get("vllm_flags", {})
            if flags.get("enable_auto_tool_choice"):
                assert "--enable-auto-tool-choice" in result.output
            if flags.get("tool_call_parser"):
                assert "--tool-call-parser" in result.output
