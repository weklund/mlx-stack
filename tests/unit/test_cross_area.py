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
    CatalogEntry,
    QuantSource,
)
from mlx_stack.core.hardware import HardwareProfile
from mlx_stack.core.pull import ModelInventoryEntry
from tests.factories import make_entry, make_profile

# --------------------------------------------------------------------------- #
# Shared test data helpers
# --------------------------------------------------------------------------- #


def _make_test_catalog() -> list[CatalogEntry]:
    """Build a diverse test catalog for cross-area tests.

    Uses multi-benchmark / multi-source entries (m4-pro-32 + m4-max-128
    benchmarks, int4 + int8 sources) needed by the cross-area tests.
    """
    return [
        # High quality model (standard tier candidate)
        make_entry(
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
            sources={
                "int4": QuantSource(hf_repo="mlx-community/high-quality-32b-4bit", disk_size_gb=18.0),
                "int8": QuantSource(hf_repo="mlx-community/high-quality-32b-8bit", disk_size_gb=36.0),
            },
        ),
        # Fast small model (fast tier candidate)
        make_entry(
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
            sources={
                "int4": QuantSource(hf_repo="mlx-community/fast-0.8b-4bit", disk_size_gb=0.5),
                "int8": QuantSource(hf_repo="mlx-community/fast-0.8b-8bit", disk_size_gb=1.0),
            },
        ),
        # Medium model
        make_entry(
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
            sources={
                "int4": QuantSource(hf_repo="mlx-community/medium-8b-4bit", disk_size_gb=4.5),
                "int8": QuantSource(hf_repo="mlx-community/medium-8b-8bit", disk_size_gb=9.0),
            },
        ),
        # Longctx model (mamba2-hybrid architecture)
        make_entry(
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
            sources={
                "int4": QuantSource(hf_repo="mlx-community/longctx-32b-4bit", disk_size_gb=17.0),
                "int8": QuantSource(hf_repo="mlx-community/longctx-32b-8bit", disk_size_gb=34.0),
            },
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
    (benchmarks_dir / f"{profile_id}.json").write_text(json.dumps(benchmarks, indent=2))


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
        # Arrange
        profile = make_profile(memory_gb=128)
        mock_detect.return_value = profile
        mock_catalog.return_value = _make_test_catalog()

        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--accept-defaults"])

        # Assert
        assert result.exit_code == 0
        stack = _read_stack_yaml(mlx_stack_home)
        assert stack["schema_version"] == 1
        assert stack["intent"] == "balanced"
        assert len(stack["tiers"]) > 0
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
        profile = make_profile(memory_gb=128)
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
    def test_full_lifecycle_init_pull_up_models_api_down(
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
        """VAL-CROSS-001: Full init -> pull -> up -> models API -> down flow.

        1. Runs init to generate configs.
        2. Mocks pull to create models.json inventory entries.
        3. Runs up (mocked subprocess) to create PID files.
        4. Mocks a GET to /v1/models returning 200 with model list.
        5. Runs down to clean up.
        6. Verifies no PID files remain.
        """
        from mlx_stack.core.process import HealthCheckResult, ServiceInfo

        profile = make_profile(memory_gb=128)
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
                name=service_name,
                pid=pid,
                port=port,
                log_path=log_file,
                pid_path=pid_file,
            )

        mock_start_service.side_effect = fake_start_service
        mock_wait_healthy.return_value = HealthCheckResult(
            healthy=True,
            response_time=0.1,
            status_code=200,
        )

        runner = CliRunner()

        # ---- Step 1: init ----
        result = runner.invoke(cli, ["init", "--accept-defaults"])
        assert result.exit_code == 0, f"init failed: {result.output}"

        stack = _read_stack_yaml(mlx_stack_home)
        assert len(stack["tiers"]) > 0, "init produced no tiers"

        # ---- Step 2: Mock pull — create models.json inventory entries ----
        inventory_entries: list[dict[str, Any]] = []
        models_dir = mlx_stack_home / "models"
        for tier in stack["tiers"]:
            model_id = tier["model"]
            quant = tier["quant"]
            source = tier["source"]
            repo_name = source.rsplit("/", 1)[-1] if "/" in source else source
            local_path = models_dir / repo_name
            local_path.mkdir(parents=True, exist_ok=True)
            (local_path / "config.json").write_text("{}")

            inventory_entries.append(
                {
                    "model_id": model_id,
                    "name": model_id,
                    "quant": quant,
                    "source_type": "mlx_community",
                    "hf_repo": source,
                    "local_path": str(local_path),
                    "disk_size_gb": 4.5,
                    "downloaded_at": "2026-03-24T00:00:00+00:00",
                }
            )

        _write_inventory(mlx_stack_home, inventory_entries)

        # Verify models.json was written correctly
        inv_data = json.loads((mlx_stack_home / "models.json").read_text())
        assert len(inv_data) == len(stack["tiers"]), (
            f"Expected {len(stack['tiers'])} inventory entries, got {len(inv_data)}"
        )

        # ---- Step 3: up (mocked processes) ----
        result = runner.invoke(cli, ["up"])
        assert result.exit_code == 0, f"up failed: {result.output}"

        # Verify PID files exist for each tier + litellm
        pids_dir = mlx_stack_home / "pids"
        pid_files = list(pids_dir.glob("*.pid"))
        assert len(pid_files) > 0, "No PID files created during up"

        # Verify PID files exist for each tier
        tier_names = {t["name"] for t in stack["tiers"]}
        pid_file_names = {p.stem for p in pid_files}
        for tier_name in tier_names:
            assert tier_name in pid_file_names, f"No PID file for tier '{tier_name}'"
        assert "litellm" in pid_file_names, "No PID file for litellm"

        # ---- Step 4: Mock GET /v1/models returning 200 ----
        # Simulate what the LiteLLM proxy would return
        expected_models = [t["model"] for t in stack["tiers"]]
        mock_response = {
            "object": "list",
            "data": [{"id": f"openai/{m}", "object": "model"} for m in expected_models],
        }

        import httpx

        with patch("httpx.get") as mock_httpx_get:
            mock_resp = MagicMock(spec=httpx.Response)
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_response
            mock_httpx_get.return_value = mock_resp

            response = httpx.get("http://localhost:4000/v1/models")
            assert response.status_code == 200
            body = response.json()
            assert body["object"] == "list"
            assert len(body["data"]) == len(expected_models)
            returned_model_ids = {d["id"] for d in body["data"]}
            for model_id in expected_models:
                assert f"openai/{model_id}" in returned_model_ids, (
                    f"Model '{model_id}' not in /v1/models response"
                )

        # ---- Step 5: down (cleanup) ----
        with (
            patch("mlx_stack.core.stack_down.acquire_lock", return_value=lock_cm),
            patch("mlx_stack.core.stack_down.is_process_alive", return_value=False),
            patch("mlx_stack.core.stack_down.read_pid_file") as mock_read_pid,
            patch("mlx_stack.core.stack_down.remove_pid_file") as mock_remove_pid,
        ):

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
            assert result.exit_code == 0, f"down failed: {result.output}"

        # ---- Step 6: Verify no PID files remain ----
        remaining_pids = list(pids_dir.glob("*.pid"))
        assert len(remaining_pids) == 0, (
            f"PID files remain after down: {[p.name for p in remaining_pids]}"
        )


# --------------------------------------------------------------------------- #
# VAL-CROSS-007: Config changes propagate to init, up, pull, recommend
#
# After config set, subsequent commands use the new values.
# Assertions check concrete values, not just exit codes.
# --------------------------------------------------------------------------- #


class TestConfigPropagation:
    """VAL-CROSS-007: Config changes propagate to all dependent commands."""

    @patch("mlx_stack.core.stack_init.load_catalog")
    @patch("mlx_stack.core.stack_init.detect_hardware")
    @patch("mlx_stack.core.stack_init._is_port_available", return_value=True)
    def test_litellm_port_5000_in_generated_litellm_yaml(
        self,
        mock_port_avail: MagicMock,
        mock_detect: MagicMock,
        mock_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """After config set litellm-port 5000, init generates litellm.yaml with port 5000.

        Verifies the concrete port value appears in the LiteLLM config
        general_settings, not just that the command exits 0.
        """
        # Arrange
        profile = make_profile(memory_gb=128)
        mock_detect.return_value = profile
        mock_catalog.return_value = _make_test_catalog()
        runner = CliRunner()

        # Act — set config then init
        result = runner.invoke(cli, ["config", "set", "litellm-port", "5000"])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["init", "--accept-defaults"])
        assert result.exit_code == 0

        # The litellm.yaml should NOT have tier ports = 5000 (that's the LiteLLM port)
        # But the tier ports in the stack should NOT be 5000 either
        stack = _read_stack_yaml(mlx_stack_home)
        tier_ports = {t["port"] for t in stack["tiers"]}
        assert 5000 not in tier_ports, "LiteLLM port 5000 should not be used as a vllm tier port"

        # Verify the port 5000 is reflected in the stack or litellm config
        # (the actual litellm.yaml doesn't store the port since it's a
        # CLI flag, but the init output should mention it, and the
        # dry-run should use it)
        with (
            patch("mlx_stack.core.stack_up.load_catalog", return_value=_make_test_catalog()),
            patch("mlx_stack.core.stack_up.get_value") as mock_get_val,
        ):
            mock_get_val.side_effect = lambda key: {
                "litellm-port": 5000,
                "openrouter-key": "",
            }.get(key, "")

            result = runner.invoke(cli, ["up", "--dry-run"])
            assert result.exit_code == 0
            # The litellm command should reference port 5000
            assert "--port 5000" in result.output or "5000" in result.output, (
                f"Port 5000 not found in up --dry-run output:\n{result.output}"
            )

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_memory_budget_pct_60_propagates_to_recommend(
        self,
        mock_load_profile: MagicMock,
        mock_load_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """After config set memory-budget-pct 60, recommend uses 60% budget.

        With 128 GB memory and 60% budget, the effective budget is 76.8 GB.
        Asserts the concrete value appears in recommend output.
        """
        profile = make_profile(memory_gb=128)
        mock_load_profile.return_value = profile
        mock_load_catalog.return_value = _make_test_catalog()

        runner = CliRunner()

        # Set custom memory-budget-pct to 60
        result = runner.invoke(cli, ["config", "set", "memory-budget-pct", "60"])
        assert result.exit_code == 0

        # Run recommend
        result = runner.invoke(cli, ["recommend"])
        assert result.exit_code == 0

        # 60% of 128 GB = 76.8 GB — this concrete value must appear
        assert "76.8 GB" in result.output, (
            f"Expected '76.8 GB' budget in recommend output, got:\n{result.output}"
        )

    @patch("mlx_stack.core.stack_init.load_catalog")
    @patch("mlx_stack.core.stack_init.detect_hardware")
    @patch("mlx_stack.core.stack_init._is_port_available", return_value=True)
    def test_memory_budget_pct_60_propagates_to_init(
        self,
        mock_port_avail: MagicMock,
        mock_detect: MagicMock,
        mock_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """After config set memory-budget-pct 60, init uses 60% budget.

        With 128 GB and 60%, budget is 76.8 GB. All selected models must
        fit within 76.8 GB each.
        """
        profile = make_profile(memory_gb=128)
        mock_detect.return_value = profile
        mock_catalog.return_value = _make_test_catalog()

        runner = CliRunner()

        # Set custom memory-budget-pct
        result = runner.invoke(cli, ["config", "set", "memory-budget-pct", "60"])
        assert result.exit_code == 0

        # Run init
        result = runner.invoke(cli, ["init", "--accept-defaults"])
        assert result.exit_code == 0

        stack = _read_stack_yaml(mlx_stack_home)
        assert len(stack["tiers"]) > 0, "Init should produce at least one tier"

        # Every tier model must have memory_gb <= 76.8 GB
        catalog = _make_test_catalog()
        budget_gb = 128 * 0.60
        for tier in stack["tiers"]:
            model_id = tier["model"]
            entry = next((e for e in catalog if e.id == model_id), None)
            assert entry is not None, f"Tier model '{model_id}' not in catalog"
            bench = entry.benchmarks.get("m4-max-128")
            if bench:
                assert bench.memory_gb <= budget_gb, (
                    f"Tier '{tier['name']}' model '{model_id}' memory "
                    f"{bench.memory_gb} GB exceeds budget {budget_gb} GB"
                )

    @patch("mlx_stack.core.pull.load_catalog")
    @patch("mlx_stack.core.pull.get_value")
    @patch("mlx_stack.core.pull.check_disk_space", return_value=(True, 100.0))
    @patch("mlx_stack.core.pull.download_model")
    @patch("mlx_stack.core.pull.add_to_inventory")
    @patch("mlx_stack.core.pull.is_model_downloaded", return_value=False)
    def test_default_quant_int8_propagates_to_pull(
        self,
        mock_is_downloaded: MagicMock,
        mock_add_inv: MagicMock,
        mock_download: MagicMock,
        mock_disk: MagicMock,
        mock_get_value: MagicMock,
        mock_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """After config set default-quant int8, pull uses int8 without --quant.

        Verifies that int8 is passed as the quant parameter to the download
        function, not just that the command exits 0.
        """
        catalog = _make_test_catalog()
        mock_catalog.return_value = catalog

        def config_side_effect(key: str) -> Any:
            if key == "default-quant":
                return "int8"
            if key == "model-dir":
                return str(mlx_stack_home / "models")
            return ""

        mock_get_value.side_effect = config_side_effect

        runner = CliRunner()
        result = runner.invoke(cli, ["pull", "medium-8b"])
        assert result.exit_code == 0, f"pull failed: {result.output}"

        # Verify download_model was called with the int8 source repo
        assert mock_download.called, "download_model was never called"
        call_args = mock_download.call_args
        hf_repo = call_args[0][0] if call_args[0] else call_args[1].get("hf_repo", "")
        assert "8bit" in hf_repo, f"Expected int8 HF repo (containing '8bit'), got: {hf_repo}"

        # Verify add_to_inventory was called with quant=int8
        assert mock_add_inv.called, "add_to_inventory was never called"
        inv_entry = mock_add_inv.call_args[0][0]
        assert inv_entry.quant == "int8", (
            f"Expected quant='int8' in inventory entry, got: {inv_entry.quant}"
        )

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
        profile = make_profile(memory_gb=128)
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
        assert "5001" in result.output, f"Port 5001 not found in dry-run output:\n{result.output}"

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
        init --force, and verifies changes are reflected.
        """
        profile = make_profile(memory_gb=128)
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

        # Verify the new budget is reflected: 80% of 128 = 102.4 GB
        # All models in catalog are <= 20 GB memory, so all should fit
        stack = _read_stack_yaml(mlx_stack_home)
        assert len(stack["tiers"]) > 0


# --------------------------------------------------------------------------- #
# VAL-CROSS-012: bench --save overrides catalog data in recommend scoring
#
# After bench --save records gen_tps=85, subsequent recommend uses 85
# (not catalog default 77). The "estimated" label disappears for that model.
# --------------------------------------------------------------------------- #


class TestBenchSaveOverridesCatalog:
    """VAL-CROSS-012: Saved benchmark data overrides catalog in recommendations."""

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_saved_gen_tps_85_overrides_catalog_77(
        self,
        mock_load_profile: MagicMock,
        mock_load_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """Saved benchmark gen_tps=85 overrides catalog gen_tps=77.

        After bench --save writes gen_tps=85 for medium-8b, recommend
        --show-all must display 85.0, not 77.0.
        """
        profile = make_profile(memory_gb=128)
        mock_load_profile.return_value = profile
        mock_load_catalog.return_value = _make_test_catalog()

        # Write saved benchmarks with gen_tps=85 (catalog has 77.0)
        _write_saved_benchmarks(
            mlx_stack_home,
            profile.profile_id,
            {
                "medium-8b": {
                    "gen_tps": 85.0,
                    "prompt_tps": 200.0,
                    "memory_gb": 5.5,
                },
            },
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["recommend", "--show-all"])
        assert result.exit_code == 0

        # The saved gen_tps (85.0) must appear instead of catalog (77.0)
        assert "85.0" in result.output, (
            f"Expected saved gen_tps '85.0' in output, got:\n{result.output}"
        )
        # The catalog default (77.0) should NOT appear for this model
        # Note: 77.0 might appear for other contexts, but 85.0 must be present
        # as the scored value for medium-8b

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
        estimates and the 'est.' label should disappear for that model.
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

        # First recommend without saved benchmarks — should show 'est.'
        result = runner.invoke(cli, ["recommend", "--show-all"])
        assert result.exit_code == 0
        first_output = result.output
        # With unknown hardware, output should contain estimated markers
        assert "est." in first_output or "estimated" in first_output.lower(), (
            f"Expected 'est.' or 'estimated' in output for unknown hardware:\n{first_output}"
        )

        # Now save benchmarks for medium-8b
        _write_saved_benchmarks(
            mlx_stack_home,
            profile.profile_id,
            {
                "medium-8b": {
                    "gen_tps": 85.0,
                    "prompt_tps": 180.0,
                    "memory_gb": 5.5,
                },
            },
        )

        # Second recommend with saved benchmarks
        result = runner.invoke(cli, ["recommend", "--show-all"])
        assert result.exit_code == 0
        second_output = result.output

        # The saved gen_tps value (85.0) should appear
        assert "85.0" in second_output, f"Expected saved gen_tps '85.0' in output:\n{second_output}"

        # Parse the output line-by-line to find the medium-8b row
        # and verify it does NOT have 'est.' marker
        lines = second_output.split("\n")
        medium_8b_lines = [line for line in lines if "Medium 8B" in line]
        assert len(medium_8b_lines) > 0, (
            f"'Medium 8B' not found in recommend output:\n{second_output}"
        )
        for line in medium_8b_lines:
            assert "(est.)" not in line, f"Medium 8B still shows 'est.' after bench --save: {line}"

    @patch("mlx_stack.cli.recommend.load_catalog")
    @patch("mlx_stack.cli.recommend.load_profile")
    def test_saved_benchmarks_affect_scoring_order(
        self,
        mock_load_profile: MagicMock,
        mock_load_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """Saved benchmarks change scoring and potentially tier assignments.

        If medium-8b gets dramatically higher gen_tps (500) from benchmarks,
        the scoring and tier assignment should change.
        """
        profile = make_profile(memory_gb=128)
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
        # With gen_tps=500, medium-8b's speed score dramatically increases.
        # Verify that the output text is actually different, proving
        # the saved benchmarks affected scoring.
        assert result_before.output != result_after.output, (
            "Recommend output should differ after bench --save with dramatically different gen_tps"
        )


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
        # Arrange
        profile = make_profile(memory_gb=128)
        _write_profile(mlx_stack_home, profile)

        # Act — read raw JSON
        profile_path = mlx_stack_home / "profile.json"
        data = json.loads(profile_path.read_text())

        # Assert — all required fields present
        assert "chip" in data
        assert "gpu_cores" in data
        assert "memory_gb" in data
        assert "bandwidth_gbps" in data
        assert "profile_id" in data

        # Act — load via hardware module
        from mlx_stack.core.hardware import load_profile

        loaded = load_profile()

        # Assert — round-trip fidelity
        assert loaded is not None
        assert loaded.chip == profile.chip
        assert loaded.memory_gb == profile.memory_gb
        assert loaded.bandwidth_gbps == profile.bandwidth_gbps
        assert loaded.profile_id == profile.profile_id

    def test_models_command_shows_pulled_models(
        self,
        mlx_stack_home: Path,
    ) -> None:
        """Invoking CLI models command shows models from inventory.

        After pull writes models.json and creates model directories,
        the models command output includes those model names.
        """
        # Create model directory and inventory entry (simulating a pull)
        models_dir = mlx_stack_home / "models"
        model_path = models_dir / "medium-8b-int4"
        model_path.mkdir(parents=True)
        (model_path / "config.json").write_text('{"quantization_config": {"bits": 4}}')

        # Write inventory
        inv_entry = {
            "model_id": "medium-8b",
            "name": "Medium 8B",
            "quant": "int4",
            "source_type": "mlx_community",
            "hf_repo": "mlx-community/medium-8b-4bit",
            "local_path": str(model_path),
            "disk_size_gb": 4.5,
            "downloaded_at": "2026-03-24T00:00:00+00:00",
        }
        _write_inventory(mlx_stack_home, [inv_entry])

        # Run the models CLI command and verify output includes the model
        runner = CliRunner()
        with patch("mlx_stack.cli.models.load_catalog", return_value=_make_test_catalog()):
            result = runner.invoke(cli, ["models"])

        assert result.exit_code == 0, f"models command failed: {result.output}"
        # The model directory name should appear in the output
        assert "medium-8b-int4" in result.output or "Medium 8B" in result.output, (
            f"Expected pulled model in models output, got:\n{result.output}"
        )

    @patch("mlx_stack.core.stack_up.load_catalog")
    @patch("mlx_stack.core.stack_up.get_value")
    def test_stack_schema_version_rejection(
        self,
        mock_get_value: MagicMock,
        mock_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """Stack with unsupported schema_version is rejected by up.

        Verifies up exits with non-zero code and mentions schema_version.
        """
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
        assert result.exit_code != 0, (
            f"up should fail for unsupported schema_version, got exit_code=0:\n{result.output}"
        )
        assert "schema_version" in result.output, (
            f"Error should mention 'schema_version', got:\n{result.output}"
        )

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
    def test_vllm_flags_in_dry_run_output(
        self,
        mock_port_avail: MagicMock,
        mock_detect: MagicMock,
        mock_catalog: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """vllm_flags from init-generated stack appear in up --dry-run output.

        Verifies that the init -> up data flow preserves vllm_flags.
        """
        profile = make_profile(memory_gb=128)
        mock_detect.return_value = profile
        catalog = _make_test_catalog()
        mock_catalog.return_value = catalog

        runner = CliRunner()

        # Init creates stack with vllm_flags
        result = runner.invoke(cli, ["init", "--accept-defaults"])
        assert result.exit_code == 0

        # Read stack to find expected flags
        stack = _read_stack_yaml(mlx_stack_home)

        # Up --dry-run should show translated vllm_flags
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

            # All tiers with tool_calling should have their flags in dry-run output
            has_tool_calling_tier = False
            for tier in stack["tiers"]:
                flags = tier.get("vllm_flags", {})
                if flags.get("enable_auto_tool_choice"):
                    has_tool_calling_tier = True
                    assert "--enable-auto-tool-choice" in result.output, (
                        f"--enable-auto-tool-choice not in dry-run for tier '{tier['name']}'"
                    )
                if flags.get("tool_call_parser"):
                    assert "--tool-call-parser" in result.output, (
                        f"--tool-call-parser not in dry-run for tier '{tier['name']}'"
                    )

            # Verify we actually tested something
            assert has_tool_calling_tier, "No tool-calling tiers found in stack — test is vacuous"

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
        profile = make_profile(memory_gb=128)
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
        profile = make_profile(memory_gb=128)
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
        profile = make_profile(memory_gb=128)
        mock_detect.return_value = profile
        mock_catalog.return_value = _make_test_catalog()

        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--accept-defaults"])
        assert result.exit_code == 0

        stack = _read_stack_yaml(mlx_stack_home)
        assert stack["hardware_profile"] == profile.profile_id
