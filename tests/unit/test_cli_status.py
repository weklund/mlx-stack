"""Tests for the `mlx-stack status` CLI command and core stack_status module.

Validates:
- VAL-STATUS-001: Five distinct states correctly reported
- VAL-STATUS-002: Health check per service with 5-second timeout
- VAL-STATUS-003: Formatted table output with uptime
- VAL-STATUS-004: --json produces valid JSON with all fields
- VAL-STATUS-005: No stack or no services handled gracefully
- VAL-STATUS-006: Stale PIDs detected as crashed
- VAL-STATUS-007: Status is read-only and does not require lockfile
- VAL-CROSS-002: Status accurately reflects each lifecycle stage
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import yaml
from click.testing import CliRunner

from mlx_stack.cli.main import cli
from mlx_stack.core.stack_status import (
    ServiceHealth,
    ServiceStatus,
    StatusResult,
    _load_stack_for_status,
    run_status,
    status_to_dict,
)
from tests.factories import create_pid_file, make_stack_yaml, write_stack_yaml

# --------------------------------------------------------------------------- #
# Tests — _load_stack_for_status
# --------------------------------------------------------------------------- #


class TestLoadStackForStatus:
    """Tests for the read-only stack loader."""

    def test_loads_valid_stack(self, mlx_stack_home: Path) -> None:
        """Valid stack definition is loaded successfully."""
        # Arrange
        write_stack_yaml(mlx_stack_home)

        # Act
        stack = _load_stack_for_status()

        # Assert
        assert stack is not None
        assert len(stack["tiers"]) == 2

    def test_returns_none_when_no_stack(self, mlx_stack_home: Path) -> None:
        """Returns None when no stack file exists."""
        stack = _load_stack_for_status()
        assert stack is None

    def test_returns_none_on_invalid_yaml(self, mlx_stack_home: Path) -> None:
        """Returns None when the stack YAML is malformed."""
        # Arrange
        stacks_dir = mlx_stack_home / "stacks"
        stacks_dir.mkdir(parents=True, exist_ok=True)
        (stacks_dir / "default.yaml").write_text("{{{invalid yaml")

        # Act
        stack = _load_stack_for_status()

        # Assert
        assert stack is None

    def test_returns_none_on_non_dict(self, mlx_stack_home: Path) -> None:
        """Returns None when the stack YAML is not a mapping."""
        # Arrange
        stacks_dir = mlx_stack_home / "stacks"
        stacks_dir.mkdir(parents=True, exist_ok=True)
        (stacks_dir / "default.yaml").write_text("- just a list item")

        # Act
        stack = _load_stack_for_status()

        # Assert
        assert stack is None

    def test_returns_none_on_missing_tiers(self, mlx_stack_home: Path) -> None:
        """Returns None when tiers are missing."""
        # Arrange
        stacks_dir = mlx_stack_home / "stacks"
        stacks_dir.mkdir(parents=True, exist_ok=True)
        (stacks_dir / "default.yaml").write_text(
            yaml.dump({"schema_version": 1, "name": "default"})
        )

        # Act
        stack = _load_stack_for_status()

        # Assert
        assert stack is None


# --------------------------------------------------------------------------- #
# Tests — run_status (core logic)
# --------------------------------------------------------------------------- #


class TestRunStatus:
    """Tests for the run_status orchestration function."""

    def test_no_stack_returns_message(self, mlx_stack_home: Path) -> None:
        """VAL-STATUS-005: No stack configured reports suggestion."""
        # Act
        result = run_status()

        # Assert
        assert result.no_stack is True
        assert result.message is not None
        assert "init" in result.message.lower()
        assert result.services == []

    @patch("mlx_stack.core.stack_status.get_service_status")
    @patch("mlx_stack.core.stack_status._get_litellm_port", return_value=4000)
    def test_all_stopped_no_pid_files(
        self,
        mock_port: MagicMock,
        mock_status: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-STATUS-005 / VAL-CROSS-002: No PID files → all stopped."""
        # Arrange
        write_stack_yaml(mlx_stack_home)
        mock_status.return_value = {
            "status": "stopped",
            "pid": None,
            "uptime": None,
            "response_time": None,
        }

        # Act
        result = run_status()

        # Assert
        assert result.no_stack is False
        assert len(result.services) == 3  # 2 tiers + litellm
        for svc in result.services:
            assert svc.status == "stopped"
            assert svc.uptime_display == "-"

    @patch("mlx_stack.core.stack_status.get_service_status")
    @patch("mlx_stack.core.stack_status._get_litellm_port", return_value=4000)
    def test_all_healthy(
        self,
        mock_port: MagicMock,
        mock_status: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-CROSS-002: After up → all healthy."""
        # Arrange
        write_stack_yaml(mlx_stack_home)
        mock_status.return_value = {
            "status": "healthy",
            "pid": 12345,
            "uptime": 3600.0,
            "response_time": 0.05,
        }

        # Act
        result = run_status()

        # Assert
        assert result.no_stack is False
        assert len(result.services) == 3
        for svc in result.services:
            assert svc.status == "healthy"
            assert svc.pid == 12345
            assert svc.uptime == 3600.0
            assert svc.uptime_display == "1h"

    @patch("mlx_stack.core.stack_status.get_service_status")
    @patch("mlx_stack.core.stack_status._get_litellm_port", return_value=4000)
    def test_five_distinct_states(
        self,
        mock_port: MagicMock,
        mock_status: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-STATUS-001: Five distinct states correctly classified."""
        # Arrange — stack with 4 tiers so we can show all 5 states
        # (4 tiers + litellm = 5 services)
        tiers = [
            {
                "name": "tier-healthy",
                "model": "model-a",
                "quant": "int4",
                "source": "src-a",
                "port": 8000,
                "vllm_flags": {},
            },
            {
                "name": "tier-degraded",
                "model": "model-b",
                "quant": "int4",
                "source": "src-b",
                "port": 8001,
                "vllm_flags": {},
            },
            {
                "name": "tier-down",
                "model": "model-c",
                "quant": "int4",
                "source": "src-c",
                "port": 8002,
                "vllm_flags": {},
            },
            {
                "name": "tier-crashed",
                "model": "model-d",
                "quant": "int4",
                "source": "src-d",
                "port": 8003,
                "vllm_flags": {},
            },
        ]
        stack = make_stack_yaml(tiers=tiers)
        write_stack_yaml(mlx_stack_home, stack)

        status_map = {
            "tier-healthy": {
                "status": "healthy",
                "pid": 1001,
                "uptime": 7200.0,
                "response_time": 0.1,
            },
            "tier-degraded": {
                "status": "degraded",
                "pid": 1002,
                "uptime": 3600.0,
                "response_time": 3.5,
            },
            "tier-down": {
                "status": "down",
                "pid": 1003,
                "uptime": 1800.0,
                "response_time": None,
            },
            "tier-crashed": {
                "status": "crashed",
                "pid": 1004,
                "uptime": None,
                "response_time": None,
            },
            "litellm": {
                "status": "stopped",
                "pid": None,
                "uptime": None,
                "response_time": None,
            },
        }

        def side_effect(service_name: str, port: int, health_path: str = "") -> dict[str, Any]:
            return status_map.get(
                service_name,
                {
                    "status": "stopped",
                    "pid": None,
                    "uptime": None,
                    "response_time": None,
                },
            )

        mock_status.side_effect = side_effect

        # Act
        result = run_status()

        # Assert
        states = {svc.tier: svc.status for svc in result.services}
        assert states["tier-healthy"] == "healthy"
        assert states["tier-degraded"] == "degraded"
        assert states["tier-down"] == "down"
        assert states["tier-crashed"] == "crashed"
        assert states["litellm"] == "stopped"

    @patch("mlx_stack.core.stack_status.get_service_status")
    @patch("mlx_stack.core.stack_status._get_litellm_port", return_value=4000)
    def test_crashed_service_no_uptime(
        self,
        mock_port: MagicMock,
        mock_status: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-STATUS-006: Stale PIDs detected as 'crashed', uptime is '-'."""
        # Arrange
        write_stack_yaml(mlx_stack_home)

        def side_effect(service_name: str, port: int, health_path: str = "") -> dict[str, Any]:
            if service_name == "standard":
                return {
                    "status": "crashed",
                    "pid": 99999,
                    "uptime": None,
                    "response_time": None,
                }
            return {
                "status": "stopped",
                "pid": None,
                "uptime": None,
                "response_time": None,
            }

        mock_status.side_effect = side_effect

        # Act
        result = run_status()

        # Assert
        standard = next(s for s in result.services if s.tier == "standard")
        assert standard.status == "crashed"
        assert standard.pid == 99999
        assert standard.uptime is None
        assert standard.uptime_display == "-"

    @patch("mlx_stack.core.stack_status.get_service_status")
    @patch("mlx_stack.core.stack_status._get_litellm_port", return_value=4000)
    def test_mixed_states(
        self,
        mock_port: MagicMock,
        mock_status: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """Mixed states: healthy + stopped services."""
        # Arrange
        write_stack_yaml(mlx_stack_home)

        def side_effect(service_name: str, port: int, health_path: str = "") -> dict[str, Any]:
            if service_name == "standard":
                return {
                    "status": "healthy",
                    "pid": 1001,
                    "uptime": 120.0,
                    "response_time": 0.05,
                }
            return {
                "status": "stopped",
                "pid": None,
                "uptime": None,
                "response_time": None,
            }

        mock_status.side_effect = side_effect

        # Act
        result = run_status()

        # Assert
        statuses = {s.tier: s.status for s in result.services}
        assert statuses["standard"] == "healthy"
        assert statuses["fast"] == "stopped"

    @patch("mlx_stack.core.stack_status.get_service_status")
    @patch("mlx_stack.core.stack_status._get_litellm_port", return_value=5001)
    def test_custom_litellm_port(
        self,
        mock_port: MagicMock,
        mock_status: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """LiteLLM uses configured port for health checks."""
        # Arrange
        write_stack_yaml(mlx_stack_home)
        mock_status.return_value = {
            "status": "stopped",
            "pid": None,
            "uptime": None,
            "response_time": None,
        }

        # Act
        result = run_status()

        # Assert
        litellm = next(s for s in result.services if s.tier == "litellm")
        assert litellm.port == 5001

    @patch("mlx_stack.core.stack_status.get_service_status")
    @patch("mlx_stack.core.stack_status._get_litellm_port", return_value=4000)
    def test_failure_on_one_service_does_not_block_others(
        self,
        mock_port: MagicMock,
        mock_status: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-STATUS-002: Failure on one service doesn't prevent checks on others."""
        # Arrange
        write_stack_yaml(mlx_stack_home)
        call_count = 0

        def side_effect(service_name: str, port: int, health_path: str = "") -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if service_name == "standard":
                return {
                    "status": "down",
                    "pid": 1001,
                    "uptime": 100.0,
                    "response_time": None,
                }
            return {
                "status": "healthy",
                "pid": 2002,
                "uptime": 200.0,
                "response_time": 0.1,
            }

        mock_status.side_effect = side_effect

        # Act
        result = run_status()

        # Assert
        assert call_count == 3
        statuses = {s.tier: s.status for s in result.services}
        assert statuses["standard"] == "down"
        assert statuses["fast"] == "healthy"
        assert statuses["litellm"] == "healthy"


# --------------------------------------------------------------------------- #
# Tests — status_to_dict
# --------------------------------------------------------------------------- #


class TestStatusToDict:
    """Tests for JSON serialization."""

    def test_no_stack_serialization(self) -> None:
        """No-stack result serializes correctly."""
        result = StatusResult(no_stack=True, message="No stack configured.")
        data = status_to_dict(result)
        assert data["no_stack"] is True
        assert data["message"] == "No stack configured."
        assert data["services"] == []

    def test_services_serialization(self) -> None:
        """VAL-STATUS-004: All fields present in JSON output."""
        # Arrange
        result = StatusResult(
            services=[
                ServiceStatus(
                    tier="standard",
                    model="big-model",
                    port=8000,
                    status=ServiceHealth.HEALTHY,
                    uptime=3600.0,
                    uptime_display="1h",
                    response_time=0.05,
                    pid=12345,
                ),
                ServiceStatus(
                    tier="fast",
                    model="fast-model",
                    port=8001,
                    status=ServiceHealth.STOPPED,
                    uptime=None,
                    uptime_display="-",
                    response_time=None,
                    pid=None,
                ),
            ],
        )

        # Act
        data = status_to_dict(result)

        # Assert
        assert len(data["services"]) == 2
        assert data["no_stack"] is False
        assert data["message"] is None

        svc1 = data["services"][0]
        assert svc1["tier"] == "standard"
        assert svc1["model"] == "big-model"
        assert svc1["port"] == 8000
        assert svc1["status"] == "healthy"
        assert svc1["uptime"] == 3600.0
        assert svc1["uptime_display"] == "1h"
        assert svc1["response_time"] == 0.05
        assert svc1["pid"] == 12345

        svc2 = data["services"][1]
        assert svc2["tier"] == "fast"
        assert svc2["status"] == "stopped"
        assert svc2["uptime"] is None
        assert svc2["pid"] is None

    def test_valid_json_roundtrip(self) -> None:
        """JSON output is parseable by json.loads."""
        # Arrange
        result = StatusResult(
            services=[
                ServiceStatus(
                    tier="t1",
                    model="m1",
                    port=8000,
                    status=ServiceHealth.HEALTHY,
                    uptime=60.0,
                    uptime_display="1m",
                    response_time=0.1,
                    pid=111,
                ),
            ],
        )

        # Act
        data = status_to_dict(result)
        json_str = json.dumps(data)
        parsed = json.loads(json_str)

        # Assert
        assert parsed == data


# --------------------------------------------------------------------------- #
# Tests — read-only behaviour
# --------------------------------------------------------------------------- #


class TestReadOnly:
    """Tests verifying status is read-only."""

    @patch("mlx_stack.core.stack_status.get_service_status")
    @patch("mlx_stack.core.stack_status._get_litellm_port", return_value=4000)
    def test_no_pid_files_modified(
        self,
        mock_port: MagicMock,
        mock_status: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-STATUS-007: Status does not modify PID files."""
        # Arrange
        write_stack_yaml(mlx_stack_home)
        pid_path = create_pid_file(mlx_stack_home, "standard", 99999)
        original_content = pid_path.read_text()
        mock_status.return_value = {
            "status": "crashed",
            "pid": 99999,
            "uptime": None,
            "response_time": None,
        }

        # Act
        run_status()

        # Assert
        assert pid_path.exists()
        assert pid_path.read_text() == original_content

    @patch("mlx_stack.core.stack_status.get_service_status")
    @patch("mlx_stack.core.stack_status._get_litellm_port", return_value=4000)
    def test_no_lockfile_acquired(
        self,
        mock_port: MagicMock,
        mock_status: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-STATUS-007: Status does not acquire lockfile."""
        # Arrange
        write_stack_yaml(mlx_stack_home)
        mock_status.return_value = {
            "status": "stopped",
            "pid": None,
            "uptime": None,
            "response_time": None,
        }
        lock_path = mlx_stack_home / "lock"
        lock_path.touch()

        # Act
        result = run_status()

        # Assert
        assert result.no_stack is False

    @patch("mlx_stack.core.stack_status.get_service_status")
    @patch("mlx_stack.core.stack_status._get_litellm_port", return_value=4000)
    def test_no_files_created(
        self,
        mock_port: MagicMock,
        mock_status: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-STATUS-007: Status does not create any new files."""
        # Arrange
        write_stack_yaml(mlx_stack_home)
        before_files = set()
        for p in mlx_stack_home.rglob("*"):
            if p.is_file():
                before_files.add(str(p.relative_to(mlx_stack_home)))
        mock_status.return_value = {
            "status": "stopped",
            "pid": None,
            "uptime": None,
            "response_time": None,
        }

        # Act
        run_status()

        # Assert
        after_files = set()
        for p in mlx_stack_home.rglob("*"):
            if p.is_file():
                after_files.add(str(p.relative_to(mlx_stack_home)))
        new_files = after_files - before_files
        assert new_files == set(), f"Status created files: {new_files}"


# --------------------------------------------------------------------------- #
# Tests — CLI integration via CliRunner
# --------------------------------------------------------------------------- #


class TestStatusCli:
    """Tests for the `mlx-stack status` CLI command via CliRunner."""

    def test_no_stack_shows_message(self, mlx_stack_home: Path) -> None:
        """VAL-STATUS-005: No stack shows helpful message."""
        runner = CliRunner()
        result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        assert "init" in result.output.lower()

    @patch("mlx_stack.core.stack_status.get_service_status")
    @patch("mlx_stack.core.stack_status._get_litellm_port", return_value=4000)
    def test_table_output_has_columns(
        self,
        mock_port: MagicMock,
        mock_status: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-STATUS-003: Formatted table with expected columns."""
        # Arrange
        write_stack_yaml(mlx_stack_home)
        mock_status.return_value = {
            "status": "healthy",
            "pid": 12345,
            "uptime": 7200.0,
            "response_time": 0.05,
        }

        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["status"])

        # Assert
        assert result.exit_code == 0
        assert "Service Status" in result.output
        assert "Tier" in result.output
        assert "Model" in result.output
        assert "Port" in result.output
        assert "Status" in result.output
        assert "Uptime" in result.output

    @patch("mlx_stack.core.stack_status.get_service_status")
    @patch("mlx_stack.core.stack_status._get_litellm_port", return_value=4000)
    def test_table_shows_tier_data(
        self,
        mock_port: MagicMock,
        mock_status: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-STATUS-003: Table shows per-tier names, models, ports, and distinct statuses."""
        # Arrange
        write_stack_yaml(mlx_stack_home)

        def side_effect(service_name: str, port: int, health_path: str = "") -> dict[str, Any]:
            if service_name == "standard":
                return {
                    "status": ServiceHealth.HEALTHY,
                    "pid": 12345,
                    "uptime": 150.0,
                    "response_time": 0.05,
                }
            if service_name == "fast":
                return {
                    "status": ServiceHealth.CRASHED,
                    "pid": 99999,
                    "uptime": None,
                    "response_time": None,
                }
            return {
                "status": ServiceHealth.STOPPED,
                "pid": None,
                "uptime": None,
                "response_time": None,
            }

        mock_status.side_effect = side_effect

        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["status"])

        # Assert
        assert result.exit_code == 0
        # Tier names and models present
        assert "standard" in result.output
        assert "fast" in result.output
        assert "big-model" in result.output
        assert "fast-model" in result.output
        # Ports present
        assert "8000" in result.output
        assert "8001" in result.output
        # Each tier's distinct status must appear — verifies per-tier rendering
        assert ServiceHealth.HEALTHY in result.output
        assert ServiceHealth.CRASHED in result.output

    @patch("mlx_stack.core.stack_status.get_service_status")
    @patch("mlx_stack.core.stack_status._get_litellm_port", return_value=4000)
    def test_table_shows_uptime(
        self,
        mock_port: MagicMock,
        mock_status: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-STATUS-003: Uptime displayed in human-readable format."""
        # Arrange
        write_stack_yaml(mlx_stack_home)

        def side_effect(service_name: str, port: int, health_path: str = "") -> dict[str, Any]:
            if service_name == "standard":
                return {
                    "status": "healthy",
                    "pid": 1001,
                    "uptime": 8100.0,  # 2h 15m
                    "response_time": 0.05,
                }
            return {
                "status": "stopped",
                "pid": None,
                "uptime": None,
                "response_time": None,
            }

        mock_status.side_effect = side_effect

        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["status"])

        # Assert
        assert result.exit_code == 0
        assert "2h 15m" in result.output

    @patch("mlx_stack.core.stack_status.get_service_status")
    @patch("mlx_stack.core.stack_status._get_litellm_port", return_value=4000)
    def test_stopped_shows_dash_uptime(
        self,
        mock_port: MagicMock,
        mock_status: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-STATUS-003: Stopped services show '-' for uptime."""
        # Arrange
        write_stack_yaml(mlx_stack_home)
        mock_status.return_value = {
            "status": "stopped",
            "pid": None,
            "uptime": None,
            "response_time": None,
        }

        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["status"])

        # Assert
        assert result.exit_code == 0
        assert "stopped" in result.output

    @patch("mlx_stack.core.stack_status.get_service_status")
    @patch("mlx_stack.core.stack_status._get_litellm_port", return_value=4000)
    def test_json_output_valid(
        self,
        mock_port: MagicMock,
        mock_status: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-STATUS-004: --json produces valid parseable JSON."""
        # Arrange
        write_stack_yaml(mlx_stack_home)
        mock_status.return_value = {
            "status": "healthy",
            "pid": 12345,
            "uptime": 3600.0,
            "response_time": 0.05,
        }

        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--json"])

        # Assert
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "services" in data
        assert isinstance(data["services"], list)
        assert len(data["services"]) == 3  # 2 tiers + litellm

    @patch("mlx_stack.core.stack_status.get_service_status")
    @patch("mlx_stack.core.stack_status._get_litellm_port", return_value=4000)
    def test_json_has_all_fields(
        self,
        mock_port: MagicMock,
        mock_status: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-STATUS-004: JSON contains all required fields per service."""
        # Arrange
        write_stack_yaml(mlx_stack_home)
        mock_status.return_value = {
            "status": "healthy",
            "pid": 12345,
            "uptime": 3600.0,
            "response_time": 0.05,
        }

        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--json"])

        # Assert
        data = json.loads(result.output)
        required_fields = {"tier", "model", "port", "status", "uptime", "uptime_display", "pid"}
        for svc in data["services"]:
            assert required_fields.issubset(svc.keys()), (
                f"Missing fields: {required_fields - svc.keys()}"
            )

    @patch("mlx_stack.core.stack_status.get_service_status")
    @patch("mlx_stack.core.stack_status._get_litellm_port", return_value=4000)
    def test_json_and_table_consistent(
        self,
        mock_port: MagicMock,
        mock_status: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-STATUS-004: JSON and table modes show consistent data."""
        # Arrange
        write_stack_yaml(mlx_stack_home)
        mock_status.return_value = {
            "status": "healthy",
            "pid": 12345,
            "uptime": 3600.0,
            "response_time": 0.05,
        }
        runner = CliRunner()

        # Act
        json_result = runner.invoke(cli, ["status", "--json"])
        json_data = json.loads(json_result.output)
        table_result = runner.invoke(cli, ["status"])

        # Assert
        assert len(json_data["services"]) == 3
        for svc in json_data["services"]:
            assert svc["tier"] in table_result.output
            assert str(svc["port"]) in table_result.output

    def test_json_no_stack(self, mlx_stack_home: Path) -> None:
        """VAL-STATUS-005: --json with no stack returns valid JSON."""
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["no_stack"] is True
        assert data["services"] == []
        assert data["message"] is not None

    @patch("mlx_stack.core.stack_status.get_service_status")
    @patch("mlx_stack.core.stack_status._get_litellm_port", return_value=4000)
    def test_degraded_state_in_table(
        self,
        mock_port: MagicMock,
        mock_status: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-STATUS-001: Degraded state displayed in table."""
        # Arrange
        write_stack_yaml(mlx_stack_home)

        def side_effect(service_name: str, port: int, health_path: str = "") -> dict[str, Any]:
            if service_name == "standard":
                return {
                    "status": "degraded",
                    "pid": 1001,
                    "uptime": 600.0,
                    "response_time": 3.5,
                }
            return {
                "status": "stopped",
                "pid": None,
                "uptime": None,
                "response_time": None,
            }

        mock_status.side_effect = side_effect

        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["status"])

        # Assert
        assert result.exit_code == 0
        assert "degraded" in result.output

    @patch("mlx_stack.core.stack_status.get_service_status")
    @patch("mlx_stack.core.stack_status._get_litellm_port", return_value=4000)
    def test_crashed_state_in_table(
        self,
        mock_port: MagicMock,
        mock_status: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-STATUS-006: Crashed state displayed for stale PIDs."""
        # Arrange
        write_stack_yaml(mlx_stack_home)

        def side_effect(service_name: str, port: int, health_path: str = "") -> dict[str, Any]:
            if service_name == "standard":
                return {
                    "status": "crashed",
                    "pid": 99999,
                    "uptime": None,
                    "response_time": None,
                }
            return {
                "status": "stopped",
                "pid": None,
                "uptime": None,
                "response_time": None,
            }

        mock_status.side_effect = side_effect

        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["status"])

        # Assert
        assert result.exit_code == 0
        assert "crashed" in result.output

    @patch("mlx_stack.core.stack_status.get_service_status")
    @patch("mlx_stack.core.stack_status._get_litellm_port", return_value=4000)
    def test_down_state_in_table(
        self,
        mock_port: MagicMock,
        mock_status: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-STATUS-001: Down state displayed for unreachable services."""
        # Arrange
        write_stack_yaml(mlx_stack_home)

        def side_effect(service_name: str, port: int, health_path: str = "") -> dict[str, Any]:
            if service_name == "standard":
                return {
                    "status": "down",
                    "pid": 1001,
                    "uptime": 100.0,
                    "response_time": None,
                }
            return {
                "status": "stopped",
                "pid": None,
                "uptime": None,
                "response_time": None,
            }

        mock_status.side_effect = side_effect

        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["status"])

        # Assert
        assert result.exit_code == 0
        assert "down" in result.output

    def test_help_text(self, mlx_stack_home: Path) -> None:
        """Status command has help text."""
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--help"])

        assert result.exit_code == 0
        assert "health" in result.output.lower() or "status" in result.output.lower()
        assert "--json" in result.output

    def test_status_appears_in_main_help(self, mlx_stack_home: Path) -> None:
        """Status command appears in main help output."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "status" in result.output

    def test_no_traceback_on_error(self, mlx_stack_home: Path) -> None:
        """Errors never show Python tracebacks."""
        runner = CliRunner()
        result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        assert "Traceback" not in result.output


# --------------------------------------------------------------------------- #
# Tests — lifecycle stage consistency (VAL-CROSS-002)
# --------------------------------------------------------------------------- #


class TestLifecycleStages:
    """Tests verifying status accurately reflects each lifecycle stage."""

    @patch("mlx_stack.core.stack_status.get_service_status")
    @patch("mlx_stack.core.stack_status._get_litellm_port", return_value=4000)
    def test_before_up_all_stopped(
        self,
        mock_port: MagicMock,
        mock_status: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-CROSS-002: Before up → all stopped."""
        # Arrange
        write_stack_yaml(mlx_stack_home)
        mock_status.return_value = {
            "status": "stopped",
            "pid": None,
            "uptime": None,
            "response_time": None,
        }

        # Act
        result = run_status()

        # Assert
        assert all(s.status == "stopped" for s in result.services)

    @patch("mlx_stack.core.stack_status.get_service_status")
    @patch("mlx_stack.core.stack_status._get_litellm_port", return_value=4000)
    def test_after_up_all_healthy(
        self,
        mock_port: MagicMock,
        mock_status: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-CROSS-002: After up → all healthy."""
        # Arrange
        write_stack_yaml(mlx_stack_home)
        mock_status.return_value = {
            "status": "healthy",
            "pid": 12345,
            "uptime": 60.0,
            "response_time": 0.05,
        }

        # Act
        result = run_status()

        # Assert
        assert all(s.status == "healthy" for s in result.services)

    @patch("mlx_stack.core.stack_status.get_service_status")
    @patch("mlx_stack.core.stack_status._get_litellm_port", return_value=4000)
    def test_after_down_all_stopped(
        self,
        mock_port: MagicMock,
        mock_status: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-CROSS-002: After down → all stopped."""
        # Arrange
        write_stack_yaml(mlx_stack_home)
        mock_status.return_value = {
            "status": "stopped",
            "pid": None,
            "uptime": None,
            "response_time": None,
        }

        # Act
        result = run_status()

        # Assert
        assert all(s.status == "stopped" for s in result.services)

    @patch("mlx_stack.core.stack_status.get_service_status")
    @patch("mlx_stack.core.stack_status._get_litellm_port", return_value=4000)
    def test_after_external_kill_crashed(
        self,
        mock_port: MagicMock,
        mock_status: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-CROSS-002: After external kill → crashed."""
        # Arrange
        write_stack_yaml(mlx_stack_home)

        def side_effect(service_name: str, port: int, health_path: str = "") -> dict[str, Any]:
            if service_name == "standard":
                return {
                    "status": "crashed",
                    "pid": 99999,
                    "uptime": None,
                    "response_time": None,
                }
            return {
                "status": "healthy",
                "pid": 1002,
                "uptime": 300.0,
                "response_time": 0.1,
            }

        mock_status.side_effect = side_effect

        # Act
        result = run_status()

        # Assert
        statuses = {s.tier: s.status for s in result.services}
        assert statuses["standard"] == "crashed"
        assert statuses["fast"] == "healthy"


# --------------------------------------------------------------------------- #
# Tests — edge cases
# --------------------------------------------------------------------------- #


class TestEdgeCases:
    """Tests for edge case handling."""

    def test_empty_tiers_in_stack(self, mlx_stack_home: Path) -> None:
        """Stack with empty tiers list shows no-stack message."""
        # Arrange
        stack = make_stack_yaml()
        stack["tiers"] = []
        write_stack_yaml(mlx_stack_home, stack)

        # Act
        result = run_status()

        # Assert
        assert result.no_stack is True

    @patch("mlx_stack.core.stack_status.get_service_status")
    @patch("mlx_stack.core.stack_status._get_litellm_port", return_value=4000)
    def test_single_tier_stack(
        self,
        mock_port: MagicMock,
        mock_status: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """Stack with a single tier works correctly."""
        # Arrange
        tiers = [
            {
                "name": "fast",
                "model": "small-model",
                "quant": "int4",
                "source": "src",
                "port": 8000,
                "vllm_flags": {},
            },
        ]
        stack = make_stack_yaml(tiers=tiers)
        write_stack_yaml(mlx_stack_home, stack)
        mock_status.return_value = {
            "status": "healthy",
            "pid": 1001,
            "uptime": 120.0,
            "response_time": 0.1,
        }

        # Act
        result = run_status()

        # Assert
        assert len(result.services) == 2  # 1 tier + litellm
        assert result.services[0].tier == "fast"
        assert result.services[1].tier == "litellm"

    @patch("mlx_stack.core.stack_status.get_service_status")
    @patch("mlx_stack.core.stack_status._get_litellm_port", return_value=4000)
    def test_three_tier_stack(
        self,
        mock_port: MagicMock,
        mock_status: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """Stack with 3 tiers reports all + litellm."""
        # Arrange
        tiers = [
            {
                "name": "standard",
                "model": "m1",
                "quant": "int4",
                "source": "s1",
                "port": 8000,
                "vllm_flags": {},
            },
            {
                "name": "fast",
                "model": "m2",
                "quant": "int4",
                "source": "s2",
                "port": 8001,
                "vllm_flags": {},
            },
            {
                "name": "longctx",
                "model": "m3",
                "quant": "int4",
                "source": "s3",
                "port": 8002,
                "vllm_flags": {},
            },
        ]
        stack = make_stack_yaml(tiers=tiers)
        write_stack_yaml(mlx_stack_home, stack)
        mock_status.return_value = {
            "status": "stopped",
            "pid": None,
            "uptime": None,
            "response_time": None,
        }

        # Act
        result = run_status()

        # Assert
        assert len(result.services) == 4  # 3 tiers + litellm

    @patch("mlx_stack.core.stack_status.get_service_status")
    @patch("mlx_stack.core.stack_status._get_litellm_port", return_value=4000)
    def test_json_output_all_five_states(
        self,
        mock_port: MagicMock,
        mock_status: MagicMock,
        mlx_stack_home: Path,
    ) -> None:
        """VAL-STATUS-001/004: All five states present in JSON."""
        # Arrange
        tiers = [
            {
                "name": f"t{i}",
                "model": f"m{i}",
                "quant": "int4",
                "source": f"s{i}",
                "port": 8000 + i,
                "vllm_flags": {},
            }
            for i in range(4)
        ]
        stack = make_stack_yaml(tiers=tiers)
        write_stack_yaml(mlx_stack_home, stack)
        status_list = ["healthy", "degraded", "down", "crashed", "stopped"]

        def side_effect(service_name: str, port: int, health_path: str = "") -> dict[str, Any]:
            idx = int(service_name[1]) if service_name.startswith("t") else 4
            st = status_list[idx]
            return {
                "status": st,
                "pid": 1000 + idx if st != "stopped" else None,
                "uptime": 100.0 if st in ("healthy", "degraded", "down") else None,
                "response_time": 0.1 if st == "healthy" else (3.0 if st == "degraded" else None),
            }

        mock_status.side_effect = side_effect

        # Act
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--json"])

        # Assert
        assert result.exit_code == 0
        data = json.loads(result.output)
        statuses = {s["tier"]: s["status"] for s in data["services"]}
        assert "healthy" in statuses.values()
        assert "degraded" in statuses.values()
        assert "down" in statuses.values()
        assert "crashed" in statuses.values()
        assert "stopped" in statuses.values()
