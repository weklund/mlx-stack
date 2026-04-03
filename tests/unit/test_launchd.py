"""Tests for the launchd core module.

Covers plist generation, file I/O, launchctl operations, status
checking, platform/prerequisite guards, environment variables,
and the high-level install/uninstall operations.
"""

from __future__ import annotations

import os
import plistlib
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from mlx_stack.core.launchd import (
    LAUNCHD_LABEL,
    PLIST_FILENAME,
    PLIST_PERMISSIONS,
    AgentStatus,
    LaunchdError,
    PlatformError,
    PrerequisiteError,
    _build_environment_variables,
    _get_agent_pid,
    _get_gui_uid,
    _resolve_mlx_stack_binary,
    check_init_prerequisite,
    check_platform,
    generate_plist,
    get_agent_status,
    get_plist_path,
    install_agent,
    load_agent,
    uninstall_agent,
    unload_agent,
    write_plist,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def stack_definition(mlx_stack_home: Path) -> dict[str, Any]:
    """Create a minimal stack definition (prerequisite for install)."""
    stacks_dir = mlx_stack_home / "stacks"
    stacks_dir.mkdir(parents=True, exist_ok=True)

    stack = {
        "schema_version": 1,
        "name": "default",
        "hardware_profile": "m4-pro-48",
        "intent": "balanced",
        "created": "2025-01-01T00:00:00",
        "tiers": [
            {
                "name": "fast",
                "model": "qwen3.5-3b",
                "quant": "int4",
                "source": "mlx-community/Qwen3.5-3B-4bit",
                "port": 8000,
                "vllm_flags": {},
            },
        ],
    }

    stack_path = stacks_dir / "default.yaml"
    stack_path.write_text(yaml.dump(stack))
    return stack


@pytest.fixture()
def plist_dir(tmp_path: Path) -> Path:
    """Provide a temporary LaunchAgents directory."""
    d = tmp_path / "Library" / "LaunchAgents"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture()
def plist_path(plist_dir: Path) -> Path:
    """Provide a plist file path."""
    return plist_dir / PLIST_FILENAME


# --------------------------------------------------------------------------- #
# Constants tests
# --------------------------------------------------------------------------- #


class TestConstants:
    """Test that launchd constants are correct."""

    def test_launchd_label(self) -> None:
        assert LAUNCHD_LABEL == "com.mlx-stack.watchdog"

    def test_plist_filename(self) -> None:
        assert PLIST_FILENAME == "com.mlx-stack.watchdog.plist"

    def test_plist_permissions(self) -> None:
        assert PLIST_PERMISSIONS == 0o644


# --------------------------------------------------------------------------- #
# AgentStatus tests
# --------------------------------------------------------------------------- #


class TestAgentStatus:
    """Tests for the AgentStatus dataclass."""

    def test_not_installed_message(self) -> None:
        status = AgentStatus(installed=False, running=False, pid=None)
        assert status.message == "not installed"

    def test_installed_and_running_message(self) -> None:
        status = AgentStatus(installed=True, running=True, pid=12345)
        assert status.message == "installed and running (PID 12345)"

    def test_installed_but_not_running_message(self) -> None:
        status = AgentStatus(installed=True, running=False, pid=None)
        assert status.message == "installed but not running"

    def test_label_default(self) -> None:
        status = AgentStatus(installed=False, running=False, pid=None)
        assert status.label == LAUNCHD_LABEL

    def test_frozen(self) -> None:
        status = AgentStatus(installed=True, running=True, pid=123)
        with pytest.raises(AttributeError):
            status.installed = False  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# get_plist_path tests
# --------------------------------------------------------------------------- #


class TestGetPlistPath:
    """Tests for get_plist_path."""

    def test_returns_home_library_launch_agents(self) -> None:
        path = get_plist_path()
        assert path == Path.home() / "Library" / "LaunchAgents" / PLIST_FILENAME

    def test_path_ends_with_plist_extension(self) -> None:
        path = get_plist_path()
        assert path.suffix == ".plist"


# --------------------------------------------------------------------------- #
# Platform check tests
# --------------------------------------------------------------------------- #


class TestCheckPlatform:
    """Tests for check_platform."""

    def test_darwin_passes(self) -> None:
        with patch("mlx_stack.core.launchd.sys") as mock_sys:
            mock_sys.platform = "darwin"
            check_platform()  # Should not raise

    def test_linux_raises_platform_error(self) -> None:
        with patch("mlx_stack.core.launchd.sys") as mock_sys:
            mock_sys.platform = "linux"
            with pytest.raises(PlatformError, match="only available on macOS"):
                check_platform()

    def test_windows_raises_platform_error(self) -> None:
        with patch("mlx_stack.core.launchd.sys") as mock_sys:
            mock_sys.platform = "win32"
            with pytest.raises(PlatformError, match="only available on macOS"):
                check_platform()

    def test_error_includes_current_platform(self) -> None:
        with patch("mlx_stack.core.launchd.sys") as mock_sys:
            mock_sys.platform = "linux"
            with pytest.raises(PlatformError, match="linux"):
                check_platform()


# --------------------------------------------------------------------------- #
# Prerequisite check tests
# --------------------------------------------------------------------------- #


class TestCheckInitPrerequisite:
    """Tests for check_init_prerequisite."""

    def test_passes_when_stack_exists(
        self, mlx_stack_home: Path, stack_definition: dict[str, Any]
    ) -> None:
        check_init_prerequisite()  # Should not raise

    def test_raises_when_no_stack(self, mlx_stack_home: Path) -> None:
        with pytest.raises(PrerequisiteError, match="No stack configuration found"):
            check_init_prerequisite()

    def test_error_suggests_init(self, mlx_stack_home: Path) -> None:
        with pytest.raises(PrerequisiteError, match="mlx-stack init"):
            check_init_prerequisite()


# --------------------------------------------------------------------------- #
# Binary resolution tests
# --------------------------------------------------------------------------- #


class TestResolveMlxStackBinary:
    """Tests for _resolve_mlx_stack_binary."""

    def test_finds_binary_on_path(self) -> None:
        with patch("shutil.which", return_value="/usr/local/bin/mlx-stack"):
            result = _resolve_mlx_stack_binary()
        assert result == "/usr/local/bin/mlx-stack"

    def test_fallback_to_sys_executable_dir(self) -> None:
        with (
            patch("shutil.which", return_value=None),
            patch("mlx_stack.core.launchd.sys") as mock_sys,
            patch("mlx_stack.core.launchd.Path") as mock_path_cls,
        ):
            mock_sys.executable = "/opt/venv/bin/python"
            mock_parent = MagicMock()
            mock_candidate = MagicMock()
            mock_candidate.exists.return_value = True
            mock_candidate.__str__ = MagicMock(return_value="/opt/venv/bin/mlx-stack")
            mock_parent.__truediv__ = lambda self, name: mock_candidate

            # Chain: Path(sys.executable).parent
            mock_path_instance = MagicMock()
            mock_path_instance.parent = mock_parent
            mock_path_cls.side_effect = (
                lambda x: mock_path_instance
                if x == "/opt/venv/bin/python"
                else MagicMock()
            )

            result = _resolve_mlx_stack_binary()
        assert result == "/opt/venv/bin/mlx-stack"

    def test_raises_when_not_found(self) -> None:
        with (
            patch("shutil.which", return_value=None),
            patch("mlx_stack.core.launchd.Path") as mock_path_cls,
        ):
            mock_instance = MagicMock()
            mock_parent = MagicMock()
            mock_candidate = MagicMock()
            mock_candidate.exists.return_value = False
            mock_parent.__truediv__ = lambda self, name: mock_candidate
            mock_instance.parent = mock_parent
            mock_path_cls.side_effect = lambda x: mock_instance

            with pytest.raises(LaunchdError, match="Could not find the mlx-stack binary"):
                _resolve_mlx_stack_binary()


# --------------------------------------------------------------------------- #
# Environment variables tests
# --------------------------------------------------------------------------- #


class TestBuildEnvironmentVariables:
    """Tests for _build_environment_variables."""

    def test_path_includes_binary_dir(self) -> None:
        env = _build_environment_variables("/opt/homebrew/bin/mlx-stack")
        assert "PATH" in env
        assert "/opt/homebrew/bin" in env["PATH"].split(":")

    def test_binary_dir_is_first_in_path(self) -> None:
        env = _build_environment_variables("/custom/dir/mlx-stack")
        components = env["PATH"].split(":")
        assert components[0] == "/custom/dir"

    def test_standard_paths_included(self) -> None:
        env = _build_environment_variables("/usr/local/bin/mlx-stack")
        path = env["PATH"]
        assert "/usr/bin" in path
        assert "/bin" in path
        assert "/usr/sbin" in path
        assert "/sbin" in path

    def test_no_duplicate_binary_dir_in_standard_paths(self) -> None:
        """If binary dir is in standard paths, it should not appear twice."""
        env = _build_environment_variables("/usr/local/bin/mlx-stack")
        components = env["PATH"].split(":")
        count = components.count("/usr/local/bin")
        assert count == 1

    def test_mlx_stack_home_not_set_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MLX_STACK_HOME", raising=False)
        env = _build_environment_variables("/usr/bin/mlx-stack")
        assert "MLX_STACK_HOME" not in env

    def test_mlx_stack_home_included_when_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MLX_STACK_HOME", "/custom/home")
        env = _build_environment_variables("/usr/bin/mlx-stack")
        assert env["MLX_STACK_HOME"] == "/custom/home"

    def test_mlx_stack_home_empty_string_not_included(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MLX_STACK_HOME", "")
        env = _build_environment_variables("/usr/bin/mlx-stack")
        assert "MLX_STACK_HOME" not in env


# --------------------------------------------------------------------------- #
# Plist generation tests
# --------------------------------------------------------------------------- #


class TestGeneratePlist:
    """Tests for generate_plist."""

    def test_required_keys_present(self, mlx_stack_home: Path) -> None:
        plist = generate_plist("/usr/local/bin/mlx-stack")
        assert plist["Label"] == LAUNCHD_LABEL
        assert plist["RunAtLoad"] is True
        assert plist["KeepAlive"] is True
        assert "ProgramArguments" in plist
        assert "StandardOutPath" in plist
        assert "StandardErrorPath" in plist
        assert "EnvironmentVariables" in plist

    def test_program_arguments(self, mlx_stack_home: Path) -> None:
        plist = generate_plist("/usr/local/bin/mlx-stack")
        args = plist["ProgramArguments"]
        assert args == ["/usr/local/bin/mlx-stack", "watch"]

    def test_log_paths(self, mlx_stack_home: Path) -> None:
        plist = generate_plist("/usr/local/bin/mlx-stack")
        assert plist["StandardOutPath"].endswith("watchdog.stdout.log")
        assert plist["StandardErrorPath"].endswith("watchdog.stderr.log")
        assert "logs" in plist["StandardOutPath"]
        assert "logs" in plist["StandardErrorPath"]

    def test_auto_resolves_binary(self, mlx_stack_home: Path) -> None:
        with patch(
            "mlx_stack.core.launchd._resolve_mlx_stack_binary",
            return_value="/auto/mlx-stack",
        ):
            plist = generate_plist()
        assert plist["ProgramArguments"][0] == "/auto/mlx-stack"

    def test_plist_is_valid_plistlib_roundtrip(self, mlx_stack_home: Path) -> None:
        """The plist dict should survive a plistlib dump/load roundtrip."""
        plist = generate_plist("/usr/local/bin/mlx-stack")
        data = plistlib.dumps(plist)
        loaded = plistlib.loads(data)
        assert loaded["Label"] == LAUNCHD_LABEL
        assert loaded["RunAtLoad"] is True
        assert loaded["KeepAlive"] is True


# --------------------------------------------------------------------------- #
# Write plist tests
# --------------------------------------------------------------------------- #


class TestWritePlist:
    """Tests for write_plist."""

    def test_creates_plist_file(self, plist_path: Path) -> None:
        plist_data = {"Label": LAUNCHD_LABEL, "RunAtLoad": True}
        result = write_plist(plist_data, plist_path)
        assert result == plist_path
        assert plist_path.exists()

    def test_file_is_valid_plist(self, plist_path: Path) -> None:
        plist_data = {
            "Label": LAUNCHD_LABEL,
            "RunAtLoad": True,
            "ProgramArguments": ["/usr/bin/mlx-stack", "watch"],
        }
        write_plist(plist_data, plist_path)

        with open(plist_path, "rb") as f:
            loaded = plistlib.load(f)

        assert loaded["Label"] == LAUNCHD_LABEL
        assert loaded["RunAtLoad"] is True
        assert loaded["ProgramArguments"] == ["/usr/bin/mlx-stack", "watch"]

    def test_sets_permissions(self, plist_path: Path) -> None:
        plist_data = {"Label": LAUNCHD_LABEL}
        write_plist(plist_data, plist_path)
        mode = plist_path.stat().st_mode & 0o777
        assert mode == PLIST_PERMISSIONS

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        deep_path = tmp_path / "new" / "dir" / PLIST_FILENAME
        plist_data = {"Label": LAUNCHD_LABEL}
        write_plist(plist_data, deep_path)
        assert deep_path.exists()

    def test_overwrites_existing_file(self, plist_path: Path) -> None:
        # Write first version
        write_plist({"Label": "old"}, plist_path)
        # Overwrite with new
        write_plist({"Label": "new"}, plist_path)

        with open(plist_path, "rb") as f:
            loaded = plistlib.load(f)
        assert loaded["Label"] == "new"

    def test_raises_on_write_failure(self, tmp_path: Path) -> None:
        bad_path = tmp_path / "nonexistent" / "deep" / "path" / PLIST_FILENAME
        # Make the first parent read-only so mkdir fails
        parent = tmp_path / "nonexistent"
        parent.mkdir()
        parent.chmod(0o444)

        with pytest.raises(LaunchdError, match="Failed to write plist"):
            write_plist({"Label": "test"}, bad_path)

        # Cleanup permissions for tmp_path cleanup
        parent.chmod(0o755)


# --------------------------------------------------------------------------- #
# launchctl operations tests
# --------------------------------------------------------------------------- #


class TestGetGuiUid:
    """Tests for _get_gui_uid."""

    def test_returns_current_uid(self) -> None:
        uid = _get_gui_uid()
        assert uid == os.getuid()


class TestLoadAgent:
    """Tests for load_agent."""

    def test_success(self, plist_path: Path) -> None:
        # Create the plist file
        write_plist({"Label": LAUNCHD_LABEL}, plist_path)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            # Should complete without raising
            load_agent(plist_path)

    def test_failure_raises(self, plist_path: Path) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stderr="Bootstrap failed: 5: Input/output error"
            )
            with pytest.raises(LaunchdError, match="launchctl bootstrap failed"):
                load_agent(plist_path)

    def test_timeout_raises(self, plist_path: Path) -> None:
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("launchctl", 30)):
            with pytest.raises(LaunchdError, match="timed out"):
                load_agent(plist_path)

    def test_missing_launchctl_raises(self, plist_path: Path) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(LaunchdError, match="launchctl not found"):
                load_agent(plist_path)


class TestUnloadAgent:
    """Tests for unload_agent."""

    def test_success(self, plist_path: Path) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            # Should complete without raising
            unload_agent(plist_path)

    def test_already_unloaded_is_nonfatal(self, plist_path: Path) -> None:
        """bootout returning 'No such process' should not raise."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=3, stderr="3: No such process"
            )
            unload_agent(plist_path)  # Should not raise

    def test_other_failure_raises(self, plist_path: Path) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stderr="Permission denied"
            )
            with pytest.raises(LaunchdError, match="launchctl bootout failed"):
                unload_agent(plist_path)

    def test_timeout_raises(self, plist_path: Path) -> None:
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("launchctl", 30)):
            with pytest.raises(LaunchdError, match="timed out"):
                unload_agent(plist_path)

    def test_missing_launchctl_raises(self, plist_path: Path) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(LaunchdError, match="launchctl not found"):
                unload_agent(plist_path)


# --------------------------------------------------------------------------- #
# Status checking tests
# --------------------------------------------------------------------------- #


class TestGetAgentPid:
    """Tests for _get_agent_pid."""

    def test_running_agent_returns_pid(self) -> None:
        output = '"PID" = 12345;\n"LimitLoadToSessionType" = "Aqua";'
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=output)
            pid = _get_agent_pid()
        assert pid == 12345

    def test_not_loaded_returns_none(self) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=113, stdout="", stderr="")
            pid = _get_agent_pid()
        assert pid is None

    def test_loaded_but_no_pid_returns_none(self) -> None:
        """Agent is loaded but has no PID line (not running)."""
        output = '"LimitLoadToSessionType" = "Aqua";'
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=output)
            pid = _get_agent_pid()
        assert pid is None

    def test_invalid_pid_value_returns_none(self) -> None:
        output = '"PID" = not_a_number;\n'
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=output)
            pid = _get_agent_pid()
        assert pid is None

    def test_timeout_returns_none(self) -> None:
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("launchctl", 10)):
            pid = _get_agent_pid()
        assert pid is None

    def test_missing_launchctl_returns_none(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            pid = _get_agent_pid()
        assert pid is None


class TestGetAgentStatus:
    """Tests for get_agent_status."""

    def test_not_installed(self) -> None:
        with patch("mlx_stack.core.launchd.get_plist_path") as mock_path:
            mock_plist = MagicMock()
            mock_plist.exists.return_value = False
            mock_path.return_value = mock_plist

            status = get_agent_status()

        assert status.installed is False
        assert status.running is False
        assert status.pid is None

    def test_installed_and_running(self) -> None:
        with (
            patch("mlx_stack.core.launchd.get_plist_path") as mock_path,
            patch("mlx_stack.core.launchd._get_agent_pid", return_value=42),
        ):
            mock_plist = MagicMock()
            mock_plist.exists.return_value = True
            mock_path.return_value = mock_plist

            status = get_agent_status()

        assert status.installed is True
        assert status.running is True
        assert status.pid == 42

    def test_installed_but_not_running(self) -> None:
        with (
            patch("mlx_stack.core.launchd.get_plist_path") as mock_path,
            patch("mlx_stack.core.launchd._get_agent_pid", return_value=None),
        ):
            mock_plist = MagicMock()
            mock_plist.exists.return_value = True
            mock_path.return_value = mock_plist

            status = get_agent_status()

        assert status.installed is True
        assert status.running is False
        assert status.pid is None


# --------------------------------------------------------------------------- #
# High-level install_agent tests
# --------------------------------------------------------------------------- #


class TestInstallAgent:
    """Tests for install_agent."""

    def test_fresh_install(
        self, mlx_stack_home: Path, stack_definition: dict[str, Any]
    ) -> None:
        with (
            patch("mlx_stack.core.launchd.check_platform"),
            patch("mlx_stack.core.launchd.get_plist_path") as mock_get_path,
            patch("mlx_stack.core.launchd.write_plist") as mock_write,
            patch("mlx_stack.core.launchd.load_agent"),
        ):
            mock_plist_path = MagicMock()
            mock_plist_path.exists.return_value = False
            mock_get_path.return_value = mock_plist_path
            mock_write.return_value = mock_plist_path

            path, was_reinstall = install_agent("/usr/bin/mlx-stack")

        assert was_reinstall is False
        assert path is mock_plist_path

        # Verify plist data contains correct binary and label
        plist_data = mock_write.call_args[0][0]
        assert plist_data["Label"] == "com.mlx-stack.watchdog"
        assert plist_data["ProgramArguments"][0] == "/usr/bin/mlx-stack"
        assert "watch" in plist_data["ProgramArguments"]

    def test_reinstall(
        self, mlx_stack_home: Path, stack_definition: dict[str, Any]
    ) -> None:
        with (
            patch("mlx_stack.core.launchd.check_platform"),
            patch("mlx_stack.core.launchd.get_plist_path") as mock_get_path,
            patch("mlx_stack.core.launchd.write_plist") as mock_write,
            patch("mlx_stack.core.launchd.load_agent"),
            patch("mlx_stack.core.launchd.unload_agent"),
        ):
            mock_plist_path = MagicMock()
            mock_plist_path.exists.return_value = True
            mock_get_path.return_value = mock_plist_path
            mock_write.return_value = mock_plist_path

            path, was_reinstall = install_agent("/usr/bin/mlx-stack")

        assert was_reinstall is True

    def test_creates_logs_dir(
        self, mlx_stack_home: Path, stack_definition: dict[str, Any]
    ) -> None:
        with (
            patch("mlx_stack.core.launchd.check_platform"),
            patch("mlx_stack.core.launchd.get_plist_path") as mock_get_path,
            patch("mlx_stack.core.launchd.write_plist") as mock_write,
            patch("mlx_stack.core.launchd.load_agent"),
        ):
            mock_plist_path = MagicMock()
            mock_plist_path.exists.return_value = False
            mock_get_path.return_value = mock_plist_path
            mock_write.return_value = mock_plist_path

            install_agent("/usr/bin/mlx-stack")

        logs_dir = mlx_stack_home / "logs"
        assert logs_dir.exists()

    def test_platform_check_runs(self, mlx_stack_home: Path) -> None:
        with patch("mlx_stack.core.launchd.check_platform", side_effect=PlatformError("not macOS")):
            with pytest.raises(PlatformError):
                install_agent()

    def test_prerequisite_check_runs(self, mlx_stack_home: Path) -> None:
        with (
            patch("mlx_stack.core.launchd.check_platform"),
        ):
            # No stack definition → PrerequisiteError
            with pytest.raises(PrerequisiteError):
                install_agent()

    def test_reinstall_best_effort_unload(
        self, mlx_stack_home: Path, stack_definition: dict[str, Any]
    ) -> None:
        """If unload fails during reinstall, install should still proceed."""
        with (
            patch("mlx_stack.core.launchd.check_platform"),
            patch("mlx_stack.core.launchd.get_plist_path") as mock_get_path,
            patch("mlx_stack.core.launchd.write_plist") as mock_write,
            patch("mlx_stack.core.launchd.load_agent"),
            patch("mlx_stack.core.launchd.unload_agent", side_effect=LaunchdError("fail")),
        ):
            mock_plist_path = MagicMock()
            mock_plist_path.exists.return_value = True
            mock_get_path.return_value = mock_plist_path
            mock_write.return_value = mock_plist_path

            # Should not raise despite unload failure
            path, was_reinstall = install_agent("/usr/bin/mlx-stack")

        assert was_reinstall is True


# --------------------------------------------------------------------------- #
# High-level uninstall_agent tests
# --------------------------------------------------------------------------- #


class TestUninstallAgent:
    """Tests for uninstall_agent."""

    def test_successful_uninstall(self) -> None:
        with (
            patch("mlx_stack.core.launchd.check_platform"),
            patch("mlx_stack.core.launchd.get_plist_path") as mock_get_path,
            patch("mlx_stack.core.launchd.unload_agent"),
        ):
            mock_plist = MagicMock()
            mock_plist.exists.return_value = True
            mock_plist.unlink = MagicMock()
            mock_get_path.return_value = mock_plist

            result = uninstall_agent()

        assert result is True

    def test_not_installed_returns_false(self) -> None:
        with (
            patch("mlx_stack.core.launchd.check_platform"),
            patch("mlx_stack.core.launchd.get_plist_path") as mock_get_path,
        ):
            mock_plist = MagicMock()
            mock_plist.exists.return_value = False
            mock_get_path.return_value = mock_plist

            result = uninstall_agent()

        assert result is False

    def test_platform_check_runs(self) -> None:
        with patch("mlx_stack.core.launchd.check_platform", side_effect=PlatformError("not macOS")):
            with pytest.raises(PlatformError):
                uninstall_agent()

    def test_best_effort_unload_no_such_process(self) -> None:
        """If unload fails with 'No such process', plist should still be removed."""
        no_proc_err = LaunchdError("No such process")
        with (
            patch("mlx_stack.core.launchd.check_platform"),
            patch("mlx_stack.core.launchd.get_plist_path") as mock_get_path,
            patch("mlx_stack.core.launchd.unload_agent", side_effect=no_proc_err),
        ):
            mock_plist = MagicMock()
            mock_plist.exists.return_value = True
            mock_plist.unlink = MagicMock()
            mock_get_path.return_value = mock_plist

            result = uninstall_agent()

        assert result is True

    def test_unload_non_trivial_error_raises(self) -> None:
        """If unload fails with a non-'No such process' error, raise LaunchdError."""
        perm_err = LaunchdError("permission denied")
        with (
            patch("mlx_stack.core.launchd.check_platform"),
            patch("mlx_stack.core.launchd.get_plist_path") as mock_get_path,
            patch("mlx_stack.core.launchd.unload_agent", side_effect=perm_err),
        ):
            mock_plist = MagicMock()
            mock_plist.exists.return_value = True
            mock_plist.unlink = MagicMock()
            mock_get_path.return_value = mock_plist

            with pytest.raises(LaunchdError, match="permission denied"):
                uninstall_agent()

        # Plist should NOT have been removed since the error was non-trivial
        mock_plist.unlink.assert_not_called()

    def test_unlink_failure_raises(self) -> None:
        with (
            patch("mlx_stack.core.launchd.check_platform"),
            patch("mlx_stack.core.launchd.get_plist_path") as mock_get_path,
            patch("mlx_stack.core.launchd.unload_agent"),
        ):
            mock_plist = MagicMock()
            mock_plist.exists.return_value = True
            mock_plist.unlink.side_effect = OSError("Permission denied")
            mock_plist.__str__ = MagicMock(return_value="/some/path")
            mock_get_path.return_value = mock_plist

            with pytest.raises(LaunchdError, match="Failed to remove plist"):
                uninstall_agent()


# --------------------------------------------------------------------------- #
# Exception hierarchy tests
# --------------------------------------------------------------------------- #


class TestExceptions:
    """Tests for the exception hierarchy."""

    def test_platform_error_is_launchd_error(self) -> None:
        assert issubclass(PlatformError, LaunchdError)

    def test_prerequisite_error_is_launchd_error(self) -> None:
        assert issubclass(PrerequisiteError, LaunchdError)

    def test_launchd_error_is_exception(self) -> None:
        assert issubclass(LaunchdError, Exception)
