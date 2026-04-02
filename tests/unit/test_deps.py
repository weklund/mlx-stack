"""Tests for mlx_stack.core.deps — dependency management module.

Covers:
- PATH lookup and binary detection
- Version detection via ``uv tool list``
- Auto-install via ``uv tool install``
- Post-install verification
- Version mismatch warnings
- Error handling (uv not found, install failure, timeout)
- Public API: check_dependency, ensure_dependency, ensure_all_dependencies

All external calls (shutil.which, subprocess.run) are mocked — no real
processes are started.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from mlx_stack.core.deps import (
    PINNED_VERSIONS,
    DependencyError,
    DependencyInstallError,
    DependencyStatus,
    _find_binary,
    _get_installed_version,
    _install_tool,
    _verify_post_install,
    check_dependency,
    ensure_all_dependencies,
    ensure_dependency,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def mock_which():
    """Patch shutil.which in the deps module."""
    with patch("mlx_stack.core.deps.shutil.which") as m:
        yield m


@pytest.fixture()
def mock_subprocess_run():
    """Patch subprocess.run in the deps module."""
    with patch("mlx_stack.core.deps.subprocess.run") as m:
        yield m


@pytest.fixture()
def mock_console():
    """Patch the Rich console in the deps module."""
    with patch("mlx_stack.core.deps._console") as m:
        yield m


# --------------------------------------------------------------------------- #
# _find_binary
# --------------------------------------------------------------------------- #


class TestFindBinary:
    """Tests for _find_binary PATH lookup."""

    def test_finds_vllm_binary(self, mock_which: MagicMock) -> None:
        mock_which.return_value = "/usr/local/bin/vllm-mlx"
        result = _find_binary("vllm-mlx")
        mock_which.assert_called_once_with("vllm-mlx")
        assert result == "/usr/local/bin/vllm-mlx"

    def test_finds_litellm_binary(self, mock_which: MagicMock) -> None:
        mock_which.return_value = "/usr/local/bin/litellm"
        result = _find_binary("litellm")
        mock_which.assert_called_once_with("litellm")
        assert result == "/usr/local/bin/litellm"

    def test_returns_none_when_not_found(self, mock_which: MagicMock) -> None:
        mock_which.return_value = None
        result = _find_binary("vllm-mlx")
        assert result is None

    def test_unknown_tool_uses_tool_name_as_binary(self, mock_which: MagicMock) -> None:
        """Tools not in _TOOL_BINARY_MAP fall back to the tool name itself."""
        mock_which.return_value = "/usr/local/bin/sometool"
        result = _find_binary("sometool")
        mock_which.assert_called_once_with("sometool")
        assert result == "/usr/local/bin/sometool"


# --------------------------------------------------------------------------- #
# _get_installed_version
# --------------------------------------------------------------------------- #


class TestGetInstalledVersion:
    """Tests for version detection via uv tool list."""

    def test_parses_version_from_uv_tool_list(
        self, mock_which: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        mock_which.return_value = "/usr/local/bin/uv"
        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout="vllm-mlx v0.2.6\nlitellm v1.83.0\n",
        )
        result = _get_installed_version("vllm-mlx")
        assert result == "0.2.6"

    def test_parses_litellm_version(
        self, mock_which: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        mock_which.return_value = "/usr/local/bin/uv"
        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout="vllm-mlx v0.2.6\nlitellm v1.83.0\n",
        )
        result = _get_installed_version("litellm")
        assert result == "1.83.0"

    def test_returns_none_when_uv_not_found(self, mock_which: MagicMock) -> None:
        mock_which.return_value = None
        result = _get_installed_version("vllm-mlx")
        assert result is None

    def test_returns_none_when_uv_fails(
        self, mock_which: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        mock_which.return_value = "/usr/local/bin/uv"
        mock_subprocess_run.return_value = MagicMock(returncode=1, stdout="")
        result = _get_installed_version("vllm-mlx")
        assert result is None

    def test_returns_none_when_tool_not_listed(
        self, mock_which: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        mock_which.return_value = "/usr/local/bin/uv"
        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout="othertool v1.0.0\n",
        )
        result = _get_installed_version("vllm-mlx")
        assert result is None

    def test_returns_none_on_timeout(
        self, mock_which: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        mock_which.return_value = "/usr/local/bin/uv"
        mock_subprocess_run.side_effect = subprocess.TimeoutExpired(cmd="uv", timeout=30)
        result = _get_installed_version("vllm-mlx")
        assert result is None

    def test_returns_none_on_os_error(
        self, mock_which: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        mock_which.return_value = "/usr/local/bin/uv"
        mock_subprocess_run.side_effect = OSError("Permission denied")
        result = _get_installed_version("vllm-mlx")
        assert result is None

    def test_handles_version_without_v_prefix(
        self, mock_which: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        mock_which.return_value = "/usr/local/bin/uv"
        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout="vllm-mlx 0.2.6\n",
        )
        result = _get_installed_version("vllm-mlx")
        assert result == "0.2.6"

    def test_handles_empty_stdout(
        self, mock_which: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        mock_which.return_value = "/usr/local/bin/uv"
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="")
        result = _get_installed_version("vllm-mlx")
        assert result is None

    def test_passes_no_color_env_to_subprocess(
        self, mock_which: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        """Verify NO_COLOR=1 is passed to prevent ANSI codes in output."""
        mock_which.return_value = "/usr/local/bin/uv"
        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout="vllm-mlx v0.2.6\n",
        )
        _get_installed_version("vllm-mlx")

        call_kwargs = mock_subprocess_run.call_args
        assert "env" in call_kwargs.kwargs or (
            len(call_kwargs.args) > 1 and isinstance(call_kwargs.args[1], dict)
        )
        env = call_kwargs.kwargs.get("env", {})
        assert env.get("NO_COLOR") == "1"

    def test_handles_realistic_uv_tool_list_output(
        self, mock_which: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        """Parse realistic uv tool list output with indented binary lines."""
        mock_which.return_value = "/usr/local/bin/uv"
        # Real uv tool list output includes indented binary entries below each tool
        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "vllm-mlx v0.2.6\n"
                "- vllm-mlx\n"
                "litellm v1.83.0\n"
                "- litellm\n"
            ),
        )
        assert _get_installed_version("vllm-mlx") == "0.2.6"
        assert _get_installed_version("litellm") == "1.83.0"


# --------------------------------------------------------------------------- #
# _install_tool
# --------------------------------------------------------------------------- #


class TestInstallTool:
    """Tests for uv tool install logic."""

    def test_successful_install(
        self,
        mock_which: MagicMock,
        mock_subprocess_run: MagicMock,
        mock_console: MagicMock,
    ) -> None:
        mock_which.return_value = "/usr/local/bin/uv"
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        _install_tool("vllm-mlx", "0.2.6")

        # Verify subprocess was called with correct args
        mock_subprocess_run.assert_called_once()
        args = mock_subprocess_run.call_args
        assert args[0][0] == ["/usr/local/bin/uv", "tool", "install", "vllm-mlx==0.2.6"]

        # Verify progress messages
        assert mock_console.print.call_count >= 2  # installing + success

    def test_uv_not_found_raises_error(self, mock_which: MagicMock) -> None:
        mock_which.return_value = None
        with pytest.raises(DependencyError, match="uv is not available"):
            _install_tool("vllm-mlx", "0.2.6")

    def test_install_failure_raises_error(
        self,
        mock_which: MagicMock,
        mock_subprocess_run: MagicMock,
        mock_console: MagicMock,
    ) -> None:
        mock_which.return_value = "/usr/local/bin/uv"
        mock_subprocess_run.return_value = MagicMock(
            returncode=1,
            stderr="Package not found: vllm-mlx==0.2.6",
        )
        with pytest.raises(DependencyInstallError, match="Failed to install vllm-mlx"):
            _install_tool("vllm-mlx", "0.2.6")

    def test_install_failure_includes_command_and_error(
        self,
        mock_which: MagicMock,
        mock_subprocess_run: MagicMock,
        mock_console: MagicMock,
    ) -> None:
        mock_which.return_value = "/usr/local/bin/uv"
        mock_subprocess_run.return_value = MagicMock(
            returncode=1,
            stderr="Network error: connection refused",
        )
        with pytest.raises(DependencyInstallError) as exc_info:
            _install_tool("vllm-mlx", "0.2.6")

        error_msg = str(exc_info.value)
        assert "uv tool install vllm-mlx==0.2.6" in error_msg
        assert "Network error: connection refused" in error_msg
        assert "Install manually with:" in error_msg

    def test_install_timeout_raises_error(
        self,
        mock_which: MagicMock,
        mock_subprocess_run: MagicMock,
        mock_console: MagicMock,
    ) -> None:
        mock_which.return_value = "/usr/local/bin/uv"
        mock_subprocess_run.side_effect = subprocess.TimeoutExpired(cmd="uv", timeout=300)
        with pytest.raises(DependencyInstallError, match="timed out"):
            _install_tool("vllm-mlx", "0.2.6")

    def test_install_os_error_raises_error(
        self,
        mock_which: MagicMock,
        mock_subprocess_run: MagicMock,
        mock_console: MagicMock,
    ) -> None:
        mock_which.return_value = "/usr/local/bin/uv"
        mock_subprocess_run.side_effect = OSError("Permission denied")
        with pytest.raises(DependencyInstallError, match="Permission denied"):
            _install_tool("vllm-mlx", "0.2.6")

    def test_install_shows_progress_message(
        self,
        mock_which: MagicMock,
        mock_subprocess_run: MagicMock,
        mock_console: MagicMock,
    ) -> None:
        mock_which.return_value = "/usr/local/bin/uv"
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        _install_tool("litellm", "1.83.0")

        # First call should be "Installing..."
        first_print = mock_console.print.call_args_list[0]
        assert "Installing litellm" in str(first_print)
        # Second call should be success
        second_print = mock_console.print.call_args_list[1]
        assert "installed successfully" in str(second_print)

    def test_litellm_installs_with_proxy_extra(
        self,
        mock_which: MagicMock,
        mock_subprocess_run: MagicMock,
        mock_console: MagicMock,
    ) -> None:
        """litellm install spec includes [proxy] extra for proxy dependencies."""
        mock_which.return_value = "/usr/local/bin/uv"
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        _install_tool("litellm", "1.83.0")

        cmd = mock_subprocess_run.call_args[0][0]
        assert cmd == ["/usr/local/bin/uv", "tool", "install", "litellm[proxy]==1.83.0"]

    def test_vllm_mlx_installs_without_extra(
        self,
        mock_which: MagicMock,
        mock_subprocess_run: MagicMock,
        mock_console: MagicMock,
    ) -> None:
        """vllm-mlx install spec has no extras."""
        mock_which.return_value = "/usr/local/bin/uv"
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        _install_tool("vllm-mlx", "0.2.6")

        cmd = mock_subprocess_run.call_args[0][0]
        assert cmd == ["/usr/local/bin/uv", "tool", "install", "vllm-mlx==0.2.6"]


# --------------------------------------------------------------------------- #
# _verify_post_install
# --------------------------------------------------------------------------- #


class TestVerifyPostInstall:
    """Tests for post-install verification."""

    def test_returns_true_when_binary_found(self, mock_which: MagicMock) -> None:
        mock_which.return_value = "/usr/local/bin/vllm-mlx"
        assert _verify_post_install("vllm-mlx") is True

    def test_returns_false_when_binary_not_found(self, mock_which: MagicMock) -> None:
        mock_which.return_value = None
        assert _verify_post_install("vllm-mlx") is False


# --------------------------------------------------------------------------- #
# check_dependency
# --------------------------------------------------------------------------- #


class TestCheckDependency:
    """Tests for check_dependency (read-only status check)."""

    def test_installed_with_matching_version(
        self, mock_which: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        # First call for _find_binary, second for _get_installed_version's uv lookup
        mock_which.side_effect = ["/usr/local/bin/vllm-mlx", "/usr/local/bin/uv"]
        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout="vllm-mlx v0.2.6\n",
        )
        status = check_dependency("vllm-mlx")
        assert status.name == "vllm-mlx"
        assert status.pinned_version == "0.2.6"
        assert status.installed is True
        assert status.installed_version == "0.2.6"
        assert status.version_match is True

    def test_installed_with_mismatched_version(
        self, mock_which: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        mock_which.side_effect = ["/usr/local/bin/vllm-mlx", "/usr/local/bin/uv"]
        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout="vllm-mlx v0.2.5\n",
        )
        status = check_dependency("vllm-mlx")
        assert status.installed is True
        assert status.installed_version == "0.2.5"
        assert status.version_match is False

    def test_not_installed(self, mock_which: MagicMock) -> None:
        mock_which.return_value = None
        status = check_dependency("vllm-mlx")
        assert status.installed is False
        assert status.installed_version is None
        assert status.version_match is None

    def test_installed_but_version_unknown(
        self, mock_which: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        mock_which.side_effect = ["/usr/local/bin/vllm-mlx", "/usr/local/bin/uv"]
        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout="othertool v1.0.0\n",
        )
        status = check_dependency("vllm-mlx")
        assert status.installed is True
        assert status.installed_version is None
        assert status.version_match is None

    def test_unknown_tool_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown dependency"):
            check_dependency("nonexistent-tool")

    def test_check_litellm(
        self, mock_which: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        mock_which.side_effect = ["/usr/local/bin/litellm", "/usr/local/bin/uv"]
        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout="litellm v1.83.0\n",
        )
        status = check_dependency("litellm")
        assert status.name == "litellm"
        assert status.pinned_version == "1.83.0"
        assert status.installed is True
        assert status.installed_version == "1.83.0"
        assert status.version_match is True


# --------------------------------------------------------------------------- #
# ensure_dependency
# --------------------------------------------------------------------------- #


class TestEnsureDependency:
    """Tests for ensure_dependency (auto-install flow)."""

    def test_already_installed_matching_version_no_install(
        self, mock_which: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        """When tool is already installed and version matches, no install occurs."""
        mock_which.side_effect = ["/usr/local/bin/vllm-mlx", "/usr/local/bin/uv"]
        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout="vllm-mlx v0.2.6\n",
        )
        status = ensure_dependency("vllm-mlx")
        assert status.installed is True
        assert status.version_match is True
        # subprocess.run should only be called once (for uv tool list, not install)
        mock_subprocess_run.assert_called_once()

    def test_missing_tool_triggers_auto_install(
        self,
        mock_which: MagicMock,
        mock_subprocess_run: MagicMock,
        mock_console: MagicMock,
    ) -> None:
        """When tool is missing, auto-install is triggered."""
        # Sequence of which calls:
        # 1. check_dependency -> _find_binary: not found
        # 2. _install_tool -> uv found
        # 3. _verify_post_install -> _find_binary: found (after install)
        # 4. check_dependency (re-check) -> _find_binary: found
        # 5. check_dependency (re-check) -> _get_installed_version -> uv found
        mock_which.side_effect = [
            None,                    # _find_binary: tool not found
            "/usr/local/bin/uv",     # _install_tool: uv found
            "/usr/local/bin/vllm-mlx",  # _verify_post_install: tool found
            "/usr/local/bin/vllm-mlx",  # re-check _find_binary
            "/usr/local/bin/uv",        # re-check _get_installed_version
        ]
        # First run = install, second run = uv tool list for version check
        mock_subprocess_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),  # install success
            MagicMock(returncode=0, stdout="vllm-mlx v0.2.6\n"),  # version check
        ]

        status = ensure_dependency("vllm-mlx")
        assert status.installed is True
        assert status.installed_version == "0.2.6"

    def test_install_failure_raises_error(
        self,
        mock_which: MagicMock,
        mock_subprocess_run: MagicMock,
        mock_console: MagicMock,
    ) -> None:
        """When install fails, DependencyInstallError is raised."""
        mock_which.side_effect = [
            None,               # _find_binary: not found
            "/usr/local/bin/uv",  # _install_tool: uv found
        ]
        mock_subprocess_run.return_value = MagicMock(
            returncode=1,
            stderr="Error: could not find package",
        )
        with pytest.raises(DependencyInstallError, match="Failed to install"):
            ensure_dependency("vllm-mlx")

    def test_post_install_not_found_raises_error(
        self,
        mock_which: MagicMock,
        mock_subprocess_run: MagicMock,
        mock_console: MagicMock,
    ) -> None:
        """When tool not found after install, error with PATH instructions."""
        mock_which.side_effect = [
            None,               # _find_binary: not found
            "/usr/local/bin/uv",  # _install_tool: uv found
            None,                # _verify_post_install: still not found
        ]
        mock_subprocess_run.return_value = MagicMock(
            returncode=0, stdout="", stderr=""
        )
        with pytest.raises(DependencyInstallError, match="not found on PATH"):
            ensure_dependency("vllm-mlx")

    def test_version_mismatch_shows_warning(
        self,
        mock_which: MagicMock,
        mock_subprocess_run: MagicMock,
        mock_console: MagicMock,
    ) -> None:
        """When installed version doesn't match pinned, a warning is shown."""
        mock_which.side_effect = ["/usr/local/bin/vllm-mlx", "/usr/local/bin/uv"]
        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout="vllm-mlx v0.2.5\n",
        )
        status = ensure_dependency("vllm-mlx")
        assert status.version_match is False

        # Verify warning was printed
        console_prints = [str(c) for c in mock_console.print.call_args_list]
        warning_found = any("version mismatch" in p for p in console_prints)
        assert warning_found, f"Expected version mismatch warning in: {console_prints}"

    def test_version_mismatch_warning_includes_upgrade_command(
        self,
        mock_which: MagicMock,
        mock_subprocess_run: MagicMock,
        mock_console: MagicMock,
    ) -> None:
        """Version mismatch warning includes the upgrade/downgrade command."""
        mock_which.side_effect = ["/usr/local/bin/litellm", "/usr/local/bin/uv"]
        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout="litellm v1.60.0\n",
        )
        ensure_dependency("litellm")

        console_prints = [str(c) for c in mock_console.print.call_args_list]
        cmd_found = any("uv tool install litellm==1.83.0" in p for p in console_prints)
        assert cmd_found, f"Expected upgrade command in: {console_prints}"

    def test_unknown_tool_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown dependency"):
            ensure_dependency("nonexistent-tool")

    def test_matching_version_no_warning(
        self,
        mock_which: MagicMock,
        mock_subprocess_run: MagicMock,
        mock_console: MagicMock,
    ) -> None:
        """When versions match, no warning is shown."""
        mock_which.side_effect = ["/usr/local/bin/vllm-mlx", "/usr/local/bin/uv"]
        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout="vllm-mlx v0.2.6\n",
        )
        ensure_dependency("vllm-mlx")
        # No prints should have been made (no install, no warning)
        mock_console.print.assert_not_called()


# --------------------------------------------------------------------------- #
# ensure_all_dependencies
# --------------------------------------------------------------------------- #


class TestEnsureAllDependencies:
    """Tests for ensure_all_dependencies."""

    def test_ensures_all_tools(self) -> None:
        """Calls ensure_dependency for each tool in PINNED_VERSIONS."""
        with patch("mlx_stack.core.deps.ensure_dependency") as mock_ensure:
            mock_ensure.return_value = DependencyStatus(
                name="test",
                pinned_version="1.0",
                installed=True,
                installed_version="1.0",
                version_match=True,
            )
            results = ensure_all_dependencies()
            assert len(results) == len(PINNED_VERSIONS)
            called_tools = [c.args[0] for c in mock_ensure.call_args_list]
            for tool in PINNED_VERSIONS:
                assert tool in called_tools

    def test_propagates_install_error(self) -> None:
        """If any dependency fails, the error is propagated."""
        with patch("mlx_stack.core.deps.ensure_dependency") as mock_ensure:
            mock_ensure.side_effect = DependencyInstallError("Failed")
            with pytest.raises(DependencyInstallError):
                ensure_all_dependencies()


# --------------------------------------------------------------------------- #
# PINNED_VERSIONS and DependencyStatus
# --------------------------------------------------------------------------- #


class TestPinnedVersions:
    """Tests for pinned version constants and data structures."""

    def test_pinned_versions_contains_vllm_mlx(self) -> None:
        assert "vllm-mlx" in PINNED_VERSIONS

    def test_pinned_versions_contains_litellm(self) -> None:
        assert "litellm" in PINNED_VERSIONS

    def test_pinned_versions_are_strings(self) -> None:
        for tool, version in PINNED_VERSIONS.items():
            assert isinstance(version, str), f"{tool} version should be str"

    def test_dependency_status_is_frozen(self) -> None:
        status = DependencyStatus(
            name="test",
            pinned_version="1.0",
            installed=True,
            installed_version="1.0",
            version_match=True,
        )
        with pytest.raises(AttributeError):
            status.name = "other"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Error message quality (VAL-DEPS-002)
# --------------------------------------------------------------------------- #


class TestErrorMessages:
    """Tests verifying error messages include required information."""

    def test_install_failure_no_traceback(
        self,
        mock_which: MagicMock,
        mock_subprocess_run: MagicMock,
        mock_console: MagicMock,
    ) -> None:
        """Error messages should not contain Python tracebacks."""
        mock_which.side_effect = [
            None,                  # _find_binary
            "/usr/local/bin/uv",   # _install_tool
        ]
        mock_subprocess_run.return_value = MagicMock(
            returncode=1,
            stderr="error: package not found",
        )
        with pytest.raises(DependencyInstallError) as exc_info:
            ensure_dependency("vllm-mlx")

        error_msg = str(exc_info.value)
        assert "Traceback" not in error_msg
        assert "uv tool install vllm-mlx==0.2.6" in error_msg
        assert "Install manually with:" in error_msg

    def test_uv_missing_error_message(self, mock_which: MagicMock) -> None:
        """When uv is not found, error gives clear instructions."""
        mock_which.return_value = None
        with pytest.raises(DependencyError, match="uv is not available"):
            _install_tool("vllm-mlx", "0.2.6")

    def test_timeout_error_includes_manual_instructions(
        self,
        mock_which: MagicMock,
        mock_subprocess_run: MagicMock,
        mock_console: MagicMock,
    ) -> None:
        """Timeout errors include manual install instructions."""
        mock_which.return_value = "/usr/local/bin/uv"
        mock_subprocess_run.side_effect = subprocess.TimeoutExpired(cmd="uv", timeout=300)
        with pytest.raises(DependencyInstallError) as exc_info:
            _install_tool("litellm", "1.83.0")

        error_msg = str(exc_info.value)
        assert "Install manually with:" in error_msg
        assert "uv tool install litellm[proxy]==1.83.0" in error_msg


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_check_dependency_all_known_tools(self, mock_which: MagicMock) -> None:
        """check_dependency works for every tool in PINNED_VERSIONS."""
        mock_which.return_value = None
        for tool in PINNED_VERSIONS:
            status = check_dependency(tool)
            assert status.name == tool
            assert status.pinned_version == PINNED_VERSIONS[tool]

    def test_uv_tool_list_with_extra_whitespace(
        self, mock_which: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        """Parser handles extra whitespace in uv tool list output."""
        mock_which.return_value = "/usr/local/bin/uv"
        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout="  vllm-mlx   v0.2.6  \n",
        )
        result = _get_installed_version("vllm-mlx")
        assert result == "0.2.6"

    def test_uv_tool_list_with_multiple_tools_finds_vllm(
        self, mock_which: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        """Parser correctly selects vllm-mlx from a multi-line list."""
        mock_which.return_value = "/usr/local/bin/uv"
        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "ruff v0.8.0\n"
                "vllm-mlx v0.2.6\n"
                "litellm v1.83.0\n"
                "mypy v1.13.0\n"
            ),
        )
        assert _get_installed_version("vllm-mlx") == "0.2.6"

    def test_uv_tool_list_with_multiple_tools_finds_litellm(
        self, mock_which: MagicMock, mock_subprocess_run: MagicMock
    ) -> None:
        """Parser correctly selects litellm from a multi-line list."""
        mock_which.return_value = "/usr/local/bin/uv"
        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "ruff v0.8.0\n"
                "vllm-mlx v0.2.6\n"
                "litellm v1.83.0\n"
                "mypy v1.13.0\n"
            ),
        )
        assert _get_installed_version("litellm") == "1.83.0"

    def test_install_tool_with_special_version(
        self,
        mock_which: MagicMock,
        mock_subprocess_run: MagicMock,
        mock_console: MagicMock,
    ) -> None:
        """Install handles version strings with pre-release suffixes."""
        mock_which.return_value = "/usr/local/bin/uv"
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        _install_tool("vllm-mlx", "0.3.0rc1")

        args = mock_subprocess_run.call_args[0][0]
        assert args[3] == "vllm-mlx==0.3.0rc1"
