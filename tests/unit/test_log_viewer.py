"""Tests for log viewing and management (core/log_viewer.py).

Covers log listing, tail reading, following with truncation detection,
archived log reading, and on-demand rotation.
"""

from __future__ import annotations

import contextlib
import threading
import time
from datetime import UTC
from pathlib import Path

from mlx_stack.core.log_viewer import (
    DEFAULT_TAIL_LINES,
    LogFileInfo,
    follow_log,
    get_available_services,
    get_log_path,
    list_log_files,
    read_all_logs,
    read_archive,
    read_log_full,
    read_log_tail,
    rotate_all_logs,
    rotate_service_log,
)
from tests.factories import create_archive, create_log_file

# --------------------------------------------------------------------------- #
# LogFileInfo
# --------------------------------------------------------------------------- #


class TestLogFileInfo:
    """Tests for the LogFileInfo data class."""

    def test_size_display_bytes(self) -> None:
        """Sizes under 1KB show in bytes."""
        from datetime import datetime

        info = LogFileInfo(
            name="test.log",
            service="test",
            size_bytes=500,
            modified=datetime.now(tz=UTC),
        )
        assert info.size_display == "500 B"

    def test_size_display_kilobytes(self) -> None:
        """Sizes 1KB-1MB show in KB."""
        from datetime import datetime

        info = LogFileInfo(
            name="test.log",
            service="test",
            size_bytes=2048,
            modified=datetime.now(tz=UTC),
        )
        assert info.size_display == "2.0 KB"

    def test_size_display_megabytes(self) -> None:
        """Sizes 1MB-1GB show in MB."""
        from datetime import datetime

        info = LogFileInfo(
            name="test.log",
            service="test",
            size_bytes=5 * 1024 * 1024,
            modified=datetime.now(tz=UTC),
        )
        assert info.size_display == "5.0 MB"

    def test_size_display_gigabytes(self) -> None:
        """Sizes >= 1GB show in GB."""
        from datetime import datetime

        info = LogFileInfo(
            name="test.log",
            service="test",
            size_bytes=2 * 1024 * 1024 * 1024,
            modified=datetime.now(tz=UTC),
        )
        assert info.size_display == "2.0 GB"

    def test_modified_display(self) -> None:
        """Modified time is formatted as expected."""
        from datetime import datetime

        dt = datetime(2025, 3, 15, 14, 30, 45, tzinfo=UTC)
        info = LogFileInfo(
            name="test.log",
            service="test",
            size_bytes=100,
            modified=dt,
        )
        assert info.modified_display == "2025-03-15 14:30:45"


# --------------------------------------------------------------------------- #
# Listing
# --------------------------------------------------------------------------- #


class TestListLogFiles:
    """Tests for list_log_files."""

    def test_empty_directory(self, mlx_stack_home: Path) -> None:
        """Empty logs directory returns empty list."""
        logs_dir = mlx_stack_home / "logs"
        logs_dir.mkdir()
        assert list_log_files() == []

    def test_no_logs_directory(self, mlx_stack_home: Path) -> None:
        """Missing logs directory returns empty list."""
        assert list_log_files() == []

    def test_lists_log_files(self, mlx_stack_home: Path) -> None:
        """Lists .log files with correct metadata."""
        # Arrange
        logs_dir = mlx_stack_home / "logs"
        create_log_file(logs_dir, "fast", "line1\nline2\n")
        create_log_file(logs_dir, "standard", "some content\n")

        # Act
        files = list_log_files()

        # Assert
        assert len(files) == 2
        assert files[0].service == "fast"
        assert files[1].service == "standard"
        assert files[0].name == "fast.log"
        assert files[0].size_bytes > 0

    def test_excludes_gz_files(self, mlx_stack_home: Path) -> None:
        """Archived .gz files are not included in listing."""
        # Arrange
        logs_dir = mlx_stack_home / "logs"
        create_log_file(logs_dir, "fast", "content")
        create_archive(logs_dir, "fast", 1, "old content")

        # Act
        files = list_log_files()

        # Assert
        assert len(files) == 1
        assert files[0].name == "fast.log"

    def test_excludes_non_log_files(self, mlx_stack_home: Path) -> None:
        """Non-.log files are excluded."""
        logs_dir = mlx_stack_home / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        create_log_file(logs_dir, "fast", "content")
        (logs_dir / "notes.txt").write_text("not a log")

        files = list_log_files()
        assert len(files) == 1


class TestGetLogPath:
    """Tests for get_log_path."""

    def test_returns_path_for_existing_log(self, mlx_stack_home: Path) -> None:
        """Returns path when log file exists."""
        logs_dir = mlx_stack_home / "logs"
        create_log_file(logs_dir, "fast", "content")

        path = get_log_path("fast")
        assert path is not None
        assert path.name == "fast.log"

    def test_returns_none_for_missing_log(self, mlx_stack_home: Path) -> None:
        """Returns None when log file does not exist."""
        logs_dir = mlx_stack_home / "logs"
        logs_dir.mkdir(parents=True)

        assert get_log_path("nonexistent") is None

    def test_returns_none_for_missing_dir(self, mlx_stack_home: Path) -> None:
        """Returns None when logs directory doesn't exist."""
        assert get_log_path("fast") is None


class TestGetAvailableServices:
    """Tests for get_available_services."""

    def test_returns_sorted_services(self, mlx_stack_home: Path) -> None:
        """Returns sorted list of services with log files."""
        # Arrange
        logs_dir = mlx_stack_home / "logs"
        create_log_file(logs_dir, "standard", "content")
        create_log_file(logs_dir, "fast", "content")
        create_log_file(logs_dir, "litellm", "content")

        # Act
        services = get_available_services()

        # Assert
        assert services == ["fast", "litellm", "standard"]

    def test_returns_empty_for_no_logs(self, mlx_stack_home: Path) -> None:
        """Returns empty list when no log files exist."""
        assert get_available_services() == []


# --------------------------------------------------------------------------- #
# Log reading
# --------------------------------------------------------------------------- #


class TestReadLogTail:
    """Tests for read_log_tail."""

    def test_reads_last_n_lines(self, tmp_path: Path) -> None:
        """Returns exactly the last N lines."""
        log = tmp_path / "test.log"
        lines = [f"line {i}" for i in range(100)]
        log.write_text("\n".join(lines))

        result = read_log_tail(log, num_lines=10)
        result_lines = result.splitlines()
        assert len(result_lines) == 10
        assert result_lines[0] == "line 90"
        assert result_lines[-1] == "line 99"

    def test_reads_all_when_fewer_than_n(self, tmp_path: Path) -> None:
        """Returns all lines when file has fewer than N lines."""
        log = tmp_path / "test.log"
        log.write_text("line 1\nline 2\nline 3")

        result = read_log_tail(log, num_lines=10)
        assert len(result.splitlines()) == 3

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        """Empty file returns empty string."""
        log = tmp_path / "test.log"
        log.write_text("")

        assert read_log_tail(log) == ""

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        """Missing file returns empty string."""
        log = tmp_path / "missing.log"
        assert read_log_tail(log) == ""

    def test_default_tail_lines(self, tmp_path: Path) -> None:
        """Default is DEFAULT_TAIL_LINES."""
        log = tmp_path / "test.log"
        lines = [f"line {i}" for i in range(100)]
        log.write_text("\n".join(lines))

        result = read_log_tail(log)
        assert len(result.splitlines()) == DEFAULT_TAIL_LINES


class TestReadLogFull:
    """Tests for read_log_full."""

    def test_reads_full_content(self, tmp_path: Path) -> None:
        """Returns complete file content."""
        log = tmp_path / "test.log"
        content = "line 1\nline 2\nline 3\n"
        log.write_text(content)

        assert read_log_full(log) == content

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        """Missing file returns empty string."""
        log = tmp_path / "missing.log"
        assert read_log_full(log) == ""


# --------------------------------------------------------------------------- #
# Following
# --------------------------------------------------------------------------- #


class TestFollowLog:
    """Tests for follow_log."""

    def test_shows_initial_tail(self, tmp_path: Path) -> None:
        """Shows initial tail content before following."""
        log = tmp_path / "test.log"
        lines = [f"line {i}" for i in range(10)]
        log.write_text("\n".join(lines))

        output: list[str] = []
        stop_event = threading.Event()

        def writer(text: str) -> None:
            output.append(text)
            stop_event.set()

        def run_follow() -> None:
            with contextlib.suppress(KeyboardInterrupt):
                follow_log(log, num_lines=5, output_callback=writer)

        thread = threading.Thread(target=run_follow, daemon=True)
        thread.start()

        # Wait for initial output
        stop_event.wait(timeout=3)

        # Simulate Ctrl-C by raising KeyboardInterrupt in follow thread
        import ctypes

        assert thread.ident is not None
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(thread.ident),
            ctypes.py_object(KeyboardInterrupt),
        )
        thread.join(timeout=3)

        assert len(output) > 0
        # Initial tail should contain last 5 lines
        initial_lines = output[0].splitlines()
        assert len(initial_lines) == 5

    def test_detects_new_content(self, tmp_path: Path) -> None:
        """Detects and outputs newly appended content."""
        log = tmp_path / "test.log"
        log.write_text("initial\n")

        output: list[str] = []
        new_content_event = threading.Event()

        def writer(text: str) -> None:
            output.append(text)
            if "new data" in text:
                new_content_event.set()

        def run_follow() -> None:
            with contextlib.suppress(KeyboardInterrupt):
                follow_log(log, num_lines=1, output_callback=writer)

        thread = threading.Thread(target=run_follow, daemon=True)
        thread.start()

        # Wait for initial output, then append
        time.sleep(0.8)
        with open(log, "a") as f:
            f.write("new data\n")

        # Wait for new content to be detected
        new_content_event.wait(timeout=5)

        import ctypes

        assert thread.ident is not None
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(thread.ident),
            ctypes.py_object(KeyboardInterrupt),
        )
        thread.join(timeout=3)

        all_text = "\n".join(output)
        assert "new data" in all_text

    def test_detects_truncation(self, tmp_path: Path) -> None:
        """Detects file truncation and resets position."""
        log = tmp_path / "test.log"
        log.write_text("old content line 1\nold content line 2\n")

        output: list[str] = []
        post_truncation_event = threading.Event()

        def writer(text: str) -> None:
            output.append(text)
            if "after truncation" in text:
                post_truncation_event.set()

        def run_follow() -> None:
            with contextlib.suppress(KeyboardInterrupt):
                follow_log(log, num_lines=2, output_callback=writer)

        thread = threading.Thread(target=run_follow, daemon=True)
        thread.start()

        # Wait for initial tail, then simulate truncation
        time.sleep(0.8)
        # Truncate and write new content (simulates copytruncate)
        open(log, "w").close()
        time.sleep(0.1)
        with open(log, "a") as f:
            f.write("after truncation\n")

        post_truncation_event.wait(timeout=5)

        import ctypes

        assert thread.ident is not None
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(thread.ident),
            ctypes.py_object(KeyboardInterrupt),
        )
        thread.join(timeout=3)

        all_text = "\n".join(output)
        assert "after truncation" in all_text


# --------------------------------------------------------------------------- #
# Archived log reading
# --------------------------------------------------------------------------- #


class TestReadArchive:
    """Tests for read_archive."""

    def test_reads_valid_archive(self, tmp_path: Path) -> None:
        """Reads and decompresses a valid gzip archive."""
        archive = create_archive(tmp_path, "fast", 1, "archived content")
        result = read_archive(archive)
        assert result == "archived content"

    def test_handles_corrupt_archive(self, tmp_path: Path) -> None:
        """Returns error message for corrupt archive."""
        corrupt = tmp_path / "fast.log.1.gz"
        corrupt.write_bytes(b"not gzip data")

        result = read_archive(corrupt)
        assert "Error reading archive" in result


class TestReadAllLogs:
    """Tests for read_all_logs."""

    def test_chronological_order(self, mlx_stack_home: Path) -> None:
        """Shows archives oldest first, then current log."""
        # Arrange -- higher archive number = older
        logs_dir = mlx_stack_home / "logs"
        create_archive(logs_dir, "fast", 3, "oldest content")
        create_archive(logs_dir, "fast", 2, "middle content")
        create_archive(logs_dir, "fast", 1, "newest archived content")
        create_log_file(logs_dir, "fast", "current content")

        # Act
        result = read_all_logs("fast")

        # Assert -- section headers appear in chronological order
        lines = result.splitlines()
        oldest_idx = next(i for i, line in enumerate(lines) if "fast.log.3.gz" in line)
        middle_idx = next(i for i, line in enumerate(lines) if "fast.log.2.gz" in line)
        newest_idx = next(i for i, line in enumerate(lines) if "fast.log.1.gz" in line)
        current_idx = next(i for i, line in enumerate(lines) if "Current:" in line)
        assert oldest_idx < middle_idx < newest_idx < current_idx

    def test_archives_only(self, mlx_stack_home: Path) -> None:
        """Works with only archived files (no current log)."""
        logs_dir = mlx_stack_home / "logs"
        create_archive(logs_dir, "fast", 1, "archived only")

        result = read_all_logs("fast")
        assert "archived only" in result

    def test_current_only(self, mlx_stack_home: Path) -> None:
        """Works with only current log (no archives)."""
        logs_dir = mlx_stack_home / "logs"
        create_log_file(logs_dir, "fast", "current only")

        result = read_all_logs("fast")
        assert "current only" in result

    def test_no_logs_returns_empty(self, mlx_stack_home: Path) -> None:
        """Returns empty string when no logs exist."""
        result = read_all_logs("nonexistent")
        assert result == ""


# --------------------------------------------------------------------------- #
# On-demand rotation
# --------------------------------------------------------------------------- #


class TestRotateServiceLog:
    """Tests for rotate_service_log."""

    def test_rotates_eligible_log(self, mlx_stack_home: Path) -> None:
        """Rotates a log file exceeding the threshold."""
        logs_dir = mlx_stack_home / "logs"
        logs_dir.mkdir(parents=True)
        log_path = logs_dir / "fast.log"
        # Create a file exceeding default threshold (50MB)
        # Use a small threshold by setting config
        log_path.write_bytes(b"x" * (2 * 1024 * 1024))  # 2MB

        # Mock config to use 1MB threshold
        from unittest.mock import patch

        with patch(
            "mlx_stack.core.log_viewer.get_value",
            side_effect=lambda k: 1 if k == "log-max-size-mb" else 5,
        ):
            result = rotate_service_log("fast")

        assert result.rotated is True
        assert result.service == "fast"
        assert result.error is None

    def test_skips_small_log(self, mlx_stack_home: Path) -> None:
        """Does not rotate a log below the threshold."""
        logs_dir = mlx_stack_home / "logs"
        create_log_file(logs_dir, "fast", "small content")

        from unittest.mock import patch

        with patch(
            "mlx_stack.core.log_viewer.get_value",
            side_effect=lambda k: 50 if k == "log-max-size-mb" else 5,
        ):
            result = rotate_service_log("fast")

        assert result.rotated is False
        assert result.error is None

    def test_missing_log_returns_error(self, mlx_stack_home: Path) -> None:
        """Missing log file returns error result."""
        logs_dir = mlx_stack_home / "logs"
        logs_dir.mkdir(parents=True)

        result = rotate_service_log("nonexistent")
        assert result.rotated is False
        assert result.error is not None
        assert "No log file found" in result.error


class TestRotateAllLogs:
    """Tests for rotate_all_logs."""

    def test_rotates_all_eligible(self, mlx_stack_home: Path) -> None:
        """Rotates all eligible log files."""
        logs_dir = mlx_stack_home / "logs"
        logs_dir.mkdir(parents=True)

        # Create two logs, one above threshold
        (logs_dir / "fast.log").write_bytes(b"x" * (2 * 1024 * 1024))  # 2MB
        (logs_dir / "standard.log").write_text("small")

        from unittest.mock import patch

        with patch(
            "mlx_stack.core.log_viewer.get_value",
            side_effect=lambda k: 1 if k == "log-max-size-mb" else 5,
        ):
            results = rotate_all_logs()

        assert len(results) == 2
        fast_result = next(r for r in results if r.service == "fast")
        std_result = next(r for r in results if r.service == "standard")
        assert fast_result.rotated is True
        assert std_result.rotated is False

    def test_empty_logs_dir(self, mlx_stack_home: Path) -> None:
        """Returns empty list when no log files exist."""
        results = rotate_all_logs()
        assert results == []
