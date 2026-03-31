"""Tests for log rotation infrastructure (core/log_rotation.py).

Covers copytruncate rotation, archive naming, shifting, compression,
truncation, max file enforcement, and edge cases (empty files, missing
files, below-threshold files, permission errors).
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from mlx_stack.core.log_rotation import LogRotationError, rotate_log  # noqa: I001

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _create_log(path: Path, size_mb: float = 0, content: str | None = None) -> None:
    """Create a log file with the given size or specific content."""
    if content is not None:
        path.write_text(content)
    else:
        # Write enough bytes to reach the target size
        size_bytes = int(size_mb * 1024 * 1024)
        if size_bytes > 0:
            path.write_bytes(b"x" * size_bytes)
        else:
            path.touch()


def _read_gz(path: Path) -> bytes:
    """Read and decompress a gzip file."""
    with gzip.open(str(path), "rb") as f:
        return f.read()


# --------------------------------------------------------------------------- #
# Basic rotation
# --------------------------------------------------------------------------- #


class TestBasicRotation:
    """Tests for the basic rotation workflow."""

    def test_rotates_file_exceeding_threshold(self, tmp_path: Path) -> None:
        """File above threshold is rotated: archive created, original truncated."""
        log = tmp_path / "service.log"
        _create_log(log, size_mb=2)

        result = rotate_log(log, max_size_mb=1, max_files=5)

        assert result is True
        # Original should be truncated to zero
        assert log.exists()
        assert log.stat().st_size == 0
        # Archive .1.gz should exist
        archive = tmp_path / "service.log.1.gz"
        assert archive.exists()
        # Archive content should match original content
        data = _read_gz(archive)
        assert len(data) == 2 * 1024 * 1024

    def test_returns_true_on_rotation(self, tmp_path: Path) -> None:
        log = tmp_path / "service.log"
        _create_log(log, size_mb=2)
        assert rotate_log(log, max_size_mb=1, max_files=5) is True

    def test_archive_is_valid_gzip(self, tmp_path: Path) -> None:
        """The archive file is valid gzip that can be decompressed."""
        log = tmp_path / "service.log"
        content = "hello world\n" * 100000
        _create_log(log, content=content)

        rotate_log(log, max_size_mb=0, max_files=5)

        archive = tmp_path / "service.log.1.gz"
        assert archive.exists()
        decompressed = _read_gz(archive).decode("utf-8")
        assert decompressed == content

    def test_original_file_truncated_to_zero(self, tmp_path: Path) -> None:
        """After rotation, the original file exists but is empty."""
        log = tmp_path / "service.log"
        _create_log(log, size_mb=2)

        rotate_log(log, max_size_mb=1, max_files=5)

        assert log.exists()
        assert log.stat().st_size == 0

    def test_uses_max_size_mb_threshold(self, tmp_path: Path) -> None:
        """Rotation uses the max_size_mb parameter as threshold."""
        log = tmp_path / "service.log"
        # Create a file just at 5MB
        _create_log(log, size_mb=5)

        # With threshold at 10MB, should NOT rotate
        result = rotate_log(log, max_size_mb=10, max_files=5)
        assert result is False
        assert log.stat().st_size == 5 * 1024 * 1024

        # With threshold at 4MB, should rotate
        result = rotate_log(log, max_size_mb=4, max_files=5)
        assert result is True
        assert log.stat().st_size == 0


# --------------------------------------------------------------------------- #
# Archive shifting
# --------------------------------------------------------------------------- #


class TestArchiveShifting:
    """Tests for sequential numbering and archive shifting."""

    def test_existing_archive_shifted_up(self, tmp_path: Path) -> None:
        """Existing .1.gz is shifted to .2.gz before new .1.gz is created."""
        log = tmp_path / "service.log"

        # First rotation
        _create_log(log, content="first content\n" * 100000)
        rotate_log(log, max_size_mb=0, max_files=5)

        # Second rotation
        _create_log(log, content="second content\n" * 100000)
        rotate_log(log, max_size_mb=0, max_files=5)

        # .1.gz should have second content, .2.gz should have first content
        archive1 = tmp_path / "service.log.1.gz"
        archive2 = tmp_path / "service.log.2.gz"
        assert archive1.exists()
        assert archive2.exists()

        data1 = _read_gz(archive1).decode("utf-8")
        data2 = _read_gz(archive2).decode("utf-8")
        assert "second content" in data1
        assert "first content" in data2

    def test_multiple_rotations_sequential_numbering(self, tmp_path: Path) -> None:
        """Multiple rotations create sequentially numbered archives."""
        log = tmp_path / "service.log"

        for i in range(4):
            _create_log(log, content=f"rotation {i}\n" * 100000)
            rotate_log(log, max_size_mb=0, max_files=5)

        # Should have archives .1 through .4
        for n in range(1, 5):
            archive = tmp_path / f"service.log.{n}.gz"
            assert archive.exists(), f"Archive {n} missing"

        # .1 = most recent (rotation 3), .4 = oldest (rotation 0)
        data1 = _read_gz(tmp_path / "service.log.1.gz").decode("utf-8")
        data4 = _read_gz(tmp_path / "service.log.4.gz").decode("utf-8")
        assert "rotation 3" in data1
        assert "rotation 0" in data4

    def test_one_is_most_recent(self, tmp_path: Path) -> None:
        """.1.gz always contains the most recently rotated content."""
        log = tmp_path / "service.log"

        _create_log(log, content="old data\n" * 100000)
        rotate_log(log, max_size_mb=0, max_files=5)

        _create_log(log, content="new data\n" * 100000)
        rotate_log(log, max_size_mb=0, max_files=5)

        data = _read_gz(tmp_path / "service.log.1.gz").decode("utf-8")
        assert "new data" in data


# --------------------------------------------------------------------------- #
# Max files enforcement
# --------------------------------------------------------------------------- #


class TestMaxFilesEnforcement:
    """Tests for max_files limiting the number of retained archives."""

    def test_oldest_deleted_when_exceeded(self, tmp_path: Path) -> None:
        """When max_files is exceeded, the oldest archive is deleted."""
        log = tmp_path / "service.log"

        # Do 6 rotations with max_files=5
        for i in range(6):
            _create_log(log, content=f"rotation {i}\n" * 100000)
            rotate_log(log, max_size_mb=0, max_files=5)

        # Should have exactly .1 through .5
        for n in range(1, 6):
            assert (tmp_path / f"service.log.{n}.gz").exists()

        # .6 should NOT exist
        assert not (tmp_path / "service.log.6.gz").exists()

        # .1 should be most recent (rotation 5)
        data1 = _read_gz(tmp_path / "service.log.1.gz").decode("utf-8")
        assert "rotation 5" in data1

        # .5 should be oldest retained (rotation 1, since rotation 0 was deleted)
        data5 = _read_gz(tmp_path / "service.log.5.gz").decode("utf-8")
        assert "rotation 1" in data5

    def test_max_files_one(self, tmp_path: Path) -> None:
        """With max_files=1, only the most recent archive is kept."""
        log = tmp_path / "service.log"

        for i in range(3):
            _create_log(log, content=f"rotation {i}\n" * 100000)
            rotate_log(log, max_size_mb=0, max_files=1)

        # Only .1.gz should exist
        assert (tmp_path / "service.log.1.gz").exists()
        assert not (tmp_path / "service.log.2.gz").exists()

        data = _read_gz(tmp_path / "service.log.1.gz").decode("utf-8")
        assert "rotation 2" in data

    def test_max_files_exact_boundary(self, tmp_path: Path) -> None:
        """At exactly max_files rotations, all archives are kept."""
        log = tmp_path / "service.log"

        for i in range(3):
            _create_log(log, content=f"rotation {i}\n" * 100000)
            rotate_log(log, max_size_mb=0, max_files=3)

        for n in range(1, 4):
            assert (tmp_path / f"service.log.{n}.gz").exists()

        assert not (tmp_path / "service.log.4.gz").exists()


# --------------------------------------------------------------------------- #
# Edge cases — skip rotation
# --------------------------------------------------------------------------- #


class TestEdgeCases:
    """Tests for edge cases where rotation should be skipped."""

    def test_missing_file_returns_false(self, tmp_path: Path) -> None:
        """Missing log file is silently skipped."""
        log = tmp_path / "nonexistent.log"
        result = rotate_log(log, max_size_mb=1, max_files=5)
        assert result is False

    def test_empty_file_returns_false(self, tmp_path: Path) -> None:
        """Empty log file is skipped."""
        log = tmp_path / "empty.log"
        log.touch()
        assert log.stat().st_size == 0

        result = rotate_log(log, max_size_mb=1, max_files=5)
        assert result is False

    def test_below_threshold_returns_false(self, tmp_path: Path) -> None:
        """File below size threshold is not rotated."""
        log = tmp_path / "small.log"
        _create_log(log, size_mb=0.5)

        result = rotate_log(log, max_size_mb=1, max_files=5)
        assert result is False
        # File should be unchanged
        assert log.stat().st_size == int(0.5 * 1024 * 1024)

    def test_no_archive_created_when_skipped(self, tmp_path: Path) -> None:
        """No archive files created when rotation is skipped."""
        log = tmp_path / "small.log"
        _create_log(log, size_mb=0.5)

        rotate_log(log, max_size_mb=1, max_files=5)

        archive = tmp_path / "small.log.1.gz"
        assert not archive.exists()


# --------------------------------------------------------------------------- #
# Permission errors
# --------------------------------------------------------------------------- #


class TestPermissionErrors:
    """Tests for graceful handling of permission errors."""

    def test_permission_error_raises_log_rotation_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Permission errors during rotation raise LogRotationError."""
        log = tmp_path / "service.log"
        _create_log(log, size_mb=2)

        # Make the directory read-only to prevent creating archives
        import mlx_stack.core.log_rotation as lr

        def _mock_copy_and_compress(src: Path, dst_gz: Path) -> None:
            raise PermissionError("Permission denied")

        monkeypatch.setattr(lr, "_copy_and_compress", _mock_copy_and_compress)

        with pytest.raises(LogRotationError, match="Permission denied"):
            rotate_log(log, max_size_mb=1, max_files=5)

    def test_os_error_raises_log_rotation_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OS errors during rotation raise LogRotationError."""
        log = tmp_path / "service.log"
        _create_log(log, size_mb=2)

        import mlx_stack.core.log_rotation as lr

        def _mock_copy_and_compress(src: Path, dst_gz: Path) -> None:
            raise OSError("Disk full")

        monkeypatch.setattr(lr, "_copy_and_compress", _mock_copy_and_compress)

        with pytest.raises(LogRotationError, match="I/O error"):
            rotate_log(log, max_size_mb=1, max_files=5)


# --------------------------------------------------------------------------- #
# Config key integration
# --------------------------------------------------------------------------- #


class TestConfigKeyIntegration:
    """Tests for the log rotation config keys in core/config.py."""

    def test_log_max_size_mb_default(self, mlx_stack_home: Path) -> None:
        from mlx_stack.core.config import get_value

        assert get_value("log-max-size-mb") == 50

    def test_log_max_files_default(self, mlx_stack_home: Path) -> None:
        from mlx_stack.core.config import get_value

        assert get_value("log-max-files") == 5

    def test_log_max_size_mb_set_and_get(self, mlx_stack_home: Path) -> None:
        from mlx_stack.core.config import get_value, set_value

        set_value("log-max-size-mb", "100")
        assert get_value("log-max-size-mb") == 100

    def test_log_max_files_set_and_get(self, mlx_stack_home: Path) -> None:
        from mlx_stack.core.config import get_value, set_value

        set_value("log-max-files", "10")
        assert get_value("log-max-files") == 10

    def test_log_max_size_mb_rejects_zero(self, mlx_stack_home: Path) -> None:
        from mlx_stack.core.config import ConfigValidationError, set_value

        with pytest.raises(ConfigValidationError, match="Must be at least 1"):
            set_value("log-max-size-mb", "0")

    def test_log_max_files_rejects_zero(self, mlx_stack_home: Path) -> None:
        from mlx_stack.core.config import ConfigValidationError, set_value

        with pytest.raises(ConfigValidationError, match="Must be at least 1"):
            set_value("log-max-files", "0")

    def test_log_max_size_mb_rejects_negative(self, mlx_stack_home: Path) -> None:
        from mlx_stack.core.config import ConfigValidationError, set_value

        with pytest.raises(ConfigValidationError, match="Must be at least 1"):
            set_value("log-max-size-mb", "-5")

    def test_log_max_files_rejects_negative(self, mlx_stack_home: Path) -> None:
        from mlx_stack.core.config import ConfigValidationError, set_value

        with pytest.raises(ConfigValidationError, match="Must be at least 1"):
            set_value("log-max-files", "-1")

    def test_log_max_size_mb_rejects_non_integer(self, mlx_stack_home: Path) -> None:
        from mlx_stack.core.config import ConfigValidationError, set_value

        with pytest.raises(ConfigValidationError, match="Expected an integer"):
            set_value("log-max-size-mb", "abc")

    def test_log_max_files_rejects_non_integer(self, mlx_stack_home: Path) -> None:
        from mlx_stack.core.config import ConfigValidationError, set_value

        with pytest.raises(ConfigValidationError, match="Expected an integer"):
            set_value("log-max-files", "abc")

    def test_config_list_includes_log_keys(self, mlx_stack_home: Path) -> None:
        from mlx_stack.core.config import get_all_config

        entries = get_all_config()
        names = {e["name"] for e in entries}
        assert "log-max-size-mb" in names
        assert "log-max-files" in names

    def test_config_reset_restores_log_defaults(self, mlx_stack_home: Path) -> None:
        from mlx_stack.core.config import get_value, reset_config, set_value

        set_value("log-max-size-mb", "200")
        set_value("log-max-files", "20")
        reset_config()
        assert get_value("log-max-size-mb") == 50
        assert get_value("log-max-files") == 5


# --------------------------------------------------------------------------- #
# Process.py append mode
# --------------------------------------------------------------------------- #


class TestStartServiceAppendMode:
    """Tests that start_service opens logs in append mode."""

    def test_log_opened_in_append_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify log file is opened with 'a' mode (not 'w')."""
        from unittest.mock import MagicMock, patch

        from mlx_stack.core.process import start_service

        monkeypatch.setenv("MLX_STACK_HOME", str(tmp_path / ".mlx-stack"))
        (tmp_path / ".mlx-stack").mkdir(parents=True, exist_ok=True)
        (tmp_path / ".mlx-stack" / "logs").mkdir(parents=True, exist_ok=True)
        (tmp_path / ".mlx-stack" / "pids").mkdir(parents=True, exist_ok=True)

        # Pre-create a log file with existing content
        log_dir = tmp_path / ".mlx-stack" / "logs"
        log_file = log_dir / "test-service.log"
        log_file.write_text("existing content\n")

        opened_modes: list[str] = []
        original_open = open

        def tracking_open(path: object, mode: str = "r", *args: object, **kwargs: object) -> object:
            if str(path) == str(log_file):
                opened_modes.append(mode)
            return original_open(path, mode, *args, **kwargs)  # type: ignore[arg-type]

        mock_proc = MagicMock()
        mock_proc.pid = 12345

        with (
            patch("builtins.open", side_effect=tracking_open),
            patch("subprocess.Popen", return_value=mock_proc),
        ):
            start_service(
                service_name="test-service",
                cmd=["echo", "test"],
                port=8000,
                log_dir=log_dir,
            )

        assert "a" in opened_modes, f"Expected 'a' mode, got modes: {opened_modes}"

    def test_existing_log_content_preserved_on_restart(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Restarting a service preserves existing log content (append mode)."""
        from unittest.mock import MagicMock, patch

        from mlx_stack.core.process import start_service

        monkeypatch.setenv("MLX_STACK_HOME", str(tmp_path / ".mlx-stack"))
        (tmp_path / ".mlx-stack").mkdir(parents=True, exist_ok=True)
        (tmp_path / ".mlx-stack" / "pids").mkdir(parents=True, exist_ok=True)

        log_dir = tmp_path / ".mlx-stack" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "test-svc.log"
        log_file.write_text("previous session content\n")

        mock_proc = MagicMock()
        mock_proc.pid = 99999

        with patch("subprocess.Popen", return_value=mock_proc):
            start_service(
                service_name="test-svc",
                cmd=["echo", "test"],
                port=8001,
                log_dir=log_dir,
            )

        # The log file should still contain previous content
        # (since we opened with "a", not "w")
        content = log_file.read_text()
        assert "previous session content" in content
