"""Log rotation infrastructure for mlx-stack.

Implements copytruncate-based rotation for service log files in
~/.mlx-stack/logs/. The rotation strategy copies the log file to an
archive, compresses it with gzip, then truncates the original in-place
so that the writing process's file descriptor remains valid.

Archive naming uses sequential numbering:
  <name>.log.1.gz = most recent rotation
  <name>.log.2.gz = second most recent
  ...
  <name>.log.N.gz = oldest retained rotation

Archives beyond max_files are deleted.
"""

from __future__ import annotations

import gzip
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class LogRotationError(Exception):
    """Raised when a log rotation operation fails."""


def rotate_log(
    log_path: Path,
    max_size_mb: int = 50,
    max_files: int = 5,
) -> bool:
    """Rotate a log file using the copytruncate strategy.

    If the log file exceeds ``max_size_mb``, this function:
    1. Shifts existing archives up by one number (.1.gz → .2.gz, etc.)
    2. Copies the current log to ``<name>.log.1.gz`` (gzip compressed)
    3. Truncates the original log file in-place (preserving the FD)
    4. Deletes archives numbered higher than ``max_files``

    Args:
        log_path: Path to the log file to rotate.
        max_size_mb: Maximum log file size in megabytes before rotation.
        max_files: Maximum number of rotated archive files to retain.

    Returns:
        True if rotation was performed, False if skipped (file missing,
        empty, or below threshold).

    Raises:
        LogRotationError: If a permission error or I/O error prevents
            rotation.
    """
    # Edge case: missing file — silently skip
    if not log_path.exists():
        return False

    # Edge case: empty file — skip
    try:
        file_size = log_path.stat().st_size
    except OSError:
        return False

    if file_size == 0:
        return False

    # Edge case: below threshold — skip
    threshold_bytes = max_size_mb * 1024 * 1024
    if file_size < threshold_bytes:
        return False

    try:
        _perform_rotation(log_path, max_files)
    except PermissionError as exc:
        msg = f"Permission denied during log rotation of {log_path}: {exc}"
        raise LogRotationError(msg) from None
    except OSError as exc:
        msg = f"I/O error during log rotation of {log_path}: {exc}"
        raise LogRotationError(msg) from None

    return True


def _perform_rotation(log_path: Path, max_files: int) -> None:
    """Execute the actual rotation steps.

    This is separated from ``rotate_log`` to keep the edge-case
    handling clean.

    Args:
        log_path: Path to the log file.
        max_files: Maximum number of archive files to keep.
    """
    base = log_path.parent
    stem = log_path.name  # e.g. "fast.log"

    # Step 1: Delete archives beyond max_files (they'll shift up by 1)
    # After shifting, the old .max_files.gz becomes .(max_files+1).gz
    # so we need to delete .max_files.gz before shifting
    _delete_excess_archives(base, stem, max_files)

    # Step 2: Shift existing archives up by one number
    _shift_archives(base, stem, max_files)

    # Step 3: Copy current log to temporary file, compress to .1.gz
    archive_path = base / f"{stem}.1.gz"
    _copy_and_compress(log_path, archive_path)

    # Step 4: Truncate the original log file in-place
    _truncate_file(log_path)


def _archive_path_for(base: Path, stem: str, number: int) -> Path:
    """Return the path for archive number N.

    Args:
        base: Parent directory.
        stem: Log file name (e.g. "fast.log").
        number: Archive number (1 = most recent).

    Returns:
        Path like ``base/fast.log.1.gz``.
    """
    return base / f"{stem}.{number}.gz"


def _delete_excess_archives(base: Path, stem: str, max_files: int) -> None:
    """Delete archives numbered higher than max_files.

    After the upcoming shift, what is currently archive N will become
    archive N+1. So we need to remove the current max_files archive
    (which would become max_files+1 after shifting) and any existing
    archives beyond max_files.

    Args:
        base: Parent directory.
        stem: Log file name.
        max_files: Maximum archives to keep.
    """
    # Delete from max_files upward (the shift would push these out of range)
    n = max_files
    while True:
        path = _archive_path_for(base, stem, n)
        if path.exists():
            path.unlink()
            n += 1
        else:
            break


def _shift_archives(base: Path, stem: str, max_files: int) -> None:
    """Shift existing archives up by one number.

    Moves .N.gz → .(N+1).gz, starting from the highest existing
    number down to 1, to avoid overwriting.

    Args:
        base: Parent directory.
        stem: Log file name.
        max_files: Maximum archives (used as upper bound for search).
    """
    # Find the highest existing archive number (up to max_files - 1,
    # since max_files was already deleted)
    highest = 0
    for i in range(1, max_files):
        if _archive_path_for(base, stem, i).exists():
            highest = i

    # Shift from highest down to 1
    for i in range(highest, 0, -1):
        src = _archive_path_for(base, stem, i)
        dst = _archive_path_for(base, stem, i + 1)
        if src.exists():
            src.rename(dst)


def _copy_and_compress(src: Path, dst_gz: Path) -> None:
    """Copy a file and compress it to a gzip archive.

    Uses shutil.copy2 to copy to a temporary file, then compresses
    the temporary file to the target gzip path.

    Args:
        src: Source log file path.
        dst_gz: Destination gzip archive path.
    """
    # Create a temporary uncompressed copy
    tmp_path = dst_gz.parent / f"{dst_gz.name}.tmp"
    try:
        shutil.copy2(str(src), str(tmp_path))

        # Compress the copy
        with open(tmp_path, "rb") as f_in:
            with gzip.open(str(dst_gz), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    finally:
        # Clean up temporary file
        if tmp_path.exists():
            tmp_path.unlink()


def _truncate_file(path: Path) -> None:
    """Truncate a file in-place, preserving the path.

    Uses ``open(path, "w").close()`` to truncate to zero bytes.

    Args:
        path: Path to the file to truncate.
    """
    open(path, "w").close()  # noqa: SIM115
