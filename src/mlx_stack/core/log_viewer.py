"""Log viewing and management module for mlx-stack.

Provides core functionality for viewing, following, listing, and rotating
service log files in ~/.mlx-stack/logs/. Supports tail-style viewing,
real-time following with truncation detection, log listing with metadata,
on-demand rotation, and archived log viewing.
"""

from __future__ import annotations

import gzip
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from mlx_stack.core.config import get_value
from mlx_stack.core.log_rotation import LogRotationError, rotate_log
from mlx_stack.core.paths import get_logs_dir

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Polling interval for --follow mode (seconds)
FOLLOW_POLL_INTERVAL = 0.5

# Default number of lines for --tail
DEFAULT_TAIL_LINES = 50

# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class LogFileInfo:
    """Metadata about a log file."""

    name: str
    service: str
    size_bytes: int
    modified: datetime
    is_archive: bool = False

    @property
    def size_display(self) -> str:
        """Return human-readable file size."""
        if self.size_bytes < 1024:
            return f"{self.size_bytes} B"
        elif self.size_bytes < 1024 * 1024:
            return f"{self.size_bytes / 1024:.1f} KB"
        elif self.size_bytes < 1024 * 1024 * 1024:
            return f"{self.size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{self.size_bytes / (1024 * 1024 * 1024):.1f} GB"

    @property
    def modified_display(self) -> str:
        """Return human-readable modification time."""
        return self.modified.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class RotationResult:
    """Result of a rotation operation."""

    service: str
    rotated: bool
    error: str | None = None


# --------------------------------------------------------------------------- #
# Log listing
# --------------------------------------------------------------------------- #


def list_log_files() -> list[LogFileInfo]:
    """List all log files in the logs directory.

    Returns current log files (*.log) sorted alphabetically. Does not
    include archived .gz files.

    Returns:
        List of LogFileInfo objects for each log file found.
    """
    logs_dir = get_logs_dir()
    if not logs_dir.exists():
        return []

    results: list[LogFileInfo] = []
    for path in sorted(logs_dir.iterdir()):
        if path.suffix == ".log" and path.is_file():
            # Extract service name: "fast.log" -> "fast"
            service = path.stem
            try:
                stat = path.stat()
                info = LogFileInfo(
                    name=path.name,
                    service=service,
                    size_bytes=stat.st_size,
                    modified=datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ),
                )
                results.append(info)
            except OSError:
                continue

    return results


def get_log_path(service: str) -> Path | None:
    """Get the path to a service's log file.

    Args:
        service: The service name (e.g. "fast", "litellm").

    Returns:
        Path to the log file, or None if it doesn't exist.
    """
    logs_dir = get_logs_dir()
    log_path = logs_dir / f"{service}.log"
    if log_path.exists() and log_path.is_file():
        return log_path
    return None


def get_available_services() -> list[str]:
    """Get list of services that have log files.

    Returns:
        Sorted list of service names with existing log files.
    """
    return [info.service for info in list_log_files()]


# --------------------------------------------------------------------------- #
# Log viewing
# --------------------------------------------------------------------------- #


def read_log_tail(log_path: Path, num_lines: int = DEFAULT_TAIL_LINES) -> str:
    """Read the last N lines from a log file.

    Args:
        log_path: Path to the log file.
        num_lines: Number of lines to return from the end.

    Returns:
        String containing the last N lines of the file.
    """
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""

    if not content:
        return ""

    lines = content.splitlines()
    tail_lines = lines[-num_lines:] if len(lines) > num_lines else lines
    return "\n".join(tail_lines)


def read_log_full(log_path: Path) -> str:
    """Read the full content of a log file.

    Args:
        log_path: Path to the log file.

    Returns:
        Full file content as a string.
    """
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


# --------------------------------------------------------------------------- #
# Log following
# --------------------------------------------------------------------------- #


def follow_log(
    log_path: Path,
    num_lines: int = DEFAULT_TAIL_LINES,
    output_callback: Callable[[str], None] | None = None,
) -> None:
    """Follow a log file, printing new content as it appears.

    Implements tail -f behavior using polling. Checks file size every
    FOLLOW_POLL_INTERVAL seconds, reads new content when size increases.
    Detects file truncation (size decrease) and resets read position.

    This function blocks until interrupted. Ctrl-C is handled cleanly
    (no traceback, exit 0).

    Args:
        log_path: Path to the log file to follow.
        num_lines: Number of initial lines to show.
        output_callback: Optional callable for writing output. If None,
            uses print(). Signature: callback(text: str) -> None.
    """
    write = output_callback or print

    # Show initial tail
    initial = read_log_tail(log_path, num_lines)
    if initial:
        write(initial)

    # Start following from current end of file
    try:
        position = log_path.stat().st_size
    except OSError:
        position = 0

    try:
        while True:
            time.sleep(FOLLOW_POLL_INTERVAL)

            try:
                current_size = log_path.stat().st_size
            except OSError:
                # File may have been removed temporarily
                continue

            # Detect truncation (copytruncate rotation)
            if current_size < position:
                position = 0

            if current_size > position:
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        f.seek(position)
                        new_content = f.read()
                        if new_content:
                            # Strip trailing newline to avoid double-spacing
                            text = new_content.rstrip("\n")
                            if text:
                                write(text)
                        position = f.tell()
                except OSError:
                    continue

    except KeyboardInterrupt:
        # Clean exit on Ctrl-C — no traceback
        pass


# --------------------------------------------------------------------------- #
# Archived log viewing
# --------------------------------------------------------------------------- #


def _get_archives_for_service(service: str) -> list[Path]:
    """Get archived log files for a service, sorted oldest first.

    Archives are named like ``<service>.log.N.gz`` where N=1 is the most
    recent. We return them sorted with the highest N first (oldest first)
    for chronological display.

    Args:
        service: The service name.

    Returns:
        List of archive paths, sorted oldest first.
    """
    logs_dir = get_logs_dir()
    if not logs_dir.exists():
        return []

    pattern = re.compile(rf"^{re.escape(service)}\.log\.(\d+)\.gz$")
    archives: list[tuple[int, Path]] = []

    for path in logs_dir.iterdir():
        match = pattern.match(path.name)
        if match and path.is_file():
            num = int(match.group(1))
            archives.append((num, path))

    # Sort by number descending (highest number = oldest)
    archives.sort(key=lambda x: x[0], reverse=True)
    return [path for _, path in archives]


def read_archive(archive_path: Path) -> str:
    """Read and decompress a gzip archive.

    Args:
        archive_path: Path to the .gz archive file.

    Returns:
        Decompressed content as a string.
    """
    try:
        with gzip.open(str(archive_path), "rb") as f:
            return f.read().decode("utf-8", errors="replace")
    except (OSError, gzip.BadGzipFile):
        return f"[Error reading archive: {archive_path.name}]\n"


def read_all_logs(service: str) -> str:
    """Read all logs for a service: archives + current, chronologically.

    Archives are shown oldest first, then the current log file.

    Args:
        service: The service name.

    Returns:
        Combined content from all archives and the current log.
    """
    parts: list[str] = []

    # Read archives oldest first
    archives = _get_archives_for_service(service)
    for archive_path in archives:
        content = read_archive(archive_path)
        if content:
            parts.append(f"--- Archive: {archive_path.name} ---")
            parts.append(content.rstrip("\n"))

    # Read current log
    log_path = get_log_path(service)
    if log_path is not None:
        current = read_log_full(log_path)
        if current:
            parts.append(f"--- Current: {log_path.name} ---")
            parts.append(current.rstrip("\n"))

    return "\n".join(parts) if parts else ""


# --------------------------------------------------------------------------- #
# On-demand rotation
# --------------------------------------------------------------------------- #


def rotate_service_log(service: str) -> RotationResult:
    """Rotate a single service's log file.

    Uses configured max_size_mb and max_files from the config module.

    Args:
        service: The service name.

    Returns:
        RotationResult indicating whether rotation was performed.
    """
    log_path = get_log_path(service)
    if log_path is None:
        return RotationResult(
            service=service,
            rotated=False,
            error=f"No log file found for service '{service}'",
        )

    max_size_mb = get_value("log-max-size-mb")
    max_files = get_value("log-max-files")

    try:
        rotated = rotate_log(log_path, max_size_mb=max_size_mb, max_files=max_files)
        return RotationResult(service=service, rotated=rotated)
    except LogRotationError as exc:
        return RotationResult(service=service, rotated=False, error=str(exc))


def rotate_all_logs() -> list[RotationResult]:
    """Rotate all eligible service log files.

    Returns:
        List of RotationResult objects, one per service.
    """
    services = get_available_services()
    if not services:
        return []

    results: list[RotationResult] = []
    for service in services:
        result = rotate_service_log(service)
        results.append(result)
    return results
