"""Fake service layer for behavioral testing of stack orchestration.

Replaces 10-deep ``@patch`` stacks with a single, configurable test double.
The default behaviour is *happy path* — everything succeeds.  Tests configure
only the failures they care about via explicit methods like ``fail_port()``,
``fail_health()``, or ``hold_lock()``.

Usage in tests::

    def test_startup(stack_on_disk, fake_services):
        # Arrange — default: all services start successfully
        # Act
        result = run_up()
        # Assert
        assert all(t.status == "healthy" for t in result.tiers)
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

from mlx_stack.core.process import (
    HealthCheckError,
    HealthCheckResult,
    LockError,
    ServiceInfo,
)
from tests.factories import make_test_catalog


class FakeServiceLayer:
    """Test double for OS-boundary functions used by ``run_up`` / watchdog.

    Defaults produce a successful startup.  Call configuration methods to
    inject specific failure scenarios before calling the code under test.
    """

    def __init__(self) -> None:
        # --- Configuration (set before act) ---
        self._port_conflicts: dict[int, tuple[int, str]] = {}
        self._health_failures: set[int] = set()
        self._start_failures: set[str] = set()
        self._alive_pids: dict[int, bool] = {}
        self._lock_held: bool = False
        self._dependency_failures: set[str] = set()
        self._model_check_failures: dict[str, str] = {}
        self._catalog = make_test_catalog()
        self._config_overrides: dict[str, Any] = {}

        # --- Recording (assert after act) ---
        self.started: list[str] = []
        self.health_checked: list[int] = []
        self.dependencies_checked: list[str] = []

        # Internal counter for deterministic PIDs
        self._next_pid = 10000

    # ------------------------------------------------------------------ #
    # Configuration methods — call these in the Arrange phase
    # ------------------------------------------------------------------ #

    def fail_port(self, port: int, pid: int = 99999, name: str = "other") -> None:
        """Make ``check_port_conflict`` report *port* as occupied."""
        self._port_conflicts[port] = (pid, name)

    def fail_health(self, port: int) -> None:
        """Make ``wait_for_healthy`` raise ``HealthCheckError`` for *port*."""
        self._health_failures.add(port)

    def fail_start(self, service_name: str) -> None:
        """Make ``start_service`` raise for *service_name*."""
        self._start_failures.add(service_name)

    def set_alive(self, pid: int, alive: bool = True) -> None:
        """Control what ``is_process_alive`` returns for a specific PID."""
        self._alive_pids[pid] = alive

    def hold_lock(self) -> None:
        """Make ``acquire_lock`` raise ``LockError``."""
        self._lock_held = True

    def fail_dependency(self, name: str) -> None:
        """Make ``ensure_dependency`` raise for *name*."""
        self._dependency_failures.add(name)

    def fail_model_check(self, tier_name: str, message: str) -> None:
        """Make ``check_local_model_exists`` return an error for *tier_name*."""
        self._model_check_failures[tier_name] = message

    def set_catalog(self, catalog: list[Any]) -> None:
        """Override the catalog returned by ``load_catalog``."""
        self._catalog = catalog

    def set_config(self, key: str, value: Any) -> None:
        """Override a config value returned by ``get_value``."""
        self._config_overrides[key] = value

    # ------------------------------------------------------------------ #
    # Fake implementations — patched into the module under test
    # ------------------------------------------------------------------ #

    def start_service(
        self,
        service_name: str,
        cmd: list[str],
        port: int,
        env: dict[str, str] | None = None,
        log_dir: Path | None = None,
    ) -> ServiceInfo:
        """Fake ``process.start_service``."""
        self.started.append(service_name)
        if service_name in self._start_failures:
            msg = f"Fake start failure for {service_name}"
            raise RuntimeError(msg)
        pid = self._next_pid + port
        return ServiceInfo(
            name=service_name,
            pid=pid,
            port=port,
            log_path=Path(f"/tmp/{service_name}.log"),
            pid_path=Path(f"/tmp/{service_name}.pid"),
        )

    def wait_for_healthy(self, **kwargs: Any) -> HealthCheckResult:
        """Fake ``process.wait_for_healthy``."""
        port = kwargs.get("port", 0)
        self.health_checked.append(port)
        if port in self._health_failures:
            msg = f"Timeout after 120s waiting for port {port}"
            raise HealthCheckError(msg)
        return HealthCheckResult(healthy=True, response_time=0.1, status_code=200)

    def check_port_conflict(self, port: int) -> tuple[int, str] | None:
        """Fake ``process.check_port_conflict``."""
        return self._port_conflicts.get(port)

    def is_process_alive(self, pid: int) -> bool:
        """Fake ``process.is_process_alive``."""
        return self._alive_pids.get(pid, False)

    def cleanup_stale_pid(self, service_name: str) -> bool:
        """Fake ``process.cleanup_stale_pid``."""
        return True

    @contextmanager
    def acquire_lock(self):
        """Fake ``process.acquire_lock``."""
        if self._lock_held:
            raise LockError("Lock held by another process (fake)")
        yield

    def ensure_dependency(self, name: str) -> None:
        """Fake ``deps.ensure_dependency``."""
        self.dependencies_checked.append(name)
        if name in self._dependency_failures:
            from mlx_stack.core.deps import DependencyInstallError

            msg = f"Failed to install {name} (fake)"
            raise DependencyInstallError(msg)

    def which(self, name: str) -> str:
        """Fake ``shutil.which``."""
        return f"/usr/local/bin/{name}"

    def check_local_model_exists(self, tier: dict[str, Any]) -> str | None:
        """Fake ``stack_up.check_local_model_exists``."""
        tier_name = tier.get("name", "")
        return self._model_check_failures.get(tier_name)

    def load_catalog(self) -> list[Any]:
        """Fake ``catalog.load_catalog``."""
        return self._catalog

    def get_value(self, key: str) -> Any:
        """Fake ``config.get_value``."""
        if key in self._config_overrides:
            return self._config_overrides[key]
        defaults: dict[str, Any] = {
            "litellm-port": 4000,
            "openrouter-key": "",
            "default-quant": "int4",
            "memory-budget-pct": 40,
            "model-dir": "~/.mlx-stack/models",
            "auto-health-check": True,
            "log-max-size-mb": 50,
            "log-max-files": 5,
        }
        return defaults.get(key, "")
