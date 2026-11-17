"""Pytest configuration and global fixtures for backend test suite."""
import ipaddress
import os
import socket

import pytest

# Fail-closed test environment isolation: prevent any automated test from reaching cloud databases
if os.environ.get("RUN_LIVE_STAGING_TESTS") != "1":
    os.environ["ENVIRONMENT"] = "test"
    os.environ["DATABASE_BACKEND"] = "sqlite"
    os.environ["QDRANT_STORAGE_PATH"] = ":memory:"
    for k in ("DATABASE_URL", "STAGING_DATABASE_URL", "SUPABASE_DATABASE_URL", "QDRANT_URL", "QDRANT_API_KEY"):
        val = os.environ.get(k, "")
        if any(remote in val for remote in ("supabase.com", "pooler", "postgres.", "aws-0-", "qdrant.io")):
            os.environ.pop(k, None)
    try:
        from app.core.config import settings
        settings.QDRANT_URL = ""
        settings.QDRANT_API_KEY = ""
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Hermetic outbound-network guard.
# LLMRouter (and other services) lazily build LIVE providers when a test does
# not inject a mock. Those leaks previously produced real HTTPS calls to
# openrouter.ai and flaky failures. Unless a live-test flag is explicitly set,
# every non-loopback DNS lookup / connect is refused and recorded.
# ---------------------------------------------------------------------------
LIVE_NETWORK_ALLOWED = (
    os.environ.get("RUN_LIVE_PROVIDER_TESTS") == "1" or os.environ.get("RUN_LIVE_STAGING_TESTS") == "1"
)
BLOCKED_NETWORK_ATTEMPTS: list = []
_LOCAL_NAMES = {"localhost", "testserver", "test", ""}


def _is_local_host(host) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "ignore")
    name = str(host).strip("[]").lower()
    if name in _LOCAL_NAMES or name.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(name.split("%")[0])
    except ValueError:
        return False
    return ip.is_loopback or ip.is_unspecified


if not LIVE_NETWORK_ALLOWED:
    _real_getaddrinfo = socket.getaddrinfo
    _real_connect = socket.socket.connect
    _real_connect_ex = socket.socket.connect_ex

    def _guarded_getaddrinfo(host, *args, **kwargs):
        if not _is_local_host(host):
            BLOCKED_NETWORK_ATTEMPTS.append(str(host))
            raise socket.gaierror(f"Hermetic test run: live network lookup blocked for {host!r}")
        return _real_getaddrinfo(host, *args, **kwargs)

    def _check_address(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6) and isinstance(address, tuple):
            if not _is_local_host(address[0]):
                BLOCKED_NETWORK_ATTEMPTS.append(str(address[0]))
                raise ConnectionRefusedError(f"Hermetic test run: live connect blocked for {address[0]!r}")

    def _guarded_connect(sock, address):
        _check_address(sock, address)
        return _real_connect(sock, address)

    def _guarded_connect_ex(sock, address):
        _check_address(sock, address)
        return _real_connect_ex(sock, address)

    socket.getaddrinfo = _guarded_getaddrinfo
    socket.socket.connect = _guarded_connect
    socket.socket.connect_ex = _guarded_connect_ex


def pytest_terminal_summary(terminalreporter):
    if BLOCKED_NETWORK_ATTEMPTS:
        counts = {}
        for host in BLOCKED_NETWORK_ATTEMPTS:
            counts[host] = counts.get(host, 0) + 1
        terminalreporter.write_sep("-", "hermetic network guard: blocked live attempts")
        for host, n in sorted(counts.items()):
            terminalreporter.write_line(f"{host}: {n}")


from app.services.cache.response_cache import response_cache  # noqa: E402
from app.services.llm.circuit_breaker import model_health_tracker  # noqa: E402


@pytest.fixture(autouse=True)
def clean_test_state():
    """Ensure every test runs with an isolated, clean response cache, health tracker, and quota state."""
    response_cache.clear()
    model_health_tracker.reset()
    try:
        from app.core.maintenance import set_maintenance_mode
        set_maintenance_mode(None)
    except Exception:  # noqa: S110, BLE001
        pass
    try:
        from app.services.workspace_store import workspace_store
        with workspace_store._get_sqlite_connection() as conn:
            conn.execute("DELETE FROM user_daily_usage")
            conn.execute("DELETE FROM library_files WHERE status = 'deleting'")
            conn.commit()
    except Exception:  # noqa: S110, BLE001
        pass
    yield
    response_cache.clear()
    model_health_tracker.reset()
    try:
        from app.core.maintenance import set_maintenance_mode
        set_maintenance_mode(None)
    except Exception:  # noqa: S110, BLE001
        pass
    try:
        from app.services.workspace_store import workspace_store
        with workspace_store._get_sqlite_connection() as conn:
            conn.execute("DELETE FROM user_daily_usage")
            conn.execute("DELETE FROM library_files WHERE status = 'deleting'")
            conn.commit()
    except Exception:  # noqa: S110, BLE001
        pass

