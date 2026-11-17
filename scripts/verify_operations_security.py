"""Run operational security tests without cloud credentials or real network access.

Run with an existing backend environment: python -B scripts/verify_operations_security.py [--full]
Artifacts and temporary files stay under this checkout's ignored scratch directory.
"""

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from pathlib import Path


def main() -> int:
    # Retrieval has set-based tie ordering. Fix the interpreter seed before imports
    # so baseline/candidate comparisons cannot depend on process hash randomization.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        # execv does not reliably propagate the child exit status on Windows.
        return subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]],
            check=False,
        ).returncode
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all backend tests except the shared-server E2E script",
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=Path(tempfile.gettempdir()) / "fastembed_cache",
        help="Existing FastEmbed cache to copy privately; no downloads are performed",
    )
    parser.add_argument("--targets", nargs="+", help="Explicit offline pytest targets")
    parser.add_argument("--evidence-name", default=None, help="JUnit filename stem")
    parser.add_argument("--coverage", action="store_true", help="Enforce existing CI app coverage floor (pytest-cov required)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    scratch = root / "scratch" / "operations-verification"
    scratch.mkdir(parents=True, exist_ok=True)
    model_name = "models--qdrant--bge-small-en-v1.5-onnx-q"
    cached_model = args.model_cache / model_name
    dest_model = scratch / "fastembed_cache" / model_name
    if cached_model.is_dir() and not dest_model.is_dir():
        shutil.copytree(
            cached_model, dest_model, dirs_exist_ok=True
        )
    # Never inherit live credentials or database/storage endpoints into this process.
    for name in (
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "HUGGINGFACE_API_KEY",
        "QDRANT_API_KEY",
        "QDRANT_URL",
        "DATABASE_URL",
        "SUPABASE_DATABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_KEY",
    ):
        os.environ[name] = ""
    os.environ.update(
        ENVIRONMENT="test",
        JWT_SECRET_KEY="synthetic-operations-verification-only",
        GOOGLE_CLIENT_ID="synthetic-client.apps.googleusercontent.com",
        DATABASE_BACKEND="sqlite",
        STORAGE_BACKEND="local",
        RUN_LIVE_PROVIDER_TESTS="0",
        OPERATIONS_RESET_ENABLED="false",
        PYTHONDONTWRITEBYTECODE="1",
        TEMP=str(scratch),
        TMP=str(scratch),
        HF_HUB_OFFLINE="1",
        FASTEMBED_CACHE_PATH=str(scratch / "fastembed_cache"),
        QDRANT_STORAGE_PATH=":memory:",
    )
    tempfile.tempdir = str(scratch)
    sys.path.insert(0, str(root / "backend"))
    sys.path.insert(0, str(root))

    def no_network(*_args, **_kwargs):
        raise OSError("Real network access is disabled during operations verification.")

    # TestClient/ASGITransport and mocked transports still work. Block outbound
    # IPv4/IPv6, including loopback, to avoid touching a shared local backend.
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo
    real_socketpair = socket.socketpair
    internal_socketpair = threading.local()

    def guarded_socketpair(*arguments, **kwargs):
        # Windows implements socketpair using its own fresh loopback listener.
        # Permit only that stdlib operation in this thread, not shared services.
        internal_socketpair.active = True
        try:
            return real_socketpair(*arguments, **kwargs)
        finally:
            internal_socketpair.active = False

    def guarded_connect(sock, *arguments):
        if sock.family in (socket.AF_INET, socket.AF_INET6) and not getattr(
            internal_socketpair, "active", False
        ):
            return no_network()
        return real_connect(sock, *arguments)

    def guarded_connect_ex(sock, *arguments):
        if sock.family in (socket.AF_INET, socket.AF_INET6) and not getattr(
            internal_socketpair, "active", False
        ):
            return no_network()
        return real_connect_ex(sock, *arguments)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.getaddrinfo = no_network
    socket.socketpair = guarded_socketpair
    try:
        import pytest

        targets = args.targets or (
            ["backend/tests"]
            if args.full
            else [
                "backend/tests/test_operations_boundary.py",
                "backend/tests/test_telemetry.py",
                "backend/tests/test_observability.py",
                "backend/tests/test_auth.py",
                "backend/tests/test_csrf.py",
                "backend/tests/test_api.py",
                "backend/tests/test_tenant_authorization_hardening.py",
            ]
        )
        # This test intentionally connects to an existing server; never run it here.
        return pytest.main(
            [
                *targets,
                *(["--cov=app", "--cov-report=term-missing", "--cov-fail-under=60"] if args.coverage else []),
                "--ignore=backend/tests/test_e2e_live.py",
                "-q",
                "--tb=short",
                "-p",
                "no:cacheprovider",
                "--basetemp",
                str(scratch / "pytest-temp"),
                "--junitxml",
                str(scratch / ((args.evidence_name or ("full" if args.full else "focused")) + ".xml")),
            ]
        )
    finally:
        socket.socket.connect = real_connect
        socket.socket.connect_ex = real_connect_ex
        socket.getaddrinfo = real_getaddrinfo
        socket.socketpair = real_socketpair


if __name__ == "__main__":
    raise SystemExit(main())
