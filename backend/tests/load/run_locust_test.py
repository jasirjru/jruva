"""Automated test harness to run a real Locust concurrent load test and report real p50/p95/p99."""
import subprocess
import time
import sys
import os
import csv
import httpx

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Pre-seed cache in application
from app.services.cache.response_cache import response_cache
from app.services.image.image_service import image_service

# Pre-seed chat cache
response_cache.set(
    tier="S+++",
    model="openai/gpt-oss-120b",
    prompt="benchmark-cached-probe-prompt",
    content="This is a verified pre-seeded benchmark response.",
    provider="groq",
    model_name="openai/gpt-oss-120b",
    total_tokens=42,
)

# Pre-seed image cache
image_service.cache.set(
    prompt="benchmark-cached-image-prompt",
    width=1024,
    height=1024,
    data={
        "success": True,
        "image_url": "https://example.com/cached-benchmark.png",
        "original_prompt": "benchmark-cached-image-prompt",
        "enhanced_prompt": "benchmark-cached-image-prompt, photorealistic 8k",
        "model": "flux",
        "seed": 12345,
        "width": 1024,
        "height": 1024,
    }
)

def run(users=None, run_time="10s", strategy=None):
    users = users or (int(sys.argv[1]) if len(sys.argv) > 1 else 50)
    strategy = strategy or (sys.argv[2] if len(sys.argv) > 2 else os.getenv("CACHE_PERSISTENCE_STRATEGY", "sync")).lower().strip()
    port = 8019
    host = f"http://127.0.0.1:{port}"
    print(f"[Harness] Starting Uvicorn server on {host} for {users} concurrent users (Strategy: {strategy})...")

    # Start uvicorn in background with production baseline rate limit (12 req/minute) and chosen strategy
    env = os.environ.copy()
    env["RATE_LIMIT_PER_MINUTE"] = "12"
    env["CACHE_PERSISTENCE_STRATEGY"] = strategy
    cmd_server = [
        sys.executable, "-m", "uvicorn", "app.main:app",
        "--host", "127.0.0.1", "--port", str(port),
        "--log-level", "warning"
    ]
    server_proc = subprocess.Popen(cmd_server, cwd="backend", env=env)

    # Wait for server to become responsive
    ready = False
    for i in range(30):
        try:
            with httpx.Client(timeout=1.0) as client:
                r = client.get(f"{host}/api/v1/models")
                if r.status_code == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.3)

    if not ready:
        print("[Harness] ERROR: Server failed to start in time.")
        server_proc.kill()
        sys.exit(1)

    print(f"[Harness] Server is ready! Triggering Locust with {users} users (spawn rate {max(5, users // 5)})...")
    csv_prefix = f"backend/tests/load/locust_u{users}_{strategy}"
    locust_cmd = [
        sys.executable, "-m", "locust",
        "-f", "backend/tests/load/locustfile.py",
        "--headless",
        "-u", str(users),
        "-r", str(max(5, users // 5)),
        "--run-time", run_time,
        "--host", host,
        "--csv", csv_prefix,
        "--csv-full-history",
    ]

    try:
        locust_res = subprocess.run(locust_cmd, capture_output=True, text=True)
        print("[Harness] Locust execution finished.")
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=3)
        except Exception:
            server_proc.kill()

    # Parse and display CSV output
    stats_file = f"{csv_prefix}_stats.csv"
    if os.path.exists(stats_file):
        print("\n" + "="*80)
        print("REAL MEASURED LOAD TEST METRICS (from Locust CSV output):")
        print("="*80)
        with open(stats_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = ["Type", "Name", "Request Count", "Failure Count", "Median Response Time", "Average Response Time", "95%", "99%", "Requests/s"]
            print(f"{'Endpoint':<35} | {'Reqs':<6} | {'Fail':<4} | {'p50(ms)':<8} | {'p95(ms)':<8} | {'p99(ms)':<8} | {'RPS':<6}")
            print("-" * 85)
            for row in reader:
                name = row.get("Name", "")
                reqs = row.get("Request Count", "")
                fails = row.get("Failure Count", "")
                p50 = row.get("50%", row.get("Median Response Time", ""))
                p95 = row.get("95%", "")
                p99 = row.get("99%", "")
                rps = row.get("Requests/s", "")
                print(f"{name:<35} | {reqs:<6} | {fails:<4} | {p50:<8} | {p95:<8} | {p99:<8} | {rps:<6}")
        print("="*80 + "\n")
    else:
        print("[Harness] Stats file not found. Raw output:")
        print(locust_res.stdout)
        print(locust_res.stderr)

if __name__ == "__main__":
    run()
