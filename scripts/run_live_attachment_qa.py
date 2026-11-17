"""Exact-deployment required-field acceptance; not general accuracy or memory validation."""
import argparse
import base64
import hashlib
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = os.getenv("BACKEND_URL", "https://jru-chatbot-backend.onrender.com").rstrip("/")
CANONICAL_FRONTEND_URL = os.getenv("FRONTEND_URL", "https://www.jruva.in").rstrip("/")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")


FIXTURES = [
    {
        "id": "Test A — JPEG",
        "path": os.path.join(PROJECT_ROOT, "fixtures", "test_a_visual_text.jpg"),
        "mime": "image/jpeg",
        "prompt": "Describe the visual objects in this certificate and transcribe all visible text, candidate ID and score exactly.",
        "ground_truth": {
            "objects": ["Gold star medal", "blue laboratory flask", "silver laptop"],
            "text": ["JRUVA RESEARCH CERTIFICATE 2026", "Candidate ID: JIdent-8842", "Score: 98.5%"],
        },
    },
    {
        "id": "Test B — PNG",
        "path": os.path.join(PROJECT_ROOT, "fixtures", "test_b_chart.png"),
        "mime": "image/png",
        "prompt": "Explain the contents of this chart and extract the exact performance percentage values for AI Lab, Robotics, Data Science, and Cybersecurity.",
        "ground_truth": {
            "title": "DEPARTMENT PERFORMANCE METRICS 2026",
            "values": {
                "AI Lab": "94.2%",
                "Robotics": "88.7%",
                "Data Science": "91.5%",
                "Cybersecurity": "85.0%",
            },
        },
    },
    {
        "id": "Test C — Text-based PDF",
        "path": os.path.join(PROJECT_ROOT, "fixtures", "test_c_text_document.pdf"),
        "mime": "application/pdf",
        "prompt": "Provide a structured summary of this document, state the exact Authorized Budget, number of Approved Projects, the Project Zephyr telemetry frequency, and the exact Identifying Verification Phrase.",
        "ground_truth": {
            "budget": "$4,850,000",
            "projects": "12",
            "telemetry": "420 megahertz",
            "phrase": "Immutable Ledger Verification Protocol Alpha",
        },
    },
    {
        "id": "Test D — Scanned PDF",
        "path": os.path.join(PROJECT_ROOT, "fixtures", "test_d_scanned_document.pdf"),
        "mime": "application/pdf",
        "prompt": "This is a scanned document with no selectable text. Extract the Permit Number, Facility Code, Date, Safety Rating, and Inspector Name exactly.",
        "ground_truth": {
            "permit": "SC-90412",
            "facility": "FAC-99",
            "date": "11 October 2026",
            "safety_rating": "Category A-1",
            "inspector": "Inspector K. Vance",
        },
    },
]


def check_backend_health():
    start = time.perf_counter()
    try:
        response = httpx.get(f"{BASE_URL}/api/v1/health", timeout=30.0)
        data = response.json()
        healthy = response.status_code == 200 and isinstance(data, dict) and data.get("status") == "healthy"
        return healthy, response.status_code, (time.perf_counter() - start) * 1000, data
    except (httpx.HTTPError, ValueError) as error:
        return False, 0, (time.perf_counter() - start) * 1000, {"error": str(error)}


def evaluate_accuracy(fixture, answer):
    """Conservative required-field checks, including chart row association."""
    normalized = re.sub(r"[*`_]", "", answer.lower())
    flat = re.sub(r"\s+", " ", normalized)
    truth = fixture["ground_truth"]
    omissions = []
    if "objects" in truth:
        for label, pattern in [
            ("gold star medal", r"gold(?:en)?[^\n]{0,80}star[^\n]{0,40}(?:medal|badge)"),
            ("blue laboratory flask", r"blue[^\n]{0,80}(?:flask|beaker)"),
            ("silver laptop", r"(?:silver|gr[ae]y)[^\n]{0,80}laptop"),
        ]:
            if not re.search(pattern, normalized):
                omissions.append(f"Missing object: {label}")
        for value in truth["text"]:
            # Labels may be reformatted into Markdown headings/bullets.
            required = value.split(": ", 1)[-1].lower()
            if required not in flat:
                omissions.append(f"Missing text: {value}")
    elif "values" in truth:
        if truth["title"].lower() not in flat:
            omissions.append("Missing chart title")
        for department, value in truth["values"].items():
            pattern = rf"{re.escape(department.lower())}[^\n\d]{{0,40}}{re.escape(value)}(?!\d)"
            if not re.search(pattern, normalized):
                omissions.append(f"Missing/misassociated chart value: {department} = {value}")
    else:
        for field, value in truth.items():
            if field == "projects":
                pattern = rf"(?:approved projects[^\d]{{0,30}}\b{value}\b|\b{value}\b\s+(?:institutional\s+)?(?:projects|initiatives))"
                found = re.search(pattern, flat) is not None
            else:
                found = value.lower() in flat
            if not found:
                omissions.append(f"Missing {field}: {value}")
    return omissions


def iter_sse_events(chunks):
    buffer = ""
    for chunk in chunks:
        buffer = (buffer + chunk).replace("\r\n", "\n")
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            lines = block.splitlines()
            data = "\n".join(line[5:].lstrip() for line in lines if line.startswith("data:"))
            if not data:
                continue
            event = json.loads(data)
            if not isinstance(event, dict):
                raise ValueError("Invalid SSE event object")
            event_name = next((line[6:].strip() for line in lines if line.startswith("event:")), "")
            if not event.get("type") and event_name in ("error", "done", "finish", "token"):
                event["type"] = event_name
            yield event
    if buffer.strip() and not buffer.lstrip().startswith(":"):
        raise ValueError("Unterminated SSE frame")


def run_single_test(fixture, round_num=1):
    file_path = Path(fixture["path"])
    file_bytes = file_path.read_bytes()
    payload = {
        "conversation_id": str(uuid.uuid4()), "session_id": str(uuid.uuid4()),
        "message": fixture["prompt"], "model": "auto", "tier": "S+++", "stream": True,
        "attachments": [{"name": file_path.name, "mime_type": fixture["mime"], "data": base64.b64encode(file_bytes).decode("ascii")}],
    }
    headers = {
        "Origin": CANONICAL_FRONTEND_URL, "Referer": CANONICAL_FRONTEND_URL + "/",
        "Accept": "text/event-stream", "Authorization": f"Bearer {AUTH_TOKEN}",
    }
    start = time.perf_counter()
    first_token = None
    text = ""
    provider = model = None
    errors = []
    clean_finish = False
    status = 0
    try:
        with httpx.Client(timeout=60.0) as client:
            with client.stream("POST", f"{BASE_URL}/api/v1/chat", headers=headers, json=payload) as response:
                status = response.status_code
                if status != 200:
                    errors.append(f"HTTP {status}")
                else:
                    for event in iter_sse_events(response.iter_text()):
                        kind = event.get("type")
                        if kind == "token":
                            if clean_finish:
                                errors.append("Token after terminal event")
                            content = event.get("content") or event.get("text") or ""
                            if content and first_token is None:
                                first_token = time.perf_counter() - start
                            text += content
                        provider = event.get("provider") or provider
                        model = event.get("model") or model
                        if kind == "error":
                            errors.append(event.get("content") or event.get("error") or "SSE error")
                        if kind in ("finish", "done", "complete"):
                            if event.get("finish_reason") == "stop":
                                clean_finish = True
                            else:
                                errors.append(f"Unsuccessful finish: {event.get('finish_reason')}")
    except (httpx.HTTPError, ValueError, TypeError) as error:
        errors.append(str(error))
    omissions = evaluate_accuracy(fixture, text)
    result = {
        "fixture_id": fixture["id"], "round": round_num,
        "file_size": len(file_bytes), "fixture_sha256": hashlib.sha256(file_bytes).hexdigest(),
        "mime": fixture["mime"], "status_code": status,
        "ttft": round(first_token, 2) if first_token is not None else None,
        "total_time": round(time.perf_counter() - start, 2),
        "clean_finish": clean_finish and not errors, "provider": provider, "model": model,
        "response": text, "error": errors or None, "omissions": omissions,
        "passed": status == 200 and clean_finish and not errors and not omissions and provider == "gemini" and bool(model),
    }
    print(f"{fixture['id']} round {round_num}: {'PASS' if result['passed'] else 'FAIL'}; {result['total_time']}s; {provider}/{model}; omissions={omissions}; errors={errors}")
    return result


def main(argv=None):
    global BASE_URL, AUTH_TOKEN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-url", default=BASE_URL)
    parser.add_argument("--expected-commit", default=os.getenv("EXPECTED_COMMIT_SHA", ""))
    parser.add_argument("--deployment-id", default=os.getenv("DEPLOYMENT_ID", ""))
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "scratch/attachment-qa/results.json")
    parser.add_argument("--interval", type=float, default=3.0)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_commit) or not args.deployment_id or args.interval < 0:
        parser.error("a full --expected-commit and --deployment-id are required; interval must be nonnegative")
    BASE_URL = args.backend_url.rstrip("/")
    AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")
    token_file = os.getenv("TOKEN_FILE")
    if not AUTH_TOKEN and token_file:
        AUTH_TOKEN = Path(token_file).read_text(encoding="utf-8").strip()
    if not AUTH_TOKEN:
        parser.error("set AUTH_TOKEN or an explicit TOKEN_FILE; credentials are never saved in the report")
    report = {
        "schema_version": 2, "started_at": datetime.now(timezone.utc).isoformat(),
        "backend_url": BASE_URL, "expected_commit": args.expected_commit,
        "deployment_id": args.deployment_id, "expected_cases": len(FIXTURES) * 2,
        "evaluation": "required-field acceptance on four synthetic fixtures repeated twice",
        "results": [], "health_checks": [], "passed": False,
    }

    def healthy_same_commit():
        healthy, status, latency, data = check_backend_health()
        report["health_checks"].append({"healthy": healthy, "status": status, "latency_ms": round(latency, 2), "data": data})
        return healthy and isinstance(data, dict) and data.get("commit") == args.expected_commit

    try:
        proceed = healthy_same_commit()
        for round_num in (1, 2):
            if not proceed:
                break
            for fixture in FIXTURES:
                result = run_single_test(fixture, round_num)
                report["results"].append(result)
                proceed = healthy_same_commit() and result["passed"]
                if not proceed:
                    break
                time.sleep(args.interval)
        report["passed"] = proceed and len(report["results"]) == report["expected_cases"]
    except (OSError, ValueError, httpx.HTTPError) as error:
        report["run_error"] = str(error)
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Acceptance: {'PASS' if report['passed'] else 'FAIL'} ({len(report['results'])}/{report['expected_cases']} cases). Report: {args.report}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
