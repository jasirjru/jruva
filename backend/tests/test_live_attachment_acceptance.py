"""Acceptance must reject omitted fields, swapped values and failed streams."""

import importlib.util
import json
from pathlib import Path

import httpx
import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "synthetic-test-token")
    path = Path(__file__).resolve().parents[2] / "scripts/run_live_attachment_qa.py"
    spec = importlib.util.spec_from_file_location("attachment_qa", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("fixture_index,answer", [
    (2, "Authorized Budget: $4,850,000. Telemetry: 420 megahertz. Immutable Ledger Verification Protocol Alpha"),
    (3, "Permit Number: SC-90412. Facility Code: FAC-99. Inspector K. Vance"),
    (1, "DEPARTMENT PERFORMANCE METRICS 2026\nAI Lab: 91.5%\nData Science: 94.2%\nRobotics: 88.7%\nCybersecurity: 85.0%"),
])
def test_omitted_or_swapped_fields_fail(runner, fixture_index, answer, monkeypatch):
    client_class = httpx.Client
    events = [
        {"type": "token", "content": answer, "provider": "gemini", "model": "gemini-3.6-flash"},
        {"type": "finish", "finish_reason": "stop"},
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
    monkeypatch.setattr(runner.httpx, "Client", lambda **kwargs: client_class(transport=transport, **kwargs))
    result = runner.run_single_test(runner.FIXTURES[fixture_index])
    assert result["passed"] is False


@pytest.mark.parametrize("healthy,commit", [(False, "a" * 40), (True, "b" * 40), (True, None)])
def test_wrong_deployment_or_failed_health_fails_before_chat(runner, monkeypatch, tmp_path, healthy, commit):
    calls = []
    monkeypatch.setattr(runner, "check_backend_health", lambda: (healthy, 200, 1.0, {"commit": commit}))
    monkeypatch.setattr(runner, "run_single_test", lambda *args: calls.append(args))
    report = tmp_path / "report.json"
    code = runner.main(["--expected-commit", "a" * 40, "--deployment-id", "synthetic", "--report", str(report)])
    assert code == 1
    assert not calls
    assert json.loads(report.read_text())["passed"] is False


def test_complete_matrix_and_post_request_health_required(runner, monkeypatch, tmp_path):
    checks = iter([(True, 200, 1.0, {"commit": "a" * 40}), (False, 503, 1.0, {})])
    monkeypatch.setattr(runner, "check_backend_health", lambda: next(checks))
    monkeypatch.setattr(runner, "run_single_test", lambda *args: {"passed": True})
    report = tmp_path / "report.json"
    assert runner.main(["--expected-commit", "a" * 40, "--deployment-id", "synthetic", "--report", str(report), "--interval", "0"]) == 1
    saved = json.loads(report.read_text())
    assert len(saved["results"]) == 1
    assert saved["passed"] is False


def test_complete_matrix_records_provenance_and_passes(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "check_backend_health", lambda: (True, 200, 1.0, {"commit": "a" * 40}))
    monkeypatch.setattr(runner, "run_single_test", lambda *args: {"passed": True})
    report = tmp_path / "report.json"
    assert runner.main(["--expected-commit", "a" * 40, "--deployment-id", "synthetic", "--report", str(report), "--interval", "0"]) == 0
    saved = json.loads(report.read_text())
    assert len(saved["results"]) == 8
    assert len(saved["health_checks"]) == 9
    assert saved["expected_commit"] == "a" * 40
    assert saved["deployment_id"] == "synthetic"


def test_sse_handles_crlf_chunk_boundaries(runner):
    events = list(runner.iter_sse_events([': ping\r', '\n\r\ndata: {"type":"token","content":"hi"}\r', '\n\r', '\n']))
    assert events == [{"type": "token", "content": "hi"}]


def test_sse_rejects_unterminated_frame(runner):
    with pytest.raises(ValueError, match="Unterminated"):
        list(runner.iter_sse_events(['data: {"type":"finish","finish_reason":"stop"}']))


@pytest.mark.parametrize("index", range(4))
def test_complete_saved_fixture_answers_are_accepted(runner, index):
    path = Path(__file__).resolve().parents[2] / "fixtures/live_qa_results.json"
    saved = json.loads(path.read_text())
    assert runner.evaluate_accuracy(runner.FIXTURES[index], saved[index]["response"]) == []


@pytest.mark.parametrize("failure", ["error_event", "error_finish"])
def test_stream_failure_cannot_pass_with_complete_answer(runner, monkeypatch, failure):
    client_class = httpx.Client
    answer = "Permit: SC-90412; Facility: FAC-99; Date: 11 October 2026; Safety Rating: Category A-1; Inspector K. Vance"
    events = [{"type": "token", "content": answer, "provider": "gemini", "model": "gemini-3.6-flash"}]
    if failure == "error_event":
        events.append({"type": "error", "content": "upstream interrupted"})
    events.append({"type": "finish", "finish_reason": "error" if failure == "error_finish" else "stop"})
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
    monkeypatch.setattr(runner.httpx, "Client", lambda **kwargs: client_class(transport=transport, **kwargs))
    result = runner.run_single_test(runner.FIXTURES[3])
    assert result["passed"] is False
