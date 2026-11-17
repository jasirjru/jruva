"""Live End-to-End integration test script against running FastAPI server."""
import json
import httpx

def test_live_e2e():
    client = httpx.Client(
        base_url="http://127.0.0.1:8001",
        timeout=30.0,
        headers={"x-benchmark-bypass": "jru-benchmark-internal-bypass-key"}
    )
    try:
        r = client.get("/api/v1/health")
    except httpx.ConnectError:
        import pytest
        pytest.skip("Live standalone server not running on port 8001")
    
    assert r.status_code == 200, f"Health check failed: {r.status_code}"
    print("[PASS] 1. Health endpoint responded 200 OK:", r.json())
    
    # 2. Turn 1: Stream message with name
    print("[RUN] 2. Sending Turn 1: 'Hello, my name is Jordan. Please remember this!'")
    session_id = None
    accumulated_turn1 = ""
    error_turn1 = None
    with client.stream("POST", "/api/v1/chat", json={
        "message": "Hello, my name is Jordan. Please remember this!",
        "stream": True
    }) as res:
        assert res.status_code == 200
        for line in res.iter_lines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                if event.get("session_id") and not session_id:
                    session_id = event["session_id"]
                if event.get("type") == "token":
                    accumulated_turn1 += event.get("content", "")
                elif event.get("type") == "error":
                    error_turn1 = event.get("content")
    
    if error_turn1:
        print(f"[ERROR] Turn 1 received error event: {error_turn1}")
    print(f"[PASS] Turn 1 Response: {accumulated_turn1.strip()}")
    print(f"[PASS] Captured Session ID: {session_id}")
    assert session_id is not None
    assert len(accumulated_turn1) > 0

    # 3. Turn 2: Verify memory retention
    print("[RUN] 3. Sending Turn 2: 'What is my name? Reply in one word.'")
    accumulated_turn2 = ""
    error_turn2 = None
    with client.stream("POST", "/api/v1/chat", json={
        "session_id": session_id,
        "message": "What is my name? Reply in one word.",
        "stream": True
    }) as res:
        assert res.status_code == 200
        for line in res.iter_lines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                if event.get("type") == "token":
                    accumulated_turn2 += event.get("content", "")
                elif event.get("type") == "error":
                    error_turn2 = event.get("content")
    
    if error_turn2:
        print(f"[ERROR] Turn 2 received error event: {error_turn2}")
    print(f"[PASS] Turn 2 Response: {accumulated_turn2.strip()}")
    assert "Jordan" in accumulated_turn2, f"Memory recall failed. Expected 'Jordan', got: '{accumulated_turn2}' (error: {error_turn2})"
    print("[PASS] Multi-turn memory successfully retained context across SSE streams!")
    
    # 4. Turn 3: Non-streaming sync response
    print("[RUN] 4. Testing synchronous fallback response...")
    sync_res = client.post("/api/v1/chat", json={
        "session_id": session_id,
        "message": "Give me a one sentence greeting.",
        "stream": False
    })
    assert sync_res.status_code == 200
    sync_data = sync_res.json()
    assert sync_data["role"] == "assistant"
    assert len(sync_data["content"]) > 0
    print(f"[PASS] Sync Response: {sync_data['content']}")
    
    # 5. Validation error handling
    print("[RUN] 5. Testing validation error (empty prompt)...")
    val_res = client.post("/api/v1/chat", json={"message": "", "stream": False})
    assert val_res.status_code == 422
    print("[PASS] 422 Unprocessable Entity returned for invalid input.")
    
    print("\n[ALL 5/5 E2E INTEGRATION SUITES PASSED SUCCESSFULLY!]")

if __name__ == "__main__":
    test_live_e2e()
