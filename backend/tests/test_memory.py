"""Unit and integration tests for User Memory Card and User-Scoped Sessions."""
import uuid
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.auth_service import auth_service
from app.services.memory_service import memory_service
from app.services.session_store import session_store

client = TestClient(app)


@pytest.fixture
def auth_tokens():
    """Create two isolated test users and return their JWT tokens."""
    uid1 = f"student_one_{uuid.uuid4().hex[:6]}"
    u1 = auth_service.authenticate_or_register_google_user(
        email=f"{uid1}@jru.ac.in",
        full_name="Student One",
        google_sub=uid1,
    )
    uid2 = f"student_two_{uuid.uuid4().hex[:6]}"
    u2 = auth_service.authenticate_or_register_google_user(
        email=f"{uid2}@jru.ac.in",
        full_name="Student Two",
        google_sub=uid2,
    )
    return {
        "user1": u1["user"],
        "token1": u1["token"],
        "user2": u2["user"],
        "token2": u2["token"],
    }


def test_user_session_isolation(auth_tokens):
    """Verify that sessions created by User 1 are only visible to User 1, not User 2."""
    token1 = auth_tokens["token1"]
    token2 = auth_tokens["token2"]
    user1_id = auth_tokens["user1"]["id"]

    sid = uuid.uuid4()
    session_store.ensure_session(sid, title="User 1 Confidential Project", user_id=user1_id)
    session_store.record_turn(sid, "user", "Hello this is my personal project.")
    session_store.record_turn(sid, "assistant", "I am here to help.")

    # User 1 lists sessions -> sees the session
    res1 = client.get("/api/v1/sessions", headers={"Authorization": f"Bearer {token1}"})
    assert res1.status_code == 200
    ids1 = [s["id"] for s in res1.json()["sessions"]]
    assert str(sid) in ids1

    # User 2 lists sessions -> does NOT see User 1's session
    res2 = client.get("/api/v1/sessions", headers={"Authorization": f"Bearer {token2}"})
    assert res2.status_code == 200
    ids2 = [s["id"] for s in res2.json()["sessions"]]
    assert str(sid) not in ids2

    # Clean up
    session_store.delete_session(sid, user_id=user1_id)


def test_memory_card_crud_api(auth_tokens):
    """Verify full CRUD lifecycle of user's personal Memory Card."""
    token1 = auth_tokens["token1"]

    # 1. Create a memory
    add_resp = client.post(
        "/api/v1/memory",
        headers={"Authorization": f"Bearer {token1}"},
        json={"fact": "B.Tech Computer Science 3rd Year", "category": "academic"},
    )
    assert add_resp.status_code == 201, add_resp.text
    created = add_resp.json()
    assert created["fact"] == "B.Tech Computer Science 3rd Year"
    assert created["category"] == "academic"
    mem_id = created["id"]

    # 2. Add second memory
    client.post(
        "/api/v1/memory",
        headers={"Authorization": f"Bearer {token1}"},
        json={"fact": "Targeting campus placement at Google", "category": "goal"},
    )

    # 3. List memories
    list_resp = client.get("/api/v1/memory", headers={"Authorization": f"Bearer {token1}"})
    assert list_resp.status_code == 200
    data = list_resp.json()
    assert data["count"] >= 2
    facts = [m["fact"] for m in data["memories"]]
    assert "B.Tech Computer Science 3rd Year" in facts

    # 4. Update memory
    up_resp = client.put(
        f"/api/v1/memory/{mem_id}",
        headers={"Authorization": f"Bearer {token1}"},
        json={"fact": "B.Tech Computer Science 4th Year (Final Sem)", "category": "academic"},
    )
    assert up_resp.status_code == 200

    # 5. Delete memory
    del_resp = client.delete(f"/api/v1/memory/{mem_id}", headers={"Authorization": f"Bearer {token1}"})
    assert del_resp.status_code == 200

    # 6. Clear all memories
    clear_resp = client.delete("/api/v1/memory", headers={"Authorization": f"Bearer {token1}"})
    assert clear_resp.status_code == 200

    list_after = client.get("/api/v1/memory", headers={"Authorization": f"Bearer {token1}"})
    assert list_after.json()["count"] == 0


def test_memory_auto_extraction():
    """Verify intelligent fact extraction from natural chat conversation."""
    test_uid = f"extractor_test_{uuid.uuid4().hex[:6]}"

    msg = "I am in 3rd year B.Tech CSE and I am preparing for GATE exam."
    facts = memory_service.extract_and_save_facts(test_uid, msg)
    assert len(facts) >= 2
    categories = [f["category"] for f in facts]
    assert "academic" in categories
    assert "goal" in categories

    prompt_block = memory_service.format_memory_prompt_block(test_uid)
    assert "User Memory Card" in prompt_block
    assert "Gate Exam" in prompt_block or "GATE" in prompt_block

    memory_service.clear_user_memories(test_uid)

    # Test name extraction (including colloquial typos like 'my ame is...')
    name_msg = "my ame is risha kahar"
    name_facts = memory_service.extract_and_save_facts(test_uid, name_msg)
    assert len(name_facts) >= 1
    assert name_facts[0]["fact"] == "Name: Risha Kahar"

    name_block = memory_service.format_memory_prompt_block(test_uid)
    assert "Name: Risha Kahar" in name_block
    assert "Memory Card" in name_block

    memory_service.clear_user_memories(test_uid)

    # Test empty memory card prompt block
    empty_block = memory_service.format_memory_prompt_block(test_uid)
    assert "Memory Card" in empty_block
    assert "sidebar" in empty_block
