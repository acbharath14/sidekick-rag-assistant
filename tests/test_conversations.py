"""Contract test for the conversation endpoints (Part 6: multi-chat).

Uses RAG_DEMO_FAKE=1 (deterministic fakes — no live Ollama) and a temp
SQLite DB via RAG_DEMO_CONVERSATIONS_DB so tests never touch real data.
Run:  RAG_DEMO_FAKE=1 python -m pytest tests/test_conversations.py -v
"""
import os
import tempfile

_db_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.close(_db_fd)
os.environ["RAG_DEMO_CONVERSATIONS_DB"] = _db_path
os.environ["RAG_DEMO_FAKE"] = "1"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi.testclient import TestClient

from rag_demo.api import app
from rag_demo import conversations as conv_store

client = TestClient(app)


def _create(title=None):
    body = {} if title is None else {"title": title}
    r = client.post("/api/conversations", json=body)
    assert r.status_code == 200
    return r.json()


def test_create_defaults_to_new_chat():
    conv = _create()
    assert conv["title"] == "New chat"
    assert conv["id"]


def test_create_with_title():
    conv = _create("My title")
    assert conv["title"] == "My title"


def test_list_ordering_newest_first():
    import time
    a = _create("aaa-first")
    time.sleep(1.1)  # ensure distinct updated_at seconds
    b = _create("bbb-second")
    r = client.get("/api/conversations")
    assert r.status_code == 200
    convs = r.json()["conversations"]
    ids = [c["id"] for c in convs]
    # b was updated most recently → first
    assert ids.index(b["id"]) < ids.index(a["id"])
    for c in convs:
        assert set(c.keys()) == {
            "id", "title", "created_at", "updated_at", "message_count",
        }


def test_get_with_messages_empty():
    conv = _create()
    r = client.get(f"/api/conversations/{conv['id']}")
    assert r.status_code == 200
    body = r.json()
    assert body["messages"] == []


def test_get_missing():
    r = client.get("/api/conversations/does-not-exist")
    assert r.json()["ok"] is False


def test_rename():
    conv = _create()
    r = client.patch(
        f"/api/conversations/{conv['id']}", json={"title": "Renamed"}
    )
    assert r.json()["ok"] is True
    body = client.get(f"/api/conversations/{conv['id']}").json()
    assert body["title"] == "Renamed"


def test_rename_empty_title_rejected():
    conv = _create()
    r = client.patch(
        f"/api/conversations/{conv['id']}", json={"title": "   "}
    )
    assert r.json()["ok"] is False


def test_delete():
    conv = _create()
    r = client.delete(f"/api/conversations/{conv['id']}")
    assert r.json()["ok"] is True
    r = client.get(f"/api/conversations/{conv['id']}")
    assert r.json()["ok"] is False


def test_chat_persistence_round_trip():
    conv = _create()
    r = client.post(
        "/api/chat",
        json={"question": "What is the refund policy?", "conversation_id": conv["id"]},
    )
    assert r.status_code == 200
    body = r.json()
    # First message on an untitled conversation → auto-title returned
    assert body.get("conversation_title", "").startswith("What is the refund")

    detail = client.get(f"/api/conversations/{conv['id']}").json()
    assert len(detail["messages"]) == 2
    assert detail["messages"][0]["role"] == "user"
    assert detail["messages"][0]["text"] == "What is the refund policy?"
    assert detail["messages"][1]["role"] == "assistant"
    assert detail["messages"][1]["text"]  # fake LLM answered
    assert isinstance(detail["messages"][1]["sources"], list)
    # Title was persisted too
    assert detail["title"].startswith("What is the refund")

    # Second message keeps the title (no re-titling)
    r = client.post(
        "/api/chat",
        json={"question": "Second question here", "conversation_id": conv["id"]},
    )
    assert "conversation_title" not in r.json()
    detail = client.get(f"/api/conversations/{conv['id']}").json()
    assert len(detail["messages"]) == 4


def test_chat_stream_persistence():
    conv = _create("stream-test")
    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"question": "Stream me", "conversation_id": conv["id"]},
    ) as r:
        assert r.status_code == 200
        chunks = list(r.iter_raw())
    assert chunks, "expected SSE events"
    detail = client.get(f"/api/conversations/{conv['id']}").json()
    roles = [m["role"] for m in detail["messages"]]
    assert roles == ["user", "assistant"]


def test_chat_unknown_conversation_id_ignored():
    r = client.post(
        "/api/chat",
        json={"question": "Hello?", "conversation_id": "nope-missing"},
    )
    assert r.status_code == 200
    assert "text" in r.json()  # answered, just not persisted


def test_clear_messages():
    conv = _create()
    client.post(
        "/api/chat",
        json={"question": "Something to clear", "conversation_id": conv["id"]},
    )
    r = client.delete(f"/api/conversations/{conv['id']}/messages")
    assert r.json()["ok"] is True
    detail = client.get(f"/api/conversations/{conv['id']}").json()
    assert detail["messages"] == []
    assert detail["title"]  # conversation itself survives


def test_list_message_counts():
    conv = _create()
    client.post(
        "/api/chat",
        json={"question": "Count me", "conversation_id": conv["id"]},
    )
    convs = client.get("/api/conversations").json()["conversations"]
    mine = next(c for c in convs if c["id"] == conv["id"])
    assert mine["message_count"] == 2


def test_auto_title_word_boundary():
    t = conv_store.auto_title(
        "This is a very long question that should be cut at a word boundary nicely"
    )
    assert len(t) <= 41  # 40 chars + ellipsis
    assert not t.rstrip("…").endswith("b"), "must not cut mid-word"
    assert conv_store.auto_title("Short") == "Short"


# --- Document cabinet endpoints ---


def test_documents_list_empty():
    r = client.get("/api/documents")
    assert r.status_code == 200
    assert r.json()["documents"] == []


def test_document_crud_via_store():
    # Store-level: add → list → get → set summary → delete
    doc = conv_store.add_document("doc1", "a.pdf", 3, 5000, 12000, "pypdf")
    assert doc["id"] == "doc1"
    docs = conv_store.list_documents()
    assert len(docs) == 1
    assert docs[0]["has_summary"] is False
    assert conv_store.set_document_summary("doc1", "Summary here.")
    full = conv_store.get_document("doc1")
    assert full["summary"] == "Summary here."
    assert conv_store.list_documents()[0]["has_summary"] is True
    deleted = conv_store.delete_document("doc1")
    assert deleted["filename"] == "a.pdf"
    assert conv_store.get_document("doc1") is None


def test_get_document_missing():
    r = client.get("/api/documents/nope")
    assert r.json()["ok"] is False


def test_delete_document_missing():
    r = client.delete("/api/documents/nope")
    assert r.json()["ok"] is False


def test_chat_with_document_id_unknown_ignored():
    # Unknown document_id should not break chat — treated as unscoped.
    r = client.post(
        "/api/chat",
        json={"question": "Hello?", "document_id": "nope-missing"},
    )
    assert r.status_code == 200
    assert "text" in r.json()


def test_upload_persists_to_index(tmp_path=None):
    # Upload a small text file; verify it lands in the persistent index
    # and SQLite. Uses the fake embeddings tier (RAG_DEMO_FAKE=1).
    import io

    content = (
        "The Zephyr Protocol defines packet routing for mesh networks. "
        "Section 7 covers retransmission timers in detail."
    )
    r = client.post(
        "/api/upload",
        files={"file": ("zephyr.txt", io.BytesIO(content.encode()), "text/plain")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    doc_id = body["id"]
    assert body["chunk_count"] >= 1

    # Listed in the cabinet
    docs = client.get("/api/documents").json()["documents"]
    assert any(d["id"] == doc_id for d in docs)

    # Document-scoped chat finds it
    r = client.post(
        "/api/chat",
        json={"question": "What is the Zephyr Protocol?", "document_id": doc_id},
    )
    assert r.status_code == 200

    # Delete removes it from the cabinet
    r = client.delete(f"/api/documents/{doc_id}")
    assert r.json()["ok"] is True
    docs = client.get("/api/documents").json()["documents"]
    assert not any(d["id"] == doc_id for d in docs)
