"""SQLite persistence for multi-chat conversations.

One DB file at {index_dir}/conversations.db (next to the FAISS index;
override with RAG_DEMO_CONVERSATIONS_DB). stdlib sqlite3 only.
Uploaded-document FAISS indexes are NOT persisted — they stay in-memory
(session-scoped) by design.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    sources TEXT,
    grounded INTEGER,
    latency_ms INTEGER,
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    char_count INTEGER NOT NULL,
    file_size INTEGER NOT NULL DEFAULT 0,
    extractor TEXT NOT NULL,
    summary TEXT,
    uploaded_at INTEGER NOT NULL
);
"""


def _db_path() -> Path:
    override = os.environ.get("RAG_DEMO_CONVERSATIONS_DB")
    if override:
        return Path(override)
    from .config import get_settings

    return get_settings().index_dir / "conversations.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    # Migrate older documents tables missing the newer columns.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(documents)").fetchall()}
    if "file_size" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN file_size INTEGER NOT NULL DEFAULT 0")
    if "summary" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN summary TEXT")
    conn.row_factory = sqlite3.Row
    return conn


def _now() -> int:
    return int(time.time())


def auto_title(text: str, max_chars: int = 40) -> str:
    """First 40 chars, truncated at a word boundary."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return (cut or text[:max_chars]).rstrip() + "…"


def create_conversation(title: str | None = None) -> dict:
    conv_id = uuid.uuid4().hex[:12]
    now = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at)"
            " VALUES (?, ?, ?, ?)",
            (conv_id, title or "New chat", now, now),
        )
    return {"id": conv_id, "title": title or "New chat", "created_at": now}


def list_conversations() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   COUNT(m.id) AS message_count,
                   (SELECT text FROM messages m2
                     WHERE m2.conversation_id = c.id
                     ORDER BY m2.id DESC LIMIT 1) AS last_message,
                   (SELECT created_at FROM messages m3
                     WHERE m3.conversation_id = c.id
                     ORDER BY m3.id DESC LIMIT 1) AS last_message_at
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            """
        ).fetchall()
    return [
        {
            "id": r["id"],
            "title": r["title"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "message_count": r["message_count"],
            "last_message": r["last_message"],
            "last_message_at": r["last_message_at"],
        }
        for r in rows
    ]


def get_conversation(conv_id: str) -> dict | None:
    with _connect() as conn:
        conv = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations"
            " WHERE id = ?",
            (conv_id,),
        ).fetchone()
        if not conv:
            return None
        msgs = conn.execute(
            "SELECT role, text, sources, grounded, latency_ms, created_at"
            " FROM messages WHERE conversation_id = ? ORDER BY id",
            (conv_id,),
        ).fetchall()
    return {
        "id": conv["id"],
        "title": conv["title"],
        "created_at": conv["created_at"],
        "updated_at": conv["updated_at"],
        "messages": [
            {
                "role": m["role"],
                "text": m["text"],
                "sources": json.loads(m["sources"]) if m["sources"] else [],
                "grounded": bool(m["grounded"]) if m["grounded"] is not None else None,
                "latency_ms": m["latency_ms"],
                "created_at": m["created_at"],
            }
            for m in msgs
        ],
    }


def rename_conversation(conv_id: str, title: str) -> bool:
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, _now(), conv_id),
        )
        return cur.rowcount > 0


def delete_conversation(conv_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM conversations WHERE id = ?", (conv_id,)
        )
        return cur.rowcount > 0


def clear_messages(conv_id: str) -> bool:
    """Delete all messages in a conversation (the conversation itself survives)."""
    with _connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
        if not exists:
            return False
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (_now(), conv_id),
        )
        return True


def add_message(
    conv_id: str,
    role: str,
    text: str,
    sources: list[str] | None = None,
    grounded: bool | None = None,
    latency_ms: int | None = None,
) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO messages"
            " (conversation_id, role, text, sources, grounded, latency_ms, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                conv_id,
                role,
                text,
                json.dumps(sources) if sources is not None else None,
                int(grounded) if grounded is not None else None,
                latency_ms,
                _now(),
            ),
        )
        return cur.lastrowid


def touch_conversation(conv_id: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (_now(), conv_id),
        )


def maybe_autotitle(conv_id: str, first_message: str) -> str | None:
    """If the conversation still has the default title, generate one from
    the first user message. Returns the new title, or None if unchanged."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT title FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
        if not row or row["title"] != "New chat":
            return None
        title = auto_title(first_message)
        conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (title, conv_id),
        )
        return title


def conversation_exists(conv_id: str) -> bool:
    with _connect() as conn:
        return (
            conn.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conv_id,)
            ).fetchone()
            is not None
        )


# --- Documents (persistent uploads) ---


def add_document(
    doc_id: str,
    filename: str,
    chunk_count: int,
    char_count: int,
    file_size: int,
    extractor: str,
) -> dict:
    now = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO documents"
            " (id, filename, chunk_count, char_count, file_size, extractor,"
            " summary, uploaded_at)"
            " VALUES (?, ?, ?, ?, ?, ?, NULL, ?)",
            (doc_id, filename, chunk_count, char_count, file_size, extractor, now),
        )
    return {
        "id": doc_id,
        "filename": filename,
        "chunk_count": chunk_count,
        "char_count": char_count,
        "file_size": file_size,
        "extractor": extractor,
        "summary": None,
        "uploaded_at": now,
    }


def list_documents() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, filename, chunk_count, char_count, file_size,"
            " extractor, summary IS NOT NULL AS has_summary, uploaded_at"
            " FROM documents ORDER BY uploaded_at DESC"
        ).fetchall()
    return [
        {
            "id": r["id"],
            "filename": r["filename"],
            "chunk_count": r["chunk_count"],
            "char_count": r["char_count"],
            "file_size": r["file_size"],
            "extractor": r["extractor"],
            "has_summary": bool(r["has_summary"]),
            "uploaded_at": r["uploaded_at"],
        }
        for r in rows
    ]


def get_document(doc_id: str) -> dict | None:
    with _connect() as conn:
        r = conn.execute(
            "SELECT id, filename, chunk_count, char_count, file_size,"
            " extractor, summary, uploaded_at"
            " FROM documents WHERE id = ?",
            (doc_id,),
        ).fetchone()
    if not r:
        return None
    return {
        "id": r["id"],
        "filename": r["filename"],
        "chunk_count": r["chunk_count"],
        "char_count": r["char_count"],
        "file_size": r["file_size"],
        "extractor": r["extractor"],
        "summary": r["summary"],
        "uploaded_at": r["uploaded_at"],
    }


def set_document_summary(doc_id: str, summary: str) -> bool:
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE documents SET summary = ? WHERE id = ?", (summary, doc_id)
        )
        return cur.rowcount > 0


def delete_document(doc_id: str) -> dict | None:
    """Delete the SQLite row; returns the row (for source filtering) or None."""
    doc = get_document(doc_id)
    if not doc:
        return None
    with _connect() as conn:
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    return doc
