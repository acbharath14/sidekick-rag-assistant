"""Per-chat persistence for the Streamlit UI.

Each chat is a JSON file under chats/: {id, title, created, messages, history}.
Hit objects (dataclasses) are serialized to plain dicts on save.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

CHATS_DIR = Path(__file__).resolve().parent.parent.parent / "chats"


def _ensure_dir() -> None:
    CHATS_DIR.mkdir(exist_ok=True)


def _hit_to_dict(h) -> dict:
    d = asdict(h) if hasattr(h, "__dataclass_fields__") else dict(h)
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


def new_chat(title: str = "New chat") -> dict:
    """Create, persist, and return a fresh chat dict."""
    _ensure_dir()
    chat = {
        "id": uuid.uuid4().hex[:8],
        "title": title,
        "created": datetime.now(timezone.utc).isoformat(),
        "messages": [],
        "history": [],
    }
    save_chat(chat)
    return chat


def save_chat(chat: dict) -> None:
    """Persist a chat, serializing Hit objects to dicts."""
    _ensure_dir()
    data = dict(chat)
    msgs = []
    for m in chat.get("messages", []):
        m2 = dict(m)
        if m2.get("hits"):
            m2["hits"] = [_hit_to_dict(h) for h in m2["hits"]]
        msgs.append(m2)
    data["messages"] = msgs
    (CHATS_DIR / f"{chat['id']}.json").write_text(json.dumps(data, indent=1))


def load_chat(chat_id: str) -> dict | None:
    p = CHATS_DIR / f"{chat_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def list_chats() -> list[dict]:
    """Newest-first list of {id, title, created}."""
    _ensure_dir()
    chats = []
    for p in CHATS_DIR.glob("*.json"):
        try:
            c = json.loads(p.read_text())
            chats.append(
                {
                    "id": c["id"],
                    "title": c.get("title", "Untitled"),
                    "created": c.get("created", ""),
                }
            )
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    chats.sort(key=lambda c: c["created"], reverse=True)
    return chats


def delete_chat(chat_id: str) -> None:
    p = CHATS_DIR / f"{chat_id}.json"
    if p.exists():
        p.unlink()


def rename_chat(chat_id: str, title: str) -> None:
    chat = load_chat(chat_id)
    if chat is not None:
        chat["title"] = title
        save_chat(chat)
