"""Opt-in JSONL audit logging for questions.

Set RAG_DEMO_AUDIT_LOG to a file path to record one JSON line per ask():
timestamp, question, retrieved sources, user groups, retrieval settings,
model, and latency. Answer text is NOT logged by default (keeps the log safe
to share); RAG_DEMO_AUDIT_LOG_ANSWERS=1 opts in.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def log_question(settings, question: str, answer, latency_ms: float) -> None:
    path = settings.audit_log_path
    if not path:
        return
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "sources": answer.sources,
        "user_groups": settings.user_groups,
        "retrieval": settings.retrieval,
        "rerank": settings.rerank_enabled,
        "model": settings.ollama_model
        if settings.llm_provider == "ollama"
        else settings.llm_provider,
        "latency_ms": round(latency_ms, 1),
    }
    if settings.audit_log_answers:
        entry["answer"] = answer.text
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
