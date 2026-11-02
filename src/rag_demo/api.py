"""FastAPI backend for the Sidekick RAG assistant.

Exposes the RAG pipeline (ask/stream_ask from chain.py) as a REST + SSE API
for the React frontend. Streamlit (app.py) is untouched and keeps working.

Run:  python -m rag_demo.api   (uvicorn on :8000)

Endpoints:
  POST /api/chat         non-streaming ask
  POST /api/chat/stream  SSE streaming (token events, then done)
  GET  /api/health       ollama reachability + index presence
  GET  /api/config       safe settings subset (no secrets)
  POST /api/feedback     thumbs up/down vote (persisted via audit.log_feedback)
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .audit import log_feedback
from .chain import Answer, ask, stream_ask
from .config import get_settings

app = FastAPI(title="Sidekick RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev
        "http://localhost:4173",  # Vite preview
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _hit_to_wire(h: Any) -> dict:
    # allowed_groups is internal (permission filtering) — never on the wire.
    return {"source": h.source, "text": h.text, "score": h.score}


def _answer_to_wire(a: Answer) -> dict:
    return {
        "text": a.text,
        "sources": a.sources,
        "hits": [_hit_to_wire(h) for h in a.hits],
        "grounded": a.grounded,
        "latency_ms": a.latency_ms,
        "standalone_question": a.standalone_question,
    }


class ChatRequest(BaseModel):
    question: str
    history: list[tuple[str, str]] | None = None


class FeedbackRequest(BaseModel):
    question: str
    vote: str  # "up" | "down"


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    answer = ask(req.question, history=req.history or [])
    return _answer_to_wire(answer)


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    def events():
        for event in stream_ask(req.question, history=req.history or []):
            if "token" in event:
                yield f"data: {json.dumps({'token': event['token']})}\n\n"
            else:
                payload = {"done": _answer_to_wire(event["answer"])}
                yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/api/health")
def health() -> dict:
    import urllib.request

    settings = get_settings()
    ollama_ok = False
    try:
        urllib.request.urlopen(f"{settings.ollama_base_url}/api/tags", timeout=2)
        ollama_ok = True
    except Exception:
        pass
    index_dir = settings.index_dir
    return {
        "status": "ok" if ollama_ok else "degraded",
        "ollama_reachable": ollama_ok,
        "index_present": index_dir.exists(),
    }


@app.get("/api/config")
def config() -> dict:
    settings = get_settings()
    return {
        "ollama_model": settings.ollama_model,
        "embedding_model": settings.embedding_model,
        "retrieval": settings.retrieval,
        "rerank": settings.rerank_enabled,
        "hybrid_fallback": settings.hybrid_fallback,
    }


@app.post("/api/feedback")
def feedback(req: FeedbackRequest) -> dict:
    if req.vote not in ("up", "down"):
        return {"ok": False, "error": "vote must be 'up' or 'down'"}
    log_feedback(get_settings(), req.question, req.vote)
    return {"ok": True}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
