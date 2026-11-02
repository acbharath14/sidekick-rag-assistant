"""FastAPI backend for the Sidekick RAG assistant.

Exposes the RAG pipeline (ask/stream_ask from chain.py) as a REST + SSE API
for the React frontend. Streamlit (app.py) is untouched and keeps working.

Run:  python -m rag_demo.api   (uvicorn on :8000)

Endpoints:
  POST /api/chat         non-streaming ask (optional upload_id for file chat)
  POST /api/chat/stream  SSE streaming (token events, then done)
  GET  /api/health       ollama reachability + index presence
  GET  /api/config       safe settings subset (no secrets)
  POST /api/feedback     thumbs up/down vote (persisted via audit.log_feedback)
  POST /api/upload       upload a document → extract, summarize, index in memory
  DELETE /api/upload/{upload_id}  drop an uploaded document index
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .audit import log_feedback
from .chain import Answer, ask, stream_ask
from .config import get_embeddings, get_settings

app = FastAPI(title="Sidekick RAG API")

# In-memory upload indexes, keyed by upload_id. Nothing is written to disk.
_upload_retrievers: dict[str, Any] = {}
_upload_names: dict[str, str] = {}

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
    upload_id: str | None = None  # when set, ask against the uploaded doc


class FeedbackRequest(BaseModel):
    question: str
    vote: str  # "up" | "down"


def _upload_retriever(upload_id: str | None):
    if not upload_id:
        return None
    return _upload_retrievers.get(upload_id)


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    answer = ask(
        req.question,
        history=req.history or [],
        retriever=_upload_retriever(req.upload_id),
    )
    return _answer_to_wire(answer)


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    retriever = _upload_retriever(req.upload_id)

    def events():
        for event in stream_ask(req.question, history=req.history or [], retriever=retriever):
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


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    """Upload a document: extract text, summarize, build an in-memory index.
    Returns an upload_id for subsequent /api/chat calls. Nothing is written
    to disk."""
    from .extract import UnsupportedFormat, extract_text

    settings = get_settings()
    max_mb = float(os.environ.get("RAG_DEMO_UPLOAD_MAX_MB", "10"))
    data = await file.read()
    if len(data) > max_mb * 1024 * 1024:
        return {"ok": False, "error": f"File too large (over {max_mb:.0f} MB)."}
    try:
        text = extract_text(file.filename or "upload", data)
    except UnsupportedFormat as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": f"Couldn't read {file.filename}: {e}"}
    if not text.strip():
        return {"ok": False, "error": f"No text found in {file.filename}."}
    # Summarize.
    from .summarize import summarize

    try:
        summary = summarize(text, settings)
    except Exception as e:
        summary = f"Summarize failed: {type(e).__name__}: {e}"
    # Build an in-memory index for follow-up questions.
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from .retriever import Retriever

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks = splitter.split_text(text)
    source_name = f"upload:{file.filename}"
    store = FAISS.from_texts(
        chunks,
        get_embeddings(settings),
        metadatas=[{"source": source_name, "allowed_groups": ["*"]} for _ in chunks],
    )
    chunk_dicts = [
        {"text": c, "source": source_name, "allowed_groups": ["*"]} for c in chunks
    ]
    upload_id = uuid.uuid4().hex[:12]
    _upload_retrievers[upload_id] = Retriever(settings, store=store, chunks=chunk_dicts)
    _upload_names[upload_id] = file.filename or "upload"
    return {
        "ok": True,
        "upload_id": upload_id,
        "filename": file.filename,
        "char_count": len(text),
        "chunk_count": len(chunks),
        "summary": summary,
    }


@app.delete("/api/upload/{upload_id}")
def delete_upload(upload_id: str) -> dict:
    _upload_retrievers.pop(upload_id, None)
    _upload_names.pop(upload_id, None)
    return {"ok": True}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
