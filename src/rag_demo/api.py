"""FastAPI backend for the Sidekick RAG assistant.

Exposes the RAG pipeline (ask/stream_ask from chain.py) as a REST + SSE API
for the React frontend. Streamlit (app.py) is untouched and keeps working.

Run:  python -m rag_demo.api   (uvicorn on :8000)

Endpoints:
  POST /api/chat         non-streaming ask (optional document_id to scope
                         retrieval, optional conversation_id for persistence)
  POST /api/chat/stream  SSE streaming (token events, then done)
  GET  /api/health       ollama reachability + index presence
  GET  /api/config       safe settings subset (no secrets)
  POST /api/feedback     thumbs up/down vote (persisted via audit.log_feedback)
  POST /api/upload       upload one or more documents into the PERSISTENT
                         FAISS index (repeated "file" field)
  GET  /api/documents            list uploaded documents
  GET  /api/documents/{id}       metadata + cached/on-demand summary
  POST /api/documents/{id}/summarize   (re)generate + cache the summary
  DELETE /api/documents/{id}     remove from index + SQLite
  GET    /api/conversations            list conversations (newest first)
  POST   /api/conversations            create ({title?})
  GET    /api/conversations/{id}       one conversation with messages
  PATCH  /api/conversations/{id}       rename ({title})
  DELETE /api/conversations/{id}       delete conversation + messages
  DELETE /api/conversations/{id}/messages  clear messages, keep conversation

Auth: when the RAG_DEMO_API_KEY env var is set (non-empty), every /api/*
endpoint except /api/health requires the matching X-API-Key header
(401 {"detail": "Invalid or missing API key"} otherwise). When the env var
is unset or empty, all endpoints are open (local dev).
"""
from __future__ import annotations

import hmac
import json
import os
import uuid
from dataclasses import asdict
from typing import Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi import status as http_status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .audit import log_feedback
from .caching import clear_cache
from .chain import Answer, ask, stream_ask
from .config import get_embeddings, get_settings
from . import conversations as conv_store

app = FastAPI(title="Sidekick RAG API")

# In-memory text cache for on-demand summarization, keyed by doc_id.
# The vectors themselves live in the persistent FAISS index on disk.
_upload_texts: dict[str, str] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev
        "http://localhost:4173",  # Vite preview
    ],
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["*"],
)


def require_api_key(
    x_api_key: str | None = Header(default=None),
) -> None:
    """Dependency: enforce X-API-Key auth when RAG_DEMO_API_KEY is set.

    - Env var unset or empty -> open access (local dev, backwards compatible).
    - Env var set -> every /api/* endpoint except /api/health requires the
      matching header, else 401 {"detail": "Invalid or missing API key"}.
    The env var is read per-request (not at import time) so tests and
    process managers can change it without a restart.
    """
    expected = os.environ.get("RAG_DEMO_API_KEY", "").strip()
    if not expected:
        return  # auth disabled
    if x_api_key is None or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


# Shorthand applied to every protected route below. /api/health is
# deliberately exempt (Docker healthchecks, frontend health indicator).
_auth = [Depends(require_api_key)]


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
        "weak_evidence": a.weak_evidence,
    }


class ChatRequest(BaseModel):
    question: str
    history: list[tuple[str, str]] | None = None
    document_id: str | None = None  # when set, scope retrieval to this doc
    conversation_id: str | None = None  # when set, persist the exchange


class FeedbackRequest(BaseModel):
    question: str
    vote: str  # "up" | "down"


class CreateConversationRequest(BaseModel):
    title: str | None = None


class RenameConversationRequest(BaseModel):
    title: str


def _document_exists(doc_id: str | None) -> str | None:
    """Validate a document_id for scoped chat; returns it or None."""
    if not doc_id:
        return None
    return doc_id if conv_store.get_document(doc_id) else None


@app.post("/api/chat", dependencies=_auth)
def chat(req: ChatRequest) -> dict:
    conv_id = req.conversation_id
    if conv_id and conv_store.conversation_exists(conv_id):
        conv_store.add_message(conv_id, "user", req.question)
        new_title = conv_store.maybe_autotitle(conv_id, req.question)
    else:
        conv_id = None
        new_title = None
    answer = ask(
        req.question,
        history=req.history or [],
        document_id=_document_exists(req.document_id),
    )
    if conv_id:
        conv_store.add_message(
            conv_id,
            "assistant",
            answer.text,
            sources=answer.sources,
            grounded=answer.grounded,
            latency_ms=answer.latency_ms,
        )
        conv_store.touch_conversation(conv_id)
    body = _answer_to_wire(answer)
    if new_title:
        body["conversation_title"] = new_title
    return body


@app.post("/api/chat/stream", dependencies=_auth)
def chat_stream(req: ChatRequest) -> StreamingResponse:
    doc_id = _document_exists(req.document_id)
    conv_id = (
        req.conversation_id
        if req.conversation_id and conv_store.conversation_exists(req.conversation_id)
        else None
    )
    if conv_id:
        conv_store.add_message(conv_id, "user", req.question)
        new_title = conv_store.maybe_autotitle(conv_id, req.question)
    else:
        new_title = None

    def events():
        final_answer: Answer | None = None
        for event in stream_ask(
            req.question, history=req.history or [], document_id=doc_id
        ):
            if "token" in event:
                yield f"data: {json.dumps({'token': event['token']})}\n\n"
            else:
                final_answer = event["answer"]
                payload = {"done": _answer_to_wire(final_answer)}
                if new_title:
                    payload["done"]["conversation_title"] = new_title
                yield f"data: {json.dumps(payload)}\n\n"
        if conv_id and final_answer is not None:
            conv_store.add_message(
                conv_id,
                "assistant",
                final_answer.text,
                sources=final_answer.sources,
                grounded=final_answer.grounded,
                latency_ms=final_answer.latency_ms,
            )
            conv_store.touch_conversation(conv_id)

    return StreamingResponse(events(), media_type="text/event-stream")


# --- Conversations (multi-chat persistence) ---


@app.get("/api/conversations", dependencies=_auth)
def list_conversations() -> dict:
    return {"conversations": conv_store.list_conversations()}


@app.post("/api/conversations", dependencies=_auth)
def create_conversation(req: CreateConversationRequest) -> dict:
    return conv_store.create_conversation(req.title)


@app.get("/api/conversations/{conv_id}", dependencies=_auth)
def get_conversation(conv_id: str) -> dict:
    conv = conv_store.get_conversation(conv_id)
    if not conv:
        return {"ok": False, "error": "Conversation not found."}
    return conv


@app.patch("/api/conversations/{conv_id}", dependencies=_auth)
def rename_conversation(conv_id: str, req: RenameConversationRequest) -> dict:
    title = (req.title or "").strip()
    if not title:
        return {"ok": False, "error": "Title must not be empty."}
    if not conv_store.rename_conversation(conv_id, title):
        return {"ok": False, "error": "Conversation not found."}
    return {"ok": True}


@app.delete("/api/conversations/{conv_id}", dependencies=_auth)
def delete_conversation(conv_id: str) -> dict:
    if not conv_store.delete_conversation(conv_id):
        return {"ok": False, "error": "Conversation not found."}
    return {"ok": True}


@app.delete("/api/conversations/{conv_id}/messages", dependencies=_auth)
def clear_conversation_messages(conv_id: str) -> dict:
    if not conv_store.clear_messages(conv_id):
        return {"ok": False, "error": "Conversation not found."}
    return {"ok": True}


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


@app.get("/api/config", dependencies=_auth)
def config() -> dict:
    settings = get_settings()
    return {
        "ollama_model": settings.ollama_model,
        "embedding_model": settings.embedding_model,
        "retrieval": settings.retrieval,
        "rerank": settings.rerank_enabled,
        "hybrid_fallback": settings.hybrid_fallback,
    }


@app.post("/api/feedback", dependencies=_auth)
def feedback(req: FeedbackRequest) -> dict:
    if req.vote not in ("up", "down"):
        return {"ok": False, "error": "vote must be 'up' or 'down'"}
    log_feedback(get_settings(), req.question, req.vote)
    return {"ok": True}


# Max files accepted in one POST /api/upload request. The per-file size
# limit (RAG_DEMO_UPLOAD_MAX_MB, default 10 MB) and per-file validation
# are unchanged; this cap only bounds the batch so one request cannot
# queue unbounded indexing work. Overridable per process with
# RAG_DEMO_UPLOAD_MAX_FILES.
MAX_UPLOAD_FILES = 10


def _ingest_upload(filename: str, data: bytes, settings: Any, max_mb: float) -> dict:
    """Ingest one uploaded file: extract, chunk, index, record.

    Returns the legacy single-file result shape: {"ok": True, **doc} on
    success, {"ok": False, "error": ...} on failure. The batch endpoint
    wraps one of these per file; the single-file path returns one
    untouched, so its response is identical to before. The extract,
    OCR, and verify imports stay function-local (as they were in the
    endpoint) so tests can patch rag_demo.extract / rag_demo.ocr.
    """
    import json as _json

    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from .extract import UnsupportedFormat, extract_text_with_meta, verify_magic_bytes

    if len(data) > max_mb * 1024 * 1024:
        return {"ok": False, "error": f"File too large (over {max_mb:.0f} MB)."}
    try:
        if not verify_magic_bytes(filename, data):
            return {"ok": False, "error": "File content doesn't match its extension."}
        text, extractor = extract_text_with_meta(filename, data)
    except UnsupportedFormat as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": f"Couldn't read {filename}: {e}"}
    if not text.strip() and filename.lower().endswith(".pdf"):
        # Scanned PDF with no embedded text: fall back to OCR, mirroring
        # the old Streamlit app. ocr.py is imported lazily so nothing breaks
        # when the tesseract/poppler binaries are absent.
        from .ocr import INSTALL_HINT, ocr_available, ocr_pdf

        if ocr_available():
            try:
                text = ocr_pdf(data)
            except RuntimeError as e:
                return {
                    "ok": False,
                    "error": f"No text found in {filename} (scanned PDF). {e}",
                }
            if text.strip():
                extractor = "ocr"
        else:
            return {
                "ok": False,
                "error": (
                    f"No text found in {filename} (scanned PDF). {INSTALL_HINT}"
                ),
            }
    if not text.strip():
        return {"ok": False, "error": f"No text found in {filename}."}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks = splitter.split_text(text)
    doc_id = uuid.uuid4().hex[:12]
    source_name = f"upload:{doc_id}"
    metadatas = [
        {"source": source_name, "allowed_groups": ["*"]} for _ in chunks
    ]

    # Append vectors to the persistent FAISS index.
    index_dir = settings.index_dir
    embeddings = get_embeddings(settings)
    if index_dir.exists():
        store = FAISS.load_local(
            str(index_dir), embeddings, allow_dangerous_deserialization=True
        )
        store.add_texts(chunks, metadatas=metadatas)
    else:
        index_dir.mkdir(parents=True, exist_ok=True)
        store = FAISS.from_texts(chunks, embeddings, metadatas=metadatas)
    store.save_local(str(index_dir))

    # Append chunk metadata for the BM25 side of hybrid retrieval.
    chunks_path = index_dir / "chunks.json"
    existing = (
        _json.loads(chunks_path.read_text(encoding="utf-8"))
        if chunks_path.exists()
        else []
    )
    existing.extend(
        {"text": c, "source": source_name, "allowed_groups": ["*"]}
        for c in chunks
    )
    chunks_path.write_text(_json.dumps(existing, ensure_ascii=False), encoding="utf-8")

    # Record in SQLite; keep the text in memory for on-demand summarization.
    doc = conv_store.add_document(
        doc_id,
        filename,
        len(chunks),
        len(text),
        len(data),
        extractor,
    )
    _upload_texts[doc_id] = text
    return {"ok": True, **doc}


@app.post("/api/upload", dependencies=_auth)
async def upload(file: list[UploadFile] = File(...)) -> dict:
    """Upload one or more documents into the PERSISTENT knowledge base.

    Accepts a standard multipart request with a repeated "file" field, so
    one request can carry several documents. Each file goes through the
    same extraction, chunking, and indexing pipeline as a single-file
    upload (including the OCR fallback for scanned PDFs with no embedded
    text). A failure in one file never rolls back the others: every file
    gets its own result entry.

    Response:
      - One file: the legacy shape, unchanged: {"ok": True, **doc} on
        success, {"ok": False, "error": ...} on failure.
      - Several files: {"ok": <True when every file succeeded>,
        "results": [{"filename": ..., "ok": True, "id": ..., ...} or
        {"filename": ..., "ok": False, "error": ...}], "succeeded": n,
        "failed": m}. Top-level "ok" is False when any file failed, so
        callers must inspect "results" for the per-file outcome.
    """
    settings = get_settings()
    max_files = int(
        os.environ.get("RAG_DEMO_UPLOAD_MAX_FILES", str(MAX_UPLOAD_FILES))
    )
    if len(file) > max_files:
        return {
            "ok": False,
            "error": f"Too many files (max {max_files} per request).",
        }
    max_mb = float(os.environ.get("RAG_DEMO_UPLOAD_MAX_MB", "10"))
    ingested = []
    for upload_file in file:
        filename = upload_file.filename or "upload"
        data = await upload_file.read()
        ingested.append((filename, _ingest_upload(filename, data, settings, max_mb)))
    succeeded = sum(1 for _, result in ingested if result["ok"])
    if succeeded:
        # The on-disk index changed, so the cached retriever would keep
        # serving the old vectors. Drop it once for the whole batch, not
        # once per file; the next question reloads. (Caching contract,
        # see caching.py.)
        clear_cache()
    if len(ingested) == 1:
        # Single-file requests keep the exact legacy response shape.
        return ingested[0][1]
    return {
        "ok": succeeded == len(ingested),
        "results": [
            {"filename": name, **result} for name, result in ingested
        ],
        "succeeded": succeeded,
        "failed": len(ingested) - succeeded,
    }


@app.get("/api/documents", dependencies=_auth)
def list_documents() -> dict:
    return {"documents": conv_store.list_documents()}


@app.get("/api/documents/{doc_id}", dependencies=_auth)
def get_document(doc_id: str) -> dict:
    """Full metadata + cached summary (generated on demand if missing)."""
    doc = conv_store.get_document(doc_id)
    if not doc:
        return {"ok": False, "error": "Document not found."}
    if not doc["summary"]:
        text = _upload_texts.get(doc_id)
        if text:
            from .summarize import summarize

            try:
                summary = summarize(text, get_settings())
                if summary:
                    conv_store.set_document_summary(doc_id, summary)
                    doc["summary"] = summary
            except Exception:
                pass
    return {"ok": True, **doc}


@app.post("/api/documents/{doc_id}/summarize", dependencies=_auth)
def summarize_document(doc_id: str) -> dict:
    """(Re)generate the LLM summary for a document and cache it in SQLite."""
    doc = conv_store.get_document(doc_id)
    if not doc:
        return {"ok": False, "error": "Document not found."}
    text = _upload_texts.get(doc_id)
    if not text:
        return {
            "ok": False,
            "error": "Document text no longer in memory (restarted since upload)."
            " Re-upload to summarize.",
        }
    from .summarize import summarize

    try:
        summary = summarize(text, get_settings())
    except Exception as e:
        return {"ok": False, "error": f"Summarize failed: {type(e).__name__}: {e}"}
    summary = summary or "Summary returned empty."
    conv_store.set_document_summary(doc_id, summary)
    return {"ok": True, "summary": summary}


@app.post("/api/documents/{doc_id}/summarize/stream", dependencies=_auth)
def summarize_document_stream(doc_id: str) -> StreamingResponse:
    """SSE stream of summarize progress events, then a done event.

    Event payloads (see rag_demo.summarize.summarize_events):
      {"type": "start", "mode": "single" | "map_reduce", "total": n}
      {"type": "progress", "done": i, "total": n, "stage": "map"}
      {"type": "progress", "stage": "reduce"}
      {"type": "done", "summary": "..."}
      {"type": "error", "error": "..."}
    The final summary is cached in SQLite exactly like the sync endpoint.
    """

    def _error_event(message: str):
        yield f"data: {json.dumps({'type': 'error', 'error': message})}\n\n"

    doc = conv_store.get_document(doc_id)
    if not doc:
        return StreamingResponse(_error_event("Document not found."),
                                 media_type="text/event-stream")
    text = _upload_texts.get(doc_id)
    if not text:
        return StreamingResponse(
            _error_event("Document text no longer in memory (restarted since upload)."
                         " Re-upload to summarize."),
            media_type="text/event-stream",
        )

    def events():
        from .summarize import summarize_events

        final: str | None = None
        for event in summarize_events(text, get_settings()):
            if event["type"] == "done":
                final = event["summary"]
            yield f"data: {json.dumps(event)}\n\n"
        if final:
            conv_store.set_document_summary(doc_id, final)

    return StreamingResponse(events(), media_type="text/event-stream")


@app.delete("/api/documents/{doc_id}", dependencies=_auth)
def delete_document(doc_id: str) -> dict:
    """Remove a document: filter its chunks out of chunks.json, rebuild the
    FAISS index from the remaining chunks, and delete the SQLite row."""
    import json as _json

    from langchain_community.vectorstores import FAISS

    doc = conv_store.delete_document(doc_id)
    if not doc:
        return {"ok": False, "error": "Document not found."}
    source_name = f"upload:{doc_id}"
    settings = get_settings()
    index_dir = settings.index_dir
    chunks_path = index_dir / "chunks.json"
    if chunks_path.exists():
        remaining = [
            c
            for c in _json.loads(chunks_path.read_text(encoding="utf-8"))
            if c.get("source") != source_name
        ]
        chunks_path.write_text(
            _json.dumps(remaining, ensure_ascii=False), encoding="utf-8"
        )
        # Rebuild the FAISS index from the surviving chunks.
        if remaining:
            store = FAISS.from_texts(
                [c["text"] for c in remaining],
                get_embeddings(settings),
                metadatas=[
                    {
                        "source": c.get("source", "?"),
                        "allowed_groups": c.get("allowed_groups", ["*"]),
                    }
                    for c in remaining
                ],
            )
            store.save_local(str(index_dir))
        # If nothing remains, leave the (now empty) index files as-is;
        # the next upload will recreate the index.
    _upload_texts.pop(doc_id, None)
    return {"ok": True}


# Legacy alias: the React UI's summarize button used /api/upload/{id}/summarize.
@app.post("/api/upload/{doc_id}/summarize", dependencies=_auth)
def summarize_upload(doc_id: str) -> dict:
    return summarize_document(doc_id)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
