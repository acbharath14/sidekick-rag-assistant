"""FastAPI backend for the Sidekick RAG assistant.

Exposes the RAG pipeline (ask/stream_ask from chain.py) as a REST + SSE API
for the React frontend. Streamlit (app.py) is untouched and keeps working.

Run:  python -m rag_demo.api   (uvicorn on :8000)

Endpoints:
  POST /api/chat         non-streaming ask (optional document_id to scope
                         retrieval, optional conversation_id for persistence)
  POST /api/chat/stream  SSE streaming (token events, then done)
  GET  /api/health       ollama reachability + index presence +
                         Ollama queue snapshot (in-flight calls, waits,
                         timeouts, retries)
  GET  /api/config       safe settings subset (no secrets)
  GET  /api/metrics      in-memory counters + latency summaries (no new dep)
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

Rate limiting: the expensive endpoints (/api/chat, /api/chat/stream,
/api/upload, /api/documents/{id}/summarize and its stream and legacy
alias) are limited per client per minute (see rag_demo.rate_limit and
Settings.rate_limit_*_per_minute). Exceeding the limit returns 429
{"detail": "Rate limit exceeded. Try again later."} with a Retry-After
header (seconds). /api/health is never limited.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import uuid
from dataclasses import asdict
from typing import Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi import status as http_status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .audit import log_feedback
from .caching import clear_cache
from .chain import Answer, ask, stream_ask
from .config import get_embeddings, get_settings
from .ollama_guard import (
    OllamaTimeoutError,
    is_transient_error,
    iter_with_idle_timeout,
    ollama_snapshot,
    run_with_timeout,
)
from . import conversations as conv_store
from .observability import (
    FILENAME_PREVIEW_CHARS,
    get_metrics_snapshot,
    log_error_event,
    log_event,
    log_summarize_event,
    log_upload_event,
    new_request_id,
    record_summarize,
    record_upload,
    truncate_preview,
)
from .rate_limit import rate_limit
from .summarize import SINGLE_CALL_CHARS

logger = logging.getLogger(__name__)

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

# Per-client rate limits (see rate_limit.py). Auth runs first, so a bad
# key gets 401 without consuming quota, and the limiter keys on the
# validated key. Chat and its stream share one bucket (both hit the LLM);
# the two summarize routes share one too. /api/health stays unthrottled,
# and the cheap endpoints (conversations, feedback, config, document
# list/get/delete) are not limited.
_rate_chat = [Depends(require_api_key), Depends(rate_limit("chat"))]
_rate_upload = [Depends(require_api_key), Depends(rate_limit("upload"))]
_rate_summarize = [Depends(require_api_key), Depends(rate_limit("summarize"))]


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


# --- Cancel-on-disconnect for the streaming endpoints ---------------------
#
# The two SSE endpoints below drive their producer (stream_ask /
# summarize_events) one item at a time in a worker thread while a small
# watcher task probes request.is_disconnected(). When the client goes
# away mid-generation the driver closes the producer iterator: that
# raises GeneratorExit inside the producer, whose finally closes the
# underlying chain stream, which closes the Ollama HTTP stream, so the
# model stops generating instead of finishing tokens nobody reads.
# The SSE contract for connected clients is unchanged: same events, same
# order, same error events. A disconnect ends the stream quietly (no
# error event, no 500): there is nobody left to tell.
#
# The endpoints are async for exactly this reason: is_disconnected() is
# a coroutine, and the sync producers must run off the event loop so the
# watcher can run while a token is in flight.

_DISCONNECT_POLL_SECONDS = 0.2
_DISCONNECT_GRACE_SECONDS = 5.0


def _pull(iterator):
    """One pull from a sync iterator, for a worker thread.

    Returns (True, item), or (False, None) at exhaustion. Producer
    errors propagate to the caller.
    """
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


def _close_iterator(iterator) -> None:
    """Close a producer iterator, best effort.

    Closing raises GeneratorExit inside the producer so it can release
    its model call. A ValueError means the producer is still executing
    in another thread (a token pull is in flight); the iterator is
    reaped when that pull lands, so there is nothing more to do here.
    """
    close = getattr(iterator, "close", None)
    if close is None:
        return
    try:
        close()
    except ValueError:
        pass


async def _watch_disconnect(request: Request) -> bool:
    """Return True when the client disconnects; never raises.

    Probes request.is_disconnected() until it says True. Starlette's
    probe waits for the next ASGI message, which only arrives on
    disconnect, so one probe usually blocks until then; if a server
    answers False at once instead, sleep briefly and probe again.
    A probe that errors is logged once and treated as still connected,
    so a broken probe can never kill a healthy stream.
    """
    logged = False
    while True:
        try:
            if await request.is_disconnected():
                return True
        except Exception:
            if not logged:
                logger.warning(
                    "disconnect probe failed; assuming client still connected",
                    exc_info=True,
                )
                logged = True
        await asyncio.sleep(_DISCONNECT_POLL_SECONDS)


async def _drive_stream(iterator, request: Request, *, label: str):
    """Yield items from a sync producer iterator, stopping on disconnect.

    Each pull runs in a worker thread so the event loop stays free to
    notice the disconnect while a token is in flight. On disconnect the
    producer is closed (see _close_iterator), one log line is written,
    and iteration ends with no error event: the client is gone. Producer
    errors pass through unchanged, so the existing error handling in the
    caller still applies.
    """
    watcher = asyncio.ensure_future(_watch_disconnect(request))
    yielded = 0
    try:
        loop = asyncio.get_running_loop()
        while True:
            pull = asyncio.ensure_future(
                loop.run_in_executor(None, _pull, iterator)
            )
            done, _pending = await asyncio.wait(
                {pull, watcher}, return_when=asyncio.FIRST_COMPLETED
            )
            if watcher in done:
                # The client left mid-generation. Wait briefly for the
                # in-flight pull to land so the producer is suspended
                # and close() can run; if it is still stuck, close()
                # is a no-op and the producer is reaped when the pull
                # lands (bounded by the stream idle timeout).
                try:
                    await asyncio.wait_for(
                        asyncio.shield(pull), timeout=_DISCONNECT_GRACE_SECONDS
                    )
                except asyncio.TimeoutError:
                    pass
                _close_iterator(iterator)
                logger.info(
                    "client disconnected; cancelled %s stream after %d events",
                    label,
                    yielded,
                )
                return
            try:
                has_item, item = pull.result()
            except BaseException:
                _close_iterator(iterator)
                raise
            if not has_item:
                return
            yielded += 1
            yield item
    finally:
        if not watcher.done():
            watcher.cancel()


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


@app.post("/api/chat", dependencies=_rate_chat)
def chat(req: ChatRequest) -> dict:
    request_id = new_request_id()
    conv_id = req.conversation_id
    if conv_id and conv_store.conversation_exists(conv_id):
        conv_store.add_message(conv_id, "user", req.question)
        new_title = conv_store.maybe_autotitle(conv_id, req.question)
    else:
        conv_id = None
        new_title = None
    # LLM failures are counted and logged inside ask(); they still
    # propagate as 500s, exactly as before.
    # Request deadline (settings.chat_timeout_seconds, 330s by default):
    # a hung Ollama call fails fast with 504 instead of hanging the
    # request. Transient Ollama errors surface as 502 once the bounded
    # retries inside the chain are exhausted.
    settings = get_settings()
    try:
        answer = run_with_timeout(
            lambda: ask(
                req.question,
                history=req.history or [],
                document_id=_document_exists(req.document_id),
                request_id=request_id,
                endpoint="/api/chat",
            ),
            settings.chat_timeout_seconds,
            label="chat",
        )
    except OllamaTimeoutError:
        raise HTTPException(
            status_code=http_status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The model took too long to answer. Try again in a moment.",
        )
    except Exception as e:
        if is_transient_error(e):
            raise HTTPException(
                status_code=http_status.HTTP_502_BAD_GATEWAY,
                detail="Ollama is unreachable right now. Try again in a moment.",
            )
        raise
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


@app.post("/api/chat/stream", dependencies=_rate_chat)
async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    request_id = new_request_id()
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

    async def events():
        final_answer: Answer | None = None
        try:
            async for event in _drive_stream(
                stream_ask(
                    req.question,
                    history=req.history or [],
                    document_id=doc_id,
                    request_id=request_id,
                    endpoint="/api/chat/stream",
                ),
                request,
                label="chat",
            ):
                if "token" in event:
                    yield f"data: {json.dumps({'token': event['token']})}\n\n"
                else:
                    final_answer = event["answer"]
                    payload = {"done": _answer_to_wire(final_answer)}
                    if new_title:
                        payload["done"]["conversation_title"] = new_title
                    yield f"data: {json.dumps(payload)}\n\n"
        except OllamaTimeoutError:
            # Stalled stream: end with a plain error event instead of
            # leaving the SSE connection open forever.
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "error",
                        "error": "The model took too long to respond. "
                        "Try again in a moment.",
                    }
                )
                + "\n\n"
            )
            return
        except Exception as e:
            if is_transient_error(e):
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "error",
                            "error": "Ollama is unreachable right now. "
                            "Try again in a moment.",
                        }
                    )
                    + "\n\n"
                )
                return
            raise
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
        # Queue visibility: in-flight calls, waits, timeouts, retries.
        # A backed-up Ollama shows up here without opening the logs.
        "ollama": ollama_snapshot(),
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
        "rate_limit": {
            "enabled": settings.rate_limit_enabled,
            "chat_per_minute": settings.rate_limit_chat_per_minute,
            "upload_per_minute": settings.rate_limit_upload_per_minute,
            "summarize_per_minute": settings.rate_limit_summarize_per_minute,
        },
    }


@app.get("/api/metrics", dependencies=_auth)
def metrics() -> dict:
    """In-memory counters and latency summaries (see observability.py).
    No new dependency, no persistence: counts reset on restart. The
    cache section reports hits/reloads when the caching module exposes
    cache_stats(); until then it reports the chain call-site counters
    and available=False."""
    return get_metrics_snapshot()


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
    import time as _time

    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from .extract import UnsupportedFormat, extract_text_with_meta, verify_magic_bytes

    request_id = new_request_id()
    _started = _time.perf_counter()

    def _fail(error: str) -> dict:
        # No behavior change: same error body as before, only counted
        # and logged (filename only, never file content).
        record_upload((_time.perf_counter() - _started) * 1000, ok=False)
        log_error_event(
            request_id=request_id,
            endpoint="/api/upload",
            where="upload",
            error_type="UploadError",
            error=error,
            filename=truncate_preview(filename, FILENAME_PREVIEW_CHARS),
        )
        return {"ok": False, "error": error}
    if len(data) > max_mb * 1024 * 1024:
        return _fail(f"File too large (over {max_mb:.0f} MB).")
    try:
        if not verify_magic_bytes(filename, data):
            return _fail("File content doesn't match its extension.")
        text, extractor = extract_text_with_meta(filename, data)
    except UnsupportedFormat as e:
        return _fail(str(e))
    except Exception as e:
        return _fail(f"Couldn't read {filename}: {e}")
    if not text.strip() and filename.lower().endswith(".pdf"):
        # Scanned PDF with no embedded text: fall back to OCR, mirroring
        # the old Streamlit app. ocr.py is imported lazily so nothing breaks
        # when the tesseract/poppler binaries are absent.
        from .ocr import INSTALL_HINT, ocr_available, ocr_pdf

        if ocr_available():
            try:
                text = ocr_pdf(data)
            except RuntimeError as e:
                return _fail(
                    f"No text found in {filename} (scanned PDF). {e}"
                )
            if text.strip():
                extractor = "ocr"
        else:
            return _fail(
                f"No text found in {filename} (scanned PDF). {INSTALL_HINT}"
            )
    if not text.strip():
        return _fail(f"No text found in {filename}.")

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
    _latency_ms = (_time.perf_counter() - _started) * 1000
    record_upload(_latency_ms, ok=True)
    log_upload_event(
        request_id=request_id,
        doc_id=doc_id,
        filename=filename,
        chunks=len(chunks),
        chars=len(text),
        size_bytes=len(data),
        extractor=extractor,
        latency_ms=_latency_ms,
    )
    return {"ok": True, **doc}


@app.post("/api/upload", dependencies=_rate_upload)
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


@app.post("/api/documents/{doc_id}/summarize", dependencies=_rate_summarize)
def summarize_document(doc_id: str) -> dict:
    """(Re)generate the LLM summary for a document and cache it in SQLite."""
    import time as _time

    request_id = new_request_id()
    _started = _time.perf_counter()
    doc = conv_store.get_document(doc_id)
    if not doc:
        log_error_event(
            request_id=request_id,
            endpoint="/api/documents/{id}/summarize",
            where="summarize",
            error_type="NotFound",
            error="Document not found.",
            doc_id=doc_id,
        )
        return {"ok": False, "error": "Document not found."}
    text = _upload_texts.get(doc_id)
    if not text:
        log_error_event(
            request_id=request_id,
            endpoint="/api/documents/{id}/summarize",
            where="summarize",
            error_type="NotFound",
            error="Document text no longer in memory (restarted since upload).",
            doc_id=doc_id,
        )
        return {
            "ok": False,
            "error": "Document text no longer in memory (restarted since upload)."
            " Re-upload to summarize.",
        }
    from .summarize import summarize

    settings = get_settings()
    try:
        summary = run_with_timeout(
            lambda: summarize(text, settings),
            settings.summarize_request_timeout_seconds,
            label="summarize",
        )
    except OllamaTimeoutError:
        return {
            "ok": False,
            "error": "Summarization timed out. Try again with a shorter document.",
        }
    except Exception as e:
        _latency_ms = (_time.perf_counter() - _started) * 1000
        record_summarize(_latency_ms, ok=False)
        log_error_event(
            request_id=request_id,
            endpoint="/api/documents/{id}/summarize",
            where="summarize",
            error_type=type(e).__name__,
            error=str(e),
            doc_id=doc_id,
        )
        return {"ok": False, "error": f"Summarize failed: {type(e).__name__}: {e}"}
    summary = summary or "Summary returned empty."
    conv_store.set_document_summary(doc_id, summary)
    _latency_ms = (_time.perf_counter() - _started) * 1000
    record_summarize(_latency_ms, ok=True)
    log_summarize_event(
        request_id=request_id,
        doc_id=doc_id,
        mode="single" if len(text) <= SINGLE_CALL_CHARS else "map_reduce",
        chars=len(text),
        latency_ms=_latency_ms,
    )
    return {"ok": True, "summary": summary}


@app.post("/api/documents/{doc_id}/summarize/stream", dependencies=_rate_summarize)
async def summarize_document_stream(doc_id: str, request: Request) -> StreamingResponse:
    """SSE stream of summarize progress events, then a done event.

    Event payloads (see rag_demo.summarize.summarize_events):
      {"type": "start", "mode": "single" | "map_reduce", "total": n}
      {"type": "progress", "done": i, "total": n, "stage": "map"}
      {"type": "progress", "stage": "reduce"}
      {"type": "done", "summary": "..."}
      {"type": "error", "error": "..."}
    The final summary is cached in SQLite exactly like the sync endpoint.
    If the client disconnects mid-summary, generation is cancelled (see
    _drive_stream) and nothing is cached: no partial summary is stored.
    """
    import time as _time

    request_id = new_request_id()
    _started = _time.perf_counter()

    def _error_event(message: str):
        yield f"data: {json.dumps({'type': 'error', 'error': message})}\n\n"

    doc = conv_store.get_document(doc_id)
    if not doc:
        log_error_event(
            request_id=request_id,
            endpoint="/api/documents/{id}/summarize/stream",
            where="summarize",
            error_type="NotFound",
            error="Document not found.",
            doc_id=doc_id,
        )
        return StreamingResponse(_error_event("Document not found."),
                                 media_type="text/event-stream")
    text = _upload_texts.get(doc_id)
    if not text:
        log_error_event(
            request_id=request_id,
            endpoint="/api/documents/{id}/summarize/stream",
            where="summarize",
            error_type="NotFound",
            error="Document text no longer in memory (restarted since upload).",
            doc_id=doc_id,
        )
        return StreamingResponse(
            _error_event("Document text no longer in memory (restarted since upload)."
                         " Re-upload to summarize."),
            media_type="text/event-stream",
        )

    async def events():
        from .summarize import summarize_events

        settings = get_settings()
        final: str | None = None
        mode = "single" if len(text) <= SINGLE_CALL_CHARS else "map_reduce"
        try:
            async for event in _drive_stream(
                iter_with_idle_timeout(
                    summarize_events(text, settings),
                    settings.stream_idle_timeout_seconds,
                    label="summarize-stream",
                ),
                request,
                label="summarize",
            ):
                if event["type"] == "done":
                    final = event["summary"]
                    _latency_ms = (_time.perf_counter() - _started) * 1000
                    record_summarize(_latency_ms, ok=True)
                    log_summarize_event(
                        request_id=request_id,
                        doc_id=doc_id,
                        mode=mode,
                        chars=len(text),
                        latency_ms=_latency_ms,
                    )
                elif event["type"] == "error":
                    _latency_ms = (_time.perf_counter() - _started) * 1000
                    record_summarize(_latency_ms, ok=False)
                    log_error_event(
                        request_id=request_id,
                        endpoint="/api/documents/{id}/summarize/stream",
                        where="summarize",
                        error_type="SummarizeError",
                        error=event.get("error", ""),
                        doc_id=doc_id,
                    )
                yield f"data: {json.dumps(event)}\n\n"
        except OllamaTimeoutError:
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "error",
                        "error": "Summarization stalled: the model produced "
                        "no output for a while. Try again in a moment.",
                    }
                )
                + "\n\n"
            )
            return
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
        log_error_event(
            request_id=new_request_id(),
            endpoint="/api/documents/{id}",
            where="delete_document",
            error_type="NotFound",
            error="Document not found.",
            doc_id=doc_id,
        )
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
    log_event(
        "document_delete",
        request_id=new_request_id(),
        doc_id=doc_id,
    )
    # chunks.json / the FAISS index changed on disk; drop the cached
    # retriever so the next question sees the deletion.
    clear_cache()
    return {"ok": True}


# Legacy alias: the React UI's summarize button used /api/upload/{id}/summarize.
@app.post("/api/upload/{doc_id}/summarize", dependencies=_rate_summarize)
def summarize_upload(doc_id: str) -> dict:
    return summarize_document(doc_id)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
