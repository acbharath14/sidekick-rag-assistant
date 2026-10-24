"""Session-level caching for expensive objects (Phase 1, S1).

Problem it solves: before this module, every question rebuilt the
embeddings object (a SentenceTransformer weight load), reloaded the FAISS
index from disk, and — in agentic mode — constructed a fresh Retriever per
tool call. The user paid model-load + index-load before generation started.

Design (deliberate, read before changing):
- Plain settings-keyed module dicts, NOT `st.cache_resource`. Rationale:
  tests, the CLI, and the MCP server all run without Streamlit, so a
  Streamlit-decorated cache could not be unit-tested here; and in the
  Streamlit app, module state persists across reruns in-process, which
  gives the same residency. If a future change wants Streamlit's cache
  management, wrap these accessors — do NOT add a second cache alongside
  this one, or invalidation will diverge.
- The FAISS store is cached as part of the Retriever, not separately:
  there is no standalone store-load path in the app (ingest creates
  stores; the upload path creates its own in-memory store), so caching
  the Retriever covers every load.

Cache-invalidation contract (Phase 4 Settings depends on this):
- Entries are keyed by exactly the settings that affect the object:
  - embeddings: (embedding_provider, embedding_model)
  - retriever: (embedding_provider, embedding_model, index_dir,
    retrieval, rerank_enabled)
  - episodic memory: (lancedb dir derived from index_dir,
    embedding_provider, embedding_model)
  Fields NOT in a key are applied at call time, not construction time
  (user_groups, top_k), so they never need invalidation.
- Changing a keyed field yields a fresh entry for the affected objects
  only: e.g. switching corpora (index_dir changes) builds a new
  Retriever but reuses the same embeddings object. That is what "clears
  exactly the affected cached resources" means here.
- Keying does NOT cover rebuilding an index in place (same index_dir,
  new vectors): the cached Retriever would keep serving the old store.
  Any code path that rebuilds an index MUST call clear_cache()
  afterwards. Today that is the enterprise index build in app.py; the
  Phase 4 Settings panel must do the same on model/corpus/index change
  and after any rebuild.
- Upload retrievers are never cached: they are session-specific
  in-memory indexes passed explicitly, and must never leak across
  sessions via this process-global cache.

Escape hatch: RAG_DEMO_NO_CACHE=1 bypasses the cache entirely
(constructs fresh every call, stores nothing). Default off. Intended
for debugging only.
"""

from __future__ import annotations

import os
import threading

from .config import Settings, get_embeddings, get_settings
from .episodic import EpisodicMemory, _episodic_dir
from .retriever import Retriever

# RLock (not Lock): get_cached_retriever/get_cached_episodic_memory call
# get_cached_embeddings while holding the lock, so the lock must be
# reentrant — a plain Lock deadlocks on first use.
_LOCK = threading.RLock()
_EMBEDDINGS: dict[tuple, object] = {}
_RETRIEVERS: dict[tuple, Retriever] = {}
_EPISODIC: dict[tuple, EpisodicMemory] = {}


def _no_cache() -> bool:
    return os.environ.get("RAG_DEMO_NO_CACHE") == "1"


def _embeddings_key(settings: Settings) -> tuple:
    return (settings.embedding_provider, settings.embedding_model)


def _retriever_key(settings: Settings) -> tuple:
    return (
        settings.embedding_provider,
        settings.embedding_model,
        str(settings.index_dir),
        settings.retrieval,
        settings.rerank_enabled,
    )


def _episodic_key(settings: Settings) -> tuple:
    return (
        str(_episodic_dir(settings)),
        settings.embedding_provider,
        settings.embedding_model,
    )


def get_cached_embeddings(settings: Settings | None = None):
    """Return the shared embeddings object for these settings.

    With RAG_DEMO_NO_CACHE=1, constructs a fresh object every call.
    """
    settings = settings or get_settings()
    if _no_cache():
        return get_embeddings(settings)
    key = _embeddings_key(settings)
    with _LOCK:
        obj = _EMBEDDINGS.get(key)
        if obj is None:
            obj = get_embeddings(settings)
            _EMBEDDINGS[key] = obj
        return obj


def get_cached_retriever(settings: Settings | None = None) -> Retriever:
    """Return the shared Retriever for these settings (default-corpus path).

    The Retriever is built once per key, sharing the cached embeddings
    object. Upload retrievers must NOT go through here (see module
    docstring); pass them explicitly to ask()/stream_ask()/agentic_ask().
    Raises exactly as Retriever() does when the index is missing.
    """
    settings = settings or get_settings()
    if _no_cache():
        return Retriever(settings, embeddings=get_embeddings(settings))
    key = _retriever_key(settings)
    with _LOCK:
        retriever = _RETRIEVERS.get(key)
        if retriever is None:
            retriever = Retriever(
                settings, embeddings=get_cached_embeddings(settings)
            )
            _RETRIEVERS[key] = retriever
        return retriever


def get_cached_episodic_memory(settings: Settings | None = None) -> EpisodicMemory:
    """Return the session-singleton EpisodicMemory for these settings.

    Shares the cached embeddings object, so episodic memory adds no
    extra model load. Construction never raises: like EpisodicMemory
    itself, a broken/missing LanceDB backend degrades to a disabled
    instance (search/add become no-ops).
    """
    settings = settings or get_settings()
    if _no_cache():
        return EpisodicMemory(settings, embeddings=get_embeddings(settings))
    key = _episodic_key(settings)
    with _LOCK:
        memory = _EPISODIC.get(key)
        if memory is None:
            memory = EpisodicMemory(
                settings, embeddings=get_cached_embeddings(settings)
            )
            _EPISODIC[key] = memory
        return memory


def clear_cache() -> None:
    """Drop all cached entries (embeddings, retrievers, episodic memory).

    Call this after rebuilding any index in place, and from Settings
    whenever model/corpus/index configuration changes (see the contract
    in the module docstring). Next access rebuilds lazily.
    """
    with _LOCK:
        _EMBEDDINGS.clear()
        _RETRIEVERS.clear()
        _EPISODIC.clear()
