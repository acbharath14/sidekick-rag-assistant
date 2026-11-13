"""Settings, all overridable via environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

# Index format version (Phase 5, A2+A3). Version 2 indexes carry chunk_id
# + section metadata on every chunk (see ingest.chunk_documents). Indexes
# built by earlier versions lack those fields, so a version marker is
# written next to every index (index_format.json) and loaders treat a
# missing or older marker as stale — never silently filename-only.
INDEX_FORMAT_VERSION = 2


@dataclass
class Settings:
    """Runtime configuration. `RAG_DEMO_FAKE=1` swaps in deterministic fakes
    (used by tests and CI — no model downloads, no API keys)."""

    docs_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("RAG_DEMO_DOCS_DIR", PROJECT_ROOT / "docs")
        )
    )
    index_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("RAG_DEMO_INDEX_DIR", PROJECT_ROOT / ".faiss_index")
        )
    )
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 8

    # Embeddings: "huggingface" (default) or "fake"
    embedding_provider: str = field(
        default_factory=lambda: os.environ.get("RAG_DEMO_EMBEDDINGS", "huggingface")
    )
    embedding_model: str = field(
        default_factory=lambda: os.environ.get(
            "RAG_DEMO_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    # LLM: "ollama" (default) or "fake". Ollama model defaults to qwen3:8b.
    llm_provider: str = field(default_factory=lambda: os.environ.get("RAG_DEMO_LLM", "ollama"))
    ollama_model: str = field(default_factory=lambda: os.environ.get("RAG_DEMO_MODEL", "qwen3:8b"))
    ollama_base_url: str = field(
        default_factory=lambda: os.environ.get("RAG_DEMO_BASE_URL", "http://localhost:11434")
    )
    # Ollama timeouts and retries (Phase 1, Wave 2). Defaults preserve the
    # old behavior: the model call itself keeps its 300s deadline, and a
    # chat request gets 30s of headroom over that for retrieval and
    # bookkeeping. The summarize per-call deadline stays 600s, exactly as
    # summarize.py already used it; it now lives here instead of being
    # read from the environment inside summarize.py.
    # RAG_DEMO_LLM_TIMEOUT: deadline for one Ollama call (seconds).
    llm_timeout_seconds: float = field(
        default_factory=lambda: float(os.environ.get("RAG_DEMO_LLM_TIMEOUT", "300"))
    )
    # RAG_DEMO_LLM_RETRY_ATTEMPTS: total tries for one Ollama call,
    # including the first attempt. Only transient errors (connection
    # errors, HTTP 5xx) are retried; timeouts never are.
    llm_retry_attempts: int = field(
        default_factory=lambda: int(os.environ.get("RAG_DEMO_LLM_RETRY_ATTEMPTS", "3"))
    )
    # RAG_DEMO_LLM_RETRY_BACKOFF: comma separated wait between retries,
    # e.g. "5,15". Same (5, 15) as summarize.py already used.
    llm_retry_backoff_seconds: tuple[float, ...] = field(
        default_factory=lambda: tuple(
            float(v)
            for v in os.environ.get("RAG_DEMO_LLM_RETRY_BACKOFF", "5,15").split(",")
            if v.strip()
        )
        or (5.0, 15.0)
    )
    # RAG_DEMO_CHAT_TIMEOUT: deadline for one /api/chat request
    # (seconds). 330 covers the 300s model call plus retrieval overhead.
    chat_timeout_seconds: float = field(
        default_factory=lambda: float(os.environ.get("RAG_DEMO_CHAT_TIMEOUT", "330"))
    )
    # RAG_DEMO_STREAM_IDLE_TIMEOUT: longest silence allowed mid-stream
    # on /api/chat/stream and the summarize stream (seconds). Tokens
    # normally arrive steadily, so 300s only fires on a genuine stall.
    stream_idle_timeout_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("RAG_DEMO_STREAM_IDLE_TIMEOUT", "300")
        )
    )
    # RAG_DEMO_SUMMARY_TIMEOUT: deadline for one summarize LLM call.
    summarize_timeout_seconds: float = field(
        default_factory=lambda: float(os.environ.get("RAG_DEMO_SUMMARY_TIMEOUT", "600"))
    )
    # RAG_DEMO_SUMMARIZE_PARALLEL_MAP: run the map-reduce map calls
    # concurrently (1 = on). Default off: the map loop stays sequential.
    # The prompts, the model, and the reduce step are identical either
    # way, so the summary is the same; only the wall time can change.
    # Whether it actually helps depends on the Ollama server (its
    # parallel-sequence capacity on CPU-only hardware), so measure it
    # on the target machine before turning it on.
    summarize_parallel_map: bool = field(
        default_factory=lambda: os.environ.get("RAG_DEMO_SUMMARIZE_PARALLEL_MAP", "0") == "1"
    )
    # RAG_DEMO_SUMMARY_REQUEST_TIMEOUT: deadline for one sync summarize
    # request (seconds). Default 0 means disabled: map-reduce total time
    # is N chunks times the per-call deadline by design, so a single
    # request cap would either break long documents or be meaningless.
    # Set it to bound a single request when that matters more.
    summarize_request_timeout_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("RAG_DEMO_SUMMARY_REQUEST_TIMEOUT", "0")
        )
    )

    # Retrieval: "hybrid" (default, BM25 + dense with reciprocal rank
    # fusion) or "dense" (pure vector search).
    retrieval: str = field(
        default_factory=lambda: os.environ.get("RAG_DEMO_RETRIEVAL", "hybrid")
    )
    # Cross-encoder reranking of the top candidates. On by default;
    # RAG_DEMO_RERANK=0 disables it. Downloads a small model on first use,
    # never on CI: under RAG_DEMO_FAKE=1 the default stays off, so tests
    # never touch sentence_transformers/torch. An explicit RAG_DEMO_RERANK
    # value always wins, exactly as before.
    rerank_enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "RAG_DEMO_RERANK",
            "0" if os.environ.get("RAG_DEMO_FAKE") == "1" else "1",
        )
        == "1"
    )
    # Relevance floor (Phase 5, A2+A3): minimum confidence for a passage
    # to feed the prompt. Hits below the floor are dropped and, when
    # nothing passes, the UI shows a weak-evidence notice instead of a
    # confident guess. 0 disables the floor. Default 0.5 is principled,
    # not golden-set tuned (no real embeddings in the sandbox to tune
    # against — see retriever.confidence_for): it is the midpoint of the
    # confidence scale, i.e. rerank logit 0 and dense L2 distance 1.0.
    # Under RAG_DEMO_FAKE=1 the default is 0: fake-embedding distances
    # are meaningless, so the fake tier never drops hits on them (same
    # fake-aware pattern as the rerank default). An explicit
    # RAG_DEMO_RELEVANCE_FLOOR value always wins.
    relevance_floor: float = field(
        default_factory=lambda: float(
            os.environ.get(
                "RAG_DEMO_RELEVANCE_FLOOR",
                "0",
            )
        )
    )
    # Query rewriting for multi-turn chat. Default on; no-op without history.
    rewrite_enabled: bool = field(
        default_factory=lambda: os.environ.get("RAG_DEMO_REWRITE", "1") == "1"
    )
    # Hybrid mode: when the corpus has no answer, fall back to the LLM's
    # general knowledge, clearly marked (never silently mixed with citations).
    hybrid_fallback: bool = field(
        default_factory=lambda: os.environ.get("RAG_DEMO_HYBRID", "1") == "1"
    )
    # Jev routing trial (part 2): one decision-model question in front of
    # retrieval. Default off; when on, ask()/stream_ask() route
    # general-knowledge questions straight to general_knowledge_answer()
    # and skip retrieval. Any routing failure degrades to the corpus path.
    jev_routing: bool = field(
        default_factory=lambda: os.environ.get("RAG_DEMO_JEV_ROUTING", "0") == "1"
    )
    # LangGraph migration (Phase 2): route ask()/stream_ask() through the
    # StateGraph in rag_demo/graph.py instead of the LCEL chain in
    # chain.py. Same steps, same behavior; off by default until the
    # graph is verified on the iMac.
    use_langgraph: bool = field(
        default_factory=lambda: os.environ.get("RAG_DEMO_USE_LANGGRAPH", "0")
        == "1"
    )
    # Agentic loop budget (Phase 2, S3): iteration cap + wall-clock cap.
    # The iteration cap preserves the old MAX_ITERATIONS default (4).
    # The wall-clock cap is generous (5 min, matching the LLM timeout) so
    # it only fires on genuinely stuck runs, not on slow-but-healthy CPU
    # runs. When either cap is hit, the agent stops with the evidence
    # gathered so far plus an explicit uncertainty note (see agent.py).
    agent_max_iterations: int = field(
        default_factory=lambda: int(os.environ.get("RAG_DEMO_AGENT_MAX_ITERATIONS", "4"))
    )
    agent_budget_seconds: float = field(
        default_factory=lambda: float(os.environ.get("RAG_DEMO_AGENT_BUDGET_SECONDS", "300"))
    )
    # GitHub connector: token (never committed) and default repo.
    github_token: str | None = field(
        default_factory=lambda: os.environ.get("RAG_DEMO_GITHUB_TOKEN")
    )
    github_repo: str = field(
        default_factory=lambda: os.environ.get("RAG_DEMO_GITHUB_REPO", "")
    )
    # Groups the asker belongs to, for permission-aware retrieval.
    user_groups: list[str] = field(
        default_factory=lambda: [
            g.strip()
            for g in os.environ.get("RAG_DEMO_USER_GROUPS", "").split(",")
            if g.strip()
        ]
    )
    # Opt-in JSONL audit log path.
    audit_log_path: str | None = field(
        default_factory=lambda: os.environ.get("RAG_DEMO_AUDIT_LOG")
    )
    audit_log_answers: bool = field(
        default_factory=lambda: os.environ.get("RAG_DEMO_AUDIT_LOG_ANSWERS") == "1"
    )
    # Per-client API rate limits (Phase 1, Wave 3): sliding window of
    # 60 seconds per API key (or per client IP when auth is off).
    # Defaults are generous for a single-user demo: chat is the LLM path,
    # uploads and summarizes are the heavy background paths. A per-bucket
    # value of 0 disables that bucket; RAG_DEMO_RATE_LIMIT_ENABLED=0
    # disables all limiting.
    rate_limit_enabled: bool = field(
        default_factory=lambda: os.environ.get("RAG_DEMO_RATE_LIMIT_ENABLED", "1") == "1"
    )
    rate_limit_chat_per_minute: int = field(
        default_factory=lambda: int(
            os.environ.get("RAG_DEMO_RATE_LIMIT_CHAT_PER_MINUTE", "30")
        )
    )
    rate_limit_upload_per_minute: int = field(
        default_factory=lambda: int(
            os.environ.get("RAG_DEMO_RATE_LIMIT_UPLOAD_PER_MINUTE", "10")
        )
    )
    rate_limit_summarize_per_minute: int = field(
        default_factory=lambda: int(
            os.environ.get("RAG_DEMO_RATE_LIMIT_SUMMARIZE_PER_MINUTE", "20")
        )
    )

    @property
    def fake(self) -> bool:
        return os.environ.get("RAG_DEMO_FAKE") == "1"


def get_settings() -> Settings:
    s = Settings()
    if s.fake:
        s.embedding_provider = "fake"
        s.llm_provider = "fake"
    return s


def get_embeddings(settings: Settings):
    """Lazy imports keep `import rag_demo` light when deps are missing."""
    if settings.embedding_provider == "fake":
        from langchain_core.embeddings import FakeEmbeddings

        return FakeEmbeddings(size=384)
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def get_llm(settings: Settings):
    if settings.llm_provider == "fake":
        from langchain_core.language_models.fake import FakeListLLM

        return FakeListLLM(
            responses=[
                "Based on the indexed docs: see the cited sources below."
            ]
        )
    from langchain_ollama import OllamaLLM

    return OllamaLLM(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        timeout=settings.llm_timeout_seconds,  # fail fast instead of hanging forever on a stalled server
        keep_alive="30m",  # keep the model in RAM between prompts; avoids reload stalls
    )
