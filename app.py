"""Sidekick RAG Assistant — chat UI over the indexed corpus.

Run:  PYTHONPATH=src streamlit run app.py
Needs the index built first (`python -m rag_demo.ingest`) and Ollama running
for real answers (or RAG_DEMO_FAKE=1 for the fake LLM).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json

import streamlit as st

from evals.regression import check_case
from rag_demo.audit import log_feedback
from rag_demo.chain import ask, stream_ask
from rag_demo.chat_store import (
    delete_chat,
    list_chats,
    load_chat,
    new_chat,
    rename_chat,
    save_chat,
)
from rag_demo.config import get_embeddings, get_settings
from rag_demo.ingest import INDEX_DIRS
from rag_demo.sources import SOURCES

st.set_page_config(
    page_title="Sidekick RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .citation-pill {
        display: inline-block;
        background: #4F46E5;
        color: white;
        border-radius: 999px;
        padding: 1px 10px;
        margin: 2px 4px 2px 0;
        font-size: 0.78rem;
        font-family: monospace;
    }
    .suggestion-chip button {
        border-radius: 999px !important;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

settings = get_settings()

CORPUS_LABELS = {
    "meridian": "Meridian Logistics (fictional)",
    "playwright-docs": "Playwright docs",
    "github-docs": "GitHub repo docs",
    "confluence-mock": "Confluence (mock)",
}

SUGGESTIONS = {
    "meridian": [
        "How do I create a shipment via the API?",
        "How long are API tokens valid?",
        "A shipment is stuck in exception. What does that mean?",
    ],
    "playwright-docs": [
        "How do I run tests in headed mode?",
        "How do I handle authentication in tests?",
    ],
    "github-docs": [
        "What does this repo do?",
        "How do I run the tests?",
    ],
    "confluence-mock": [
        "What is the deploy checklist?",
        "What is the webhook retry policy?",
    ],
}

def _ollama_reachable(base_url: str) -> bool:
    try:
        import urllib.request

        urllib.request.urlopen(f"{base_url}/api/tags", timeout=2)
        return True
    except Exception:  # noqa: BLE001 — unreachable is a status, not an error
        return False


FALLBACK_BADGE = '<span class="citation-pill">🌐 General knowledge</span>'


def _chat_markdown(messages: list[dict]) -> str:
    lines = ["# Sidekick chat export", ""]
    for m in messages:
        lines.append(f"## {'You' if m['role'] == 'user' else 'Sidekick'}")
        lines.append(m["text"])
        if m.get("sources"):
            lines.append(f"*Sources: {', '.join(m['sources'])}*")
        lines.append("")
    return "\n".join(lines)


def _load_regression_cases() -> list[dict]:
    """Standard prompts shared with evals/regression.py (single source of truth)."""
    path = Path(__file__).resolve().parent / "evals" / "regression_cases.json"
    try:
        return json.loads(path.read_text())
    except OSError:
        return []


def _persist_active_chat() -> None:
    """Write session-state messages/history back to the active chat file."""
    chat_id = st.session_state.get("active_chat_id")
    if not chat_id:
        return
    chat = load_chat(chat_id) or {"id": chat_id, "title": "New chat"}
    chat["messages"] = st.session_state.get("messages", [])
    chat["history"] = st.session_state.get("history", [])
    # Auto-title from the first user message.
    if chat.get("title") in (None, "New chat"):
        for m in chat["messages"]:
            if m.get("role") == "user" and m.get("text"):
                chat["title"] = m["text"][:40]
                break
    save_chat(chat)


# ---------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.header("📚 Sidekick")
    corpus_options = [c for c in SOURCES if c in CORPUS_LABELS]
    if st.session_state.get("upload_retriever") is not None:
        corpus_options.append("upload")
        CORPUS_LABELS["upload"] = (
            f"Uploaded: {st.session_state.get('upload_name', 'file')}"
        )
    default_corpus = st.session_state.pop("corpus_override", "meridian")
    if default_corpus not in corpus_options:
        default_corpus = "meridian"
    corpus = st.radio(
        "Corpus",
        options=corpus_options,
        format_func=lambda c: CORPUS_LABELS.get(c, c),
        index=corpus_options.index(default_corpus),
        help="Build an index first: python -m rag_demo.ingest --source <name>",
    )
    upload_retriever = (
        st.session_state.get("upload_retriever") if corpus == "upload" else None
    )
    if corpus != "meridian" and corpus != "upload":
        settings.index_dir = settings.index_dir.parent / INDEX_DIRS[corpus]

    st.divider()
    st.subheader("Session")
    user_groups = st.text_input(
        "Your groups (comma-separated)",
        value=",".join(settings.user_groups),
        help="Permission-aware retrieval demo: try 'ops-all' vs 'ops-leads' on the Confluence corpus.",
    )
    groups = [g.strip() for g in user_groups.split(",") if g.strip()]
    if st.button("➕ New chat", use_container_width=True):
        _persist_active_chat()
        _chat = new_chat()
        st.session_state.active_chat_id = _chat["id"]
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.subheader("💬 Chats")
    _chat_list = list_chats()
    if _chat_list:
        _labels = {c["id"]: c["title"] for c in _chat_list}
        _ids = [c["id"] for c in _chat_list]
        _cur = st.session_state.get("active_chat_id")
        _idx = _ids.index(_cur) if _cur in _ids else 0
        _picked = st.radio(
            "Conversations",
            options=_ids,
            format_func=lambda cid: _labels.get(cid, cid),
            index=_idx,
            label_visibility="collapsed",
        )
        if _picked != st.session_state.get("active_chat_id"):
            _persist_active_chat()
            st.session_state.active_chat_id = _picked
            _loaded = load_chat(_picked) or {"messages": [], "history": []}
            st.session_state.messages = _loaded.get("messages", [])
            st.session_state.history = _loaded.get("history", [])
            st.rerun()
        with st.expander("✏️ Rename / delete"):
            _new_title = st.text_input(
                "Title",
                value=_labels.get(st.session_state.active_chat_id, ""),
                key="chat-rename",
            )
            _rc1, _rc2 = st.columns(2)
            with _rc1:
                if st.button("Rename", use_container_width=True):
                    rename_chat(
                        st.session_state.active_chat_id, _new_title or "Untitled"
                    )
                    st.rerun()
            with _rc2:
                if st.button("🗑️ Delete", use_container_width=True):
                    delete_chat(st.session_state.active_chat_id)
                    _remaining = list_chats()
                    if _remaining:
                        st.session_state.active_chat_id = _remaining[0]["id"]
                        _loaded = load_chat(_remaining[0]["id"]) or {}
                        st.session_state.messages = _loaded.get("messages", [])
                        st.session_state.history = _loaded.get("history", [])
                    else:
                        _chat = new_chat()
                        st.session_state.active_chat_id = _chat["id"]
                        st.session_state.messages = []
                        st.session_state.history = []
                    st.rerun()
    _search = st.text_input("🔍 Search chat", key="chat-search")
    if st.session_state.get("messages"):
        st.download_button(
            "📥 Export chat",
            data=_chat_markdown(st.session_state.messages),
            file_name="sidekick-chat.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.divider()
    st.subheader("Engine")
    st.markdown(
        f"**LLM** `{settings.ollama_model if settings.llm_provider == 'ollama' else settings.llm_provider}`  \n"
        f"**Embeddings** `{settings.embedding_provider}`  \n"
        f"**Retrieval** `{settings.retrieval}`"
        + (" + rerank" if settings.rerank_enabled else "")
    )
    if settings.fake or settings.llm_provider != "ollama":
        st.caption("🎭 Fake LLM (RAG_DEMO_FAKE=1)")
    elif _ollama_reachable(settings.ollama_base_url):
        st.caption("🟢 Ollama connected")
    else:
        st.caption("🔴 Ollama unreachable — start it with `ollama serve`")
    settings.hybrid_fallback = st.checkbox(
        "🌐 General-knowledge fallback",
        value=settings.hybrid_fallback,
        help="When the corpus has no answer, fall back to the LLM's general "
        "knowledge — clearly marked, never silently mixed with citations. "
        "Off = strict corpus-only (says 'I don't know').",
    )

    st.divider()
    with st.expander("🧪 Regression prompts"):
        st.caption(
            "Standard prompts through the full pipeline, with live checks. "
            "Targets the Meridian demo corpus."
        )
        if st.button("▶ Run all", key="reg-run-all", use_container_width=True):
            st.session_state["reg_run_all"] = True
            st.rerun()
        for _case in _load_regression_cases():
            if st.button(
                f"▶ {_case['id']}",
                key=f"reg-{_case['id']}",
                help=_case["question"],
                use_container_width=True,
            ):
                st.session_state.pending_question = _case["question"]
                st.session_state.pending_case = _case
                st.rerun()

    st.divider()
    st.subheader("📎 Upload a document")
    uploaded = st.file_uploader(
        "PDF, DOCX, TXT, MD, or CSV",
        type=["pdf", "docx", "txt", "md", "csv"],
        help="Extracted in memory only — nothing is written to disk or committed.",
    )
    if uploaded is not None:
        import os

        max_mb = float(os.environ.get("RAG_DEMO_UPLOAD_MAX_MB", "10"))
        if uploaded.size > max_mb * 1024 * 1024:
            st.error(f"File too large (over {max_mb:.0f} MB).")
        elif st.session_state.get("upload_name") != uploaded.name:
            from rag_demo.extract import UnsupportedFormat, extract_text

            data = uploaded.read()
            try:
                text = extract_text(uploaded.name, data)
            except UnsupportedFormat as e:
                st.error(str(e))
            except Exception as e:  # noqa: BLE001 — corrupt/encrypted files
                st.error(f"Couldn't read {uploaded.name}: {e}")
            else:
                ocr_used = False
                if not text.strip() and uploaded.name.lower().endswith(".pdf"):
                    from rag_demo.ocr import ocr_available, ocr_pdf

                    if ocr_available():
                        with st.spinner("No embedded text — running OCR…"):
                            try:
                                text = ocr_pdf(data)
                                ocr_used = bool(text.strip())
                            except RuntimeError as e:
                                st.warning(str(e))
                if not text.strip():
                    st.warning(
                        f"No text found in {uploaded.name}. It's likely a scanned "
                        "PDF — install OCR support (conda install -c conda-forge "
                        "tesseract poppler), or copy the text into a .txt file."
                    )
                else:
                    st.session_state.upload_name = uploaded.name
                    st.session_state.upload_text = text
                    st.session_state.upload_summary = None
                    st.session_state.upload_retriever = None
                    msg = f"Extracted {len(text):,} characters from {uploaded.name}."
                    if ocr_used:
                        msg += " (via OCR — may contain recognition errors)"
                    st.success(msg)
    if st.session_state.get("upload_text"):
        if st.button("📝 Summarize", use_container_width=True):
            from rag_demo.summarize import summarize

            with st.spinner("Summarizing…"):
                st.session_state.upload_summary = summarize(
                    st.session_state.upload_text, settings
                )
        if st.button("💬 Ask about this file", use_container_width=True):
            from langchain_community.vectorstores import FAISS
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            from rag_demo.retriever import Retriever

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
            )
            chunks = splitter.split_text(st.session_state.upload_text)
            source_name = f"upload:{st.session_state.upload_name}"
            store = FAISS.from_texts(
                chunks,
                get_embeddings(settings),
                metadatas=[
                    {"source": source_name, "allowed_groups": ["*"]} for _ in chunks
                ],
            )
            chunk_dicts = [
                {"text": c, "source": source_name, "allowed_groups": ["*"]}
                for c in chunks
            ]
            st.session_state.upload_retriever = Retriever(
                settings, store=store, chunks=chunk_dicts
            )
            st.session_state.corpus_override = "upload"
            st.rerun()
        if st.button("🗑️ Clear upload", use_container_width=True):
            for key in (
                "upload_name",
                "upload_text",
                "upload_summary",
                "upload_retriever",
                "corpus_override",
            ):
                st.session_state.pop(key, None)
            st.rerun()

# ------------------------------------------------------------------ state ---
# Per-chat state: the active chat file is the source of truth; session state
# holds the working copy.
if "active_chat_id" not in st.session_state:
    _chats = list_chats()
    if _chats:
        st.session_state.active_chat_id = _chats[0]["id"]
        _loaded = load_chat(_chats[0]["id"]) or {}
        st.session_state.messages = _loaded.get("messages", [])
        st.session_state.history = _loaded.get("history", [])
    else:
        _chat = new_chat()
        st.session_state.active_chat_id = _chat["id"]
        st.session_state.messages = []
        st.session_state.history = []

index_ok = settings.index_dir.exists() or upload_retriever is not None

# ------------------------------------------------------- regression run-all ---
if st.session_state.pop("reg_run_all", False) and index_ok:
    _cases = _load_regression_cases()
    _results: list[dict] = []
    _progress = st.progress(0, text="Running regression suite…")
    for _i, _case in enumerate(_cases):
        try:
            _ans = ask(
                _case["question"],
                settings,
                history=[tuple(h) for h in _case.get("history", [])],
                user_groups=groups,
                retriever=upload_retriever,
            )
            _checks = check_case(_case, _ans, real_llm=not settings.fake)
            _results.append(
                {
                    "id": _case["id"],
                    "question": _case["question"],
                    "answer": _ans.text,
                    "ok": all(_p for _, _p in _checks),
                    "checks": _checks,
                    "sources": _ans.sources,
                }
            )
        except Exception as e:  # noqa: BLE001 — a crash is a failed case
            _results.append({"id": _case["id"], "ok": False, "error": str(e)})
        _progress.progress((_i + 1) / len(_cases), text=f"Ran {_case['id']}…")
    _progress.empty()
    st.session_state.reg_results = _results
    st.rerun()

if st.session_state.get("reg_results"):
    _results = st.session_state.reg_results
    _passed = sum(1 for _r in _results if _r["ok"])
    with st.expander(
        f"🧪 Regression results: {_passed}/{len(_results)} passed", expanded=True
    ):
        for _r in _results:
            _mark = "✅" if _r["ok"] else "❌"
            if _r.get("error"):
                st.markdown(f"{_mark} `{_r['id']}` — crashed: {_r['error']}")
                continue
            with st.expander(f"{_mark} `{_r['id']}`", expanded=False):
                st.markdown(f"**Q:** {_r['question']}")
                st.markdown(_r["answer"])
                for _name, _passed_check in _r["checks"]:
                    st.markdown(f"{'✓' if _passed_check else '✗'} {_name}")
                st.caption(f"sources: {_r['sources']}")
        if st.button("Dismiss results"):
            st.session_state.pop("reg_results", None)
            st.rerun()

# ------------------------------------------------------------------ summary ---
if st.session_state.get("upload_summary"):
    with st.expander("📝 Document summary", expanded=True):
        st.markdown(st.session_state.upload_summary)
        if st.button("Dismiss summary"):
            st.session_state.upload_summary = None
            st.rerun()

# ------------------------------------------------------------------- hero ---
if not st.session_state.messages:
    st.markdown(
        '<div class="hero-title">📚 Sidekick RAG Assistant</div>'
        '<div class="hero-sub">Answers grounded in your indexed corpus — every fact cited.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
**What Sidekick can do:**
- 📚 Answer questions from your corpus — every fact cited to its source
- 🌐 Fall back to general knowledge when the corpus doesn't cover it (always marked)
- 📎 Read your uploaded documents — PDF, DOCX, TXT, MD, CSV (OCR for scanned PDFs)
"""
    )
    if not index_ok:
        st.warning(
            f"No index at `{settings.index_dir}`. Build it first:\n\n"
            "`PYTHONPATH=src python -m rag_demo.ingest --source " + corpus + "`"
        )
    st.write("Try one of these:")
    _sugs = SUGGESTIONS.get(corpus, [])
    for _row in range(0, len(_sugs), 2):
        cols = st.columns(2)
        for col, suggestion in zip(cols, _sugs[_row : _row + 2]):
            with col:
                if st.button(
                    suggestion,
                    key=f"sug-{suggestion[:20]}",
                    use_container_width=True,
                ):
                    st.session_state.pending_question = suggestion
                    st.rerun()

# --------------------------------------------------------------- history ---
def _hit_field(h, key):
    return h[key] if isinstance(h, dict) else getattr(h, key)


for msg in st.session_state.messages:
    if _search and _search.lower() not in msg.get("text", "").lower():
        continue
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "📚"):
        st.markdown(msg["text"])
        if msg.get("sources"):
            pills = "".join(
                f'<span class="citation-pill">{s}</span>' for s in msg["sources"]
            )
            st.markdown(pills, unsafe_allow_html=True)
        if msg.get("latency_ms") is not None:
            st.caption(
                f"⚡ {msg['latency_ms']/1000:.1f}s · {len(msg.get('hits', []))} passages"
            )
        if msg.get("grounded") is False:
            st.markdown(FALLBACK_BADGE, unsafe_allow_html=True)
        if msg.get("hits"):
            with st.expander("Retrieved passages"):
                for h in msg["hits"]:
                    st.markdown(
                        f"**[{_hit_field(h, 'source')}]** "
                        f"(score {_hit_field(h, 'score'):.3f})"
                    )
                    st.caption(_hit_field(h, "text")[:500])
        if msg["role"] == "assistant" and msg.get("text"):
            with st.expander("📋 Copyable answer"):
                st.code(msg["text"], language="markdown")
        if msg.get("checks"):
            _ok = all(_p for _, _p in msg["checks"])
            with st.expander(
                f"{'✅' if _ok else '❌'} Regression checks ({msg.get('case_id')})"
            ):
                for _name, _passed in msg["checks"]:
                    st.markdown(f"{'✓' if _passed else '✗'} {_name}")

# ------------------------------------------------------------------ chat ---
question = st.session_state.pop("pending_question", None) or st.chat_input(
    "Ask anything…", disabled=not index_ok
)
reg_case = st.session_state.pop("pending_case", None)
if question and index_ok:
    # Regression runs use the case's canned history (e.g. the follow-up case).
    ask_history = (
        [tuple(h) for h in reg_case.get("history", [])]
        if reg_case
        else st.session_state.history
    )
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="📚"):
        holder: dict = {}

        def token_stream():
            for event in stream_ask(
                question,
                settings,
                history=ask_history,
                user_groups=groups,
                retriever=upload_retriever,
            ):
                if "token" in event:
                    yield event["token"]
                else:
                    holder["answer"] = event["answer"]

        try:
            st.write_stream(token_stream())
        except Exception as e:  # noqa: BLE001 — show, don't crash the UI
            st.error(f"Couldn't answer: {e}")
            st.stop()

        answer = holder.get("answer")
        if answer is None:
            st.error("Couldn't answer: empty response from the chain.")
            st.stop()

        if answer.sources:
            pills = "".join(
                f'<span class="citation-pill">{s}</span>' for s in answer.sources
            )
            st.markdown(pills, unsafe_allow_html=True)
        if answer.latency_ms is not None:
            st.caption(
                f"⚡ {answer.latency_ms/1000:.1f}s · {len(answer.hits)} passages"
            )
        if not answer.grounded:
            st.markdown(FALLBACK_BADGE, unsafe_allow_html=True)
        if answer.standalone_question:
            st.caption(f"🔍 Understood as: {answer.standalone_question}")
        with st.expander("Retrieved passages"):
            for h in answer.hits:
                st.markdown(f"**[{h.source}]** (score {h.score:.3f})")
                st.caption(h.text[:500])
        with st.expander("📋 Copyable answer"):
            st.code(answer.text, language="markdown")
        fb_col1, fb_col2 = st.columns([1, 1])
        with fb_col1:
            if st.button("👍", key=f"up-{len(st.session_state.messages)}"):
                log_feedback(settings, question, "up")
                st.toast("Thanks for the feedback!")
        with fb_col2:
            if st.button("👎", key=f"down-{len(st.session_state.messages)}"):
                log_feedback(settings, question, "down")
                st.toast("Thanks — we'll use this to improve retrieval.")

    reg_checks = (
        check_case(reg_case, answer, real_llm=not settings.fake)
        if reg_case
        else None
    )
    if reg_checks:
        with st.expander(
            f"{'✅' if all(p for _, p in reg_checks) else '❌'} "
            f"Regression checks ({reg_case['id']})",
            expanded=True,
        ):
            for _name, _passed in reg_checks:
                st.markdown(f"{'✓' if _passed else '✗'} {_name}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "text": answer.text,
            "sources": answer.sources,
            "hits": answer.hits,
            "checks": reg_checks,
            "case_id": reg_case["id"] if reg_case else None,
            "latency_ms": answer.latency_ms,
            "grounded": answer.grounded,
        }
    )
    st.session_state.history.append((question, answer.text))
    st.session_state.history = st.session_state.history[-6:]
    _persist_active_chat()
