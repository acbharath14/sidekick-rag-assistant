"""Sidekick RAG Assistant — chat UI over the indexed corpus.

Run:  PYTHONPATH=src streamlit run app.py
Needs the index built first (`python -m rag_demo.ingest`) and Ollama running
for real answers (or RAG_DEMO_FAKE=1 for the fake LLM).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from rag_demo.chain import stream_ask
from rag_demo.config import get_settings
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

# ---------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.header("📚 Sidekick")
    corpus = st.radio(
        "Corpus",
        options=[c for c in SOURCES if c in CORPUS_LABELS],
        format_func=lambda c: CORPUS_LABELS.get(c, c),
        help="Build an index first: python -m rag_demo.ingest --source <name>",
    )
    if corpus != "meridian":
        settings.index_dir = settings.index_dir.parent / INDEX_DIRS[corpus]

    st.divider()
    st.subheader("Session")
    user_groups = st.text_input(
        "Your groups (comma-separated)",
        value=",".join(settings.user_groups),
        help="Permission-aware retrieval demo: try 'eng-all' vs 'eng-leads' on the Confluence corpus.",
    )
    groups = [g.strip() for g in user_groups.split(",") if g.strip()]
    if st.button("🧹 New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.subheader("Engine")
    st.markdown(
        f"**LLM** `{settings.ollama_model if settings.llm_provider == 'ollama' else settings.llm_provider}`  \n"
        f"**Embeddings** `{settings.embedding_provider}`  \n"
        f"**Retrieval** `{settings.retrieval}`"
        + (" + rerank" if settings.rerank_enabled else "")
    )

# ------------------------------------------------------------------ state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

index_ok = settings.index_dir.exists()

# ------------------------------------------------------------------- hero ---
if not st.session_state.messages:
    st.markdown(
        '<div class="hero-title">📚 Sidekick RAG Assistant</div>'
        '<div class="hero-sub">Answers grounded in your indexed corpus — every fact cited.</div>',
        unsafe_allow_html=True,
    )
    if not index_ok:
        st.warning(
            f"No index at `{settings.index_dir}`. Build it first:\n\n"
            "`PYTHONPATH=src python -m rag_demo.ingest --source " + corpus + "`"
        )
    st.write("Try one of these:")
    cols = st.columns(len(SUGGESTIONS.get(corpus, [])) or 1)
    for col, suggestion in zip(cols, SUGGESTIONS.get(corpus, [])):
        with col:
            if st.button(suggestion, key=f"sug-{suggestion[:20]}", use_container_width=True):
                st.session_state.pending_question = suggestion
                st.rerun()

# --------------------------------------------------------------- history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "📚"):
        st.markdown(msg["text"])
        if msg.get("sources"):
            pills = "".join(
                f'<span class="citation-pill">{s}</span>' for s in msg["sources"]
            )
            st.markdown(pills, unsafe_allow_html=True)
        if msg.get("hits"):
            with st.expander("Retrieved passages"):
                for h in msg["hits"]:
                    st.markdown(f"**[{h.source}]** (score {h.score:.3f})")
                    st.caption(h.text[:500])
        if msg["role"] == "assistant" and msg.get("text"):
            with st.expander("📋 Copyable answer"):
                st.code(msg["text"], language="markdown")

# ------------------------------------------------------------------ chat ---
question = st.session_state.pop("pending_question", None) or st.chat_input(
    "Ask about the corpus…", disabled=not index_ok
)
if question and index_ok:
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="📚"):
        answer = None

        def token_stream():
            nonlocal answer
            for event in stream_ask(
                question,
                settings,
                history=st.session_state.history,
                user_groups=groups,
            ):
                if "token" in event:
                    yield event["token"]
                else:
                    answer = event["answer"]

        try:
            st.write_stream(token_stream())
        except Exception as e:  # noqa: BLE001 — show, don't crash the UI
            st.error(f"Couldn't answer: {e}")
            st.stop()

        if answer is None:
            st.error("Couldn't answer: empty response from the chain.")
            st.stop()

        if answer.sources:
            pills = "".join(
                f'<span class="citation-pill">{s}</span>' for s in answer.sources
            )
            st.markdown(pills, unsafe_allow_html=True)
        with st.expander("Retrieved passages"):
            for h in answer.hits:
                st.markdown(f"**[{h.source}]** (score {h.score:.3f})")
                st.caption(h.text[:500])
        with st.expander("📋 Copyable answer"):
            st.code(answer.text, language="markdown")
        fb_col1, fb_col2 = st.columns([1, 1])
        with fb_col1:
            if st.button("👍", key=f"up-{len(st.session_state.messages)}"):
                st.toast("Thanks for the feedback!")
        with fb_col2:
            if st.button("👎", key=f"down-{len(st.session_state.messages)}"):
                st.toast("Thanks — we'll use this to improve retrieval.")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "text": answer.text,
            "sources": answer.sources,
            "hits": answer.hits,
        }
    )
    st.session_state.history.append((question, answer.text))
    st.session_state.history = st.session_state.history[-6:]
