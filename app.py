"""Streamlit chat UI over the RAG index.

Run:  PYTHONPATH=src streamlit run app.py
Needs the index built first (`python -m rag_demo.ingest`) and Ollama running
for real answers (or RAG_DEMO_FAKE=1 for the fake LLM).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve() / "src"))

import streamlit as st

from rag_demo.chain import ask
from rag_demo.config import get_settings
from rag_demo.sources import SOURCES
from rag_demo.ingest import INDEX_DIRS

st.set_page_config(page_title="RAG Assistant Demo", page_icon="📚")

st.title("📚 RAG Assistant Demo")
st.caption(
    "Retrieval-augmented generation over an indexed corpus. "
    "Answers are constrained to retrieved context and cite their sources."
)

settings = get_settings()

CORPUS_LABELS = {
    "meridian": "Meridian Logistics (fictional, default)",
    "playwright-docs": "Playwright docs (optional index)",
    "github-docs": "GitHub repo docs (optional index)",
    "confluence-mock": "Confluence mock (optional index)",
}

with st.sidebar:
    st.header("Corpus")
    corpus = st.radio(
        "Which index to query",
        options=[c for c in SOURCES if c in CORPUS_LABELS],
        format_func=lambda c: CORPUS_LABELS.get(c, c),
        help="Build an index first: python -m rag_demo.ingest --source <name>",
    )
    if corpus != "meridian":
        settings.index_dir = settings.index_dir.parent / INDEX_DIRS[corpus]
    st.divider()
    st.caption(
        f"LLM: `{settings.llm_provider}`\n\nEmbeddings: `{settings.embedding_provider}`"
    )
    if st.button("Rebuild index"):
        st.info("Run `PYTHONPATH=src python -m rag_demo.ingest` in a terminal.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        if msg.get("hits"):
            with st.expander("Retrieved passages"):
                for h in msg["hits"]:
                    st.markdown(f"**[{h.source}]** (score {h.score:.3f})")
                    st.caption(h.text[:500])

if not settings.index_dir.exists():
    st.warning(
        f"No index at `{settings.index_dir}`. "
        "Build it first: `PYTHONPATH=src python -m rag_demo.ingest`"
    )
    st.stop()

question = st.chat_input("Ask about the corpus…")
if question:
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving…"):
            try:
                answer = ask(question, settings)
            except Exception as e:  # noqa: BLE001 — show, don't crash the UI
                st.error(f"Couldn't answer: {e}")
                st.stop()
        st.markdown(answer.text)
        if answer.sources:
            st.caption("Sources: " + ", ".join(f"`{s}`" for s in answer.sources))
        with st.expander("Retrieved passages"):
            for h in answer.hits:
                st.markdown(f"**[{h.source}]** (score {h.score:.3f})")
                st.caption(h.text[:500])
    st.session_state.messages.append(
        {"role": "assistant", "text": answer.text, "hits": answer.hits}
    )
