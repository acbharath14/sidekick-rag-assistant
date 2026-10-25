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
from rag_demo.modes import (
    CORPUS_DISPLAY,
    DEMO_CORPORA,
    WORKSPACE_CORPORA,
    WORKSPACE_ENTERPRISE_SOURCES,
    connected_sources,
    default_mode,
    source_status,
)
from rag_demo.preflight import run_preflight
from rag_demo.walkthrough import SPECIMEN_TEXT, run_walkthrough
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
    /* ---------- Base ---------- */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ---------- Citation pills ---------- */
    .citation-pill {
        display: inline-block;
        background: linear-gradient(135deg, #6366F1, #8B5CF6);
        color: white;
        border-radius: 999px;
        padding: 2px 12px;
        margin: 3px 6px 3px 0;
        font-size: 0.75rem;
        font-weight: 500;
        font-family: monospace;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
    }

    /* ---------- Hero / welcome ---------- */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        background: linear-gradient(135deg, #A5B4FC, #C4B5FD);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        color: #94A3B8;
        font-size: 1.15rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .capability-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s, transform 0.2s;
    }
    .capability-card:hover {
        border-color: #6366F1;
        transform: translateY(-2px);
    }

    /* ---------- Chat messages ---------- */
    [data-testid="stChatMessage"] {
        animation: fadeIn 0.3s ease-in;
        padding: 1rem 1.2rem !important;
        border-radius: 16px !important;
        margin-bottom: 0.8rem !important;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: #0B1220 !important;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #C4B5FD !important;
        font-weight: 700 !important;
        margin-top: 1.2rem !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #1E293B !important;
        margin: 1rem 0 !important;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
        border: 1px solid #334155 !important;
    }
    .stButton > button:hover {
        border-color: #6366F1 !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25) !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
        border: none !important;
    }

    /* ---------- Expanders ---------- */
    [data-testid="stExpander"] {
        border: 1px solid #1E293B !important;
        border-radius: 10px !important;
        background: #0F172A !important;
    }

    /* ---------- Chat input ---------- */
    [data-testid="stChatInput"] {
        border-radius: 16px !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }

    /* ---------- Thinking indicator ---------- */
    .thinking-dots::after {
        content: '';
        animation: dots 1.2s infinite;
    }
    @keyframes dots {
        0%, 20% { content: ''; }
        40% { content: '.'; }
        60% { content: '..'; }
        80%, 100% { content: '...'; }
    }

    /* ---------- Scrollbar ---------- */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0F172A;
    }
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #475569;
    }
</style>
""",
    unsafe_allow_html=True,
)

settings = get_settings()

# Phase 4 (U1+U2): source labels live in modes.py (Demo labels carry the
# fictional marking; the GitHub label names the repo explicitly).
# CORPUS_DISPLAY is mutated with the "upload" entry at runtime, exactly
# as the old CORPUS_LABELS dict was.

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

FALLBACK_BADGE = '<span class="citation-pill">🌐 General knowledge</span>'


def _demo_sources_caption(is_demo: bool) -> None:
    """Fictional-data marking above demo citations (Phase 4, U1+U2):
    demo answers wear the marking on their sources, not just on the mode
    banner, so a copied answer cannot pass as real-company evidence."""
    if is_demo:
        st.caption("🎭 Demo sources — fictional data, for trial only.")


def _chat_markdown(messages: list[dict], demo: bool = False) -> str:
    lines = ["# Sidekick chat export", ""]
    if demo:
        lines.append("*Demo mode: sources below are fictional data, for trial only.*")
        lines.append("")
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


def _enterprise_ui():
    """Sidebar UI for enterprise connector credentials and index building."""
    import os

    # --- Confluence ---
    with st.expander("📝 Confluence", expanded=False):
        c_url = st.text_input(
            "URL", value=st.session_state.get("ent_confluence_url", ""),
            placeholder="https://your-domain.atlassian.net/wiki",
            key="ent_c_url",
        )
        c_token = st.text_input(
            "Token (PAT)", value=st.session_state.get("ent_confluence_token", ""),
            type="password", key="ent_c_token",
        )
        c_space = st.text_input(
            "Space key (optional)", value=st.session_state.get("ent_confluence_space", ""),
            key="ent_c_space",
        )
        if st.button("Build Confluence index", key="ent_c_build"):
            st.session_state["ent_confluence_url"] = c_url
            st.session_state["ent_confluence_token"] = c_token
            st.session_state["ent_confluence_space"] = c_space
            os.environ["RAG_DEMO_CONFLUENCE_URL"] = c_url
            os.environ["RAG_DEMO_CONFLUENCE_TOKEN"] = c_token
            if c_space:
                os.environ["RAG_DEMO_CONFLUENCE_SPACE"] = c_space
            _build_enterprise_index("confluence")

    # --- Jira ---
    with st.expander("🎫 Jira", expanded=False):
        j_url = st.text_input(
            "URL", value=st.session_state.get("ent_jira_url", ""),
            placeholder="https://your-domain.atlassian.net",
            key="ent_j_url",
        )
        j_token = st.text_input(
            "Token (PAT)", value=st.session_state.get("ent_jira_token", ""),
            type="password", key="ent_j_token",
        )
        j_jql = st.text_input(
            "JQL (optional)", value=st.session_state.get("ent_jira_jql", ""),
            placeholder="updated >= -30d ORDER BY updated DESC",
            key="ent_j_jql",
        )
        if st.button("Build Jira index", key="ent_j_build"):
            st.session_state["ent_jira_url"] = j_url
            st.session_state["ent_jira_token"] = j_token
            st.session_state["ent_jira_jql"] = j_jql
            os.environ["RAG_DEMO_JIRA_URL"] = j_url
            os.environ["RAG_DEMO_JIRA_TOKEN"] = j_token
            if j_jql:
                os.environ["RAG_DEMO_JIRA_JQL"] = j_jql
            _build_enterprise_index("jira")

    # --- Outlook ---
    with st.expander("📧 Outlook", expanded=False):
        o_token = st.text_input(
            "Graph API token", value=st.session_state.get("ent_outlook_token", ""),
            type="password", key="ent_o_token",
            help="OAuth2 bearer token for Microsoft Graph",
        )
        o_folder = st.text_input(
            "Folder", value=st.session_state.get("ent_outlook_folder", "inbox"),
            key="ent_o_folder",
        )
        if st.button("Build Outlook index", key="ent_o_build"):
            st.session_state["ent_outlook_token"] = o_token
            st.session_state["ent_outlook_folder"] = o_folder
            os.environ["RAG_DEMO_OUTLOOK_TOKEN"] = o_token
            os.environ["RAG_DEMO_OUTLOOK_FOLDER"] = o_folder
            _build_enterprise_index("outlook")

    # --- AWS S3 ---
    with st.expander("☁️ AWS S3", expanded=False):
        s3_bucket = st.text_input(
            "Bucket", value=st.session_state.get("ent_s3_bucket", ""),
            key="ent_s3_bucket",
        )
        s3_profile = st.text_input(
            "Profile (for SSO/federation)", value=st.session_state.get("ent_s3_profile", ""),
            placeholder="my-sso-profile",
            help="AWS profile name. Run `aws sso login --profile NAME` first.",
            key="ent_s3_profile",
        )
        s3_prefix = st.text_input(
            "Prefix filter (optional)", value=st.session_state.get("ent_s3_prefix", ""),
            key="ent_s3_prefix",
        )
        if st.button("Build S3 index", key="ent_s3_build"):
            st.session_state["ent_s3_bucket"] = s3_bucket
            st.session_state["ent_s3_profile"] = s3_profile
            st.session_state["ent_s3_prefix"] = s3_prefix
            os.environ["RAG_DEMO_AWS_S3_BUCKET"] = s3_bucket
            if s3_profile:
                os.environ["RAG_DEMO_AWS_PROFILE"] = s3_profile
            if s3_prefix:
                os.environ["RAG_DEMO_AWS_S3_PREFIX"] = s3_prefix
            _build_enterprise_index("aws-s3")


def _build_enterprise_index(source: str):
    """Build the FAISS index for an enterprise source."""
    from rag_demo.ingest import INDEX_DIRS
    from rag_demo.config import get_settings
    from rag_demo.sources import SOURCES
    from rag_demo.ingest import build_index

    settings = get_settings()
    settings.index_dir = settings.index_dir.parent / INDEX_DIRS[source]
    source_cls = SOURCES[source]
    src = source_cls()

    with st.spinner(f"Building {source} index..."):
        try:
            store = build_index(settings, src)
            if store:
                # Phase 1 (S1): a rebuild in place must not leave the
                # cached retriever serving the old index (see caching.py
                # invalidation contract). Only on success — a failed
                # build leaves the on-disk index (and cache) untouched.
                from rag_demo.caching import clear_cache

                clear_cache()
                st.success(f"✅ Indexed {store.index.ntotal} chunks from {source}")
                st.info(f"Select '{source}' in the Corpus dropdown to query it.")
            else:
                st.warning("No documents indexed.")
        except Exception as e:
            st.error(f"Failed: {e}")


def _build_demo_index() -> bool:
    """Build the demo corpora (Meridian + Confluence mock) in the UI —
    the same path as `python -m rag_demo.ingest`, minus the terminal
    (Phase 4, U1+U2). Returns True when both indexes built; on failure
    the on-disk index (and cache) is left untouched, like the
    enterprise build above. Skips an index that already exists."""
    from rag_demo.ingest import build_index
    from rag_demo.sources import MarkdownDirectorySource, MockConfluenceSource
    from rag_demo.preflight import index_format_current, index_present

    settings = get_settings()
    demo_sources = [
        ("meridian", MarkdownDirectorySource(settings.docs_dir)),
        ("confluence-mock", MockConfluenceSource()),
    ]
    with st.status("Building demo indexes…", expanded=True) as status:
        for source_key, src in demo_sources:
            index_dir = settings.index_dir.parent / INDEX_DIRS[source_key]
            if index_present(index_dir) and index_format_current(index_dir):
                st.write(f"{source_key}: index already exists — skipping.")
                continue
            if index_present(index_dir):
                # Phase 5: a stale index is rebuilt, not skipped — skipping
                # it would leave filename-only citations in place.
                st.write(f"{source_key}: index is from an older version — rebuilding.")
            st.write(f"Indexing {source_key}…")
            build_settings = get_settings()
            build_settings.index_dir = index_dir
            try:
                store = build_index(build_settings, src)
            except Exception as e:
                status.update(label="❌ Demo index build failed", state="error")
                st.error(f"Failed building {source_key}: {e}")
                return False
            if store:
                st.write(f"{source_key}: indexed {store.index.ntotal} chunks.")
            else:
                st.warning(f"{source_key}: no documents indexed.")
        status.update(label="✅ Demo indexes ready", state="complete")
    # A rebuild in place must not leave the cached retriever serving
    # the old index (see caching.py invalidation contract).
    from rag_demo.caching import clear_cache

    clear_cache()
    st.success("✅ Demo indexes built. Chat input is now enabled.")
    return True


with st.sidebar:
    st.header("📚 Sidekick")
    # --- Mode switch: one visible control, always on top ---
    # "Connected" = a workspace source with a built index on disk
    # (see modes.py: credentials alone do not count).
    _connected = connected_sources(settings.index_dir.parent, INDEX_DIRS)
    if "mode-selector" not in st.session_state:
        st.session_state["mode-selector"] = default_mode(_connected)
    app_mode = st.radio(
        "Mode",
        options=["demo", "workspace"],
        format_func=lambda m: (
            "🎭 Demo (fictional data)" if m == "demo" else "💼 Workspace"
        ),
        key="mode-selector",
        help="Demo serves fictional data for trial. Workspace serves your own sources.",
    )
    if app_mode == "demo":
        st.caption("🎭 Demo mode: all data here is fictional, for trial only.")
    else:
        st.caption("💼 Workspace: your own documentation and enterprise sources.")
    # Switching mode resets the source picker to that mode's first
    # option — otherwise the old mode's corpus would linger as a stale
    # widget value that is not in the new mode's options.
    _prev_mode = st.session_state.get("_prev_mode")
    if _prev_mode is not None and _prev_mode != app_mode:
        st.session_state.pop("corpus-selector", None)
    st.session_state["_prev_mode"] = app_mode

    tab_chat, tab_sources, tab_settings = st.tabs(
        ["💬 Chat", "📚 Sources", "⚙️ Settings"]
    )

    with tab_chat:
        # Agentic mode toggle (ReAct: Thought → Action → Observation).
        agentic_mode = st.toggle(
            "🤖 Agentic mode",
            value=st.session_state.get("agentic_mode", False),
            help="Multi-step reasoning: plans searches, executes tools, verifies, refines.",
            key="agentic-toggle",
        )
        st.session_state["agentic_mode"] = agentic_mode
        if agentic_mode:
            show_plan = st.toggle(
                "Show plan before executing",
                value=st.session_state.get("agentic_show_plan", True),
                help="Display the agent's plan as a flowchart for approval.",
                key="agentic-show-plan",
            )
            st.session_state["agentic_show_plan"] = show_plan

        if st.button(
            "🔍 How it works — pipeline walkthrough",
            use_container_width=True,
            help="See a document move through chunking → embedding → retrieval, with scores. No chat, no generation — just the pipeline.",
        ):
            st.session_state["show_walkthrough"] = True
            st.rerun()

        st.subheader("Session")
        user_groups = st.text_input(
            "Your groups (comma-separated)",
            value=",".join(settings.user_groups),
            help="Permission-aware retrieval demo: try 'eng-all' vs 'eng-leads' on the Confluence corpus.",
        )
        groups = [g.strip() for g in user_groups.split(",") if g.strip()]
        if st.button("➕ New chat", use_container_width=True):
            _persist_active_chat()
            _chat = new_chat()
            st.session_state.active_chat_id = _chat["id"]
            st.session_state.messages = []
            st.session_state.history = []
            # Explicitly set the radio widget to the new chat (not just clear it)
            # to prevent stale selection from reloading old messages on next rerun.
            st.session_state["chat-selector"] = _chat["id"]
            st.rerun()

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
                key="chat-selector",
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
                        # Explicitly sync the radio widget to prevent stale selection.
                        st.session_state["chat-selector"] = st.session_state.active_chat_id
                        st.rerun()
        _search = st.text_input("🔍 Search chat", key="chat-search")
        if st.session_state.get("messages"):
            st.download_button(
                "📥 Export chat",
                data=_chat_markdown(
                    st.session_state.messages, demo=(app_mode == "demo")
                ),
                file_name="sidekick-chat.md",
                mime="text/markdown",
                use_container_width=True,
            )

    with tab_sources:
        st.subheader("📚 Sources")
        if app_mode == "demo":
            # Demo mode: fictional corpora only, never a workspace source.
            st.caption(
                "🎭 Fictional demo data — for trial only, not real company data."
            )
            if st.button(
                "🔨 Build demo index",
                key="build_demo_index",
                help="Build the Meridian + Confluence mock demo indexes inside the app — no terminal needed. Chat input enables automatically when the build finishes.",
            ):
                if _build_demo_index():
                    st.rerun()
            for _src in DEMO_CORPORA:
                _dir = settings.index_dir.parent / INDEX_DIRS[_src]
                st.caption(f"{CORPUS_DISPLAY[_src]} — {source_status(_dir.exists())}")
            mode_options = [c for c in DEMO_CORPORA if c in SOURCES]
        else:
            # Workspace mode: Documentation + Your workspace. The
            # workspace path never shows a demo corpus.
            st.subheader("Documentation")
            st.caption(
                "Public docs. GitHub repo docs are this repo "
                "(sidekick-rag-assistant) itself; the repo indexed comes from "
                "RAG_DEMO_GITHUB_REPO."
            )
            st.subheader("Your workspace")
            for _src in WORKSPACE_ENTERPRISE_SOURCES:
                _dir = settings.index_dir.parent / INDEX_DIRS[_src]
                _line = f"{CORPUS_DISPLAY[_src]} — {source_status(_dir.exists())}"
                if not _dir.exists():
                    _line += " (set up below)"
                st.caption(_line)
            mode_options = [c for c in WORKSPACE_CORPORA if c in SOURCES]
        if st.session_state.get("upload_retriever") is not None:
            mode_options.append("upload")
            CORPUS_DISPLAY["upload"] = (
                f"Uploaded: {st.session_state.get('upload_name', 'file')}"
            )
        _default_corpus = st.session_state.pop("corpus_override", None)
        if _default_corpus not in mode_options:
            _default_corpus = mode_options[0]
        corpus = st.radio(
            "Demo corpus" if app_mode == "demo" else "Source",
            options=mode_options,
            format_func=lambda c: (
                CORPUS_DISPLAY.get(c, c)
                + (
                    f" — {source_status((settings.index_dir.parent / INDEX_DIRS[c]).exists())}"
                    if c in WORKSPACE_ENTERPRISE_SOURCES
                    else ""
                )
            ),
            index=mode_options.index(_default_corpus),
            help="Build an index first: python -m rag_demo.ingest --source <name>",
            key="corpus-selector",
        )

        if app_mode == "workspace":
            # Enterprise connectors: PAT/token configuration + index building.
            # Tokens are kept in session state (in-memory only, cleared on restart).
            with st.expander("🔌 Enterprise Connectors", expanded=False):
                st.caption("⚠️ Tokens are stored in-memory only and cleared on restart. Never commit them.")
                _enterprise_ui()

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
                    try:
                        result = summarize(
                            st.session_state.upload_text, settings
                        )
                        if not result or not result.strip():
                            st.error("Summarize returned empty — check the terminal for errors.")
                        else:
                            st.session_state.upload_summary = result
                    except Exception as e:
                        st.error(f"Summarize failed: {type(e).__name__}: {e}")
                        # Also print to terminal for debugging
                        print(f"[summarize] FAILED: {type(e).__name__}: {e}")
            if st.button("💬 Chat with this file", use_container_width=True,
                           help="Index this file and switch the chatbot to answer from it"):
                from langchain_community.vectorstores import FAISS
                from langchain_text_splitters import RecursiveCharacterTextSplitter

                from rag_demo.retriever import Retriever

                with st.spinner(f"Indexing {st.session_state.upload_name}…"):
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
                    # Explicitly sync the corpus radio widget.
                    st.session_state["corpus-selector"] = "upload"
                st.success(
                    f"Now chatting with **{st.session_state.upload_name}** "
                    f"({len(chunks)} chunks). Ask anything about it below. "
                    "Use the Sources tab to switch back."
                )
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

    # --- Post-selection: the selected corpus drives the index dir, and
    # the pre-flight result drives chat readiness (U3). Computed once
    # per rerun, before the Settings tab renders (tab order on screen
    # follows declaration, not render order).
    upload_retriever = (
        st.session_state.get("upload_retriever") if corpus == "upload" else None
    )
    if corpus != "meridian" and corpus != "upload":
        settings.index_dir = settings.index_dir.parent / INDEX_DIRS[corpus]
    preflight = run_preflight(settings, corpus, settings.index_dir)
    chat_ready = preflight.all_ok or upload_retriever is not None

    with tab_settings:
        st.subheader("⚙️ Settings")
        # Retrieval toggles (Phase 3 defaults, Phase 4 UI). Wired to the
        # Phase 1 cache-invalidation contract: retrieval and rerank are
        # part of the cached-Retriever key, so changing either yields a
        # fresh retriever automatically — no stale index can be served.
        # The widget values persist in session state across reruns; the
        # first render seeds them from settings (env overrides intact).
        if "settings-retrieval" not in st.session_state:
            st.session_state["settings-retrieval"] = settings.retrieval
        settings.retrieval = st.selectbox(
            "Retrieval mode",
            options=["hybrid", "dense"],
            key="settings-retrieval",
            help="Hybrid runs BM25 + dense and fuses ranks (default). Dense is pure vector search.",
        )
        if "settings-rerank" not in st.session_state:
            st.session_state["settings-rerank"] = settings.rerank_enabled
        settings.rerank_enabled = st.checkbox(
            "Rerank results (cross-encoder)",
            key="settings-rerank",
            disabled=settings.fake,
            help="Re-scores the top candidates for finer ranking. Downloads a small model on first use; disabled under RAG_DEMO_FAKE=1 so tests never touch torch.",
        )
        settings.hybrid_fallback = st.checkbox(
            "🌐 General-knowledge fallback",
            value=settings.hybrid_fallback,
            help="When the corpus has no answer, fall back to the LLM's general "
            "knowledge — clearly marked, never silently mixed with citations. "
            "Off = strict corpus-only (says 'I don't know').",
        )

        st.subheader("Engine")
        st.markdown(
            f"**LLM** `{settings.ollama_model if settings.llm_provider == 'ollama' else settings.llm_provider}`  \n"
            f"**Embeddings** `{settings.embedding_provider}`  \n"
            f"**Retrieval** `{settings.retrieval}`"
            + (" + rerank" if settings.rerank_enabled else "")
        )
        st.caption(
            "Model is set via RAG_DEMO_MODEL (default qwen3:8b); "
            "there is no in-app model switcher yet."
        )
        if settings.fake or settings.llm_provider != "ollama":
            st.caption("🎭 Fake LLM (RAG_DEMO_FAKE=1)")

        st.subheader("Diagnostics")
        st.caption(
            "🟢 Ollama connected" if preflight.ollama_ok
            else "🔴 Ollama unreachable — start it with `ollama serve`"
        )
        st.caption(
            f"🤖 Model `{settings.ollama_model}` present" if preflight.model_ok
            else f"🤖 Model `{settings.ollama_model}` missing — run `ollama pull {settings.ollama_model}`"
        )
        st.caption(
            f"📚 Index for '{corpus}' present" if preflight.index_ok
            else f"📚 Index for '{corpus}' missing — build it from the Sources tab"
        )

        st.subheader("Web search")
        # Tavily API key for reliable web search.
        tavily_key = st.text_input(
            "Tavily API key",
            value=st.session_state.get("tavily_key", ""),
            type="password",
            help="Get a free key at tavily.com (1,000 searches/month). Enables reliable web search.",
            key="tavily_key_input",
        )
        if tavily_key:
            import os
            os.environ["RAG_DEMO_TAVILY_API_KEY"] = tavily_key
            st.session_state["tavily_key"] = tavily_key
        # Web search diagnostic.
        if st.button("🌐 Test web search", key="test_web_search",
                     help="Check which search backend is active and if it works"):
            with st.spinner("Testing..."):
                import os, requests
                tavily_key = os.environ.get("RAG_DEMO_TAVILY_API_KEY")
                if tavily_key:
                    # Test Tavily backend.
                    try:
                        resp = requests.post(
                            "https://api.tavily.com/search",
                            headers={"Content-Type": "application/json"},
                            json={"api_key": tavily_key, "query": "test",
                                  "max_results": 1},
                            timeout=15,
                        )
                        if resp.status_code == 200:
                            n = len(resp.json().get("results", []))
                            st.success(f"✅ Tavily working ({n} result(s) for test query)")
                        elif resp.status_code == 401:
                            st.error("❌ Tavily: invalid API key (401)")
                        else:
                            st.error(f"❌ Tavily HTTP {resp.status_code}: "
                                     f"{resp.text[:200]}")
                    except requests.Timeout:
                        st.error("❌ Tavily timeout — network issue")
                    except Exception as e:
                        st.error(f"❌ Tavily error: {e}")
                else:
                    st.info("ℹ️ No Tavily key set — using DuckDuckGo fallback")
                    try:
                        resp = requests.post(
                            "https://html.duckduckgo.com/html/",
                            data={"q": "test"},
                            headers={"User-Agent": "Mozilla/5.0"},
                            timeout=10,
                        )
                        if resp.status_code == 200 and "result__a" in resp.text:
                            st.success("✅ Web search working")
                        elif resp.status_code == 200:
                            st.warning("⚠️ Reached DuckDuckGo but no results parsed "
                                       "(HTML changed?)")
                        else:
                            st.error(f"❌ HTTP {resp.status_code}")
                    except requests.Timeout:
                        st.error("❌ Timeout — network slow or DuckDuckGo blocked")
                    except requests.ConnectionError as e:
                        st.error(f"❌ Connection failed: {e}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        # Search cache management.
        if st.button("🗑️ Clear search cache", key="clear_search_cache",
                     help="Clear cached web search results (they expire after 30 min anyway)"):
            try:
                from rag_demo.agent import _search_cache
                n = len(_search_cache)
                _search_cache.clear()
                st.success(f"✅ Cleared {n} cached result(s)")
            except Exception:
                st.info("Cache is empty")
        if st.button("🗑️ Clear model/index cache", key="clear_model_cache",
                     help="Drop cached embeddings, retrievers, and episodic memory (Phase 1). Next question rebuilds them."):
            from rag_demo.caching import clear_cache

            clear_cache()
            st.success("✅ Cleared cached models and indexes")

        st.subheader("🧪 Regression prompts")
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

# Phase 1 (S1): one cached retriever for the default-corpus path. The
# upload retriever is session-specific and bypasses the cache by design.
# Guarded: if cached construction fails (e.g. hybrid mode without
# chunks.json), fall back to per-question construction so the failure
# surfaces at ask time exactly as before, instead of crashing the render.
active_retriever = upload_retriever
if active_retriever is None and index_ok:
    try:
        from rag_demo.caching import get_cached_retriever

        active_retriever = get_cached_retriever(settings)
    except Exception:
        active_retriever = None

# ------------------------------------------- context + pre-flight banners ---
# Phase 4 (U1+U2+U3): the always-on context indicator moved out of the
# sidebar so it sits above the chat, next to the persistent Demo-mode
# fictional marking and the pre-flight fix-it banner. `preflight` and
# `chat_ready` come from the sidebar's post-selection section.
if app_mode == "demo":
    st.warning(
        "🎭 Demo mode — answers come from fictional demo data, for trial only. "
        "Switch to Workspace in the sidebar for your own sources.",
        icon="🎭",
    )
_corpus_descriptions = {
    "meridian": "📚 **Meridian docs (fictional demo)** — logistics API, deployment, security",
    "playwright-docs": "🎭 **Playwright docs** — browser automation",
    "confluence-mock": "📝 **Confluence mock (fictional demo)** — simulated wiki pages",
    "github-docs": "🐙 **GitHub repo docs (sidekick-rag-assistant)** — repository markdown files",
    "confluence": "📝 **Confluence** — real wiki pages",
    "jira": "🎫 **Jira** — issues and tickets",
    "outlook": "📧 **Outlook** — emails",
    "aws-s3": "☁️ **AWS S3** — bucket documents",
}
if corpus == "upload":
    _ctx_label = (
        f"📄 **Uploaded file: {st.session_state.get('upload_name', 'file')}** — "
        "answers come from this file only"
    )
else:
    _ctx_label = _corpus_descriptions.get(corpus, f"📚 **{corpus}**")
st.info(
    f"{_ctx_label} · "
    "Change the **Mode** or source in the sidebar's Sources tab to switch context.",
    icon="🧭",
)
if not chat_ready:
    st.error("Chat is disabled until the items below are fixed:")
    for _problem in preflight.problems:
        st.markdown(f"- {_problem}")
# Phase 5 (A2+A3): a stale index still answers, so this is a warning,
# not a chat blocker — but it must be prominent, never silent.
if getattr(preflight, "index_stale", False):
    st.warning(
        "⚠️ This index was built before chunk-level citations — answers cite "
        "filenames only. Rebuild it for passage citations: "
        f"`PYTHONPATH=src python -m rag_demo.ingest --source {corpus}`, or use "
        "Build demo index in the Sources tab.",
        icon="⚠️",
    )

# --------------------------------------------------- pipeline walkthrough ---
# Follow-up (not a phase): a "How it works" view showing one document move
# through chunking → embedding → retrieval, with scores. It uses its own
# document (specimen or pasted text), never the current corpus, and stops
# at retrieval — no generation, no summarize step. The view switch uses
# st.stop(), the same pattern as the agentic plan-approval flow below, so
# the chat UI never renders underneath the walkthrough. Results are cached
# in session state keyed by document + query + retrieval settings, so a
# rerun (e.g. collapsing an expander) does not re-embed.
def _render_walkthrough(settings):
    import hashlib

    st.header("🔍 How it works — the pipeline, stage by stage")
    st.caption(
        "This walkthrough uses its own document — not the current corpus — "
        "and stops at retrieval. There is no generation or summarize step "
        "here; ask the same question in chat to see the full flow."
    )
    _choice = st.radio(
        "Document",
        ["Specimen document", "Paste your own text"],
        horizontal=True,
        key="walkthrough-doc-choice",
    )
    if _choice == "Specimen document":
        _doc_text = SPECIMEN_TEXT
        st.caption(
            "Specimen — written to demonstrate the pipeline. Not corpus content."
        )
        with st.expander("Specimen document"):
            st.code(_doc_text, language="markdown")
    else:
        _doc_text = st.text_area(
            "Paste markdown or plain text",
            height=200,
            key="walkthrough-doc-text",
        )
    _query = st.text_input(
        "Question to retrieve with",
        value="How does chunking work?",
        key="walkthrough-query",
    )
    if st.button("▶ Run walkthrough", type="primary"):
        _key = hashlib.sha256(
            f"{_doc_text}\n{_query}\n{settings.retrieval}\n"
            f"{settings.rerank_enabled}\n{settings.relevance_floor}\n"
            f"{settings.top_k}".encode()
        ).hexdigest()
        if st.session_state.get("walkthrough-key") != _key:
            try:
                st.session_state["walkthrough-result"] = run_walkthrough(
                    _doc_text, "walkthrough-doc", _query, settings
                )
                st.session_state["walkthrough-key"] = _key
            except Exception as e:  # noqa: BLE001 — show, don't crash the UI
                st.error(f"Walkthrough failed: {e}")
                st.session_state.pop("walkthrough-result", None)
    _result = st.session_state.get("walkthrough-result")
    if _result is None:
        st.info(
            "Press “Run walkthrough” to see the document move through "
            "chunking → embedding → retrieval."
        )
        return
    st.subheader(
        f"1 · Chunking — {_result.chunk_count} chunks "
        f"(size {settings.chunk_size}, overlap {settings.chunk_overlap})"
    )
    for _c in _result.chunks:
        with st.expander(
            f"`{_c.chunk_id}` · {_c.section or 'no section'} · {_c.chars} chars"
        ):
            st.caption(_c.text)
    st.subheader("2 · Embedding")
    _emb = _result.embedding
    if _emb is None:
        st.warning("No chunks to embed.")
    else:
        if _emb.fake:
            st.warning(
                "Fake embeddings are active (RAG_DEMO_FAKE=1): vectors below "
                "show shape only — scores in stage 3 are not meaningful."
            )
        st.write(
            f"Model: `{_emb.model}` · provider: `{_emb.provider}` · "
            f"dimension: {_emb.dim} · vectors: {_emb.vector_count}"
        )
        st.caption(f"First chunk's vector, first 5 components: {_emb.preview}")
    st.subheader(f"3 · Retrieval — query: “{_result.query}”")
    st.caption(
        f"Stages that ran: {', '.join(_result.retrieval_stages)} · "
        f"candidates: {_result.candidates} · "
        f"dropped by floor: {_result.floor_dropped}"
    )
    if not _result.hits:
        st.warning("No passages passed retrieval.")
    for _h in _result.hits:
        with st.expander(
            f"#{_h.rank} · {_h.confidence} confidence · [{_h.citation}]"
        ):
            st.caption(f"raw score: {_h.score:.3f} · chunk: `{_h.chunk_id}`")
            st.caption(_h.text)


if st.session_state.get("show_walkthrough"):
    _render_walkthrough(settings)
    if st.button("← Back to chat"):
        st.session_state["show_walkthrough"] = False
        st.rerun()
    st.stop()

# ------------------------------------------------------- regression run-all ---
if st.session_state.pop("reg_run_all", False) and chat_ready:
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
                retriever=active_retriever,
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
<div class="capability-card">
📚 <strong>Corpus Q&A</strong> — Answers grounded in your documents, every fact cited to its source.
</div>
<div class="capability-card">
🌐 <strong>General knowledge</strong> — Falls back when the corpus doesn't cover it. Always clearly marked.
</div>
<div class="capability-card">
📎 <strong>Your files</strong> — Upload PDF, DOCX, TXT, MD, CSV. Scanned PDFs get OCR automatically.
</div>
""",
        unsafe_allow_html=True,
    )
    # No index warning here: the pre-flight fix-it banner above covers
    # it (and the Sources tab's Build demo index button fixes it).
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
def _hit_field(h, key, default=""):
    # Tolerant: chats persisted before Phase 5 lack chunk_id/section/
    # confidence keys, and dict-vs-object varies by load path.
    if isinstance(h, dict):
        return h.get(key, default)
    return getattr(h, key, default)


def _render_hit(h, rank):
    """Phase 5 passage line: rank + confidence band + passage citation.
    Raw score and chunk id are debug detail, shown small — never the
    headline the old `(score …)` line was."""
    from rag_demo.retriever import confidence_band

    source = _hit_field(h, "source")
    section = _hit_field(h, "section")
    citation = f"{source} › {section}" if section else source
    band = confidence_band(_hit_field(h, "confidence", None))
    st.markdown(f"**#{rank} · {band} confidence · [{citation}]**")
    st.caption(_hit_field(h, "text")[:500])
    _debug = []
    _chunk = _hit_field(h, "chunk_id")
    if _chunk:
        _debug.append(f"chunk `{_chunk}`")
    _score = _hit_field(h, "score", None)
    if _score is not None:
        _debug.append(f"raw score {_score:.3f} (debug)")
    if _debug:
        st.caption(" · ".join(_debug))


WEAK_EVIDENCE_NOTICE = (
    "⚠️ Weak evidence — no retrieved passage passed the relevance floor, "
    "so this answer is not grounded in the corpus. Try rephrasing the question."
)


for msg in st.session_state.messages:
    if _search and _search.lower() not in msg.get("text", "").lower():
        continue
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "📚"):
        # Strip any think blocks that leaked into stored messages
        # (from before the filter fix).
        from rag_demo.chain import strip_think

        st.markdown(strip_think(msg["text"]))
        if msg.get("sources"):
            pills = "".join(
                f'<span class="citation-pill">{s}</span>' for s in msg["sources"]
            )
            st.markdown(pills, unsafe_allow_html=True)
            _demo_sources_caption(app_mode == "demo")
        if msg.get("latency_ms") is not None:
            st.caption(
                f"⚡ {msg['latency_ms']/1000:.1f}s · {len(msg.get('hits', []))} passages"
            )
        if msg.get("grounded") is False:
            st.markdown(FALLBACK_BADGE, unsafe_allow_html=True)
        if msg.get("weak_evidence"):
            st.warning(WEAK_EVIDENCE_NOTICE, icon="⚠️")
        if msg.get("hits"):
            with st.expander("Retrieved passages"):
                for _rank, h in enumerate(msg["hits"], start=1):
                    _render_hit(h, _rank)
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
    "Ask anything…", disabled=not chat_ready
)
reg_case = st.session_state.pop("pending_case", None)

# Handle pending agentic plan approval (from previous rerun).
# This must come before the `if question` block because clicking
# Execute/Cancel triggers a rerun where `question` is None.
agentic_pending = st.session_state.get("agentic_pending")
if agentic_pending and chat_ready:
    from rag_demo.agent import format_plan_as_ascii, stream_agentic_ask

    pending_q = agentic_pending["question"]
    pending_plan = agentic_pending["plan"]
    # Display the user message (already in messages, just show plan UI).
    with st.chat_message("assistant", avatar="📚"):
        # If already approved (Execute was clicked), skip plan display and run.
        if st.session_state.get("agentic_approved"):
            st.session_state.pop("agentic_pending", None)
            st.session_state.pop("agentic_approved", None)
            with st.status("🤖 Running agent...", expanded=True) as status:
                # Live progress updates via callback.
                _phase_icons = {
                    "thought": "🧠", "action": "🔍",
                    "observation": "🔎", "synthesize": "📝",
                }
                def _on_progress(phase: str, message: str):
                    icon = _phase_icons.get(phase, "•")
                    st.write(f"{icon} **{phase.title()}:** {message}")

                st.write("_This takes 1-2 minutes on CPU (multiple LLM calls)._")
                try:
                    # Phase 2 (S3): run the user-approved plan (no second
                    # planning call) and stream synthesis tokens.
                    holder: dict = {}
                    def _agentic_tokens():
                        for event in stream_agentic_ask(
                            pending_q, settings,
                            verbose=False, show_plan=False,
                            on_progress=_on_progress,
                            history=st.session_state.get("history", []),
                            retriever=active_retriever,
                            plan=pending_plan,
                        ):
                            if "token" in event:
                                yield event["token"]
                            else:
                                holder["answer"] = event["answer"]
                    st.write_stream(_agentic_tokens())
                    answer = holder["answer"]
                except Exception as e:
                    status.update(label="❌ Failed", state="error")
                    st.error(f"Agent failed: {e}")
                    st.stop()
                status.update(label="✅ Done", state="complete")

            st.markdown(answer.text)
            _demo_sources_caption(app_mode == "demo")
            if answer.sources:
                with st.expander("Sources"):
                    for s in answer.sources:
                        st.markdown(f"- {s}")
            st.session_state.messages.append({
                "role": "assistant", "text": answer.text,
                "sources": answer.sources,
            })
            st.session_state.history.append((pending_q, answer.text))
            st.stop()

        # Not yet approved: show the plan with Execute/Cancel.
        st.markdown("**🤖 Agent Plan:**")
        # ASCII flowchart (always works, no external dependency).
        st.code(format_plan_as_ascii(pending_plan, pending_q))
        st.caption("Each step shows the tool (📚 local docs / 🌐 web search) and the query.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Execute", key="agentic_exec"):
                st.session_state["agentic_approved"] = True
                st.rerun()
        with col2:
            if st.button("❌ Cancel", key="agentic_cancel"):
                st.session_state.pop("agentic_pending", None)
                st.session_state.pop("agentic_approved", None)
                st.info("Plan cancelled.")
                st.stop()
        st.info("Review the plan above, then click Execute.")
        st.stop()

if question and chat_ready:
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
        thinking = st.empty()

        # Agentic mode: ReAct loop with optional plan approval.
        if st.session_state.get("agentic_mode", False):
            from rag_demo.agent import _parse_thought, THOUGHT_PROMPT, stream_agentic_ask
            from rag_demo.chain import get_llm, strip_think

            thinking.markdown(
                '<span class="thinking-dots">🤖 Planning</span>',
                unsafe_allow_html=True,
            )
            # THOUGHT: generate plan.
            from datetime import date
            llm = get_llm(settings)
            hist = st.session_state.get("history", [])
            hist_text = "\n".join(
                f"Q: {q}\nA: {a[:300]}" for q, a in hist[-3:]
            ) if hist else "No prior conversation."
            out = llm.invoke(THOUGHT_PROMPT.format(
                question=question, current_date=date.today().isoformat(),
                history=hist_text
            ))
            thought_text = out.content if hasattr(out, "content") else str(out)
            plan = _parse_thought(strip_think(thought_text.strip()))
            if not plan:
                plan = [{"tool": "search_docs", "query": question}]
            thinking.empty()

            # Store plan in session state and rerun to show approval UI.
            # (The approval UI is handled in the agentic_pending block above,
            #  which runs before this on the next rerun.)
            if st.session_state.get("agentic_show_plan", True):
                st.session_state["agentic_pending"] = {
                    "question": question,
                    "plan": plan,
                }
                st.rerun()
            # If show_plan is off, execute directly (no approval).

            # Execute the agentic loop (Phase 2, S3): run the plan the UI
            # already built (no second planning call) and stream synthesis.
            with st.status("🤖 Running agent...", expanded=True) as status:
                st.write("**Thought:** Planning searches...")
                holder = {}
                def _agentic_tokens():
                    for event in stream_agentic_ask(
                        question, settings,
                        verbose=False, show_plan=False,
                        history=st.session_state.get("history", []),
                        retriever=active_retriever,
                        plan=plan,
                    ):
                        if "token" in event:
                            yield event["token"]
                        else:
                            holder["answer"] = event["answer"]
                st.write_stream(_agentic_tokens())
                answer = holder.get("answer")
                if answer is None:
                    st.error("Couldn't answer: empty response from the agent.")
                    st.stop()
                st.write(f"**Action:** Executed {len(answer.sources)} tool calls")
                st.write("**Observation:** Synthesizing answer...")
                status.update(label="✅ Done", state="complete")

            st.markdown(answer.text)
            _demo_sources_caption(app_mode == "demo")
            if answer.sources:
                with st.expander("Sources"):
                    for s in answer.sources:
                        st.markdown(f"- {s}")
            # Save to history.
            st.session_state.messages.append({
                "role": "assistant", "text": answer.text,
                "sources": answer.sources,
            })
            st.session_state.history.append((question, answer.text))
        else:
            thinking.markdown(
                '<span class="thinking-dots">🧠 Thinking</span>',
                unsafe_allow_html=True,
            )

            def token_stream():
                first_token = True
                for event in stream_ask(
                    question,
                    settings,
                    history=ask_history,
                    user_groups=groups,
                    retriever=active_retriever,
                ):
                    if "token" in event:
                        if first_token:
                            thinking.empty()
                            first_token = False
                        yield event["token"]
                    else:
                        holder["answer"] = event["answer"]

        try:
            st.write_stream(token_stream())
        except Exception as e:  # noqa: BLE001 — show, don't crash the UI
            thinking.empty()
            st.error(f"Couldn't answer: {e}")
            st.stop()
        finally:
            thinking.empty()

        answer = holder.get("answer")
        if answer is None:
            st.error("Couldn't answer: empty response from the chain.")
            st.stop()

        if answer.sources:
            pills = "".join(
                f'<span class="citation-pill">{s}</span>' for s in answer.sources
            )
            st.markdown(pills, unsafe_allow_html=True)
            _demo_sources_caption(app_mode == "demo")
        if answer.latency_ms is not None:
            st.caption(
                f"⚡ {answer.latency_ms/1000:.1f}s · {len(answer.hits)} passages"
            )
        if not answer.grounded:
            st.markdown(FALLBACK_BADGE, unsafe_allow_html=True)
        if answer.weak_evidence:
            st.warning(WEAK_EVIDENCE_NOTICE, icon="⚠️")
        # Only show the rewrite when it led somewhere useful — displaying
        # "Understood as:" above a "No relevant answer" abstention is confusing.
        _is_abstention = "no relevant answer" in answer.text.lower()[:80]
        if answer.standalone_question and not _is_abstention:
            st.caption(f"🔍 Understood as: {answer.standalone_question}")
        with st.expander("Retrieved passages"):
            for _rank, h in enumerate(answer.hits, start=1):
                _render_hit(h, _rank)
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
            "weak_evidence": answer.weak_evidence,
        }
    )
    st.session_state.history.append((question, answer.text))
    st.session_state.history = st.session_state.history[-6:]
    _persist_active_chat()
