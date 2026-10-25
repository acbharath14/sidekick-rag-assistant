# Sidekick RAG Assistant

A miniature **retrieval-augmented generation** assistant: documents get indexed
into a vector store, and a chat UI / CLI / MCP tool answers questions with
source citations — grounded in retrieved passages, never hallucinated from
thin air.

> **Clean-room demo.** The default `docs/` corpus is fictional (Meridian
> Logistics, a made-up freight company). Nothing here comes from any employer
> or client system.

## How it works

```
Documents ──chunk──▶ passages ──embed──▶ index ──retrieve──▶ cited answer
```

- **Sources.** Documents come from two places that never mix. **Demo mode**
  uses fictional data for zero-config trial (a made-up freight company's
  docs and simulated Confluence pages), marked fictional everywhere — mode
  label, banner, citations, and exports. **Workspace mode** uses real
  sources: documentation (Playwright docs, this repo's own docs) and your
  workspace (Confluence, Jira, Outlook, AWS S3). Each source gets its own
  index.
- **From documents to answers.** Documents are split into passages,
  embedded, and indexed. A question retrieves the most relevant passages,
  and the answer is built from those passages only, citing each fact to
  its source passage (`[source.md › Section]`). If nothing relevant is
  found, the assistant says "I don't know" instead of guessing — and if
  the best passages are too weak to trust, it says so explicitly rather
  than answering confidently.
- **Pipeline walkthrough.** The sidebar's "How it works" button shows the
  chunking → embedding → retrieval stages on a sample document or text
  you paste, with similarity scores. It runs on its own document, never
  your corpus, and stops before generating an answer.
- **More ways to ask.** The same retrieval powers a chat UI, a terminal
  chat, and a one-tool MCP server (`search_docs`) for MCP clients. An
  optional agentic mode plans multi-step research across local docs and
  the web. Past questions and answers are remembered across restarts, so
  follow-ups can build on earlier exchanges.

See `ARCHITECTURE.md` for the module-level detail: the connector pattern,
the retrieval pipeline, caching, and scale notes.

## No API keys needed

| Component | Default | Override |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, no key) | `RAG_DEMO_EMBEDDINGS=fake` |
| LLM | Ollama `qwen3:8b` at `localhost:11434` | `RAG_DEMO_MODEL`, `RAG_DEMO_BASE_URL` |
| Retrieval | hybrid (BM25+dense), `top_k=8` | `RAG_DEMO_RETRIEVAL=dense` (pure vector search) |
| Rerank | on (off under `RAG_DEMO_FAKE=1`) | `RAG_DEMO_RERANK=0` to disable (cross-encoder, downloads once) |
| Relevance floor | 0.5 confidence (off under `RAG_DEMO_FAKE=1`) | `RAG_DEMO_RELEVANCE_FLOOR=0` to disable, or any 0–1 value |
| Session cache | embeddings + FAISS retriever + episodic memory, built once per session | `RAG_DEMO_NO_CACHE=1` (bypass) |
| Agentic budget | 4 iterations, 300s wall-clock | `RAG_DEMO_AGENT_MAX_ITERATIONS`, `RAG_DEMO_AGENT_BUDGET_SECONDS` |
| Web search (agentic) | DuckDuckGo scraping (unreliable) | `RAG_DEMO_TAVILY_API_KEY` (recommended) |
| Tests/CI | deterministic fakes | `RAG_DEMO_FAKE=1` |

> **Intel Mac note:** PyTorch ships no Intel-macOS wheels past 2.2.2, and
> Ollama can't use AMD GPUs (Metal backend is Apple-Silicon-only), so this
> machine runs everything on CPU. `requirements.txt` pins the compatible
> trio (`numpy<2`, `transformers<5`) automatically. For snappier answers on
> CPU, try `RAG_DEMO_MODEL=qwen2.5-coder:7b`.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Build the index (downloads the embedding model once, ~90 MB)
PYTHONPATH=src python -m rag_demo.ingest

# 2a. Chat in the browser
PYTHONPATH=src streamlit run app.py

# 2b. Or chat in the terminal (needs Ollama running)
PYTHONPATH=src python -m rag_demo.cli
# > How long are API tokens valid?
#   ... answer with [api-reference.md] citation
# > /quit

# 3. Evaluate retrieval quality against the golden set
PYTHONPATH=src python -m evals.eval_retrieval

# 3b. Run the end-to-end regression suite (standard prompts, full pipeline)
RAG_DEMO_FAKE=1 PYTHONPATH=src python -m evals.regression
# With Ollama running, also check answer content:
PYTHONPATH=src python -m evals.regression --real-llm

# 4. Run the test suite (fakes — no models, no keys)
RAG_DEMO_FAKE=1 PYTHONPATH=src pytest tests/ -q
```

### Setup script (fresh clone)

`python scripts/setup.py` is the idempotent bootstrap: it checks
dependencies, installs/starts Ollama if absent, pulls the default model
(`qwen3:8b`), builds the demo indexes (Meridian + Confluence mock), and
verifies with one test question. Re-running is safe — every step is
skipped when already done. Honest limits: model weights are gigabytes
and are fetched, not cloned, so the repo stays light; the macOS/Windows
install branches are written carefully but not yet verified on real
machines (if an automatic install fails, the script prints the manual
steps instead of pretending success). Docker Compose is an optional
self-contained alternative, not the default path.

### Sidebar, Demo mode, and pre-flight checks

The sidebar has three tabs. **💬 Chat** holds the agentic-mode toggle,
the **🔍 How it works** pipeline-walkthrough button, the session, and the
chat list. **📚 Sources** holds source/mode
selection, index status, the **🔨 Build demo index** button (builds the
demo indexes inside the app — no terminal needed; chat input enables
automatically when the build finishes), enterprise-connector setup, and
document upload. **⚙️ Settings** holds the retrieval-mode and rerank
toggles (defaults: hybrid, rerank on), the general-knowledge fallback,
the Tavily key, cache clears, diagnostics, and the regression runner.

The **Mode** switch at the top of the sidebar is one visible control:
**🎭 Demo (fictional data)** or **💼 Workspace**. First open lands in
Workspace if any workspace source has a built index, else Demo; Demo
mode carries its fictional marking persistently (banner, context line,
citations, export), and the workspace path never shows a demo corpus.
On startup the app runs **pre-flight checks** (Ollama reachable, model
present, index present); when anything is missing, chat input is
disabled and a fix-it banner says what is missing and how to fix it,
instead of failing mid-question.

### Optional: other corpora

```bash
# Real Playwright docs (sparse checkout at a pinned tag)
PYTHONPATH=src python -m rag_demo.ingest --source playwright-docs --language python

# Any GitHub repo's markdown docs via the API (public repos need no token)
PYTHONPATH=src python -m rag_demo.ingest --source github-docs --repo owner/name
# Private repos: export RAG_DEMO_GITHUB_TOKEN=<token>  (never commit it)

# Simulated Confluence (JSON fixtures mirroring the REST API shape)
PYTHONPATH=src python -m rag_demo.ingest --source confluence-mock

# Real enterprise sources (need credentials; the sidebar's Sources tab in
# Workspace mode also has setup panels)
PYTHONPATH=src python -m rag_demo.ingest --source confluence   # RAG_DEMO_CONFLUENCE_URL + RAG_DEMO_CONFLUENCE_TOKEN
PYTHONPATH=src python -m rag_demo.ingest --source jira         # RAG_DEMO_JIRA_URL + RAG_DEMO_JIRA_TOKEN
PYTHONPATH=src python -m rag_demo.ingest --source outlook      # RAG_DEMO_OUTLOOK_TOKEN
PYTHONPATH=src python -m rag_demo.ingest --source aws-s3       # RAG_DEMO_AWS_S3_BUCKET
```

Each source gets its own index (`.faiss_index*/`). The sidebar's Sources
tab (Demo or Workspace mode) lets you switch sources; the CLI reads
`RAG_DEMO_INDEX_DIR` if you want it there too.

### ⚠️ Required index rebuild (chunk-level citations)

Indexes built before this change lack chunk-id and section metadata, so
their citations degrade to filenames only. **Rebuild every index you use**
after pulling this change — the app shows a rebuild banner above the chat
while an index is stale, and the Sources tab's Build demo index button
rebuilds stale demo indexes instead of skipping them:

```bash
PYTHONPATH=src python -m rag_demo.ingest --source meridian
PYTHONPATH=src python -m rag_demo.ingest --source confluence-mock
PYTHONPATH=src python -m rag_demo.ingest --source playwright-docs
PYTHONPATH=src python -m rag_demo.ingest --source github-docs   # needs --repo or RAG_DEMO_GITHUB_REPO
PYTHONPATH=src python -m rag_demo.ingest --source confluence    # needs Confluence credentials
PYTHONPATH=src python -m rag_demo.ingest --source jira          # needs Jira credentials
PYTHONPATH=src python -m rag_demo.ingest --source outlook       # needs Outlook token
PYTHONPATH=src python -m rag_demo.ingest --source aws-s3        # needs RAG_DEMO_AWS_S3_BUCKET
```

A rebuild writes an `index_format.json` marker next to the index; anything
without the current marker is treated as stale. Incremental rebuilds
(`--incremental`) never skip a stale index, even when documents are
unchanged. Nothing is lost by waiting — a stale index still answers, it
just cites filenames until rebuilt.

### Try the permission demo

```bash
PYTHONPATH=src python -m rag_demo.ingest --source confluence-mock
RAG_DEMO_USER_GROUPS=eng-all PYTHONPATH=src streamlit run app.py
# Ask "What are the salary bands?" -> the model can't see that page.
# Now restart with RAG_DEMO_USER_GROUPS=eng-leads -> it can.
```

Restricted chunks are filtered *before* the prompt — the LLM never sees
what the user may not. ACL is default-deny: with no groups you see public
docs only. See `ARCHITECTURE.md`.

### Hybrid mode: general-knowledge fallback

By default (`RAG_DEMO_HYBRID=1`), when the corpus has no answer the assistant
falls back to the LLM's general knowledge — clearly marked with a
🌐 badge and a "General knowledge (not from your corpus)" header, never
silently mixed with cited facts. Uncheck it in the sidebar's Settings tab
(or set `RAG_DEMO_HYBRID=0`) for strict corpus-only mode, where it says
"I don't know" instead.

### Regression prompts (in the UI)

The sidebar's Settings tab has **🧪 Regression prompts**: 9 standard prompts
(factual Q&A, code-identifier lookup, multi-turn follow-up, abstention,
small-talk) that run through the full pipeline with live pass/fail checks —
or **Run all** for the whole suite. Same cases as the CLI runner
(`python -m evals.regression`), so the two can't drift apart.

### Feedback, export, and health

- 👍/👎 under each answer persist to `feedback.jsonl` (next to the audit log
  when `RAG_DEMO_AUDIT_LOG` is set) — the start of a golden-set flywheel.
- **📥 Export chat** in the sidebar's Chat tab downloads the conversation as Markdown.
- The Settings tab's Diagnostics show **🟢/🔴 Ollama status**, model
  presence, and index presence, and each answer reports its latency and
  passage count.
- The always-on context banner above the chat shows which corpus each answer
  comes from, so a wrong-corpus answer can't pass silently.

### Chat management

The sidebar's Chat tab keeps multiple named conversations —
switch between them, rename, or delete. Chats persist to `chats/*.json`
(gitignored) and survive restarts; new chats auto-title from the first
question. **🔍 Search chat** filters the active conversation. A fresh chat
opens on a welcome screen describing what Sidekick can do, with clickable
example prompts.

### Upload a document

Upload a PDF, DOCX, TXT, MD, or CSV from the sidebar's Sources tab: it's extracted
in-memory, summarized on demand, and queryable without touching the main
index. **Scanned PDFs** (images, no embedded text) fall back to Tesseract
OCR automatically — first install the binaries:

```bash
conda install -c conda-forge tesseract poppler
```

In the sidebar's Sources tab: **📎 Upload a document** (PDF, DOCX, TXT, MD, CSV —
10 MB cap). The app extracts the text in memory, offers a **📝 Summarize**
button (map-reduce for long docs), and **💬 Ask about this file** builds an
in-memory index so you can chat over it with `[upload:filename]` citations.
Nothing is written to disk or committed; with local Ollama, bytes never
leave the machine.

### Nightly refresh

```bash
# Rebuild only when documents changed (no-op otherwise — cron-friendly)
PYTHONPATH=src python -m rag_demo.ingest --source confluence-mock --incremental
```

## MCP client setup

Add to your MCP client config (e.g. Claude Code):

```json
{
  "mcpServers": {
    "meridian-docs": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "rag_demo.mcp_server"],
      "cwd": "/path/to/sidekick-rag-assistant",
      "env": { "PYTHONPATH": "src" }
    }
  }
}
```

Then ask your client to "search the Meridian docs for the rollback procedure"
— it will call `search_docs` and answer from the corpus.

## Agentic RAG

Beyond single-shot retrieval, the agent (`rag_demo/agent.py`) runs a **ReAct loop**
(Thought → Action → Observation) that plans multi-step research, calls tools,
evaluates evidence, and refines its approach — up to 4 iterations
(`RAG_DEMO_AGENT_MAX_ITERATIONS`) and a 300s wall-clock budget
(`RAG_DEMO_AGENT_BUDGET_SECONDS`) before synthesizing. On budget exhaustion
it stops with the evidence gathered so far plus an explicit uncertainty note,
never silently and never as a complete answer.

**Tools:**
- `search_docs` — local FAISS corpus (technical documentation)
- `search_web` — web search via Tavily API (recommended) or DuckDuckGo fallback

**Features:**
- **Parallel search dispatch** — the plan's search calls (FAISS, Tavily) run
  concurrently with order preserved for synthesis; LLM calls (planning,
  observation, synthesis) stay sequential, since they contend on the same CPU
  model.
- **Single planning** — the UI-built plan is passed into the agent, so it
  doesn't plan twice; direct API/CLI callers still get one fallback plan.
- **Streamed synthesis** — agentic answers stream tokens like classic mode
  instead of returning one-shot at the end.
- **Structural sources** — agentic answers cite the retrieval result itself
  (source + section + chunk id for corpus hits, link titles for web hits),
  not bracketed tokens scraped out of tool text — so tool markers like
  `[TOOL_ERROR]` or `[CACHED …]` can never appear as "sources".
- **Smart routing** — the planner picks `search_docs` for API/code questions and
  `search_web` for everything else (prices, news, general knowledge). A fast-path
  heuristic skips the LLM planning call for obvious web queries, saving 20-30s on CPU.
- **Tavily backend** — set `RAG_DEMO_TAVILY_API_KEY` (free tier: 1,000 searches/month
  at tavily.com) for reliable web search. Falls back to DuckDuckGo HTML scraping
  without a key (often blocked/rate-limited). Enter the key in the sidebar's Settings tab
  under Agentic mode, or as an environment variable.
- **Conversation memory** — follow-ups like "what about Target?" are rewritten to
  standalone questions using chat history, before routing. History is passed to all
  agent prompts for coherent multi-turn research.
- **Search cache** — web results are cached per-session (30-min TTL) to save API
  credits. Cached hits are marked `[CACHED Xs ago]`; stale entries auto-refresh.
  Clear via the sidebar's Settings tab ("🗑️ Clear search cache" button).
- **Tool-failure honesty** — `[TOOL_ERROR]` (network/timeout/blocked) is distinguished
  from `[NO_RESULTS]` (genuinely empty). The agent never claims "doesn't exist" from
  a tool failure.
- **Plan approval** — optional ASCII flowchart preview with Execute/Edit/Cancel
  before the agent runs (CLI and Streamlit).

Enable in the sidebar's Chat tab ("🤖 Agentic mode") or via CLI:
```bash
PYTHONPATH=src python -m rag_demo.agentic_cli "Walmart Black Friday deals 2026"
```

## Project layout

```
├── docs/                    # fictional corpus (the default knowledge base)
├── app.py                   # Streamlit chat UI (streaming, citations, ACL demo, Demo/Workspace modes, pre-flight)
├── scripts/setup.py         # idempotent fresh-clone bootstrap (Ollama, model, demo indexes, test question)
├── .streamlit/config.toml   # UI theme
├── ARCHITECTURE.md          # pipeline, connector pattern, scale notes
├── src/rag_demo/
│   ├── config.py            # env-based settings + model factories
│   ├── modes.py             # Demo vs Workspace mode rules, picker labels, source statuses
│   ├── preflight.py         # startup checks: Ollama, model, index (+ fix-it messages, stale-index banner)
│   ├── sources.py           # DocumentSource: 8 corpora (see enterprise.py for the real ones)
│   ├── enterprise.py        # real Confluence / Jira / Outlook / AWS S3 connectors
│   ├── ingest.py            # source → chunks → FAISS (+ --incremental, per-source indexes, chunk ids + sections)
│   ├── retriever.py         # dense / hybrid (BM25+RRF) + ACL filtering + relevance floor + passage citations
│   ├── rerank.py            # cross-encoder reranking (on by default)
│   ├── rewrite.py           # multi-turn query rewriting
│   ├── chain.py             # RAG chain with citations (+ streaming)
│   ├── caching.py           # per-session cache: embeddings, retriever, episodic memory
│   ├── episodic.py          # LanceDB episodic memory (past Q&A recall)
│   ├── chat_store.py        # persisted sidebar chats (chats/*.json)
│   ├── agent.py             # agentic RAG: ReAct loop, parallel dispatch, streamed synthesis, budgets
│   ├── agentic_cli.py       # CLI for agentic mode with plan approval
│   ├── audit.py             # opt-in JSONL audit log
│   ├── extract.py           # file text extraction (pdf/docx/txt/md/csv)
│   ├── summarize.py         # map-reduce document summarization
│   ├── cli.py               # conversational REPL (with history)
│   └── mcp_server.py        # MCP server (search_docs tool)
├── fixtures/confluence/     # mock Confluence REST fixtures (incl. restricted page)
├── evals/
│   ├── golden.json          # 20 Q&A pairs with expected source docs
│   ├── eval_retrieval.py    # hit-rate@k, exits non-zero below threshold
│   ├── eval_compare.py      # dense vs hybrid vs hybrid+rerank table
│   ├── baselines.json       # per-mode hit-rate floors
│   ├── regression.py        # end-to-end regression runner (standard prompts)
│   ├── regression_cases.json # 9 demo cases: factual, abstention, small-talk, follow-up
│   └── mcp_demo.py          # MCP server demo script
└── tests/                   # pytest suite (fake embeddings/LLM): test_rag.py, test_phase4.py (modes, pre-flight, setup)
```

## Adapting it

- **Your own docs**: point `MarkdownDirectorySource` at any folder
  (`RAG_DEMO_DOCS_DIR`), rebuild the index, done.
- **A new source**: implement `DocumentSource.load()` (see
  `PlaywrightDocsSource` for a worked example), register it in `SOURCES`,
  add its index dir to `ingest.INDEX_DIRS`.
- **Different LLM/embeddings**: `config.py` factories read env vars; add a
  provider there without touching the chain.

## Operations

**Nightly refresh** (cron-friendly — no-op when nothing changed):

```cron
0 2 * * * cd /path/to/sidekick-rag-assistant && PYTHONPATH=src .venv/bin/python -m rag_demo.ingest --source confluence-mock --incremental
```

**Credentials**: connectors read secrets from env vars only
(`RAG_DEMO_GITHUB_TOKEN`). Never commit tokens — `.gitignore` covers `.env`.

**Audit log**: set `RAG_DEMO_AUDIT_LOG=/var/log/rag-demo/audit.jsonl` to record
one JSON line per question (timestamp, question, sources, groups, model,
latency). Answer text is excluded unless `RAG_DEMO_AUDIT_LOG_ANSWERS=1`.

**Project status**: `STATUS.md` at the repo root is a dated status log (last
updated 2026-10-08) — useful history, but not kept current per session. The
code and this README are the current record.

## License

MIT — see [LICENSE](LICENSE).
