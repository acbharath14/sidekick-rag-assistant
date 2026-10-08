# RAG Assistant Demo

A miniature **retrieval-augmented generation** assistant: documents get indexed
into a vector store, and a chat UI / CLI / MCP tool answers questions with
source citations — grounded in retrieved passages, never hallucinated from
thin air.

> **Clean-room demo.** The default `docs/` corpus is fictional (Meridian
> Logistics, a made-up freight company). Nothing here comes from any employer
> or client system.

## How it works

```
DocumentSource ──load──▶ chunks ──embed──▶ FAISS index
                                              │
                    ┌─────────────────────────┼──────────────────────────┐
                    ▼                         ▼                          ▼
             Streamlit chat UI         conversational CLI         MCP `search_docs` tool
             (app.py)                  (python -m rag_demo.cli)   (python -m rag_demo.mcp_server)

                                                        retrieval eval (evals/eval_retrieval.py)
```

- **Sources** (`rag_demo/sources.py`): a `DocumentSource` is anything that
  produces documents. Ships with two — a local markdown folder (default) and
  an optional Playwright-docs fetcher. Enterprise connectors (Confluence,
  SharePoint, …) implement the same interface; see `ARCHITECTURE.md`.
- **Ingestion** (`rag_demo/ingest.py`): load → `RecursiveCharacterTextSplitter`
  (500 chars, 50 overlap) → embeddings → FAISS, persisted to `.faiss_index/`.
- **Retrieval** (`rag_demo/retriever.py`): similarity search with scores; every
  hit carries its source file for citations.
- **Answering** (`rag_demo/chain.py`): LCEL chain — retrieve → stuff into
  prompt → LLM. The system prompt forces answers from context only, with
  `[source.md]` citations; no relevant passages means "I don't know".
- **MCP** (`rag_demo/mcp_server.py`): one tool, `search_docs(query, k)`, over
  stdio. Add it to any MCP client config.

## No API keys needed

| Component | Default | Override |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, no key) | `RAG_DEMO_EMBEDDINGS=fake` |
| LLM | Ollama `qwen3:8b` at `localhost:11434` | `RAG_DEMO_MODEL`, `RAG_DEMO_BASE_URL` |
| Retrieval | dense vector search | `RAG_DEMO_RETRIEVAL=hybrid` (BM25+dense) |
| Rerank | off | `RAG_DEMO_RERANK=1` (cross-encoder, downloads once) |
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

### Optional: other corpora

```bash
# Real Playwright docs (sparse checkout at a pinned tag)
PYTHONPATH=src python -m rag_demo.ingest --source playwright-docs --language python

# Any GitHub repo's markdown docs via the API (public repos need no token)
PYTHONPATH=src python -m rag_demo.ingest --source github-docs --repo acbharath14/sidekick-rag-assistant
# Private repos: export RAG_DEMO_GITHUB_TOKEN=<token>  (never commit it)

# Simulated Confluence (JSON fixtures mirroring the REST API shape)
PYTHONPATH=src python -m rag_demo.ingest --source confluence-mock
```

Each source gets its own index (`.faiss_index*/`). The Streamlit sidebar
lets you switch corpora; the CLI reads `RAG_DEMO_INDEX_DIR` if you want it
there too.

### Try the permission demo

```bash
PYTHONPATH=src python -m rag_demo.ingest --source confluence-mock
RAG_DEMO_USER_GROUPS=ops-all PYTHONPATH=src streamlit run app.py
# Ask "What are the driver pay bands?" -> the model can't see that page.
# Now restart with RAG_DEMO_USER_GROUPS=ops-leads -> it can.
```

Restricted chunks are filtered *before* the prompt — the LLM never sees
what the user may not. ACL is default-deny: with no groups you see public
docs only. See `ARCHITECTURE.md`.

### Hybrid mode: general-knowledge fallback

By default (`RAG_DEMO_HYBRID=1`), when the corpus has no answer the assistant
falls back to the LLM's general knowledge — clearly marked with a
🌐 badge and a "General knowledge (not from your corpus)" header, never
silently mixed with cited facts. Uncheck it in the sidebar (or set
`RAG_DEMO_HYBRID=0`) for strict corpus-only mode, where it says
"I don't know" instead.

### Regression prompts (in the UI)

The sidebar has a **🧪 Regression prompts** expander: 8 standard prompts
(factual Q&A, code-identifier lookup, multi-turn follow-up, abstention,
small-talk) that run through the full pipeline with live pass/fail checks —
or **Run all** for the whole suite. Same cases as the CLI runner
(`python -m evals.regression`), so the two can't drift apart.

### Feedback, export, and health

- 👍/👎 under each answer persist to `feedback.jsonl` (next to the audit log
  when `RAG_DEMO_AUDIT_LOG` is set) — the start of a golden-set flywheel.
- **📥 Export chat** in the sidebar downloads the conversation as Markdown.
- The Engine panel shows **🟢/🔴 Ollama status**, and each answer reports
  its latency and passage count.

### Chat management

The sidebar's **💬 Chats** section keeps multiple named conversations —
switch between them, rename, or delete. Chats persist to `chats/*.json`
(gitignored) and survive restarts; new chats auto-title from the first
question. **🔍 Search chat** filters the active conversation. A fresh chat
opens on a welcome screen describing what Sidekick can do, with clickable
example prompts.

### Upload a document

Upload a PDF, DOCX, TXT, MD, or CSV from the sidebar: it's extracted
in-memory, summarized on demand, and queryable without touching the main
index. **Scanned PDFs** (images, no embedded text) fall back to Tesseract
OCR automatically — first install the binaries:

```bash
conda install -c conda-forge tesseract poppler
```

In the Streamlit sidebar: **📎 Upload a document** (PDF, DOCX, TXT, MD, CSV —
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

## Project layout

```
├── docs/                    # fictional corpus (the default knowledge base)
├── app.py                   # Streamlit chat UI (streaming, citations, ACL demo)
├── .streamlit/config.toml   # UI theme
├── ARCHITECTURE.md          # pipeline, connector pattern, scale notes
├── src/rag_demo/
│   ├── config.py            # env-based settings + model factories
│   ├── sources.py           # DocumentSource: meridian, playwright, github, confluence-mock
│   ├── ingest.py            # source → chunks → FAISS (+ --incremental, per-source indexes)
│   ├── retriever.py         # dense / hybrid (BM25+RRF) + ACL filtering
│   ├── rerank.py            # cross-encoder reranking (opt-in)
│   ├── rewrite.py           # multi-turn query rewriting
│   ├── chain.py             # RAG chain with citations (+ streaming)
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
│   └── regression_cases.json # 8 demo cases: factual, abstention, small-talk, follow-up
└── tests/test_rag.py        # pytest suite (fake embeddings/LLM)
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

**Project status**: `STATUS.md` at the repo root is the durable record of
where things stand, next actions, and decisions — kept current at the end of
every work session (convention from
[dev-harness](https://github.com/acbharath14/dev-harness), where this repo is
the pilot project).

## License

MIT — see [LICENSE](LICENSE).
