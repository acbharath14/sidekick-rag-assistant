# Sidekick RAG Assistant — Full Feature Specification

**Status:** Draft for approval · **Date:** 2026-10-04
**Repo:** `acbharath14/sidekick-rag-assistant` (PR #1 open with initial build)

This document describes the complete feature set with implementation details.
Approve as-is, or reply with what to change — no typing marathons needed.

---

## 1. What this is

A retrieval-augmented generation (RAG) assistant distilled from a real
enterprise system into a runnable demo. Documents are indexed into a vector
store; a chat UI, CLI, and MCP tool answer questions **grounded in retrieved
passages with citations** — never hallucinated.

**Design constraints:**
- Zero API keys required (local embeddings + Ollama, fakes for CI/tests)
- Clone-and-run in minutes
- Clean-room: fictional corpus, no employer/client data
- Every stage is measurable (golden eval set)

---

## 2. Architecture

```
DocumentSource ──load──▶ Documents ──split──▶ chunks ──embed──▶ FAISS index
                                                                       │
                              ┌────────────────────────────────────────┤
                              ▼                                        ▼
                        Retriever                                 RAG chain
                  (similarity + scores)                   (retrieve → prompt → LLM)
                              │                                        │
                    ┌─────────┼─────────┐                              ▼
                    ▼         ▼         ▼                    ┌─────────────────────┐
              MCP tool   eval harness  (future API)          │ Streamlit UI / CLI  │
                                                            └─────────────────────┘
```

One job per stage, one seam per stage. Full detail in `ARCHITECTURE.md`.

---

## 3. Feature set

### 3.1 Document sources — the connector pattern (`sources.py`)

**What:** A `DocumentSource` is anything that produces LangChain `Document`
objects with a `source` metadata key (used for citations). Swapping corpora
never touches indexing code.

**Implementations:**

| Source | Name | How it works |
|---|---|---|
| `MarkdownDirectorySource` | `meridian` | Loads `**/*.md` via LangChain `DirectoryLoader` + `TextLoader` (UTF-8). Citation = filename. |
| `PlaywrightDocsSource` | `playwright-docs` | Sparse-checkouts only `docs/src/*.md` from `microsoft/playwright` at pinned tag **v1.63.0**. Cached in `.playwright-docs/` so rebuilds don't re-download. Optional `--language` filter (e.g. `python` keeps `*-python.md` + language-agnostic pages). Citations look like `[playwright-docs/auth.md]`. Needs git + network (opt-in only). |

**Enterprise mapping:** a Confluence connector is the same interface
(`load()` → paginated CQL query → `Document` per page with URL/space/
permissions metadata). Documented in `ARCHITECTURE.md` with a code sketch.

### 3.2 Ingestion pipeline (`ingest.py`)

**What:** Source → chunks → embeddings → persisted FAISS index.

**Details:**
- `build_index(settings, source)` — orchestrates the pipeline, saves to
  `settings.index_dir`
- `chunk_documents()` — `RecursiveCharacterTextSplitter`, **chunk_size=500**,
  **chunk_overlap=50** (tunable in `Settings`)
- `load_documents()` — backward-compat shim for the default source
- CLI: `python -m rag_demo.ingest [--source meridian|playwright-docs]
  [--language python]`
- Playwright corpus gets its **own index** (`.faiss_index_playwright/`) so the
  default stays pristine
- Raises a clear error if a source produces zero documents

### 3.3 Embeddings (`config.py` → `get_embeddings()`)

| Mode | Implementation | When |
|---|---|---|
| Default | `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface` (local, ~90 MB download once) | Real use |
| Fake | `FakeEmbeddings(size=384)` — deterministic, no download | Tests, CI (`RAG_DEMO_FAKE=1`) |

Override via `RAG_DEMO_EMBEDDINGS` / `RAG_DEMO_EMBEDDING_MODEL`.

### 3.4 LLM (`config.py` → `get_llm()`)

| Mode | Implementation | When |
|---|---|---|
| Default | Ollama `qwen3:8b` at `localhost:11434` | Real use |
| Fake | `FakeListLLM` with a canned cited response | Tests, CI |

Override via `RAG_DEMO_LLM`, `RAG_DEMO_MODEL`, `RAG_DEMO_BASE_URL`.

### 3.5 Retrieval (`retriever.py`)

**What:** Similarity search over the FAISS index with scores and source
tracking.

**Details:**
- `Retriever.search(query, k)` → list of `Hit(text, source, score)` using
  `similarity_search_with_score`; `k` defaults to `Settings.top_k` (4)
- `as_langchain_retriever()` — exposes the LangChain retriever interface
- Raises a clear error if the index doesn't exist ("run ingest first")
- Every hit carries its source file → citations downstream

### 3.6 RAG chain (`chain.py`)

**What:** Retrieve → stuff into prompt → LLM → parsed answer, with citations.

**Details:**
- LCEL chain: `{context: retrieve|format, question: passthrough}` → prompt →
  LLM → `StrOutputParser`
- System prompt: *"Answer using ONLY the context below. If the context
  doesn't contain the answer, say you don't know. Cite the source for each
  fact, like [api-reference.md]."*
- `ask(question)` returns `Answer(text, sources, hits)` — text plus the
  sorted unique source list plus raw hits (for UI display)
- The "I don't know" guardrail is the hallucination control

### 3.7 Interfaces

**Streamlit chat UI (`app.py`)** — `streamlit run app.py`
- Chat history, per-message retrieved-passage expander (source + score +
  text snippet)
- Sidebar corpus switcher (meridian / playwright-docs)
- Shows active LLM/embedding providers; warns clearly if the index is missing
- New dependency: `streamlit>=1.30`

**Conversational CLI (`cli.py`)** — `python -m rag_demo.cli`
- REPL with `/quit`; prints answer + `Sources: ...` line

**MCP server (`mcp_server.py`)** — `python -m rag_demo.mcp_server`
- One tool over stdio: `search_docs(query, k=4)` → formatted passages with
  scores
- Handles both `mcp` v1 (`FastMCP`) and v2 (`MCPServer`) import paths

### 3.8 Evaluation (`evals/`)

**What:** Retrieval quality is measured, not assumed.

- `golden.json` — 12 question/answer pairs, each with `expected_sources`
- `eval_retrieval.py` — runs the set against the retriever, reports
  **hit-rate@k**, exits non-zero below a threshold (CI-gateable)
- Discipline: every chunking/embedding change gets a number before it ships;
  the golden set is versioned with the corpus

### 3.9 Configuration (`config.py`)

Single `Settings` dataclass; everything overridable by env:

| Setting | Env var | Default |
|---|---|---|
| Docs dir | `RAG_DEMO_DOCS_DIR` | `./docs` |
| Index dir | `RAG_DEMO_INDEX_DIR` | `./.faiss_index` |
| Chunk size / overlap | — | 500 / 50 |
| Top-k | — | 4 |
| Embedding provider/model | `RAG_DEMO_EMBEDDINGS` / `RAG_DEMO_EMBEDDING_MODEL` | huggingface / all-MiniLM-L6-v2 |
| LLM provider/model/URL | `RAG_DEMO_LLM` / `RAG_DEMO_MODEL` / `RAG_DEMO_BASE_URL` | ollama / qwen3:8b / localhost:11434 |
| Fake mode | `RAG_DEMO_FAKE=1` | off |

---

## 4. Corpus

**Default:** 5 fictional Meridian Logistics docs (clean-room, clearly labeled):
`api-reference.md`, `deployment-runbook.md`, `onboarding.md`,
`security-policy.md`, `troubleshooting.md`.

**Optional:** real Playwright docs (pinned v1.63.0) via
`--source playwright-docs`.

---

## 5. Testing (`tests/test_rag.py`)

11 tests, all run with `RAG_DEMO_FAKE=1` (deterministic, no downloads):

1. Ingest indexes all docs (index non-empty)
2. Chunks carry `.md` source metadata
3. Retriever returns hits with sources (k=3)
4. Retriever returns exactly k hits (k=4)
5. Chain answers with citations (sources non-empty)
6. `format_hits` marks sources correctly
7. Golden eval runs end-to-end, rate in [0,1]
8. Golden file well-formed (≥10 items, question + expected_sources)
9. **NEW:** Markdown source loads exactly the 5 expected files
10. **NEW:** `build_index` accepts an explicit source
11. **NEW:** Source registry contains both sources

---

## 6. CI (`.github/workflows/ci.yml` — you add manually)

- Triggers: push/PR touching `src/`, `tests/`, `evals/`, `requirements.txt`;
  weekly Monday schedule; manual dispatch
- Python 3.12, pip cache, `pip install -r requirements.txt`
- `RAG_DEMO_FAKE=1 PYTHONPATH=src pytest tests/ -q`

---

## 7. Deliberate omissions

Not implemented — documented as design notes in `ARCHITECTURE.md` instead:

- Real enterprise connectors (no credentials, no proprietary systems)
- Permission-aware retrieval (pattern documented: ACLs in chunk metadata,
  filter before the prompt — never rely on the LLM to withhold)
- User identity / audit logging, multi-tenant isolation
- Incremental indexing (pattern documented: `last_modified` tracking,
  index sharding per source)

---

## 8. File manifest (22 files in PR #1)

```
├── docs/                    # fictional corpus (5 files)
├── app.py                   # Streamlit chat UI
├── ARCHITECTURE.md          # pipeline, connectors, scale, omissions
├── LICENSE / README.md / requirements.txt / .gitignore
├── src/rag_demo/
│   ├── config.py            # settings + model factories
│   ├── sources.py           # DocumentSource interface + 2 implementations
│   ├── ingest.py            # pipeline + --source CLI
│   ├── retriever.py         # similarity search wrapper
│   ├── chain.py             # RAG chain with citations
│   ├── cli.py               # REPL
│   └── mcp_server.py        # search_docs tool (stdio)
├── evals/                   # golden.json (12 Q&A) + eval_retrieval.py
└── tests/test_rag.py        # 11 tests
```

---

## 9. For approval

Reply with one of these (or your own words):

- **"Approved"** — PR #1 merges as-is; you add `ci.yml`; CI validates
- **"Approved with changes:"** — list what to cut/add/change, I rebuild
- **"Rejected"** — tell me what's wrong with the direction and I'll re-scope
