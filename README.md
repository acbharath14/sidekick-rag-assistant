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
| Tests/CI | deterministic fakes | `RAG_DEMO_FAKE=1` |

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

# 4. Run the test suite (fakes — no models, no keys)
RAG_DEMO_FAKE=1 PYTHONPATH=src pytest tests/ -q
```

### Optional: index the real Playwright docs

```bash
PYTHONPATH=src python -m rag_demo.ingest --source playwright-docs --language python
```

Sparse-checkouts `docs/src/*.md` from `microsoft/playwright` at a pinned tag
into a separate index (`.faiss_index_playwright/`). Needs git + network.
The Streamlit sidebar lets you switch corpora; the CLI reads
`RAG_DEMO_INDEX_DIR` if you want it there too.

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
├── app.py                   # Streamlit chat UI
├── ARCHITECTURE.md          # pipeline, connector pattern, scale notes
├── src/rag_demo/
│   ├── config.py            # env-based settings + model factories
│   ├── sources.py           # DocumentSource interface + implementations
│   ├── ingest.py            # source → chunks → FAISS (+ --source flag)
│   ├── retriever.py         # similarity search wrapper
│   ├── chain.py             # RAG chain with citations
│   ├── cli.py               # conversational REPL
│   └── mcp_server.py        # MCP server (search_docs tool)
├── evals/
│   ├── golden.json          # 12 Q&A pairs with expected source docs
│   └── eval_retrieval.py    # hit-rate@k, exits non-zero below threshold
└── tests/test_rag.py        # pytest suite (fake embeddings/LLM)
```

## Adapting it

- **Your own docs**: point `MarkdownDirectorySource` at any folder
  (`RAG_DEMO_DOCS_DIR`), rebuild the index, done.
- **A new source**: implement `DocumentSource.load()` (see
  `PlaywrightDocsSource` for a worked example), add it to `ingest.py`'s
  `--source` choices.
- **Different LLM/embeddings**: `config.py` factories read env vars; add a
  provider there without touching the chain.

## License

MIT — see [LICENSE](LICENSE).
