# Architecture

How this demo is put together, and how the same shape scales to a production
enterprise assistant. This doc is the bridge between the runnable demo and the
real-world system it distills.

## Pipeline

```
DocumentSource ──load──▶ Documents ──split──▶ chunks ──embed──▶ FAISS
                                                                    │
                              ┌─────────────────────────────────────┤
                              ▼                                     ▼
                        Retriever                            RAG chain
                   (similarity + scores)              (retrieve → prompt → LLM)
                              │                                     │
                    ┌─────────┼─────────┐                           ▼
                    ▼         ▼         ▼                      CLI / Streamlit
              MCP tool    eval harness   (future: API)
```

Each stage has one job and one seam:

| Stage | Module | Seam |
|---|---|---|
| Load | `sources.py` | `DocumentSource` interface |
| Split | `ingest.py` | chunk size/overlap in `Settings` |
| Embed | `config.py` | `get_embeddings()` factory |
| Retrieve | `retriever.py` | `top_k`, relevance floor |
| Generate | `chain.py` | system prompt, LLM factory |

## The connector pattern

`DocumentSource` is the whole enterprise story in one interface:

```python
class DocumentSource(ABC):
    name: str
    def load(self) -> list[Document]: ...
```

The eight sidebar corpora share this interface, split into two paths that
never mix (`modes.py`): **Demo mode** serves fictional data for zero-config
trial — `meridian` (local markdown folder) and `confluence-mock` (simulated
Confluence fixtures), both marked fictional everywhere. **Workspace mode**
serves real sources — Documentation (`PlaywrightDocsSource`, sparse git
checkout; `GitHubDocsSource`, repo markdown via the API, public repos
needing no token) and Your workspace (four real enterprise connectors in
`enterprise.py`: `ConfluenceSource`, `JiraSource`, `OutlookSource`,
`AWSS3Source`). Each source gets its own FAISS index
(`ingest.INDEX_DIRS`); the mode rules (`DEMO_CORPORA`,
`WORKSPACE_CORPORA`) live in `modes.py`. A connector is the same shape —
the real Confluence connector in `enterprise.py` follows it:

```python
class ConfluenceSource(DocumentSource):
    name = "confluence"
    def __init__(self, client, space_keys, ...): ...
    def load(self) -> list[Document]:
        pages = self.client.get_pages(...)      # paginated CQL query
        return [Document(page.body, metadata={
            "source": page.title,
            "url": page.url,
            "space": page.space,
            "allowed_groups": page.restrictions,   # see below
        }) for page in pages]
```

Nothing downstream changes — `ingest.build_index(settings, source)` doesn't
care where documents came from. That's the point: **the corpus is a plugin.**

## Permission-aware retrieval (implemented)

In an enterprise, not every user may see every document. Implemented via the
safe path:

1. **At index time**, each chunk's metadata carries `allowed_groups`
   (Confluence page restrictions; `["*"]` for public sources).
2. **At query time**, the asker's groups (`RAG_DEMO_USER_GROUPS`) are resolved
   once; `Retriever.search()` post-filters hits *before* the prompt —
   over-fetching 20 candidates so restricted hits don't eat the result budget.
   ACLs survive RRF fusion and cross-encoder reranking (`Hit.allowed_groups`
   is carried through both).

Never rely on the LLM to "know" what the user may see — a model that has
seen a secret in context *will* leak it under the right prompt. Filter
before the prompt, not after the answer.

Try it: index `--source confluence-mock`, then ask about "driver pay bands" with
`RAG_DEMO_USER_GROUPS=ops-all` (invisible) vs `ops-leads` (visible).

## Retrieval pipeline (data flow)

```
question ──▶ rewrite (multi-turn) ──▶┬──▶ dense (FAISS, top-20) ──┐
                                     └──▶ BM25 (chunks.json, top-20) ─┤
                                                                      ├─▶ RRF fuse ─▶ ACL filter ─▶ rerank? ─▶ floor ─▶ top-k ─▶ prompt ─▶ LLM
```

- Default is hybrid: dense (FAISS, top-20 candidates) + BM25 (over the
  `chunks.json` the index build writes, top-20 candidates), fused with
  reciprocal rank fusion (`RRF_K = 60`). `RAG_DEMO_RETRIEVAL=dense` skips
  the BM25 branch.
- Rerank is on by default: the fused candidates are re-scored with a
  cross-encoder (`rerank.py`). `RAG_DEMO_RERANK=0` disables it; under
  `RAG_DEMO_FAKE=1` the default stays off so tests never touch torch.
- Relevance floor (`RAG_DEMO_RELEVANCE_FLOOR`, default 0.5, 0 disables):
  applied to the full candidate pool before the top-k trim. It never runs
  on RRF scores (purely relative ranks, no absolute signal) — it uses the
  rerank logit (sigmoid) when reranked, else the dense L2 distance
  (0.5 = distance 1.0). A hit with no absolute signal (BM25-only, rerank
  off) is kept, not dropped on a meaningless number. When nothing passes,
  the answer carries a weak-evidence flag and the UI shows an explicit
  notice instead of a confident guess, with the prompt taking the
  abstain-friendly path.
- Passage identity: ingest (`ingest.py`, `RecursiveCharacterTextSplitter`,
  500 chars / 50 overlap from `Settings`) stamps every chunk with a chunk
  id (document id + section + ordinal) and the section path from the
  document's headings, so citations point at a passage, not just a file.
  `Hit` carries `chunk_id` + `section`; prompts cite
  `[source › section]`. Retrieved passages display rank plus a qualitative
  confidence band (High/Medium/Low); raw scores appear only as debug
  detail.
- `evals/eval_compare.py` measures dense vs hybrid vs hybrid+rerank on the
  golden set; baselines live in `evals/baselines.json`.

## Per-session caching (`caching.py`)

Expensive objects are built once per settings combination, not once per
question: the embeddings object, the FAISS retriever, and the episodic
memory. The cache key is (embedding provider, embedding model, index dir,
retrieval mode, rerank enabled), so changing a setting yields a fresh
retriever; the cache is cleared after an enterprise index rebuild.
`RAG_DEMO_NO_CACHE=1` bypasses it. The cache is module dicts, not
`@st.cache_resource`, so it stays testable without Streamlit.

## Answering (`chain.py`)

LCEL chain — retrieve → stuff into prompt → LLM. The system prompt forces
answers from context only, with `[source.md › Section]` citations; no
relevant passages means "I don't know". Multi-turn follow-ups are
rewritten to standalone questions from at most the last 3 turns of
history before retrieval; chat history itself never grows the
generation prompt.

## MCP server (`mcp_server.py`)

One tool, `search_docs(query, k)`, over stdio, served from the same
retriever as chat. Point any MCP client at the process; see the README's
MCP client setup for the config block.

## Episodic memory (`episodic.py`)

Past questions and answers live in a persistent LanceDB table
(`.lancedb_episodic/`), so follow-ups can recall earlier exchanges across
Streamlit restarts. At most 3 relevant past exchanges feed follow-up
rewriting; if LanceDB isn't installed, this quietly disables itself
instead of breaking answers.

## Pipeline walkthrough (`walkthrough.py`)

A stage-by-stage view of chunking → embedding → retrieval on a specimen
document or pasted text, with similarity scores at retrieval. It runs the
real pipeline code paths — `ingest.chunk_documents` (500/50 splitter,
chunk ids + sections), `config.get_embeddings`, and a `Retriever` over an
in-memory FAISS store built from exactly those chunks (the same
store+chunks injection the upload path uses, so hybrid/RRF/rerank/floor
behave as in chat). The stage list is exactly (chunking, embedding,
retrieval): it builds its own index, so it runs even when pre-flight
disables chat, and it stops at retrieval — no generation, no summarize
step.

## Diagrams (`diagrams.py`)

Explicit-trigger only (a chat question asking for a diagram, or the
📊 Show as diagram button under an answer). The model answers with a
closed ```mermaid flowchart, which `diagrams.py` extracts and validates
heuristically (header, balanced quotes/brackets, no HTML tags except
`<br/>`, no script content — not a full parser) and renders with a
built-in HTML component loading mermaid.js from a CDN. The raw code is
always shown under any rendered diagram; weak evidence means the code is
shown but not rendered, and invalid syntax gets a "didn't render" note.
No auto-detection, no repair retry in v1.

## Scale notes (demo → production)

The demo indexes hundreds of chunks into a single local FAISS file. A real
deployment (hundreds of thousands of pages) changes the *operations*, not the
*shape*:

- **Index sharding**: one FAISS index per source/space, merged at query time —
  isolates rebuilds and keeps any single index small enough to load fast.
- **Incremental updates**: track `last_modified` per document; re-embed only
  what changed instead of rebuilding the world nightly.
- **Refresh scheduling**: a nightly (or change-driven) job re-runs ingestion;
  the query path never blocks on indexing. The demo's manual
  `python -m rag_demo.ingest` is the degenerate case of this job.
- **Embedding cost**: at scale, batch embedding jobs beat per-query encoding;
  cache aggressively — document text changes far less often than people ask
  about it.
- **Retrieval quality**: `top_k` and chunk size are the two knobs that matter
  most. Tune them against the golden eval set (`evals/`), not by vibes.

## What the demo deliberately omits

- Real SharePoint connector (Confluence, Jira, Outlook, and AWS S3
  connectors exist in `enterprise.py`; SharePoint appears only as a
  docstring example). Credentials are still never in the repo —
  connectors read env vars only.
- User identity / auth UI (groups come from `RAG_DEMO_USER_GROUPS`, as they
  would from a gateway header in production).
- Multi-tenant isolation.

## Evaluation

`evals/eval_retrieval.py` runs a golden Q&A set against the retriever and
reports hit-rate@k, exiting non-zero below a threshold (`--retrieval` and
`--rerank` flags select the pipeline). `evals/eval_compare.py` prints a
dense-vs-hybrid-vs-rerank table and fails on regression vs
`evals/baselines.json` (real-embedding runs). `evals/per_source.py` runs
one prompt set per source (accuracy, retrieval, rerank before-vs-after,
precision at chunk + section) and skips with a reason — never crashes —
on a missing index or unconnected enterprise source. The discipline this installs:

- Retrieval is measured, not assumed — every chunking or embedding change
  gets a number before it ships.
- The golden set is versioned with the corpus; when docs change, the set gets
  reviewed, not silently trusted.
