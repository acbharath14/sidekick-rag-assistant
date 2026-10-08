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
| Retrieve | `retriever.py` | `top_k`, score threshold |
| Generate | `chain.py` | system prompt, LLM factory |

## The connector pattern

`DocumentSource` is the whole enterprise story in one interface:

```python
class DocumentSource(ABC):
    name: str
    def load(self) -> list[Document]: ...
```

The demo ships two: `MarkdownDirectorySource` (local files) and
`PlaywrightDocsSource` (sparse git checkout). A Confluence connector is the
same shape:

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
            "permissions": page.restrictions,   # see below
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
                                                                      ├─▶ RRF fuse ─▶ ACL filter ─▶ rerank? ─▶ top-k ─▶ prompt ─▶ LLM
```

- `RAG_DEMO_RETRIEVAL=dense` (default) skips the BM25 branch.
- `RAG_DEMO_RERANK=1` reranks the fused top-20 with a cross-encoder.
- `evals/eval_compare.py` measures dense vs hybrid vs hybrid+rerank on the
  golden set; baselines live in `evals/baselines.json`.

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

- Real Confluence/SharePoint/Jira connectors (no credentials by design —
  the mock source demonstrates the connector contract).
- User identity / auth UI (groups come from `RAG_DEMO_USER_GROUPS`, as they
  would from a gateway header in production).
- Multi-tenant isolation.

## Evaluation

`evals/eval_retrieval.py` runs a golden Q&A set against the retriever and
reports hit-rate@k, exiting non-zero below a threshold (`--retrieval` and
`--rerank` flags select the pipeline). `evals/eval_compare.py` prints a
dense-vs-hybrid-vs-rerank table and fails on regression vs
`evals/baselines.json` (real-embedding runs). The discipline this installs:

- Retrieval is measured, not assumed — every chunking or embedding change
  gets a number before it ships.
- The golden set is versioned with the corpus; when docs change, the set gets
  reviewed, not silently trusted.
