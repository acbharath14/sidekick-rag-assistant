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

## Permission-aware retrieval (design note)

In an enterprise, not every user may see every document. The demo skips this
(deliberately — the corpus is fictional and public), but the pattern is:

1. **At index time**, store each chunk's ACLs in metadata (space permissions,
   page restrictions, group membership).
2. **At query time**, resolve the asker's groups *once*, then filter:
   - cheap path: metadata filter on the vector query, or
   - safe path: post-filter retrieved hits before they reach the prompt.

Never rely on the LLM to "know" what the user may see — a model that has
seen a secret in context *will* leak it under the right prompt. Filter
before the prompt, not after the answer.

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

## Evaluation

`evals/eval_retrieval.py` runs a golden Q&A set against the retriever and
reports hit-rate@k, exiting non-zero below a threshold. The discipline this
installs:

- Retrieval is measured, not assumed — every chunking or embedding change
  gets a number before it ships.
- The golden set is versioned with the corpus; when docs change, the set gets
  reviewed, not silently trusted.

## What the demo deliberately omits

- Real enterprise connectors (no credentials, no proprietary systems).
- Permission filtering (documented above, not implemented).
- User identity / audit logging.
- Multi-tenant isolation.

These are production concerns. Their *shapes* are documented here so the demo
reads as a credible distillation, not a toy that never considered them.
