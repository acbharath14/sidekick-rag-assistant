# Enterprise connectors

Sidekick can build its knowledge base from enterprise systems, not just
local files and uploads. Each source is a small client behind one
interface, and sync is incremental: only what changed since the last
run is fetched.

This tree currently ships the **GitHub** connector. The interface is
designed so Confluence, Outlook, and other sources register the same
way (see "Adding a source" below).

## The framework

- `src/rag_demo/connectors/base.py` — the `Connector` interface:
  `source_name`, `connect(config)`, and `fetch_since(since, state)`
  yielding `ConnectorDoc` records (title, text, `source_label`,
  `updated_at`, url, `doc_id`). `to_documents()` adapts records to
  langchain Documents for indexing.
- `src/rag_demo/connectors/sync.py` — the sync layer and CLI. It loads
  the per-source `sync-state.json` (last sync time + working state such
  as blob SHAs), calls the connector, merges fetched records into
  `connector-documents.json` keyed by `doc_id`, prunes docs the source
  no longer holds, and rebuilds the FAISS index through the existing
  `ingest.build_index`. Each source gets its own index directory, so
  the default corpus stays pristine.
- `tests/mock_servers.py` — stdlib `ThreadingHTTPServer` mocks. Every
  connector is built and verified against a mock first: the client runs
  its real code path with only the base URL changed, so no live
  credential is needed to develop or test.

## Token rules (every connector)

- Tokens come from environment variables only, never from a CLI flag
  (so they never land in shell history) and never from a file in the
  repo.
- A token is held in memory on the connector instance, sent only as an
  `Authorization` header, and never written to disk, logs, or output.
- Tests use literal `FAKE-...` tokens only.

## GitHub connector

Fetches, per configured repo:

- the README,
- `docs/*.md` files,
- issues (title + body + state + updated time),
- pull requests (title + body + state + updated time).

Citations are readable: `github:<owner>/<repo>/<path>` for docs and
`github:<owner>/<repo>#<number>` for issues and PRs.

### Setup

Create a fine-grained personal access token:

1. GitHub → Settings → Developer settings → Personal access tokens →
   Fine-grained tokens → Generate new token.
2. Resource owner: your account or org. Repository access: **Only
   select repositories**, and pick the repos to index.
3. Repository permissions: **Contents: Read-only**, **Issues:
   Read-only**, **Pull requests: Read-only**. (Metadata is read-only
   by default.) No other permissions are needed.
4. Set an expiration (the shortest that works for you) and copy the
   token once — GitHub never shows it again.

Then:

```bash
export GITHUB_TOKEN=<the token>   # never commit this
export GITHUB_REPO=owner/name     # or GITHUB_REPOS=owner/a,owner/b
```

### Run a sync

```bash
# Full sync (first run), index lands in .faiss_index_github
python -m rag_demo.connectors.sync --source github --full

# Later runs are incremental
python -m rag_demo.connectors.sync --source github
```

Useful flags: `--repo owner/name` (repeatable, overrides the env
vars), `--branch` (default `main`), `--index-dir`, `--state-file`,
`--store-file`. `GITHUB_BASE_URL` points the client at a different API
root (the tests point it at a mock).

To ask questions against the synced index, point the app at it:

```bash
RAG_DEMO_INDEX_DIR=.faiss_index_github ./scripts/dev.sh
```

### Mock mode (no token needed)

```bash
RAG_DEMO_EMBEDDINGS=fake python scripts/mock_sync_demo.py
```

This starts a local mock GitHub server serving the fixtures in
`tests/fixtures/github/` (a fake README, three docs, two issues, three
PRs), runs the real sync CLI against it, and prints the fetched docs
with their source labels.

### How incremental sync works

- **Docs:** the git tree is fetched and blob SHAs are diffed against
  the stored state, so only new or changed files are downloaded.
  Files deleted from the repo are pruned from the store.
- **Issues:** the issues endpoint supports a server-side `since`
  filter.
- **Pull requests:** the pulls endpoint has no `since` parameter, so
  the client lists newest-first and filters by `updated_at` itself.

### Limitations and follow-ups

- **Raw code files are not fetched** in this pass (docs + issues +
  PRs only). Indexing code is the follow-up: it needs a file picker
  (which paths/extensions) and a size cap, since repos are mostly code.
- The README has no updated timestamp in the API response, so it is
  tracked by blob SHA instead.
- The git-tree endpoint truncates very large repos; a repo that hits
  truncation fails loudly rather than indexing silently partial docs.
- The pulls list is filtered client-side, so a repo with thousands of
  PRs downloads the full list on every incremental run.

## Adding a source

1. Add a client in `src/rag_demo/connectors/<name>.py` implementing
   the `Connector` interface, reading its config from env vars.
2. Register it in `sync.py` (`REGISTRY` + `INDEX_DIRS`) and add its
   env-var mapping in `main()`.
3. Add a mock server in `tests/mock_servers.py`, fixtures under
   `tests/fixtures/<name>/`, and tests in `tests/test_connectors.py`
   covering full + incremental fetch, pagination, missing-config
   errors, source labels, token-never-in-logs, and a sync end-to-end.
