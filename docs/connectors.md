# Enterprise connectors

Sidekick can index documents from enterprise systems, not just local files.
Each system is a **connector**: a small client that fetches documents and
hands them to the existing chunk/embed/index pipeline. Teams is out of scope
for now.

## Sources

| Source | Auth | Env vars |
| --- | --- | --- |
| Confluence | Personal Access Token (Server/DC) or API token (Cloud) | `CONFLUENCE_URL`, `CONFLUENCE_PAT`, `CONFLUENCE_SPACE` (optional) |
| Outlook | OAuth2 bearer token for Microsoft Graph | `OUTLOOK_BEARER_TOKEN`, `OUTLOOK_FOLDER` (default `inbox`), `OUTLOOK_MAX` (default `100`) |

## How it works

- `src/rag_demo/connectors/base.py` — the interface every source
  implements: `source_name`, `connect(config)`, and
  `fetch_since(since)`, yielding `ConnectorDoc` records (title, text,
  source label, updated timestamp, URL when the source has one).
- `src/rag_demo/connectors/confluence.py` — Confluence Cloud REST API.
  CQL search by space and `lastModified` date for incremental sync; page
  content from the storage-format body, stripped to plain text.
- `src/rag_demo/connectors/outlook.py` — Microsoft Graph messages.
  `$filter=receivedDateTime ge <ts>` for incremental sync; follows
  `@odata.nextLink` pagination; HTML bodies stripped to plain text.
- `src/rag_demo/connectors/sync.py` — the sync path. Fetches only what
  changed since the stored timestamp, merges into a local document store,
  and rebuilds the index through the existing `ingest.build_index`
  pipeline. Every chunk is labeled with a readable source:
  `confluence:<space>/<title>` for pages, `outlook:<subject>` for emails.

Sync is incremental on fetch, full on index: the state file
(`<index-dir>/sync-state.json`) records the last-sync timestamp, but the
index is rebuilt from the merged document set each run. That is the honest
trade-off for a demo-scale corpus; a production sync would update the
index in place.

One quirk to know: Confluence CQL `lastModified` is date-granular, so a
second sync re-fetches pages updated on the same day as the last sync.
The store dedupes by doc_id, so nothing is indexed twice and the
timestamp does not advance spuriously.

## Running a sync

```bash
python -m rag_demo.connectors.sync --source confluence
python -m rag_demo.connectors.sync --source outlook --full
```

Flags: `--index-dir` (default: the per-source dir from `ingest.INDEX_DIRS`),
`--state-file` (default: `<index-dir>/sync-state.json`), `--full` (ignore
the stored timestamp and fetch everything). There is deliberately no
`--token` flag: tokens on a command line leak into shell history.

## Mock mode (no live tokens needed)

Everything is verified against local mock servers, so the full flow runs
with no real credential:

- `tests/mock_servers.py` — a stdlib HTTP server that pretends to be
  Confluence Cloud (`/rest/api/content/search`, CQL filtering, start/limit
  pagination) or Microsoft Graph (`/me/mailFolders/.../messages`, `$filter`,
  `$top`, `$skiptoken` pagination). It serves the JSON fixtures on disk and
  records requests as (method, path, auth-scheme) only.
- Fixtures: `fixtures/confluence/pages-*.json` (6 pages, space OPS) and
  `tests/fixtures/outlook/messages-*.json` (4 emails).
- Run the suite: `PYTHONPATH=src:. pytest tests/test_connectors.py`
- Run the real CLI against the mocks (fake embeddings, fake tokens):

```bash
RAG_DEMO_EMBEDDINGS=fake python scripts/mock_sync_demo.py
```

To point a connector at the mock by hand, start the server (see
`tests/mock_servers.py:mock_server`) and set `CONFLUENCE_URL` (or
`OUTLOOK_BASE_URL`) to its `http://127.0.0.1:<port>` address with a fake
token. The same code path runs against the mock and the real service —
only the base URL changes.

## Token-handling rules

- Tokens come from environment variables only, or from a `config` dict
  passed to `connect()`. Never hardcoded, never committed, never in a
  fixture (tests use literal `FAKE-...` values).
- A connector holds its token on the instance only. Requests are logged
  with method, URL path, and status — never headers, never the token.
- The sync state file, the document store, `chunks.json`, and the manifest
  never contain a token. A test (`test_token_never_appears_in_logs_or_output`
  plus the sync tests) asserts this on every run.

## Outlook OAuth follow-up (not in this build)

The Outlook connector takes an already-acquired bearer token. Acquiring
one needs, in order:

1. An Azure AD app registration with the delegated `Mail.Read`
   permission (admin consent in most tenants).
2. An authorization-code flow (user signs in once, the app receives a
   code) or a device-code flow for headless use.
3. Exchange the code for an access token (and a refresh token) at the
   Microsoft identity token endpoint; refresh it before expiry (Graph
   access tokens typically live 1 hour).
4. Pass the access token as `OUTLOOK_BEARER_TOKEN`.

A future change can add a small `outlook_auth.py` that performs steps
2–3 and hands the token to this client; the client itself does not need
to change.

## Relation to the older enterprise clients

`src/rag_demo/enterprise.py` already has full-load clients
(`ConfluenceSource`, `JiraSource`, `OutlookSource`) behind the
`DocumentSource` interface, used by `python -m rag_demo.ingest --source`.
This package does not replace them: it adds incremental sync, a per-source
state file, mock-first verification, and readable source labels. Jira and
GitLab are the natural next connectors on this interface.
