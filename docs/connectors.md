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
| GitHub | Fine-grained Personal Access Token | `GITHUB_TOKEN`, `GITHUB_REPO` (or `GITHUB_REPOS`, comma-separated), `GITHUB_BRANCH` (default `main`) |
| Gmail | OAuth2 access token | `GMAIL_ACCESS_TOKEN`, `GMAIL_QUERY` (default `in:inbox`), `GMAIL_MAX` (default `100`) |

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
- `src/rag_demo/connectors/github.py` — GitHub REST API. README plus
  `docs/*.md` (git-tree SHA diff for incremental sync), issues (server-side
  `since` filter), and pull requests (newest-first list, client-side
  `updated_at` filter). Labels: `github:<owner>/<repo>/<path>` for docs,
  `github:<owner>/<repo>#<number>` for issues and PRs.
- `src/rag_demo/connectors/gmail.py` — Gmail REST API. `messages.list`
  (newest first, `pageToken` pagination) plus `messages.get` per message;
  plain-text body preferred, HTML stripped, snippet fallback. Incremental
  sync filters client-side by `internalDate` and stops at the first
  message at or before the stored cutoff. Labels: `gmail:<subject>`.
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
python -m rag_demo.connectors.sync --source github --full
python -m rag_demo.connectors.sync --source gmail --full
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
  pagination), Microsoft Graph (`/me/mailFolders/.../messages`, `$filter`,
  `$top`, `$skiptoken` pagination), GitHub (`/repos/.../readme`,
  `/git/trees`, `/git/blobs`, `/issues` with a `since` filter, `/pulls`,
  `Link` pagination; 401 without an Authorization header), or Gmail
  (`/gmail/v1/users/me/messages` with `maxResults`/`pageToken`
  pagination, `/messages/{id}` full format; 401 without an Authorization
  header). It serves the JSON fixtures on disk and records requests as
  (method, path, auth-scheme) only.
- Fixtures: `fixtures/confluence/pages-*.json` (6 pages, space OPS),
  `tests/fixtures/outlook/messages-*.json` (4 emails),
  `tests/fixtures/github/` (a fake README, three docs, two issues, three
  PRs for `acme/widgets`), and `tests/fixtures/gmail/messages-*.json`
  (4 emails: plain text, HTML-only, multipart, and one with no body
  parts, which falls back to the snippet).

## GitHub connector

Fetches, per configured repo:

- the README,
- `docs/*.md` files,
- issues (title + body + state + updated time),
- pull requests (title + body + state + updated time).

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
export GITHUB_TOKEN=<your token>
export GITHUB_REPO=owner/name     # or GITHUB_REPOS=owner/a,owner/b
python -m rag_demo.connectors.sync --source github --full
```

Useful flags: `--repo owner/name` (repeatable, overrides the env
vars), `--branch` (default `main`), `--index-dir`, `--state-file`.
The synced index lands in `.faiss_index_github_connector` (distinct
from the `github-docs` source's `.faiss_index_github`). To ask
questions against it:

```bash
RAG_DEMO_INDEX_DIR=.faiss_index_github_connector ./scripts/dev.sh
```

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

## Gmail connector

Fetches messages matching `GMAIL_QUERY` (default `in:inbox`): subject,
from, date, and body text. Plain-text parts win over HTML; HTML-only
messages are stripped to text; a message with no body parts falls back
to its snippet. Citations read `gmail:<subject>`, with a link back to
the message in Gmail web.

### Setup

The connector takes an already-acquired OAuth2 access token (the same
shape as Outlook: a bearer token, not a password or app password).
Acquiring one is the documented follow-up below; once you hold one:

```bash
export GMAIL_ACCESS_TOKEN=<your token>
export GMAIL_QUERY='in:inbox'   # optional; any Gmail search query works
python -m rag_demo.connectors.sync --source gmail --full
```

The synced index lands in `.faiss_index_gmail`. To ask questions
against it:

```bash
RAG_DEMO_INDEX_DIR=.faiss_index_gmail ./scripts/dev.sh
```

### How incremental sync works

- The Gmail list endpoint returns messages newest first. The client
  walks the list and stops at the first message whose `internalDate`
  is at or before the stored cutoff, so an incremental run only fetches
  (and downloads the full body of) messages newer than the last sync.
- The newest message date is stored as `last_sync`; the newest
  `internalDate` in milliseconds is also kept in the connector state
  (`connector_state.newest_internal_date_ms`).
- Messages are append-only: a message that leaves the inbox (archived
  or deleted) is **not** pruned from the store. A mailbox is a stream,
  not a directory listing, so once a message is indexed it stays in
  the knowledge base.

### Limitations and follow-ups

- **OAuth acquisition is not built.** The token must be acquired
  outside Sidekick for now (see below).
- **No `historyId` sync.** The date cutoff cannot see label changes
  (read/unread, starred) or deletions, and two messages can share
  the same `internalDate` millisecond. A `historyId`-based sync is
  the follow-up: it reports exactly what changed since the last
  sync, including deletions.
- The list is walked page by page, so a mailbox with thousands of
  new messages downloads the full message of each one on the first
  sync. `GMAIL_MAX` caps a run; narrow `GMAIL_QUERY` to scope it.

## Gmail OAuth follow-up (not in this build)

The Gmail connector takes an already-acquired access token. Acquiring
one needs, in order:

1. A Google Cloud project with the Gmail API enabled and an OAuth
   consent screen (external user type for a personal account; the app
   stays in testing mode unless published).
2. OAuth client credentials (Desktop app type) with the
   `https://www.googleapis.com/auth/gmail.readonly` scope.
3. An authorization-code flow (user signs in once in a browser, the
   app receives a code) and exchange the code for an access token
   (and a refresh token) at Google's token endpoint; refresh it
   before expiry (Google access tokens typically live 1 hour).
4. Pass the access token as `GMAIL_ACCESS_TOKEN`.

A future change can add a small `gmail_auth.py` that performs steps
2–3 and hands the token to this client; the client itself does not
need to change.

## Token-handling rules
- Run the suite: `PYTHONPATH=src:. pytest tests/test_connectors.py`
- Run the real CLI against the mocks (fake embeddings, fake tokens):

```bash
RAG_DEMO_EMBEDDINGS=fake python scripts/mock_sync_demo.py
```

To point a connector at the mock by hand, start the server (see
`tests/mock_servers.py:mock_server`) and set `CONFLUENCE_URL` (or
`OUTLOOK_BASE_URL`, `GITHUB_BASE_URL`, `GMAIL_BASE_URL`) to its
`http://127.0.0.1:<port>` address with a fake token. The same code
path runs against the mock and the real service — only the base URL
changes.

## GUI (Connectors view)

The React GUI has a **Connectors** tab next to Chat and Documents. Each
source gets a card: configured or not (with the missing env vars),
last sync, document count, and a Sync button. When a source has no
token configured, the card shows a token input; the pasted token is
used once for that sync and never stored. A Full sync checkbox ignores
the stored timestamp.

The view talks to three endpoints (all behind the same API-key auth as
the rest of the API):

- `GET /api/connectors` — every source with configured + last-sync status.
- `POST /api/connectors/{source}/sync` — starts a sync in a background
  thread and returns a job id. Body: `{token?, full?}`. One sync per
  source at a time (a second start returns 409).
- `GET /api/connectors/jobs/{job_id}` — poll the job until `done` or
  `error`. Job payloads never contain a token.

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
