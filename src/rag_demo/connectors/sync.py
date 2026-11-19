"""Sync a connector source into the FAISS index.

    python -m rag_demo.connectors.sync --source confluence
    python -m rag_demo.connectors.sync --source outlook --full

Incremental by design: a small state file per source records the last-sync
timestamp, and each run fetches only what changed since. Fetched documents
merge into a local document store (``connector-documents.json`` in the index
dir, keyed by doc_id), and the index is rebuilt from the merged set through
the existing chunk/embed/index pipeline (``ingest.build_index``) — no
indexing code is duplicated here.

Tokens never appear on the command line (no --token flag: that would leak
them into shell history) and never in output; only counts and source
labels are printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..config import Settings
from ..ingest import INDEX_DIRS, build_index
from ..sources import DocumentSource
from .base import Connector, to_documents
from .confluence import ConfluenceConnector
from .github import GitHubConnector
from .gmail import GmailConnector
from .outlook import OutlookConnector

CONNECTORS: dict[str, type[Connector]] = {
    "confluence": ConfluenceConnector,
    "outlook": OutlookConnector,
    "github": GitHubConnector,
    "gmail": GmailConnector,
}

# Display metadata for the GUI (and anything else listing sources).
# `token_config_key` is the connect() config key a pasted token maps to.
SOURCE_INFO: dict[str, dict] = {
    "confluence": {
        "label": "Confluence",
        "auth": "Personal Access Token",
        "required_env": ["CONFLUENCE_URL", "CONFLUENCE_PAT"],
        "token_env": "CONFLUENCE_PAT",
        "token_config_key": "pat",
        "hint": "Set CONFLUENCE_URL and CONFLUENCE_PAT, or paste a token below.",
    },
    "outlook": {
        "label": "Outlook",
        "auth": "OAuth2 bearer token",
        "required_env": ["OUTLOOK_BEARER_TOKEN"],
        "token_env": "OUTLOOK_BEARER_TOKEN",
        "token_config_key": "bearer_token",
        "hint": "Paste a Microsoft Graph bearer token below (see docs/connectors.md for how to get one).",
    },
    "github": {
        "label": "GitHub",
        "auth": "Personal Access Token",
        "required_env": ["GITHUB_TOKEN", "GITHUB_REPO"],
        "token_env": "GITHUB_TOKEN",
        "token_config_key": "token",
        "hint": "Create a fine-grained PAT (Contents, Issues, Pull requests: read-only) and set GITHUB_REPO, or paste a token below.",
    },
    "gmail": {
        "label": "Gmail",
        "auth": "OAuth2 access token",
        "required_env": ["GMAIL_ACCESS_TOKEN"],
        "token_env": "GMAIL_ACCESS_TOKEN",
        "token_config_key": "access_token",
        "hint": "Paste a Gmail OAuth2 access token below (see docs/connectors.md for how to get one).",
    },
}

_STATE_FILE = "sync-state.json"
_DOCS_FILE = "connector-documents.json"


class _ListSource(DocumentSource):
    """A DocumentSource over an in-memory list, for build_index reuse."""

    name = "connector-sync"

    def __init__(self, docs):
        self._docs = docs

    def load(self):
        return list(self._docs)


def _read_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def index_dir_for(source: str, settings: Settings | None = None) -> Path:
    """Per-source index directory (sibling of the default index dir)."""
    from ..config import get_settings

    settings = settings or get_settings()
    return Path(settings.index_dir).parent / INDEX_DIRS[source]


def read_source_status(source: str, settings: Settings | None = None) -> dict:
    """Last sync + doc count for a source, from its state/store files.

    Returns last_sync None and total 0 when the source has never synced.
    Never raises: a missing or corrupt file just means "never synced".
    """
    index_dir = index_dir_for(source, settings)
    state = _read_json(index_dir / _STATE_FILE, {})
    store = _read_json(index_dir / _DOCS_FILE, {})
    return {
        "source": source,
        "last_sync": state.get("last_sync"),
        "total_docs": len(store),
        "index_dir": str(index_dir),
    }


def is_configured(source: str) -> tuple[bool, list[str]]:
    """Whether a source can sync from env vars alone.

    Returns (configured, missing_env_vars). A token pasted at sync time
    is a runtime override, not configuration: the source still reports
    as not configured until its vars are in the environment.
    """
    import os

    if source == "github":
        token_ok = bool(
            os.environ.get("GITHUB_TOKEN") or os.environ.get("RAG_DEMO_GITHUB_TOKEN")
        )
        repo_ok = bool(
            os.environ.get("GITHUB_REPO")
            or os.environ.get("GITHUB_REPOS")
            or os.environ.get("RAG_DEMO_GITHUB_REPO")
        )
        missing = [
            name
            for name, ok in (("GITHUB_TOKEN", token_ok), ("GITHUB_REPO", repo_ok))
            if not ok
        ]
        return (not missing, missing)
    info = SOURCE_INFO[source]
    missing = [name for name in info["required_env"] if not os.environ.get(name)]
    return (not missing, missing)


def sync_source(
    connector: Connector,
    settings: Settings,
    state_path: Path | None = None,
    full: bool = False,
    config: dict | None = None,
) -> dict:
    """Run one incremental sync. Returns a summary dict (counts only).

    `config` is passed to ``connector.connect()`` (e.g. a token pasted
    at sync time); when omitted the connector reads env vars. The token
    is never written to disk — only counts and source labels land in
    the state/store files.
    """
    index_dir = Path(settings.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_path or (index_dir / _STATE_FILE)
    docs_path = index_dir / _DOCS_FILE

    state = _read_json(state_path, {})
    since = None if full else state.get("last_sync")
    # Hand the connector its persisted working state (e.g. GitHub blob
    # SHAs); it updates it in place and we persist it back below.
    if hasattr(connector, "state"):
        connector.state = state.get("connector_state", {}) or {}
    connector.connect(config)

    fetched = list(connector.fetch_since(since))
    store = _read_json(docs_path, {})
    for doc in fetched:
        key = doc.doc_id or doc.source_label
        store[key] = {
            "title": doc.title,
            "text": doc.text,
            "source_label": doc.source_label,
            "updated_at": doc.updated_at,
            "url": doc.url,
            "doc_id": doc.doc_id,
        }
    # Prune docs the source no longer holds. Only path-style ids for
    # this source are pruned ("#" ids are issues/PRs: append-only, so
    # they are never pruned). Connectors that cannot enumerate their
    # docs leave present_doc_ids empty and nothing is pruned.
    present = getattr(connector, "present_doc_ids", set()) or set()
    if present:
        prefix = f"{connector.source_name}:"
        for doc_id in list(store):
            if "#" in doc_id or not doc_id.startswith(prefix):
                continue
            if doc_id not in present:
                del store[doc_id]
    docs_path.write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")

    if not store:
        return {"source": connector.source_name, "fetched": 0, "total": 0, "last_sync": since}

    from langchain_core.documents import Document

    documents = [
        Document(
            page_content=entry["text"],
            metadata={
                "source": entry["source_label"],
                "doc_id": entry["doc_id"] or entry["source_label"],
                "allowed_groups": ["*"],
                "last_modified": entry["updated_at"],
            },
        )
        for entry in store.values()
    ]
    build_index(settings, source=_ListSource(documents))

    if fetched:
        # last_sync is the newest timestamp the source actually gave us.
        # A run that only fetched undated docs (e.g. GitHub README/docs)
        # leaves last_sync where it was; the next run still diffs docs by
        # SHA, so nothing is indexed twice.
        dated = [d.updated_at for d in fetched if d.updated_at]
        last_sync = max(dated) if dated else since
        state = {
            "source": connector.source_name,
            "last_sync": last_sync,
            "connector_state": getattr(connector, "state", {}) or {},
        }
        state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")
    else:
        last_sync = since

    return {
        "source": connector.source_name,
        "fetched": len(fetched),
        "total": len(store),
        "last_sync": last_sync,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync a connector source into the FAISS index.")
    parser.add_argument("--source", choices=sorted(CONNECTORS), default="confluence")
    parser.add_argument("--index-dir", default=None, help="index directory (default: per-source)")
    parser.add_argument("--state-file", default=None, help="sync state file (default: <index-dir>/sync-state.json)")
    parser.add_argument("--full", action="store_true", help="ignore the stored timestamp and fetch everything")
    parser.add_argument("--repo", action="append", default=None, help="repo as owner/name (repeatable; GitHub only)")
    parser.add_argument("--branch", default=None, help="branch to read docs from (GitHub only)")
    args = parser.parse_args(argv)

    index_dir = Path(args.index_dir) if args.index_dir else Path(INDEX_DIRS[args.source])
    settings = Settings(index_dir=index_dir)
    connector = CONNECTORS[args.source]()
    config: dict = {}
    if args.repo:
        config["repos"] = args.repo
    if args.branch:
        config["branch"] = args.branch
    summary = sync_source(
        connector,
        settings,
        state_path=Path(args.state_file) if args.state_file else None,
        full=args.full,
        config=config or None,
    )
    print(
        f"synced {summary['source']}: fetched {summary['fetched']} new/changed, "
        f"{summary['total']} total, last_sync={summary['last_sync']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
