"""Sync CLI: fetch from a connector, merge into the local store, rebuild.

Usage (GitHub)::

    GITHUB_TOKEN=... GITHUB_REPO=owner/name \\
        python -m rag_demo.connectors.sync --source github --full

The sync is incremental: ``sync-state.json`` records the last sync time
plus per-source state (e.g. GitHub blob SHAs), and ``connector-
documents.json`` holds the merged store keyed by doc id. The FAISS
index rebuilds through the existing ``ingest.build_index`` — no indexing
code is duplicated here. Each source gets its own index directory so the
default corpus stays pristine.

New sources register in ``REGISTRY`` as (module, class) and pick up the
same CLI; their env-var mapping is added in ``main()`` the same way.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..config import Settings, get_settings
from ..ingest import build_index
from ..sources import DocumentSource
from .base import ConnectorDoc, to_documents

REGISTRY = {
    # source name -> (module under rag_demo.connectors, class name)
    "github": ("rag_demo.connectors.github", "GitHubConnector"),
}

INDEX_DIRS = {
    "github": ".faiss_index_github",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


class _JsonDocumentsSource(DocumentSource):
    """DocumentSource backed by the merged connector store."""

    def __init__(self, records: list[dict], source_name: str):
        self.name = f"{source_name}-connector"
        self._records = records

    def load(self):
        return to_documents([ConnectorDoc(**r) for r in self._records])


def _load_connector(source_name: str):
    module_name, class_name = REGISTRY[source_name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)()


def sync_source(
    source_name: str,
    config: dict,
    settings: Settings | None = None,
    *,
    full: bool = False,
    index_dir: str | Path | None = None,
    state_file: str | Path | None = None,
    store_file: str | Path | None = None,
) -> dict:
    """Run one sync. Returns a summary dict (source, fetched, total,
    last_sync, index_dir)."""
    settings = settings or get_settings()
    index_dir = Path(index_dir) if index_dir else (
        settings.index_dir.parent / INDEX_DIRS[source_name]
    )
    settings.index_dir = index_dir
    state_file = Path(state_file) if state_file else index_dir / "sync-state.json"
    store_file = (
        Path(store_file) if store_file else index_dir / "connector-documents.json"
    )

    state = _load_json(state_file, {"last_sync": None, "repos": {}})
    store_records = _load_json(store_file, [])
    store = {r["doc_id"]: r for r in store_records}

    since = None if full else _parse_iso(state.get("last_sync"))
    connector = _load_connector(source_name)
    connector.connect(config)
    fetched = list(connector.fetch_since(since, state=state))

    for doc in fetched:
        store[doc.doc_id] = {
            "title": doc.title,
            "text": doc.text,
            "source_label": doc.source_label,
            "updated_at": doc.updated_at,
            "url": doc.url,
            "doc_id": doc.doc_id,
        }

    # Prune docs the source no longer holds. GitHub doc ids are path-style
    # (github:owner/repo/docs/guide.md); issues/PRs carry "#" and are
    # append-only, so they are never pruned here.
    present = getattr(connector, "present_doc_ids", set())
    repos = getattr(connector, "repos", [])
    if present and repos:
        for doc_id in list(store):
            if "#" in doc_id:
                continue
            if any(doc_id.startswith(f"github:{repo}/") for repo in repos):
                if doc_id not in present:
                    del store[doc_id]

    state["last_sync"] = _now_iso()
    _save_json(store_file, list(store.values()))
    _save_json(state_file, connector.state)

    if store:
        build_index(settings, source=_JsonDocumentsSource(list(store.values()), source_name), incremental=True)
    else:
        print(f"synced {source_name}: store is empty, nothing to index")

    summary = {
        "source": source_name,
        "fetched": len(fetched),
        "total": len(store),
        "last_sync": state["last_sync"],
        "index_dir": str(index_dir),
    }
    print(
        f"synced {source_name}: fetched {summary['fetched']} new/changed, "
        f"{summary['total']} total, last_sync={summary['last_sync']}"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sync an enterprise connector into the Sidekick index."
    )
    parser.add_argument("--source", choices=sorted(REGISTRY), required=True)
    parser.add_argument("--full", action="store_true", help="ignore last_sync and fetch everything")
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--state-file", default=None)
    parser.add_argument("--store-file", default=None)
    parser.add_argument("--repo", action="append", default=None,
                        help="repo as owner/name (repeatable; GitHub only)")
    parser.add_argument("--base-url", default=None,
                        help="API base URL override (tests point this at a mock)")
    parser.add_argument("--branch", default=None, help="branch to read docs from (GitHub only)")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.source == "github":
        repos = list(args.repo or [])
        if not repos:
            multi = os.environ.get("GITHUB_REPOS", "")
            repos = [r.strip() for r in multi.split(",") if r.strip()]
        if not repos:
            single = os.environ.get("GITHUB_REPO") or settings.github_repo
            repos = [single] if single else []
        config = {
            "token": os.environ.get("GITHUB_TOKEN") or settings.github_token,
            "repos": repos,
            "base_url": args.base_url or os.environ.get("GITHUB_BASE_URL"),
            "branch": args.branch or os.environ.get("GITHUB_BRANCH"),
        }
    else:  # pragma: no cover - new sources add their env mapping here
        config = {}

    sync_source(
        args.source,
        config,
        settings,
        full=args.full,
        index_dir=args.index_dir,
        state_file=args.state_file,
        store_file=args.store_file,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
