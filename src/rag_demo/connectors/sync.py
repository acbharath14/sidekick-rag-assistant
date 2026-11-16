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
from .outlook import OutlookConnector

CONNECTORS: dict[str, type[Connector]] = {
    "confluence": ConfluenceConnector,
    "outlook": OutlookConnector,
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


def sync_source(
    connector: Connector,
    settings: Settings,
    state_path: Path | None = None,
    full: bool = False,
) -> dict:
    """Run one incremental sync. Returns a summary dict (counts only)."""
    index_dir = Path(settings.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_path or (index_dir / _STATE_FILE)
    docs_path = index_dir / _DOCS_FILE

    state = _read_json(state_path, {})
    since = None if full else state.get("last_sync")
    connector.connect()

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
        last_sync = max(d.updated_at for d in fetched if d.updated_at) or since
        state = {"source": connector.source_name, "last_sync": last_sync}
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
    args = parser.parse_args(argv)

    index_dir = Path(args.index_dir) if args.index_dir else Path(INDEX_DIRS[args.source])
    settings = Settings(index_dir=index_dir)
    connector = CONNECTORS[args.source]()
    summary = sync_source(
        connector,
        settings,
        state_path=Path(args.state_file) if args.state_file else None,
        full=args.full,
    )
    print(
        f"synced {summary['source']}: fetched {summary['fetched']} new/changed, "
        f"{summary['total']} total, last_sync={summary['last_sync']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
