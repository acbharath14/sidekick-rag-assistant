"""Mock GitHub sync demo: runs the real sync CLI against a local mock.

No token, no network. The mock serves the fixtures in
tests/fixtures/github/ (a fake README, three docs, two issues, three
PRs) and the sync indexes them with fake embeddings into a temp dir.

Usage:
    RAG_DEMO_EMBEDDINGS=fake python scripts/mock_sync_demo.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("RAG_DEMO_FAKE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from mock_servers import GitHubMockHandler, mock_server  # noqa: E402
from rag_demo.config import get_settings  # noqa: E402
from rag_demo.connectors.sync import sync_source  # noqa: E402
from rag_demo.retriever import Retriever  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "github"
REPO = "acme/widgets"


def _load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def main() -> None:
    GitHubMockHandler.reset(
        {
            "repos": {
                REPO: {
                    "readme": _load("readme.json"),
                    "tree": _load("tree.json"),
                    "blobs": _load("blobs.json"),
                    "issues": _load("issues.json"),
                    "pulls": _load("pulls.json"),
                }
            }
        }
    )
    with mock_server(GitHubMockHandler) as base_url:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            settings = get_settings()
            summary = sync_source(
                "github",
                {
                    "token": "FAKE-GITHUB-TOKEN",
                    "repos": [REPO],
                    "base_url": base_url,
                },
                settings,
                full=True,
                index_dir=tmp_path / "index",
                state_file=tmp_path / "sync-state.json",
                store_file=tmp_path / "connector-documents.json",
            )
            store = json.loads(
                (tmp_path / "connector-documents.json").read_text()
            )
            print("synced docs:")
            for record in store:
                print(f"  - {record['source_label']}")
            hits = Retriever(settings).search("widget catalog bulk import", k=3)
            print("sample search hit:", hits[0].source if hits else "(none)")
            assert summary["total"] == 9, summary
            assert all(
                h.source.startswith("github:acme/widgets") for h in hits
            ), [h.source for h in hits]
    print("mock sync demo OK")


if __name__ == "__main__":
    main()
