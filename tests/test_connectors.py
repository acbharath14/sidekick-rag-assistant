"""GitHub connector tests (mock-first, RAG_DEMO_FAKE=1: no network,
no live tokens, deterministic fake embeddings).

The mock GitHub server serves the fixtures in tests/fixtures/github/;
the connector runs its real code path with only the base URL changed.
"""

import os

os.environ["RAG_DEMO_FAKE"] = "1"

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_demo.config import get_settings
from rag_demo.connectors.github import GitHubConnector
from rag_demo.connectors.sync import main as sync_main
from rag_demo.connectors.sync import sync_source
from rag_demo.retriever import Retriever

from mock_servers import GitHubMockHandler, mock_server

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "github"
FAKE_TOKEN = "FAKE-GITHUB-TOKEN"
REPO = "acme/widgets"


def _load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _repo_data(tree="tree.json"):
    return {
        "readme": _load("readme.json"),
        "tree": _load(tree),
        "blobs": _load("blobs.json"),
        "issues": _load("issues.json"),
        "pulls": _load("pulls.json"),
    }


@pytest.fixture()
def github_mock():
    GitHubMockHandler.reset({"repos": {REPO: _repo_data()}})
    with mock_server(GitHubMockHandler) as base_url:
        yield base_url


def _connector(base_url, repos=(REPO,), **kw):
    connector = GitHubConnector(base_url=base_url, **kw)
    connector.connect({"token": FAKE_TOKEN, "repos": list(repos)})
    return connector


def _labels(docs):
    return {d.source_label for d in docs}


# --- fetch ---------------------------------------------------------------


def test_github_full_fetch(github_mock):
    docs = list(_connector(github_mock).fetch_since(None))
    assert len(docs) == 9  # readme 1 + docs 3 + issues 2 + pulls 3
    assert _labels(docs) == {
        "github:acme/widgets/README.md",
        "github:acme/widgets/docs/guide.md",
        "github:acme/widgets/docs/api.md",
        "github:acme/widgets/docs/old.md",
        "github:acme/widgets#1",
        "github:acme/widgets#2",
        "github:acme/widgets#4",
        "github:acme/widgets#5",
        "github:acme/widgets#6",
    }
    # The issues endpoint also returns PRs; they must not leak into issues.
    assert not any(d.doc_id == "github:acme/widgets#issue-3" for d in docs)
    assert {d.doc_id for d in docs if "#pr-" in d.doc_id} == {
        "github:acme/widgets#pr-4",
        "github:acme/widgets#pr-5",
        "github:acme/widgets#pr-6",
    }


def test_github_source_labels(github_mock):
    docs = list(_connector(github_mock).fetch_since(None))
    by_label = {d.source_label: d for d in docs}
    assert by_label["github:acme/widgets/docs/guide.md"].doc_id == (
        "github:acme/widgets/docs/guide.md"
    )
    assert by_label["github:acme/widgets#1"].doc_id == (
        "github:acme/widgets#issue-1"
    )
    assert by_label["github:acme/widgets#4"].doc_id == (
        "github:acme/widgets#pr-4"
    )
    # Labels carry no raw code paths: only README + docs/*.md are fetched.
    assert not any("src/main.py" in label for label in by_label)


def test_github_pagination(github_mock):
    docs = list(_connector(github_mock, per_page=2).fetch_since(None))
    pulls = [d for d in docs if "#pr-" in d.doc_id]
    assert len(pulls) == 3
    pulls_paths = [p for _, p, _ in GitHubMockHandler.requests if "/pulls" in p]
    assert any("page=2" in p for p in pulls_paths), pulls_paths


def test_github_incremental_fetch(github_mock):
    connector = _connector(github_mock)
    state: dict = {}
    first = list(connector.fetch_since(None, state=state))
    assert len(first) == 9

    # Change the world: guide updated, old.md deleted, new.md added,
    # one new issue. The API clock is now past the fixture updates.
    data = GitHubMockHandler.data["repos"][REPO]
    data["tree"] = _load("tree-updated.json")
    data["issues"] = list(data["issues"]) + [
        {
            "id": 7,
            "number": 7,
            "title": "Bulk import needs a progress bar",
            "body": "Large CSVs take minutes with no feedback.",
            "state": "open",
            "updated_at": "2026-10-10T09:00:00Z",
            "html_url": "https://github.com/acme/widgets/issues/7",
        }
    ]
    GitHubMockHandler.requests = []
    since = datetime(2026, 10, 6, tzinfo=timezone.utc)
    second = list(connector.fetch_since(since, state=state))

    assert _labels(second) == {
        "github:acme/widgets/docs/guide.md",
        "github:acme/widgets/docs/new.md",
        "github:acme/widgets#7",
    }
    blob_paths = [p for _, p, _ in GitHubMockHandler.requests if "/git/blobs/" in p]
    assert any("blob-sha-guide-2" in p for p in blob_paths)
    assert any("blob-sha-new-1" in p for p in blob_paths)
    assert not any("blob-sha-api-1" in p for p in blob_paths)  # unchanged: not re-fetched
    issues_paths = [p for _, p, _ in GitHubMockHandler.requests if "/issues" in p]
    assert issues_paths and any("since=2026-10-06" in p for p in issues_paths)
    assert "github:acme/widgets/docs/old.md" not in connector.present_doc_ids


def test_github_missing_config(github_mock):
    with pytest.raises(ValueError, match="token"):
        GitHubConnector(base_url=github_mock).connect({"repos": [REPO]})
    with pytest.raises(ValueError, match="repo"):
        GitHubConnector(base_url=github_mock).connect({"token": FAKE_TOKEN})


def test_github_mock_requires_auth(github_mock):
    import requests

    resp = requests.get(f"{github_mock}/repos/{REPO}/readme", timeout=10)
    assert resp.status_code == 401


def test_github_token_never_logged(github_mock, caplog):
    connector = _connector(github_mock)
    list(connector.fetch_since(None))
    assert FAKE_TOKEN not in caplog.text
    assert FAKE_TOKEN not in json.dumps(GitHubMockHandler.requests)
    assert {scheme for _, _, scheme in GitHubMockHandler.requests} == {"Bearer"}


def test_github_multiple_repos(github_mock):
    GitHubMockHandler.data["repos"]["acme/gadgets"] = _repo_data()
    docs = list(
        _connector(github_mock, repos=(REPO, "acme/gadgets")).fetch_since(None)
    )
    assert len(docs) == 18
    assert "github:acme/gadgets/docs/guide.md" in _labels(docs)
    assert "github:acme/widgets/docs/guide.md" in _labels(docs)


# --- sync ----------------------------------------------------------------


def _sync_config(base_url, repos=(REPO,)):
    return {"token": FAKE_TOKEN, "repos": list(repos), "base_url": base_url}


def test_github_sync_end_to_end(github_mock, tmp_path):
    settings = get_settings()
    summary = sync_source(
        "github",
        _sync_config(github_mock),
        settings,
        full=True,
        index_dir=tmp_path / "index",
        state_file=tmp_path / "sync-state.json",
        store_file=tmp_path / "connector-documents.json",
    )
    assert summary["fetched"] == 9 and summary["total"] == 9
    assert (tmp_path / "index" / "index.faiss").exists()
    store = json.loads((tmp_path / "connector-documents.json").read_text())
    assert len(store) == 9

    hits = Retriever(settings).search("widget catalog bulk import", k=3)
    assert hits
    assert all(h.source.startswith("github:acme/widgets") for h in hits)


def test_github_sync_second_run_fetches_nothing(github_mock, tmp_path):
    settings = get_settings()
    kwargs = dict(
        index_dir=tmp_path / "index",
        state_file=tmp_path / "sync-state.json",
        store_file=tmp_path / "connector-documents.json",
    )
    first = sync_source("github", _sync_config(github_mock), settings, full=True, **kwargs)
    second = sync_source("github", _sync_config(github_mock), settings, **kwargs)
    assert first["fetched"] == 9
    assert second["fetched"] == 0 and second["total"] == 9


def test_github_sync_prunes_deleted_docs(github_mock, tmp_path):
    settings = get_settings()
    kwargs = dict(
        index_dir=tmp_path / "index",
        state_file=tmp_path / "sync-state.json",
        store_file=tmp_path / "connector-documents.json",
    )
    sync_source("github", _sync_config(github_mock), settings, full=True, **kwargs)
    GitHubMockHandler.data["repos"][REPO]["tree"] = _load("tree-updated.json")
    sync_source("github", _sync_config(github_mock), settings, **kwargs)
    store = json.loads((tmp_path / "connector-documents.json").read_text())
    labels = {r["source_label"] for r in store}
    assert "github:acme/widgets/docs/old.md" not in labels
    assert "github:acme/widgets/docs/new.md" in labels
    assert len(store) == 9


def test_github_sync_cli(github_mock, tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("GITHUB_REPO", REPO)
    monkeypatch.setenv("GITHUB_BASE_URL", github_mock)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sync",
            "--source",
            "github",
            "--full",
            "--index-dir",
            str(tmp_path / "index"),
            "--state-file",
            str(tmp_path / "sync-state.json"),
            "--store-file",
            str(tmp_path / "connector-documents.json"),
        ],
    )
    sync_main()
    assert (tmp_path / "index" / "index.faiss").exists()
    store = json.loads((tmp_path / "connector-documents.json").read_text())
    assert len(store) == 9
