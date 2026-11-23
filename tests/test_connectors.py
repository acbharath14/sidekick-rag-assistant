"""Enterprise connector tests (mock-first, fake tokens only).

Every test runs against the local mock servers in mock_servers.py — no live
Confluence, no live Graph, no real credential anywhere. The only token
values in this file are obvious fakes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from mock_servers import MockConfluenceHandler, MockGraphHandler, mock_server
from rag_demo.config import Settings
from rag_demo.connectors.base import ConnectorDoc, to_documents
from rag_demo.connectors.confluence import ConfluenceConnector
from rag_demo.connectors.outlook import OutlookConnector
from rag_demo.connectors.sync import main as sync_main
from rag_demo.connectors.sync import sync_source
from rag_demo.retriever import Retriever

FAKE_PAT = "FAKE-CONF-PAT-9d4e2a71"
FAKE_BEARER = "FAKE-OUTLOOK-TOKEN-5b1c8f33"


@pytest.fixture
def fake_embeddings(monkeypatch):
    """FakeEmbeddings for the chunk/embed/index pipeline in sync tests."""
    from langchain_core.embeddings import FakeEmbeddings

    factory = lambda settings: FakeEmbeddings(size=384)  # noqa: E731
    monkeypatch.setattr("rag_demo.ingest.get_embeddings", factory)
    return FakeEmbeddings(size=384)


# ---------------------------------------------------------------------------
# Source labels
# ---------------------------------------------------------------------------


def test_to_documents_stamps_readable_source_labels():
    docs = to_documents(
        [
            ConnectorDoc(
                title="TMS deploy checklist",
                text="# TMS deploy checklist\n\nStage it.",
                source_label="confluence:OPS/TMS deploy checklist",
                updated_at="2026-09-01T10:00:00Z",
                url="https://example.invalid/pages/101",
                doc_id="confluence:101",
            ),
            ConnectorDoc(
                title="TMS deploy window Friday",
                text="# TMS deploy window Friday\n\nFreeze Thursday.",
                source_label="outlook:TMS deploy window Friday",
                updated_at="2026-09-28T15:30:00Z",
                doc_id="outlook:msg-1",
            ),
        ]
    )
    assert [d.metadata["source"] for d in docs] == [
        "confluence:OPS/TMS deploy checklist",
        "outlook:TMS deploy window Friday",
    ]
    assert docs[0].metadata["doc_id"] == "confluence:101"
    assert docs[0].metadata["allowed_groups"] == ["*"]


# ---------------------------------------------------------------------------
# Confluence client against the mock
# ---------------------------------------------------------------------------


def test_confluence_fetch_all_pages(monkeypatch):
    with mock_server(MockConfluenceHandler) as (base_url, log):
        conn = ConfluenceConnector()
        conn.connect({"url": base_url, "pat": FAKE_PAT, "space": "OPS"})
        docs = list(conn.fetch_since(None))
    assert [d.title for d in docs] == [
        "TMS deploy checklist",
        "EDI retry policy",
        "Shipment incident review template",
        "Driver pay bands FY27",
        "API key rotation schedule",
        "Incident severity levels",
    ]
    assert docs[0].source_label == "confluence:OPS/TMS deploy checklist"
    assert docs[0].doc_id == "confluence:101"
    assert "dispatch_error_rate" in docs[0].text
    assert "<h1>" not in docs[0].text  # storage HTML is stripped
    # The mock saw a Bearer auth header (scheme only — never the token).
    assert log and all(entry["auth"] == "Bearer" for entry in log)
    assert not any(FAKE_PAT in str(entry) for entry in log)


def test_confluence_fetch_since_is_incremental(monkeypatch):
    with mock_server(MockConfluenceHandler) as (base_url, log):
        conn = ConfluenceConnector()
        conn.connect({"url": base_url, "pat": FAKE_PAT, "space": "OPS"})
        docs = list(conn.fetch_since("2026-09-15T00:00:00Z"))
    assert [d.title for d in docs] == [
        "Shipment incident review template",
        "Driver pay bands FY27",
        "API key rotation schedule",
        "Incident severity levels",
    ]


def test_confluence_paginates(monkeypatch):
    monkeypatch.setattr("rag_demo.connectors.confluence._PAGE_SIZE", 2)
    with mock_server(MockConfluenceHandler) as (base_url, log):
        conn = ConfluenceConnector()
        conn.connect({"url": base_url, "pat": FAKE_PAT, "space": "OPS"})
        docs = list(conn.fetch_since(None))
    assert len(docs) == 6
    # 3 full pages, then a 4th request that returns empty — the client
    # cannot know the last page was the last without asking once more.
    assert len([e for e in log if e["path"].endswith("/rest/api/content/search")]) == 4


def test_confluence_missing_config_raises(monkeypatch):
    monkeypatch.delenv("CONFLUENCE_URL", raising=False)
    monkeypatch.delenv("CONFLUENCE_PAT", raising=False)
    with pytest.raises(ValueError, match="CONFLUENCE_URL"):
        ConfluenceConnector().connect({})
    with pytest.raises(ValueError, match="CONFLUENCE_PAT"):
        ConfluenceConnector().connect({"url": "https://example.invalid/wiki"})


# ---------------------------------------------------------------------------
# Outlook client against the mock
# ---------------------------------------------------------------------------


def test_outlook_fetch_all_messages():
    with mock_server(MockGraphHandler) as (base_url, log):
        conn = OutlookConnector()
        conn.connect({"bearer_token": FAKE_BEARER, "base_url": base_url})
        docs = list(conn.fetch_since(None))
    assert [d.title for d in docs] == [
        "API key rotation — action required",
        "Driver pay bands FY27 — effective dates",
        "EDI outage resolved",
        "TMS deploy window Friday",
    ]
    assert docs[0].source_label == "outlook:API key rotation — action required"
    assert docs[0].doc_id == "outlook:msg-4"
    assert "October 15" in docs[0].text
    assert "<html>" not in docs[0].text  # HTML bodies are stripped
    assert log and all(entry["auth"] == "Bearer" for entry in log)
    assert not any(FAKE_BEARER in str(entry) for entry in log)


def test_outlook_fetch_since_is_incremental():
    with mock_server(MockGraphHandler) as (base_url, log):
        conn = OutlookConnector()
        conn.connect({"bearer_token": FAKE_BEARER, "base_url": base_url})
        docs = list(conn.fetch_since("2026-10-03T00:00:00Z"))
    assert [d.title for d in docs] == [
        "API key rotation — action required",
        "Driver pay bands FY27 — effective dates",
    ]


def test_outlook_paginates(monkeypatch):
    monkeypatch.setattr("rag_demo.connectors.outlook._PAGE_SIZE", 2)
    with mock_server(MockGraphHandler) as (base_url, log):
        conn = OutlookConnector()
        conn.connect({"bearer_token": FAKE_BEARER, "base_url": base_url})
        docs = list(conn.fetch_since(None))
    assert len(docs) == 4
    assert len([e for e in log if e["path"].endswith("/messages")]) == 2


def test_outlook_missing_token_raises(monkeypatch):
    monkeypatch.delenv("OUTLOOK_BEARER_TOKEN", raising=False)
    with pytest.raises(ValueError, match="OUTLOOK_BEARER_TOKEN"):
        OutlookConnector().connect({})


# ---------------------------------------------------------------------------
# Token never appears in logs or output
# ---------------------------------------------------------------------------


def test_token_never_appears_in_logs_or_output(monkeypatch, caplog, capsys):
    with caplog.at_level(logging.DEBUG, logger="rag_demo.connectors.confluence"), \
            caplog.at_level(logging.DEBUG, logger="rag_demo.connectors.outlook"):
        with mock_server(MockConfluenceHandler) as (conf_url, _):
            conn = ConfluenceConnector()
            conn.connect({"url": conf_url, "pat": FAKE_PAT, "space": "OPS"})
            list(conn.fetch_since(None))
        with mock_server(MockGraphHandler) as (graph_url, _):
            conn = OutlookConnector()
            conn.connect({"bearer_token": FAKE_BEARER, "base_url": graph_url})
            list(conn.fetch_since(None))
    assert FAKE_PAT not in caplog.text
    assert FAKE_BEARER not in caplog.text
    assert FAKE_PAT not in capsys.readouterr().out
    assert FAKE_BEARER not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Sync end to end: mock source -> indexed -> cited answer
# ---------------------------------------------------------------------------


def test_sync_confluence_end_to_end(monkeypatch, tmp_path, fake_embeddings):
    with mock_server(MockConfluenceHandler) as (base_url, _):
        monkeypatch.setenv("CONFLUENCE_URL", base_url)
        monkeypatch.setenv("CONFLUENCE_PAT", FAKE_PAT)
        monkeypatch.setenv("CONFLUENCE_SPACE", "OPS")
        index_dir = tmp_path / "index"
        settings = Settings(index_dir=index_dir)
        summary = sync_source(ConfluenceConnector(), settings)
    assert summary == {
        "source": "confluence",
        "fetched": 6,
        "total": 6,
        "last_sync": "2026-10-05T10:00:00Z",
    }
    state = json.loads((index_dir / "sync-state.json").read_text(encoding="utf-8"))
    assert state["last_sync"] == "2026-10-05T10:00:00Z"
    assert FAKE_PAT not in (index_dir / "sync-state.json").read_text(encoding="utf-8")
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    assert "confluence:OPS/TMS deploy checklist" in {c["source"] for c in chunks}
    assert FAKE_PAT not in (index_dir / "chunks.json").read_text(encoding="utf-8")

    retriever = Retriever(Settings(index_dir=index_dir), embeddings=fake_embeddings)
    hits = retriever.search("TMS deploy checklist", rerank=False)
    assert "confluence:OPS/TMS deploy checklist" in {h.source for h in hits}


def test_sync_is_incremental_on_second_run(monkeypatch, tmp_path, fake_embeddings):
    with mock_server(MockConfluenceHandler) as (base_url, _):
        monkeypatch.setenv("CONFLUENCE_URL", base_url)
        monkeypatch.setenv("CONFLUENCE_PAT", FAKE_PAT)
        monkeypatch.setenv("CONFLUENCE_SPACE", "OPS")
        index_dir = tmp_path / "index"
        settings = Settings(index_dir=index_dir)
        first = sync_source(ConfluenceConnector(), settings)
        second = sync_source(ConfluenceConnector(), settings)
    assert first["fetched"] == 6
    # CQL lastModified is date-granular, so the page updated on the same day
    # as last_sync is re-fetched; the store dedupes by doc_id, so the total
    # stays 6 and last_sync does not advance.
    assert second == {
        "source": "confluence",
        "fetched": 1,
        "total": 6,
        "last_sync": "2026-10-05T10:00:00Z",
    }


def test_sync_cli_entry_point(monkeypatch, tmp_path, fake_embeddings, capsys):
    with mock_server(MockConfluenceHandler) as (base_url, _):
        monkeypatch.setenv("CONFLUENCE_URL", base_url)
        monkeypatch.setenv("CONFLUENCE_PAT", FAKE_PAT)
        monkeypatch.setenv("CONFLUENCE_SPACE", "OPS")
        index_dir = tmp_path / "index"
        rc = sync_main(["--source", "confluence", "--index-dir", str(index_dir), "--full"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fetched 6 new/changed" in out
    assert "6 total" in out
    assert FAKE_PAT not in out
    assert (index_dir / "index.faiss").exists()

# ---------------------------------------------------------------------------
# GitHub client against the mock
# ---------------------------------------------------------------------------

from mock_servers import MockGitHubHandler  # noqa: E402
from rag_demo.connectors.github import GitHubConnector  # noqa: E402

FAKE_GITHUB_TOKEN = "FAKE-GITHUB-TOKEN-7e3a91c2"
GITHUB_REPO = "acme/widgets"
GITHUB_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "github"


def _github_fixture(name: str):
    return json.loads((GITHUB_FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture()
def github_mock():
    MockGitHubHandler.reset()
    with mock_server(MockGitHubHandler) as (base_url, log):
        yield base_url, log


def _github_connector(base_url, repos=(GITHUB_REPO,), **kw):
    connector = GitHubConnector(base_url=base_url, **kw)
    connector.connect({"token": FAKE_GITHUB_TOKEN, "repos": list(repos)})
    return connector


def _github_env(monkeypatch, base_url):
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_GITHUB_TOKEN)
    monkeypatch.setenv("GITHUB_REPO", GITHUB_REPO)
    monkeypatch.setenv("GITHUB_BASE_URL", base_url)


def test_github_fetch_all(monkeypatch, github_mock):
    base_url, log = github_mock
    docs = list(_github_connector(base_url).fetch_since(None))
    assert len(docs) == 9  # readme 1 + docs 3 + issues 2 + pulls 3
    assert {d.source_label for d in docs} == {
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
    assert {d.doc_id for d in docs if "#pr-" in (d.doc_id or "")} == {
        "github:acme/widgets#pr-4",
        "github:acme/widgets#pr-5",
        "github:acme/widgets#pr-6",
    }
    assert log and all(entry["auth"] == "Bearer" for entry in log)
    assert not any(FAKE_GITHUB_TOKEN in str(entry) for entry in log)


def test_github_source_labels(github_mock):
    base_url, _ = github_mock
    docs = list(_github_connector(base_url).fetch_since(None))
    by_label = {d.source_label: d for d in docs}
    assert by_label["github:acme/widgets/docs/guide.md"].doc_id == (
        "github:acme/widgets/docs/guide.md"
    )
    assert by_label["github:acme/widgets#1"].doc_id == "github:acme/widgets#issue-1"
    assert by_label["github:acme/widgets#4"].doc_id == "github:acme/widgets#pr-4"
    # Labels carry no raw code paths: only README + docs/*.md are fetched.
    assert not any("src/main.py" in label for label in by_label)


def test_github_paginates(github_mock):
    base_url, log = github_mock
    docs = list(_github_connector(base_url, per_page=2).fetch_since(None))
    pulls = [d for d in docs if "#pr-" in (d.doc_id or "")]
    assert len(pulls) == 3
    pulls_paths = [e["path"] for e in log if "/pulls" in e["path"]]
    assert any("page=2" in p for p in pulls_paths), pulls_paths


def test_github_fetch_since_is_incremental(github_mock):
    base_url, log = github_mock
    connector = _github_connector(base_url)
    first = list(connector.fetch_since(None))
    assert len(first) == 9

    # Change the world: guide updated, old.md deleted, new.md added,
    # one new issue. The API clock is now past the fixture updates.
    data = MockGitHubHandler.data["repos"][GITHUB_REPO]
    data["tree"] = _github_fixture("tree-updated.json")
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
    MockGitHubHandler.requests_log.clear()  # clear in place: `log` aliases this list
    second = list(connector.fetch_since("2026-10-06T00:00:00Z"))
    assert {d.source_label for d in second} == {
        "github:acme/widgets/docs/guide.md",
        "github:acme/widgets/docs/new.md",
        "github:acme/widgets#7",
    }
    blob_paths = [e["path"] for e in log if "/git/blobs/" in e["path"]]
    assert any("blob-sha-guide-2" in p for p in blob_paths)
    assert any("blob-sha-new-1" in p for p in blob_paths)
    assert not any("blob-sha-api-1" in p for p in blob_paths)  # unchanged: not re-fetched
    issues_paths = [e["path"] for e in log if "/issues" in e["path"]]
    assert issues_paths and any("since=2026-10-06" in p for p in issues_paths)
    assert "github:acme/widgets/docs/old.md" not in connector.present_doc_ids


def test_github_missing_config_raises(monkeypatch):
    for var in ("GITHUB_TOKEN", "RAG_DEMO_GITHUB_TOKEN", "GITHUB_REPO", "GITHUB_REPOS", "RAG_DEMO_GITHUB_REPO"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError, match="token"):
        GitHubConnector().connect({"repos": [GITHUB_REPO]})
    with pytest.raises(ValueError, match="repo"):
        GitHubConnector().connect({"token": FAKE_GITHUB_TOKEN})


def test_github_mock_requires_auth(github_mock):
    import requests

    base_url, _ = github_mock
    resp = requests.get(f"{base_url}/repos/{GITHUB_REPO}/readme", timeout=10)
    assert resp.status_code == 401


def test_github_token_never_appears_in_logs_or_output(monkeypatch, caplog, capsys, github_mock):
    base_url, log = github_mock
    with caplog.at_level(logging.DEBUG, logger="rag_demo.connectors.github"):
        connector = _github_connector(base_url)
        list(connector.fetch_since(None))
    assert FAKE_GITHUB_TOKEN not in caplog.text
    assert FAKE_GITHUB_TOKEN not in capsys.readouterr().out
    assert FAKE_GITHUB_TOKEN not in json.dumps(log)
    assert {entry["auth"] for entry in log} == {"Bearer"}


def test_github_multiple_repos(github_mock):
    base_url, _ = github_mock
    MockGitHubHandler.data["repos"]["acme/gadgets"] = {
        "readme": _github_fixture("readme.json"),
        "tree": _github_fixture("tree.json"),
        "blobs": _github_fixture("blobs.json"),
        "issues": _github_fixture("issues.json"),
        "pulls": _github_fixture("pulls.json"),
    }
    docs = list(
        _github_connector(base_url, repos=(GITHUB_REPO, "acme/gadgets")).fetch_since(None)
    )
    assert len(docs) == 18
    assert "github:acme/gadgets/docs/guide.md" in {d.source_label for d in docs}
    assert "github:acme/widgets/docs/guide.md" in {d.source_label for d in docs}


# ---------------------------------------------------------------------------
# GitHub sync end to end: mock source -> indexed -> cited answer
# ---------------------------------------------------------------------------


def test_sync_github_end_to_end(monkeypatch, tmp_path, fake_embeddings, github_mock):
    base_url, _ = github_mock
    _github_env(monkeypatch, base_url)
    index_dir = tmp_path / "index"
    settings = Settings(index_dir=index_dir)
    summary = sync_source(GitHubConnector(), settings)
    assert summary == {
        "source": "github",
        "fetched": 9,
        "total": 9,
        "last_sync": "2026-10-05T08:00:00Z",
    }
    state = json.loads((index_dir / "sync-state.json").read_text(encoding="utf-8"))
    assert state["last_sync"] == "2026-10-05T08:00:00Z"
    assert state["connector_state"]["repos"][GITHUB_REPO]["readme_sha"] == "readme-sha-001"
    assert FAKE_GITHUB_TOKEN not in (index_dir / "sync-state.json").read_text(encoding="utf-8")
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    assert "github:acme/widgets/docs/guide.md" in {c["source"] for c in chunks}
    assert FAKE_GITHUB_TOKEN not in (index_dir / "chunks.json").read_text(encoding="utf-8")

    retriever = Retriever(Settings(index_dir=index_dir), embeddings=fake_embeddings)
    hits = retriever.search("widget catalog bulk import", rerank=False)
    assert hits
    assert all(h.source.startswith("github:acme/widgets") for h in hits)


def test_sync_github_is_incremental_on_second_run(monkeypatch, tmp_path, fake_embeddings, github_mock):
    base_url, _ = github_mock
    _github_env(monkeypatch, base_url)
    index_dir = tmp_path / "index"
    settings = Settings(index_dir=index_dir)
    first = sync_source(GitHubConnector(), settings)
    second = sync_source(GitHubConnector(), settings)
    assert first["fetched"] == 9
    assert second == {
        "source": "github",
        "fetched": 0,
        "total": 9,
        "last_sync": "2026-10-05T08:00:00Z",
    }


def test_sync_github_prunes_deleted_docs(monkeypatch, tmp_path, fake_embeddings, github_mock):
    base_url, _ = github_mock
    _github_env(monkeypatch, base_url)
    index_dir = tmp_path / "index"
    settings = Settings(index_dir=index_dir)
    sync_source(GitHubConnector(), settings)
    MockGitHubHandler.data["repos"][GITHUB_REPO]["tree"] = _github_fixture("tree-updated.json")
    second = sync_source(GitHubConnector(), settings)
    assert second["fetched"] == 2  # guide.md updated + new.md added
    assert second["total"] == 9  # old.md pruned, new.md added
    store = json.loads((index_dir / "connector-documents.json").read_text(encoding="utf-8"))
    labels = {entry["source_label"] for entry in store.values()}
    assert "github:acme/widgets/docs/old.md" not in labels
    assert "github:acme/widgets/docs/new.md" in labels


def test_sync_github_cli_entry_point(monkeypatch, tmp_path, fake_embeddings, capsys, github_mock):
    base_url, _ = github_mock
    _github_env(monkeypatch, base_url)
    index_dir = tmp_path / "index"
    rc = sync_main(["--source", "github", "--index-dir", str(index_dir), "--full"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fetched 9 new/changed" in out
    assert "9 total" in out
    assert FAKE_GITHUB_TOKEN not in out
    assert (index_dir / "index.faiss").exists()

# ---------------------------------------------------------------------------
# Gmail client against the mock
# ---------------------------------------------------------------------------

from mock_servers import MockGmailHandler  # noqa: E402
from rag_demo.connectors.gmail import GmailConnector  # noqa: E402

FAKE_GMAIL_TOKEN = "FAKE-GMAIL-TOKEN-3f7a19d4"
GMAIL_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "gmail"


@pytest.fixture()
def gmail_mock():
    with mock_server(MockGmailHandler) as (base_url, log):
        yield base_url, log


def _gmail_connector(base_url, **kw):
    connector = GmailConnector()
    connector.connect({"access_token": FAKE_GMAIL_TOKEN, "base_url": base_url, **kw})
    return connector


def _gmail_env(monkeypatch, base_url):
    monkeypatch.setenv("GMAIL_ACCESS_TOKEN", FAKE_GMAIL_TOKEN)
    monkeypatch.setenv("GMAIL_BASE_URL", base_url)


def test_gmail_fetch_all(gmail_mock):
    base_url, log = gmail_mock
    docs = list(_gmail_connector(base_url).fetch_since(None))
    assert [d.title for d in docs] == [
        "API key rotation — action required",
        "Driver pay bands FY27 — effective dates",
        "EDI outage resolved",
        "TMS deploy window Friday",
    ]
    assert docs[0].source_label == "gmail:API key rotation — action required"
    assert docs[0].doc_id == "gmail:msg-g4"
    assert docs[0].updated_at == "2026-10-06T08:45:00Z"
    assert docs[0].url == "https://mail.google.com/mail/u/0/#inbox/msg-g4"
    assert "Reminder: API keys rotate" in docs[0].text
    assert "<html>" not in docs[0].text  # HTML bodies are stripped
    assert log and all(entry["auth"] == "Bearer" for entry in log)
    assert not any(FAKE_GMAIL_TOKEN in str(entry) for entry in log)


def test_gmail_source_labels(gmail_mock):
    base_url, _ = gmail_mock
    docs = list(_gmail_connector(base_url).fetch_since(None))
    by_label = {d.source_label: d for d in docs}
    assert by_label["gmail:TMS deploy window Friday"].doc_id == "gmail:msg-g1"
    assert by_label["gmail:EDI outage resolved"].updated_at == "2026-10-02T09:12:00Z"
    assert all(d.source_label.startswith("gmail:") for d in docs)


def test_gmail_paginates(monkeypatch, gmail_mock):
    monkeypatch.setattr("rag_demo.connectors.gmail._PAGE_SIZE", 2)
    base_url, log = gmail_mock
    docs = list(_gmail_connector(base_url).fetch_since(None))
    assert len(docs) == 4
    list_paths = [
        e["path"] for e in log
        if e["path"].split("?")[0].endswith("/messages")
    ]
    assert len(list_paths) == 2  # two list pages of 2 ids each


def test_gmail_fetch_since_is_incremental(gmail_mock):
    base_url, log = gmail_mock
    docs = list(_gmail_connector(base_url).fetch_since("2026-10-03T00:00:00Z"))
    assert [d.title for d in docs] == [
        "API key rotation — action required",
        "Driver pay bands FY27 — effective dates",
    ]
    # The client stops at the first message at/before the cutoff, so the
    # oldest message is never even fetched.
    assert not any("/msg-g1" in e["path"] for e in log)


def test_gmail_multipart_prefers_plain_text(gmail_mock):
    base_url, _ = gmail_mock
    docs = list(_gmail_connector(base_url).fetch_since(None))
    by_id = {d.doc_id: d for d in docs}
    text = by_id["gmail:msg-g3"].text
    assert "Driver pay bands FY27 take effect October 1" in text
    assert "managers should confirm each driver's band" in text
    assert "<html>" not in text
    assert "<p>" not in text


def test_gmail_html_body_stripped(gmail_mock):
    base_url, _ = gmail_mock
    docs = list(_gmail_connector(base_url).fetch_since(None))
    by_id = {d.doc_id: d for d in docs}
    text = by_id["gmail:msg-g2"].text
    assert "dead-letter queue for 30 days" in text
    assert "<html>" not in text
    assert "<p>" not in text


def test_gmail_snippet_fallback(gmail_mock):
    base_url, _ = gmail_mock
    docs = list(_gmail_connector(base_url).fetch_since(None))
    by_id = {d.doc_id: d for d in docs}
    # msg-g4 carries no body parts; the snippet is the document text.
    assert "Reminder: API keys rotate" in by_id["gmail:msg-g4"].text


def test_gmail_missing_token_raises(monkeypatch):
    monkeypatch.delenv("GMAIL_ACCESS_TOKEN", raising=False)
    with pytest.raises(ValueError, match="GMAIL_ACCESS_TOKEN"):
        GmailConnector().connect({})


def test_gmail_mock_requires_auth(gmail_mock):
    import requests

    base_url, _ = gmail_mock
    resp = requests.get(f"{base_url}/gmail/v1/users/me/messages", timeout=10)
    assert resp.status_code == 401


def test_gmail_token_never_appears_in_logs_or_output(monkeypatch, caplog, capsys, gmail_mock):
    base_url, log = gmail_mock
    with caplog.at_level(logging.DEBUG, logger="rag_demo.connectors.gmail"):
        list(_gmail_connector(base_url).fetch_since(None))
    assert FAKE_GMAIL_TOKEN not in caplog.text
    assert FAKE_GMAIL_TOKEN not in capsys.readouterr().out
    assert FAKE_GMAIL_TOKEN not in json.dumps(log)
    assert {entry["auth"] for entry in log} == {"Bearer"}


# ---------------------------------------------------------------------------
# Gmail sync end to end: mock source -> indexed -> cited answer
# ---------------------------------------------------------------------------


def test_sync_gmail_end_to_end(monkeypatch, tmp_path, fake_embeddings, gmail_mock):
    base_url, _ = gmail_mock
    _gmail_env(monkeypatch, base_url)
    index_dir = tmp_path / "index"
    settings = Settings(index_dir=index_dir)
    summary = sync_source(GmailConnector(), settings)
    assert summary == {
        "source": "gmail",
        "fetched": 4,
        "total": 4,
        "last_sync": "2026-10-06T08:45:00Z",
    }
    state = json.loads((index_dir / "sync-state.json").read_text(encoding="utf-8"))
    assert state["last_sync"] == "2026-10-06T08:45:00Z"
    assert state["connector_state"]["newest_internal_date_ms"] == 1791276300000
    assert FAKE_GMAIL_TOKEN not in (index_dir / "sync-state.json").read_text(encoding="utf-8")
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    assert "gmail:API key rotation — action required" in {c["source"] for c in chunks}
    assert FAKE_GMAIL_TOKEN not in (index_dir / "chunks.json").read_text(encoding="utf-8")

    retriever = Retriever(Settings(index_dir=index_dir), embeddings=fake_embeddings)
    hits = retriever.search("API key rotation schedule", rerank=False)
    assert hits
    assert all(h.source.startswith("gmail:") for h in hits)


def test_sync_gmail_is_incremental_on_second_run(monkeypatch, tmp_path, fake_embeddings, gmail_mock):
    base_url, _ = gmail_mock
    _gmail_env(monkeypatch, base_url)
    index_dir = tmp_path / "index"
    settings = Settings(index_dir=index_dir)
    first = sync_source(GmailConnector(), settings)
    second = sync_source(GmailConnector(), settings)
    assert first["fetched"] == 4
    assert second == {
        "source": "gmail",
        "fetched": 0,
        "total": 4,
        "last_sync": "2026-10-06T08:45:00Z",
    }


def test_sync_gmail_cli_entry_point(monkeypatch, tmp_path, fake_embeddings, capsys, gmail_mock):
    base_url, _ = gmail_mock
    _gmail_env(monkeypatch, base_url)
    index_dir = tmp_path / "index"
    rc = sync_main(["--source", "gmail", "--index-dir", str(index_dir), "--full"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fetched 4 new/changed" in out
    assert "4 total" in out
    assert FAKE_GMAIL_TOKEN not in out
    assert (index_dir / "index.faiss").exists()
