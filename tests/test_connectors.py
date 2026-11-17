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
