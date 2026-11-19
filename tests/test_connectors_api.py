"""API tests for the enterprise-connectors GUI endpoints.

GET /api/connectors, POST /api/connectors/{source}/sync (background job),
GET /api/connectors/jobs/{job_id}. The real sync is monkeypatched out —
these tests prove the HTTP contract and the token rules, not the sync
itself (that is covered by tests/test_connectors.py).
"""

import json
import os
import threading
import time

os.environ["RAG_DEMO_FAKE"] = "1"

import pytest
from fastapi.testclient import TestClient

import rag_demo.api as api
from rag_demo.api import app
from rag_demo.config import Settings

client = TestClient(app)

FAKE_TOKEN = "FAKE-GUI-TOKEN-4c2d88a1"


@pytest.fixture()
def tmp_settings(monkeypatch, tmp_path):
    """Point the API's settings at a temp dir (per-source index dirs live under it)."""
    settings = Settings(index_dir=tmp_path / "index")
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    return settings


def _clear_connector_env(monkeypatch):
    for var in (
        "CONFLUENCE_URL",
        "CONFLUENCE_PAT",
        "OUTLOOK_BEARER_TOKEN",
        "GITHUB_TOKEN",
        "RAG_DEMO_GITHUB_TOKEN",
        "GITHUB_REPO",
        "GITHUB_REPOS",
        "RAG_DEMO_GITHUB_REPO",
        "GMAIL_ACCESS_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)


def _poll_job(job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/connectors/jobs/{job_id}")
        assert r.status_code == 200, r.text
        job = r.json()
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} still running after {timeout}s")


def test_list_connectors_unconfigured(monkeypatch, tmp_settings):
    _clear_connector_env(monkeypatch)
    r = client.get("/api/connectors")
    assert r.status_code == 200
    sources = {c["source"]: c for c in r.json()["connectors"]}
    assert sorted(sources) == ["confluence", "github", "gmail", "outlook"]
    for conn in sources.values():
        assert conn["configured"] is False
        assert conn["missing_env"]
        assert conn["last_sync"] is None
        assert conn["total_docs"] == 0
        assert conn["setup_hint"]
    assert sources["github"]["missing_env"] == ["GITHUB_TOKEN", "GITHUB_REPO"]


def test_list_connectors_configured_from_env(monkeypatch, tmp_settings):
    _clear_connector_env(monkeypatch)
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("GITHUB_REPO", "acme/widgets")
    r = client.get("/api/connectors")
    sources = {c["source"]: c for c in r.json()["connectors"]}
    assert sources["github"]["configured"] is True
    assert sources["github"]["missing_env"] == []
    assert sources["confluence"]["configured"] is False


def test_list_connectors_reads_sync_state(monkeypatch, tmp_settings):
    _clear_connector_env(monkeypatch)
    index_dir = tmp_settings.index_dir.parent / ".faiss_index_github_connector"
    index_dir.mkdir(parents=True)
    (index_dir / "sync-state.json").write_text(
        json.dumps({"source": "github", "last_sync": "2026-10-05T08:00:00Z"}),
        encoding="utf-8",
    )
    (index_dir / "connector-documents.json").write_text(
        json.dumps({"a": {"doc_id": "a"}, "b": {"doc_id": "b"}}),
        encoding="utf-8",
    )
    r = client.get("/api/connectors")
    sources = {c["source"]: c for c in r.json()["connectors"]}
    assert sources["github"]["last_sync"] == "2026-10-05T08:00:00Z"
    assert sources["github"]["total_docs"] == 2
    assert sources["confluence"]["last_sync"] is None


def test_sync_unknown_source_is_404(tmp_settings):
    r = client.post("/api/connectors/jira/sync", json={})
    assert r.status_code == 404


def test_sync_job_unknown_is_404(tmp_settings):
    assert client.get("/api/connectors/jobs/nope123").status_code == 404


def test_sync_runs_in_background_and_reports_done(monkeypatch, tmp_settings):
    calls = []

    def fake_sync(connector, settings, full=False, config=None):
        calls.append({"source": connector.source_name, "full": full, "config": config})
        return {"source": "github", "fetched": 9, "total": 9, "last_sync": "2026-10-05T08:00:00Z"}

    monkeypatch.setattr(api, "sync_source", fake_sync)
    r = client.post("/api/connectors/github/sync", json={"token": FAKE_TOKEN})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert r.json()["status"] == "running"
    job = _poll_job(job_id)
    assert job["status"] == "done"
    assert job["summary"] == {"source": "github", "fetched": 9, "total": 9, "last_sync": "2026-10-05T08:00:00Z"}
    assert job["error"] is None
    assert calls and calls[0]["config"] == {"token": FAKE_TOKEN}
    assert calls[0]["full"] is False
    # The token never appears in the job payload.
    assert FAKE_TOKEN not in json.dumps(job)


def test_sync_without_token_passes_no_config(monkeypatch, tmp_settings):
    calls = []

    def fake_sync(connector, settings, full=False, config=None):
        calls.append(config)
        return {"source": "outlook", "fetched": 1, "total": 1, "last_sync": None}

    monkeypatch.setattr(api, "sync_source", fake_sync)
    r = client.post("/api/connectors/outlook/sync", json={"full": True})
    job = _poll_job(r.json()["job_id"])
    assert job["status"] == "done"
    assert calls == [None]


def test_sync_second_run_while_running_is_409(monkeypatch, tmp_settings):
    started = threading.Event()
    release = threading.Event()

    def blocking_sync(connector, settings, full=False, config=None):
        started.set()
        assert release.wait(timeout=10)
        return {"source": "github", "fetched": 0, "total": 0, "last_sync": None}

    monkeypatch.setattr(api, "sync_source", blocking_sync)
    try:
        r = client.post("/api/connectors/github/sync", json={})
        assert r.status_code == 200
        assert started.wait(timeout=5)
        r2 = client.post("/api/connectors/github/sync", json={})
        assert r2.status_code == 409
        # A different source is unaffected.
        assert client.post("/api/connectors/outlook/sync", json={}).status_code == 200
    finally:
        release.set()


def test_sync_error_is_reported_on_the_job(monkeypatch, tmp_settings):
    def failing_sync(connector, settings, full=False, config=None):
        raise RuntimeError("GitHub rejected the token (401).")

    monkeypatch.setattr(api, "sync_source", failing_sync)
    r = client.post("/api/connectors/github/sync", json={"token": FAKE_TOKEN})
    job = _poll_job(r.json()["job_id"])
    assert job["status"] == "error"
    assert "401" in (job["error"] or "")
    assert job["summary"] is None
    assert FAKE_TOKEN not in json.dumps(job)
