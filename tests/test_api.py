"""API contract test for the FastAPI backend (Part 1 of the overhaul spec).

Hits /api/chat with RAG_DEMO_FAKE=1 (deterministic fakes — no live Ollama,
no model downloads). Asserts the response shape the React frontend depends
on. Run:  RAG_DEMO_FAKE=1 python -m pytest tests/test_api.py -v
"""
import os

os.environ["RAG_DEMO_FAKE"] = "1"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi.testclient import TestClient

from rag_demo.api import app

client = TestClient(app)


def test_chat_shape():
    r = client.post("/api/chat", json={"question": "What is this?"})
    assert r.status_code == 200
    body = r.json()
    for key in ("text", "sources", "hits", "grounded", "latency_ms",
                "standalone_question"):
        assert key in body, f"missing key: {key}"
    assert isinstance(body["sources"], list)
    assert isinstance(body["hits"], list)
    for h in body["hits"]:
        assert set(h.keys()) == {"source", "text", "score"}, \
            f"hit keys wrong: {h.keys()}"
        assert "allowed_groups" not in h  # internal only, never on the wire


def test_health_shape():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"status", "ollama_reachable", "index_present"}


def test_config_no_secrets():
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    blob = str(body).lower()
    assert "token" not in blob and "secret" not in blob and "key" not in blob, \
        "config endpoint leaked a secret-bearing field"


def test_feedback_validation():
    r = client.post("/api/feedback",
                    json={"question": "q", "vote": "sideways"})
    assert r.json()["ok"] is False


def _upload(client, filename="scan.pdf", content=b"%PDF-1.4 fake scanned bytes"):
    return client.post(
        "/api/upload",
        files={"file": (filename, content, "application/pdf")},
    )


def test_upload_scanned_pdf_falls_back_to_ocr():
    """Regression test: the FastAPI rewrite dropped the OCR fallback the old
    Streamlit app had — scanned PDFs must OCR instead of 'No text found'."""
    from unittest.mock import patch

    with (
        patch(
            "rag_demo.extract.extract_text_with_meta",
            return_value=("", "pypdf"),
        ),
        patch("rag_demo.ocr.ocr_available", return_value=True),
        patch("rag_demo.ocr.ocr_pdf", return_value="OCR read this page."),
    ):
        r = _upload(client)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True, body
    assert body["extractor"] == "ocr"
    doc_id = body["id"]
    # Cleanup: remove the test document from the index + SQLite.
    d = client.delete(f"/api/documents/{doc_id}")
    assert d.json()["ok"] is True


def test_upload_scanned_pdf_without_ocr_surfaces_install_hint():
    from unittest.mock import patch

    with (
        patch(
            "rag_demo.extract.extract_text_with_meta",
            return_value=("", "pypdf"),
        ),
        patch("rag_demo.ocr.ocr_available", return_value=False),
    ):
        r = _upload(client)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "No text found" in body["error"]
    assert "tesseract" in body["error"]


def test_upload_scanned_pdf_ocr_failure_surfaces_cause():
    from unittest.mock import patch

    with (
        patch(
            "rag_demo.extract.extract_text_with_meta",
            return_value=("", "pypdf"),
        ),
        patch("rag_demo.ocr.ocr_available", return_value=True),
        patch(
            "rag_demo.ocr.ocr_pdf",
            side_effect=RuntimeError("OCR needs the tesseract binary"),
        ),
    ):
        r = _upload(client)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "tesseract" in body["error"]
