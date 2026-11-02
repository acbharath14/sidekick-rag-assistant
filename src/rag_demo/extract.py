"""Ad-hoc file text extraction for the upload feature.

Supported: .pdf, .docx, .txt/.md (utf-8), .csv (rendered as a markdown
table, capped at 200 rows). Anything else raises UnsupportedFormat.

PDF extraction tries backends in order (inspired by doc-extract-eval):
pymupdf (fastest) → pdfplumber (best tables) → pypdf (pure Python).
Each is optional — missing backends are skipped silently.

verify_magic_bytes() guards uploads against extension spoofing (e.g. a
script renamed to .pdf) using stdlib-only magic-byte checks.
"""

from __future__ import annotations

import csv
import io

SUPPORTED = (".pdf", ".docx", ".txt", ".md", ".csv")


class UnsupportedFormat(Exception):
    pass


def _extension(filename: str) -> str:
    return "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def verify_magic_bytes(filename: str, data: bytes) -> bool:
    """Check that the file's leading bytes match its extension.

    Stdlib-only (no python-magic). Raises UnsupportedFormat for unknown
    extensions; returns True/False for known ones. An empty payload is
    valid UTF-8 (so .txt passes) but fails binary signatures (so .pdf
    and .docx fail) — emptiness itself is rejected later by the
    "No text found" check in the upload endpoint.
    """
    ext = _extension(filename)
    if ext == ".pdf":
        return data.startswith(b"%PDF-")
    if ext == ".docx":
        # docx is a ZIP archive; local file header signature.
        return data.startswith(b"PK\x03\x04")
    if ext in (".txt", ".md", ".csv"):
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            return False
        return True
    raise UnsupportedFormat(
        f"{filename!r}: supported types are {', '.join(SUPPORTED)}"
    )


def extract_text(filename: str, data: bytes) -> str:
    """Extract text. Returns (text, extractor_name) for PDFs so callers can
    report which backend was used."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == ".pdf":
        return _extract_pdf(data)
    if ext == ".docx":
        return _extract_docx(data)
    if ext in (".txt", ".md"):
        return data.decode("utf-8")
    if ext == ".csv":
        return _extract_csv(data)
    raise UnsupportedFormat(
        f"{filename!r}: supported types are {', '.join(SUPPORTED)}"
    )


def extract_text_with_meta(filename: str, data: bytes) -> tuple[str, str]:
    """Like extract_text but also returns the extractor backend name."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == ".pdf":
        return _extract_pdf_with_fallback(data)
    text = extract_text(filename, data)
    return text, ext.lstrip(".") if ext else "text"


def _extract_pdf(data: bytes) -> str:
    text, _ = _extract_pdf_with_fallback(data)
    return text


def _extract_pdf_with_fallback(data: bytes) -> tuple[str, str]:
    """Try PDF backends in order; return (text, backend_name)."""
    for backend in ("pymupdf", "pdfplumber", "pypdf"):
        try:
            if backend == "pymupdf":
                text = _extract_pdf_pymupdf(data)
            elif backend == "pdfplumber":
                text = _extract_pdf_pdfplumber(data)
            else:
                text = _extract_pdf_pypdf(data)
            if text.strip():
                return text, backend
        except ImportError:
            continue  # backend not installed — try the next one
        except Exception:
            continue  # backend failed on this file — try the next one
    # All backends failed or produced nothing.
    return "", "none"


def _extract_pdf_pymupdf(data: bytes) -> str:
    import fitz  # pymupdf

    doc = fitz.open(stream=data, filetype="pdf")
    return "\n\n".join(page.get_text() or "" for page in doc).strip()


def _extract_pdf_pdfplumber(data: bytes) -> str:
    import pdfplumber

    parts = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n\n".join(parts).strip()


def _extract_pdf_pypdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()


def _extract_docx(data: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _extract_csv(data: bytes, max_rows: int = 200) -> str:
    rows = list(csv.reader(io.StringIO(data.decode("utf-8"))))
    if not rows:
        return ""
    header, body = rows[0], rows[1 : max_rows + 1]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    if len(rows) > max_rows + 1:
        lines.append(f"\n*({len(rows) - max_rows - 1} more rows truncated)*")
    return "\n".join(lines)
