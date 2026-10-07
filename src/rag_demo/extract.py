"""Ad-hoc file text extraction for the upload feature.

Supported: .pdf (pypdf), .docx (python-docx), .txt/.md (utf-8), .csv
(rendered as a markdown table, capped at 200 rows). Anything else raises
UnsupportedFormat. Pure-Python deps only — Intel-Mac and CI safe.
"""

from __future__ import annotations

import csv
import io

SUPPORTED = (".pdf", ".docx", ".txt", ".md", ".csv")


class UnsupportedFormat(Exception):
    pass


def extract_text(filename: str, data: bytes) -> str:
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


def _extract_pdf(data: bytes) -> str:
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
