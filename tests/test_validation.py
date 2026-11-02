"""Magic-byte validation tests for document uploads.

Pure stdlib — no RAG_DEMO_FAKE needed. Run:
    python -m pytest tests/test_validation.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_demo.extract import UnsupportedFormat, verify_magic_bytes


def test_valid_pdf_header():
    data = b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"
    assert verify_magic_bytes("report.pdf", data) is True


def test_fake_pdf_text_file_renamed():
    # A shell script masquerading as a PDF must be rejected.
    data = b"#!/bin/bash\necho pwned\n"
    assert verify_magic_bytes("evil.pdf", data) is False


def test_valid_docx_zip_header():
    # docx is a ZIP archive; local file header signature PK\x03\x04.
    data = b"PK\x03\x04\x14\x00\x06\x00\x08\x00!\x00" + b"\x00" * 64
    assert verify_magic_bytes("notes.docx", data) is True


def test_script_renamed_to_docx():
    data = b"#!/usr/bin/env python3\nprint('hi')\n"
    assert verify_magic_bytes("evil.docx", data) is False


def test_valid_txt_utf8():
    assert verify_magic_bytes("notes.txt", "héllo wörld\n".encode("utf-8")) is True


def test_binary_garbage_as_txt():
    assert verify_magic_bytes("junk.txt", b"\xff\xfe\x00\x28binary") is False


def test_valid_csv_utf8():
    assert verify_magic_bytes("data.csv", b"a,b,c\n1,2,3\n") is True


def test_empty_file_txt_passes_utf8():
    # Empty payload is valid UTF-8; emptiness is rejected later by the
    # "No text found" check, not by magic-byte validation.
    assert verify_magic_bytes("empty.txt", b"") is True


def test_empty_file_pdf_fails_signature():
    assert verify_magic_bytes("empty.pdf", b"") is False


def test_empty_file_docx_fails_signature():
    assert verify_magic_bytes("empty.docx", b"") is False


def test_unsupported_extension_raises():
    with pytest.raises(UnsupportedFormat):
        verify_magic_bytes("archive.zip", b"PK\x03\x04")


def test_case_insensitive_extension():
    assert verify_magic_bytes("REPORT.PDF", b"%PDF-1.4\n") is True
    assert verify_magic_bytes("NOTES.DOCX", b"PK\x03\x04") is True
