"""Optional OCR fallback for scanned PDFs (no embedded text to extract).

Requires on PATH:
  - tesseract binary  →  conda install -c conda-forge tesseract
  - poppler utils     →  conda install -c conda-forge poppler
plus the Python wrappers (pip install pytesseract pdf2image — in
requirements.txt).

Never imported unless a PDF yields no embedded text, so nothing breaks when
the binaries are absent: the caller gets a clear RuntimeError instead.
"""

from __future__ import annotations

INSTALL_HINT = (
    "OCR needs the tesseract and poppler binaries on PATH "
    "(conda install -c conda-forge tesseract poppler)"
)


def ocr_available() -> bool:
    """True if the Python wrappers are installed (binaries checked at use)."""
    try:
        import pytesseract  # noqa: F401
        import pdf2image  # noqa: F401

        return True
    except ImportError:
        return False


def ocr_pdf(data: bytes, max_pages: int = 20, dpi: int = 200) -> str:
    """OCR a scanned PDF, up to max_pages. Raises RuntimeError with install
    instructions when the backend is missing or fails."""
    try:
        from pdf2image import convert_from_bytes

        import pytesseract
    except ImportError as e:
        raise RuntimeError(INSTALL_HINT) from e
    try:
        images = convert_from_bytes(data, dpi=dpi, first_page=1, last_page=max_pages)
    except Exception as e:  # noqa: BLE001 — usually poppler missing
        raise RuntimeError(f"{INSTALL_HINT} (poppler error: {e})") from e
    try:
        texts = [pytesseract.image_to_string(img) or "" for img in images]
    except Exception as e:  # noqa: BLE001 — usually tesseract missing
        raise RuntimeError(f"{INSTALL_HINT} (tesseract error: {e})") from e
    return "\n\n".join(texts).strip()
