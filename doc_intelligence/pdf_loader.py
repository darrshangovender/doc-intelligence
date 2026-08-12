"""PDF loader. Uses pdfplumber for native-text PDFs; falls back to OCR for scans.

Decision rule: a page is considered "native text" if pdfplumber extracts at
least ``min_chars_for_native`` non-whitespace characters. Otherwise we render
the page to an image and run Tesseract on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union


def has_text_layer(pdf_path: Union[str, Path], min_chars_for_native: int = 40) -> bool:
    """True if the PDF appears to have a usable text layer on at least one page."""
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pdfplumber required") from exc
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = (page.extract_text() or "").strip()
            if len(text) >= min_chars_for_native:
                return True
    return False


def extract_native_text(pdf_path: Union[str, Path]) -> str:
    """Extract text from a PDF that has a text layer."""
    import pdfplumber

    chunks: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                chunks.append(text)
    return "\n\n".join(chunks)


def extract_ocr_text(pdf_path: Union[str, Path]) -> str:
    """Render each page to an image and OCR it. Slow path."""
    try:
        import pdfplumber
        import pytesseract
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pdfplumber + pytesseract required for OCR fallback") from exc

    chunks: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            img = page.to_image(resolution=200).original
            chunks.append(pytesseract.image_to_string(img))
    return "\n\n".join(chunks)


def load_pdf_text(pdf_path: Union[str, Path], min_chars_for_native: int = 40) -> str:
    """Pick native extraction or OCR depending on whether the PDF has a text layer."""
    if has_text_layer(pdf_path, min_chars_for_native=min_chars_for_native):
        return extract_native_text(pdf_path)
    return extract_ocr_text(pdf_path)
