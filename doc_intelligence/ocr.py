"""OCR wrapper. Uses pytesseract for image inputs; also accepts pre-OCR'd text.

The wrapper is intentionally narrow — extractors only need ``read(source)`` to
return a plain string. The dispatch between native PDF text, OCR, and raw text
lives in :mod:`doc_intelligence.pdf_loader`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union


# Tesseract is heavy and optional at import-time so unit tests don't require it.
def _tesseract_image_to_string(image_path: Path) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pytesseract / Pillow required for OCR") from exc
    with Image.open(image_path) as img:
        return pytesseract.image_to_string(img)


def ocr_image(image_path: Union[str, Path]) -> str:
    """Run Tesseract on an image file."""
    return _tesseract_image_to_string(Path(image_path))


def read_text_file(text_path: Union[str, Path]) -> str:
    """Read a pre-OCR'd .txt file (text-only mode)."""
    return Path(text_path).read_text(encoding="utf-8")


def read(source: Union[str, Path, bytes]) -> str:
    """Generic entry point.

    * ``str`` that looks like a path → dispatch by suffix
    * ``Path`` → dispatch by suffix
    * raw ``str`` (no file exists) → returned as-is (text-only mode)
    * ``bytes`` → decoded as UTF-8 (text-only mode)
    """
    if isinstance(source, bytes):
        return source.decode("utf-8", errors="replace")
    if isinstance(source, Path) or (isinstance(source, str) and Path(source).exists()):
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return read_text_file(path)
        if suffix in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}:
            return ocr_image(path)
        if suffix == ".pdf":
            # Defer to pdf_loader to handle native-text vs OCR
            from doc_intelligence.pdf_loader import load_pdf_text

            return load_pdf_text(path)
        raise ValueError(f"Unsupported file type: {suffix}")
    # Assume it's a raw text string
    return str(source)
