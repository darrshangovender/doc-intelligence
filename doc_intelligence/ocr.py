"""OCR wrapper. Uses pytesseract for image inputs; also accepts pre-OCR'd text.

The wrapper is intentionally narrow — extractors only need ``read(source)`` to
return a plain string. The dispatch between native PDF text, OCR, and raw text
lives in :mod:`doc_intelligence.pdf_loader`.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def tesseract_available() -> bool:
    """True if OCR can actually run here — bindings *and* the Tesseract binary.

    The binary is the piece CI lacks. ``pip install pytesseract`` succeeds on a
    bare ubuntu runner, so an import check passes and then ``image_to_string``
    dies at exec time. Callers and tests should gate on this instead.
    """
    try:
        import pytesseract
        from PIL import Image  # noqa: F401
    except ImportError:
        return False
    cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
    return shutil.which(str(cmd)) is not None or Path(str(cmd)).exists()


# Tesseract is heavy and optional at import-time so unit tests don't require it.
def _tesseract_image_to_string(image_path: Path) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pytesseract / Pillow required for OCR. `pip install doc-intelligence[ocr]`"
        ) from exc
    with Image.open(image_path) as img:
        return pytesseract.image_to_string(img)


def ocr_image(image_path: str | Path) -> str:
    """Run Tesseract on an image file."""
    return _tesseract_image_to_string(Path(image_path))


def read_text_file(text_path: str | Path) -> str:
    """Read a pre-OCR'd .txt file (text-only mode)."""
    return Path(text_path).read_text(encoding="utf-8")


def read(source: str | Path | bytes) -> str:
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
