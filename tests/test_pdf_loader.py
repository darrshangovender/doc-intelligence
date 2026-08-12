"""PDF dispatch tests — native vs OCR routing.

We don't ship sample PDFs in the repo; instead we build a tiny one on the fly
using reportlab (a dev dependency). If reportlab isn't installed the test
suite skips the PDF-specific test rather than failing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_intelligence import pdf_loader


pytest.importorskip("reportlab", reason="reportlab needed to synthesize test PDFs")
pytest.importorskip("pdfplumber", reason="pdfplumber is the PDF reader under test")


def _make_native_pdf(out: Path, text: str = "Hello native PDF — INV-42, total 100.00") -> Path:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    c = canvas.Canvas(str(out), pagesize=letter)
    # Two lines so we comfortably clear the min_chars threshold
    c.drawString(72, 720, text)
    c.drawString(72, 700, "Second line so we comfortably clear the text-layer threshold.")
    c.save()
    return out


def test_native_pdf_routed_to_native_extraction(tmp_path):
    pdf = _make_native_pdf(tmp_path / "native.pdf")
    assert pdf_loader.has_text_layer(pdf) is True

    text = pdf_loader.load_pdf_text(pdf)
    assert "INV-42" in text
    assert "100.00" in text


def test_has_text_layer_returns_false_for_empty_pdf(tmp_path, monkeypatch):
    """Simulate a scan: empty text layer → has_text_layer is False."""
    pdf = _make_native_pdf(tmp_path / "empty.pdf", text=" ")  # whitespace-only

    # Patch extract_text to return empty so we don't depend on reportlab quirks.
    import pdfplumber

    original_open = pdfplumber.open

    class FakePage:
        def extract_text(self):
            return ""

    class FakePdf:
        pages = [FakePage()]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(pdfplumber, "open", lambda *_a, **_kw: FakePdf())
    assert pdf_loader.has_text_layer(pdf) is False
    monkeypatch.setattr(pdfplumber, "open", original_open)


def test_ocr_fallback_invoked_for_text_layerless_pdf(tmp_path, monkeypatch):
    """When has_text_layer is False, load_pdf_text dispatches to OCR."""
    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-fake")

    monkeypatch.setattr(pdf_loader, "has_text_layer", lambda *_a, **_kw: False)

    sentinel = "OCR FALLBACK CALLED"
    monkeypatch.setattr(pdf_loader, "extract_ocr_text", lambda _p: sentinel)
    monkeypatch.setattr(pdf_loader, "extract_native_text", lambda _p: "SHOULD NOT BE CALLED")

    assert pdf_loader.load_pdf_text(pdf) == sentinel
