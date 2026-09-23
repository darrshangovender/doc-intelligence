"""OCR dispatch tests.

The text-only paths must work on a machine with no Tesseract binary — that is
the CI case. The one test that genuinely needs OCR is skipped rather than
dropped, so the coverage returns automatically wherever Tesseract is present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_intelligence import ocr

requires_tesseract = pytest.mark.skipif(
    not ocr.tesseract_available(),
    reason="Tesseract binary not installed (expected on stock CI runners)",
)


class TestReadDispatch:
    def test_bytes_are_decoded_as_text(self) -> None:
        assert ocr.read(b"INV-42 total 100.00") == "INV-42 total 100.00"

    def test_undecodable_bytes_do_not_raise(self) -> None:
        # A mangled scan should degrade, not crash the pipeline.
        assert "INV" in ocr.read(b"\xff\xfeINV-42")

    def test_raw_string_that_is_not_a_path_is_returned_as_is(self) -> None:
        assert ocr.read("Invoice 12345, total R100") == "Invoice 12345, total R100"

    def test_text_file_is_read_from_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.txt"
        path.write_text("ACME WIDGETS PTY LTD", encoding="utf-8")
        assert ocr.read(path) == "ACME WIDGETS PTY LTD"

    def test_markdown_is_treated_as_text(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.md"
        path.write_text("# Invoice\n\nINV-1", encoding="utf-8")
        assert "INV-1" in ocr.read(path)

    def test_unsupported_suffix_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.docx"
        path.write_bytes(b"not supported")
        with pytest.raises(ValueError, match="Unsupported file type"):
            ocr.read(path)

    def test_pdf_is_delegated_to_pdf_loader(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The native-vs-OCR decision belongs to pdf_loader; ocr.read must not
        # duplicate it.
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-fake")
        from doc_intelligence import pdf_loader

        monkeypatch.setattr(pdf_loader, "load_pdf_text", lambda _p: "DELEGATED")
        assert ocr.read(pdf) == "DELEGATED"

    def test_image_suffix_routes_to_ocr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Routing is asserted without invoking Tesseract, so this holds on CI.
        img = tmp_path / "scan.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        monkeypatch.setattr(ocr, "_tesseract_image_to_string", lambda _p: "OCR TEXT")
        assert ocr.read(img) == "OCR TEXT"


class TestAvailabilityProbe:
    def test_probe_returns_a_bool(self) -> None:
        assert isinstance(ocr.tesseract_available(), bool)

    def test_probe_is_false_when_bindings_are_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def no_pytesseract(name: str, *args: object, **kwargs: object):
            if name == "pytesseract":
                raise ImportError("simulated missing binding")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_pytesseract)
        assert ocr.tesseract_available() is False


class TestRealOcr:
    @requires_tesseract
    def test_round_trips_rendered_text(self, tmp_path: Path) -> None:
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (400, 80), "white")
        ImageDraw.Draw(img).text((10, 30), "INVOICE 12345", fill="black")
        path = tmp_path / "scan.png"
        img.save(path)

        assert "12345" in ocr.ocr_image(path)
