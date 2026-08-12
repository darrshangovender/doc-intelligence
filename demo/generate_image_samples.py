"""Render the text samples in ``demo/sample_docs/`` as PNG receipts via PIL.

These PNGs can then be fed through the real OCR path (Tesseract) for a more
realistic end-to-end demo. The text files remain the source of truth for the
unit tests, which run offline.

Usage::

    python demo/generate_image_samples.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).parent
SAMPLES = HERE / "sample_docs"
OUT = HERE / "sample_images"


def render(text: str, out_path: Path) -> None:
    width = 720
    # Try a monospace TTF if available; fall back to PIL default bitmap font.
    try:
        font = ImageFont.truetype("consola.ttf", 16)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", 16)
        except OSError:
            font = ImageFont.load_default()

    lines = text.splitlines() or [""]
    line_h = 22
    pad = 24
    height = pad * 2 + line_h * len(lines)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((pad, pad + i * line_h), line, fill="black", font=font)
    img.save(out_path)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for txt in sorted(SAMPLES.glob("*.txt")):
        render(txt.read_text(encoding="utf-8"), OUT / f"{txt.stem}.png")
        print(f"rendered {txt.stem}.png")


if __name__ == "__main__":
    main()
