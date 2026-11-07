"""Find the render dpi dots-mocr tolerates (scratch probe, not part of the suite).

Run: uv run python scripts/probe_ocr_dpi.py /pfad/seite.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

from arbiter.ocr import PROMPT_DOTS, VisionOcr, dots_to_text, render_png


def main() -> None:
    pdf = Path(sys.argv[1])
    client = VisionOcr(base_url="http://127.0.0.1:8080", model="dots-mocr")
    for dpi in (100, 150, 200, 250, 300):
        try:
            png = render_png(pdf, 1, dpi)
        except Exception as exc:
            print(f"{dpi} dpi: Rendern fehlgeschlagen ({exc})")
            continue
        try:
            raw = client.text_from_png(png, prompt=PROMPT_DOTS)
            text = dots_to_text(raw)
            print(f"{dpi} dpi: {len(png) // 1024} KB PNG, {len(text)} Zeichen Text")
            print("   " + text[:160].replace("\n", " | "))
        except Exception as exc:
            print(f"{dpi} dpi: {exc}")
    client.close()


if __name__ == "__main__":
    main()
