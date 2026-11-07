"""Zeitmessung für eine OCR-Seite mit beiden Motoren (dots-mocr und Paddle)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from arbiter.config import load_config
from arbiter.ocr import VisionOcr, dots_to_text, render_png, word_overlap
from arbiter.pipeline import PADDLE_PROMPT

cfg = load_config(Path("arbiter.yaml"), os.environ)
backends = cfg.backends
pfad = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "Downloads" / "ix.26.13.pdf")
seiten = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["74"])]

dots = VisionOcr(backends.base_url, backends.ocr_dots_model)
paddle = VisionOcr(backends.base_url, backends.ocr_paddle_model)

for nummer in seiten:
    start = time.monotonic()
    png = render_png(pfad, nummer, cfg.rags["downloads"].ocr_dpi)
    index = dots_to_text(dots.text_from_png(png))
    zweit = paddle.text_from_png(png, prompt=PADDLE_PROMPT)
    dauer = time.monotonic() - start
    print(
        f"S.{nummer}: {dauer:.1f}s | dots {len(index)} Zeichen | paddle {len(zweit)} Zeichen "
        f"| Überlappung {word_overlap(index, zweit):.2f}",
        flush=True,
    )
    print(f"  dots:   {' '.join(index.split())[:150]}")
    print(f"  paddle: {' '.join(zweit.split())[:150]}")
