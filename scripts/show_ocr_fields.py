"""Show the OCR fields of the indexed scan page (inspection, not part of the suite)."""

from __future__ import annotations

import os
from pathlib import Path

from arbiter.config import load_config
from arbiter.ocr import word_overlap
from arbiter.store import Store

cfg = load_config(Path("arbiter.yaml"), os.environ)
store = Store(cfg.db_path)
hits = store.search_fts("downloads", "TRENNBLATT", k=3)
for hit in hits:
    print(f"id={hit.id} ocr={hit.ocr} ueberlappung={hit.ocr_ueberlappung}")
    print("  dots :", " ".join(hit.text.split())[:160])
    print("  paddle:", " ".join(hit.ocr_paddle.split())[:160])
    print("  neu berechnet:", round(word_overlap(hit.text, hit.ocr_paddle), 3))
