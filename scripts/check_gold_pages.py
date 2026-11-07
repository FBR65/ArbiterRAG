"""Print the indexed text of every gold page, so the labels can be checked (A1)."""

from __future__ import annotations

import os
from pathlib import Path

from arbiter.config import load_config
from arbiter.eval import load_goldset
from arbiter.store import Store

cfg = load_config(Path("arbiter.yaml"), os.environ)
store = Store(cfg.db_path)
rows = store._table("downloads").to_arrow().to_pylist()

for nummer, gold in enumerate(load_goldset(Path("goldset/downloads.yaml")), start=1):
    print(f"\n### {nummer}. {gold.frage}")
    if not gold.erwartet:
        print("   (nicht beantwortbar)")
        continue
    for quelle, seite in gold.erwartet:
        treffer = [row for row in rows if quelle in row["quelle"] and row["seite"] == seite]
        if not treffer:
            print(f"   FEHLT: {quelle} S{seite}")
            continue
        text = " ".join(treffer[0]["text"].split())[:230]
        print(f"   {quelle[:40]} S{seite}: {text}")
