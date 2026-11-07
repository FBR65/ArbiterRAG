"""Print a short outline of the indexed corpus (inspection, not part of the suite)."""

from __future__ import annotations

import os
from pathlib import Path

from arbiter.config import load_config
from arbiter.store import Store

cfg = load_config(Path("arbiter.yaml"), os.environ)
store = Store(cfg.db_path)
table = store._table("downloads")
rows = sorted(table.to_arrow().to_pylist(), key=lambda row: row["id"])
current = None
for row in rows:
    head = " ".join(row["text"].split())[:110]
    if row["quelle"] != current:
        current = row["quelle"]
        print(f"\n== {current}")
    print(f"  S{row['seite']:>3} {row['id'].split('#')[-1]:>2}: {head}")
