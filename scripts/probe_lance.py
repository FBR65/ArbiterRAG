"""Probe the installed LanceDB API surface used by arbiter.store (not part of the suite)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import lancedb
import pyarrow as pa

DIM = 4


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="lance-probe-"))
    try:
        db = lancedb.connect(str(root))
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("quelle", pa.string()),
                pa.field("seite", pa.int32()),
                pa.field("hash", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), DIM)),
            ]
        )
        table = db.create_table("probe", schema=schema)
        rows = [
            {
                "id": "a",
                "text": "Die Miete betraegt 900 Euro",
                "quelle": "a.pdf",
                "seite": 1,
                "hash": "h1",
                "vector": [1.0, 0.0, 0.0, 0.0],
            },
            {
                "id": "b",
                "text": "Das Wetter ist sonnig",
                "quelle": "b.pdf",
                "seite": 2,
                "hash": "h2",
                "vector": [0.0, 1.0, 0.0, 0.0],
            },
        ]
        table.add(rows)
        print("count:", table.count_rows())

        table.create_fts_index("text", replace=True, lower_case=True, ascii_folding=True)
        print("fts index ok")

        fts = table.search("Wetter", query_type="fts").limit(5).to_list()
        print("fts rows:", [{k: v for k, v in row.items() if k != "vector"} for row in fts])

        dense = table.search([0.9, 0.1, 0.0, 0.0]).limit(5).to_list()
        print(
            "dense rows (default metric):",
            [{k: v for k, v in row.items() if k != "vector"} for row in dense],
        )

        builder = table.search([0.9, 0.1, 0.0, 0.0])
        print("has distance_type:", hasattr(builder, "distance_type"))
        if hasattr(builder, "distance_type"):
            dense_cos = builder.distance_type("cosine").limit(5).to_list()
            print(
                "dense rows (cosine):",
                [{k: v for k, v in row.items() if k != "vector"} for row in dense_cos],
            )

        table.delete("id = 'a'")
        print("count after delete:", table.count_rows())

        names = db.table_names()
        print("table_names:", names)
        print("open_table:", db.open_table("probe").count_rows())
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
