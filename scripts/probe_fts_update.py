"""Probe FTS behaviour after an update (scratch, not part of the suite)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import lancedb
import pyarrow as pa

DIM = 4


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="lance-fts-"))
    try:
        db = lancedb.connect(str(root))
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), DIM)),
            ]
        )
        table = db.create_table("t", schema=schema)
        table.add([{"id": "a#1", "text": "Die Miete betraegt 900 Euro", "vector": [1, 0, 0, 0]}])
        table.create_fts_index("text", replace=True)
        print("vorher 950:", len(table.search("950", query_type="fts").limit(5).to_list()))

        table.delete("id = 'a#1'")
        table.add([{"id": "a#1", "text": "Die Miete betraegt 950 Euro", "vector": [1, 0, 0, 0]}])
        table.create_fts_index("text", replace=True)
        for query in ("950", "Miete", "betraegt", "Miete 950"):
            rows = table.search(query, query_type="fts").limit(5).to_list()
            print(f"nachher {query!r}:", len(rows), [r["text"] for r in rows])

        print("count:", table.count_rows())
        print("rows:", [r["id"] for r in table.to_arrow().to_pylist()])
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
