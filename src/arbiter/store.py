"""LanceDB store (spec B3, B4, B7, C1, C2).

One table per RAG. Both halves of the hybrid search live here: dense vectors
and a full-text index over the same rows, so a chunk can never be found by one
half and be missing from the other.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa


class StoreError(Exception):
    """Raised when the store cannot satisfy a request."""


@dataclass(frozen=True)
class Record:
    id: str
    text: str
    quelle: str
    seite: int | None
    hash: str
    vector: list[float]


@dataclass(frozen=True)
class Hit:
    id: str
    text: str
    quelle: str
    seite: int | None
    score: float


class Store:
    """Thin wrapper around LanceDB; no query logic beyond storage."""

    def __init__(self, db_path: Path, dim: int | None = None) -> None:
        self._db = lancedb.connect(str(db_path))
        self._dim = dim

    def count(self, rag: str) -> int:
        return int(self._table(rag).count_rows())

    def upsert(self, rag: str, records: Iterable[Record]) -> int:
        """Insert new chunks, replace changed ones, drop chunks that vanished."""
        batch = list(records)
        if not batch:
            return 0

        dim = self._resolve_dim(batch)
        table = self._table_or_create(rag, dim)
        existing = {row["id"]: row["hash"] for row in table.to_arrow().to_pylist()}

        by_quelle: dict[str, list[Record]] = {}
        for record in batch:
            by_quelle.setdefault(record.quelle, []).append(record)

        stale: list[str] = []
        changed: list[Record] = []
        for quelle, group in by_quelle.items():
            keep = {record.id for record in group}
            stale.extend(
                rid for rid in existing if rid.startswith(f"{quelle}#") and rid not in keep
            )
            changed.extend(record for record in group if existing.get(record.id) != record.hash)

        self._delete(table, sorted(set(stale) | {r.id for r in changed}))
        if changed:
            table.add([_row(record) for record in changed])
        return len(changed)

    def _resolve_dim(self, batch: list[Record]) -> int:
        """The vector dimension comes from the model, so the first batch fixes it."""
        lengths = {len(record.vector) for record in batch}
        if len(lengths) != 1:
            raise StoreError(f"Vektoren mit unterschiedlicher Länge: {sorted(lengths)}")
        dim = lengths.pop()
        if self._dim is not None and dim != self._dim:
            raise StoreError(f"Vektor hat {dim} Dimensionen, erwartet {self._dim}")
        if self._dim is None:
            self._dim = dim
        return dim

    def ensure_fts(self, rag: str) -> None:
        table = self._table(rag)
        try:
            table.create_fts_index("text", replace=True, lower_case=True, ascii_folding=True)
        except Exception as exc:
            raise StoreError(f"Volltextindex für '{rag}' nicht erstellbar: {exc}") from exc

    def search_dense(self, rag: str, vector: list[float], k: int) -> list[Hit]:
        table = self._table(rag)
        if self._dim is not None and len(vector) != self._dim:
            raise StoreError(f"Vektor hat {len(vector)} Dimensionen, erwartet {self._dim}")
        rows = table.search(vector).distance_type("cosine").limit(k).to_list()
        return [
            Hit(
                id=row["id"],
                text=row["text"],
                quelle=row["quelle"],
                seite=row["seite"],
                score=1.0 - float(row["_distance"]),
            )
            for row in rows
        ]

    def search_fts(self, rag: str, query: str, k: int) -> list[Hit]:
        table = self._table(rag)
        if not query.strip():
            return []
        try:
            rows = table.search(query, query_type="fts").limit(k).to_list()
        except Exception as exc:
            raise StoreError(
                f"Volltextsuche in '{rag}' fehlgeschlagen — fehlt der Volltextindex? ({exc})"
            ) from exc
        return [
            Hit(
                id=row["id"],
                text=row["text"],
                quelle=row["quelle"],
                seite=row["seite"],
                score=float(row.get("_score", 0.0)),
            )
            for row in rows
        ]

    def _table(self, rag: str) -> Any:
        try:
            return self._db.open_table(rag)
        except Exception as exc:
            known = ", ".join(_table_name(t) for t in self._db.list_tables()) or "keine"
            raise StoreError(
                f"Kein Index für '{rag}' (vorhanden: {known}). Erst 'arbiter index {rag}'."
            ) from exc

    def _table_or_create(self, rag: str, dim: int) -> Any:
        try:
            return self._db.open_table(rag)
        except Exception:
            return self._db.create_table(rag, schema=self._schema(dim))

    def _schema(self, dim: int) -> pa.Schema:
        return pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("quelle", pa.string()),
                pa.field("seite", pa.int32()),
                pa.field("hash", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), dim)),
            ]
        )

    def _delete(self, table: Any, ids: list[str]) -> None:
        if not ids:
            return
        quoted = ", ".join("'" + rid.replace("'", "''") + "'" for rid in ids)
        table.delete(f"id IN ({quoted})")


def _table_name(entry: object) -> str:
    """LanceDB returns table listings as names or tuples, depending on version."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, (tuple, list)) and entry:
        return str(entry[0])
    return str(entry)


def _row(record: Record) -> dict[str, object]:
    return {
        "id": record.id,
        "text": record.text,
        "quelle": record.quelle,
        "seite": -1 if record.seite is None else record.seite,
        "hash": record.hash,
        "vector": record.vector,
    }
