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
    ocr: bool = False
    ocr_paddle: str = ""
    ocr_ueberlappung: float = 0.0
    eltern: str = ""
    seite_ende: int | None = None
    strategie: str = ""


@dataclass(frozen=True)
class Hit:
    id: str
    text: str
    quelle: str
    seite: int | None
    score: float
    ocr: bool = False
    ocr_paddle: str = ""
    ocr_ueberlappung: float = 0.0
    eltern: str = ""


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
        existing = {
            row["id"]: (row["hash"], str(row.get("strategie", "") or ""))
            for row in table.to_arrow().to_pylist()
        }

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
            changed.extend(
                record
                for record in group
                if existing.get(record.id) != (record.hash, record.strategie)
            )

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

    def purge(self, rag: str, vorhandene: set[str]) -> int:
        """Remove rows of files that are no longer in the corpus.

        Without this, a file that was removed from the folder (or recognised as
        a duplicate) keeps its rows and keeps stealing ranks.
        """
        table = self._table(rag)
        bekannt = {row["quelle"] for row in table.to_arrow().to_pylist()}
        verschwunden = sorted(bekannt - vorhandene)
        if not verschwunden:
            return 0
        quoted = ", ".join("'" + quelle.replace("'", "''") + "'" for quelle in verschwunden)
        table.delete(f"quelle IN ({quoted})")
        return len(verschwunden)

    def ensure_fts(self, rag: str, stem: bool = False) -> None:
        table = self._table(rag)
        try:
            table.create_fts_index(
                "text",
                replace=True,
                lower_case=True,
                ascii_folding=True,
                language="German",
                stem=stem,
            )
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
                **_ocr_fields(row),
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
                **_ocr_fields(row),
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

    def _ensure_ocr_columns(self, table: Any) -> None:
        """Add the OCR columns to a table written by an older version.

        Without this, every schema addition would force a full re-index and the
        upsert would fail with "'ocr' does not exist in table schema".
        """
        present = set(table.schema.names)
        missing = {
            name: literal
            for name, literal in (
                ("ocr", "false"),
                ("ocr_paddle", "''"),
                ("ocr_ueberlappung", "0.0"),
                ("eltern", "''"),
                ("seite_ende", "-1"),
                ("strategie", "''"),
            )
            if name not in present
        }
        if missing:
            table.add_columns(missing)

    def fingerprints(self, rag: str) -> dict[str, str]:
        """File fingerprints of the documents already in the index.

        The check has to happen *before* a file is read, otherwise OCR and embedding
        are paid again for an unchanged file.
        """
        name = f"{rag}_ingest"
        try:
            table = self._db.open_table(name)
        except Exception:
            return {}
        return {row["quelle"]: row["fingerabdruck"] for row in table.to_arrow().to_pylist()}

    def set_fingerprint(self, rag: str, quelle: str, fingerabdruck: str) -> None:
        name = f"{rag}_ingest"
        try:
            table = self._db.open_table(name)
        except Exception:
            table = self._db.create_table(
                name,
                schema=pa.schema(
                    [pa.field("quelle", pa.string()), pa.field("fingerabdruck", pa.string())]
                ),
            )
        table.delete(f"quelle = '{quelle.replace(chr(39), chr(39) * 2)}'")
        table.add([{"quelle": quelle, "fingerabdruck": fingerabdruck}])

    def page_text(self, rag: str, quelle: str, seite: int) -> str:
        """All chunks that touch one page as one text (spec O3).

        The sentence check compares an answer sentence with the page it cites, not with
        the 600-character window that happened to be retrieved: the statement may stand
        two sentences further on the same page. A chunk that starts on page 3 and ends
        on page 4 belongs to both pages, otherwise a cited page would look unsupported.
        """
        table = self._table(rag)
        sicher = quelle.replace("'", "''")
        try:
            rows = (
                table.search()
                .where(f"quelle = '{sicher}'")
                .select(["id", "text", "seite", "seite_ende"])
                .limit(5000)
                .to_list()
            )
        except Exception:
            return ""
        treffer = [row for row in rows if _deckt(row, seite)]
        if not treffer:
            # Textdateien haben keine Seitenzahl; dann gilt das ganze Dokument.
            treffer = rows
        if not treffer:
            return ""
        return " ".join(str(row["text"]) for row in sorted(treffer, key=lambda r: str(r["id"])))

    def _table_or_create(self, rag: str, dim: int) -> Any:
        try:
            table = self._db.open_table(rag)
        except Exception:
            return self._db.create_table(rag, schema=self._schema(dim))
        self._ensure_ocr_columns(table)
        return table

    def _schema(self, dim: int) -> pa.Schema:
        return pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("quelle", pa.string()),
                pa.field("seite", pa.int32()),
                pa.field("hash", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), dim)),
                pa.field("ocr", pa.bool_()),
                pa.field("ocr_paddle", pa.string()),
                pa.field("ocr_ueberlappung", pa.float64()),
                pa.field("eltern", pa.string()),
                pa.field("seite_ende", pa.int32()),
                pa.field("strategie", pa.string()),
            ]
        )

    def _delete(self, table: Any, ids: list[str]) -> None:
        if not ids:
            return
        quoted = ", ".join("'" + rid.replace("'", "''") + "'" for rid in ids)
        table.delete(f"id IN ({quoted})")


def _ocr_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Rows written before a column existed simply have no values."""
    return {
        "ocr": bool(row.get("ocr", False)),
        "ocr_paddle": str(row.get("ocr_paddle", "") or ""),
        "ocr_ueberlappung": float(row.get("ocr_ueberlappung", 0.0) or 0.0),
        "eltern": str(row.get("eltern", "") or ""),
    }


def _deckt(row: dict[str, Any], seite: int) -> bool:
    """True when the row's page range includes ``seite`` (spec O3).

    Rows from an older index have ``seite_ende`` = -1; for them the start page counts.
    """
    start = int(row.get("seite", -1) or -1)
    ende = int(row.get("seite_ende", -1) or -1)
    if start < 0:
        return False
    return start <= seite <= max(start, ende)


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
        "ocr": record.ocr,
        "ocr_paddle": record.ocr_paddle,
        "ocr_ueberlappung": record.ocr_ueberlappung,
        "eltern": record.eltern,
        "seite_ende": -1 if record.seite_ende is None else record.seite_ende,
        "strategie": record.strategie,
    }
