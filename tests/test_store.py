"""Spec B3/B4/B7 and C1/C2: LanceDB store."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from arbiter.store import Hit, Record, Store, StoreError

DIM = 4


def _record(rid: str, text: str, vector: list[float], quelle: str = "a.pdf") -> Record:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return Record(id=rid, text=text, quelle=quelle, seite=1, hash=digest, vector=vector)


def _store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "db", dim=DIM)
    store.upsert("miethaus", [_record("a#1", "Die Miete betraegt 900 Euro", [1, 0, 0, 0])])
    store.upsert("miethaus", [_record("b#1", "Das Wetter ist sonnig", [0, 1, 0, 0])])
    store.ensure_fts("miethaus")
    return store


def test_upsert_and_count(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.count("miethaus") == 2


def test_resame_id_replaces_instead_of_duplicating(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert("miethaus", [_record("a#1", "Die Miete betraegt 950 Euro", [1, 0, 0, 0])])
    # The pipeline rebuilds the full-text index once after all upserts (B7);
    # new rows are not searchable before that.
    store.ensure_fts("miethaus")
    assert store.count("miethaus") == 2
    hits = store.search_fts("miethaus", "950", k=5)
    assert len(hits) == 1
    assert "950" in hits[0].text


def test_dense_search_orders_by_similarity(tmp_path: Path) -> None:
    store = _store(tmp_path)
    hits = store.search_dense("miethaus", [0.9, 0.1, 0, 0], k=2)
    assert hits[0].id == "a#1"
    assert isinstance(hits[0], Hit)
    assert hits[0].quelle == "a.pdf"
    assert hits[0].seite == 1


def test_full_text_search_finds_german_term(tmp_path: Path) -> None:
    store = _store(tmp_path)
    hits = store.search_fts("miethaus", "Wetter", k=5)
    assert [h.id for h in hits] == ["b#1"]


def test_search_without_fts_index_is_an_error(tmp_path: Path) -> None:
    store = Store(tmp_path / "db", dim=DIM)
    store.upsert("miethaus", [_record("a#1", "Text", [1, 0, 0, 0])])
    with pytest.raises(StoreError) as err:
        store.search_fts("miethaus", "Text", k=5)
    assert "Volltextindex" in str(err.value)


def test_unknown_table_is_an_error(tmp_path: Path) -> None:
    store = Store(tmp_path / "db", dim=DIM)
    with pytest.raises(StoreError) as err:
        store.search_dense("gibtsnicht", [1, 0, 0, 0], k=1)
    assert "gibtsnicht" in str(err.value)


def test_ensure_fts_is_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.ensure_fts("miethaus")
    assert store.count("miethaus") == 2
