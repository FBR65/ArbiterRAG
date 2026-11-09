"""Spec O: Indexierung je Datei, Seitenzuordnung, Strategie im Bericht (Stufe 7.1)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from arbiter.config import Backends, Config, Rag
from arbiter.pipeline import Pipeline, seite_bei, verkette_seiten
from arbiter.store import Record, Store

HANDBUCH = """\
2 Anreise
2.1 Torzeiten
Das untere Tor oeffnet um 17:00. Es schliesst nach Abschnitt 2.2.

2.2 Dunkles Fenster
Ein dunkles Fenster ist jede Zeitspanne zwischen Daemmerung und Nacht.
"""


class FlatEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _pipeline(tmp_path: Path, text: str = HANDBUCH, **rag_aenderungen: object) -> Pipeline:
    corpus = tmp_path / "korpus"
    corpus.mkdir(exist_ok=True)
    (corpus / "handbuch.txt").write_text(text, encoding="utf-8")
    werte = {
        "name": "k",
        "description": "Test",
        "corpus_path": corpus,
        "extensions": (".txt",),
        "rerank": False,
    }
    werte.update(rag_aenderungen)
    cfg = Config(
        db_path=tmp_path / "db",
        backends=Backends(
            base_url="http://test", embed_model="m", rerank_model="r", chat_model="c"
        ),
        rags={"k": Rag(**werte)},  # type: ignore[arg-type]
    )
    return Pipeline(cfg, cfg.rags["k"], embedder=FlatEmbedder())


def test_seiten_werden_mit_position_verkettet() -> None:
    text, anfaenge = verkette_seiten(["Seite eins.", "Seite zwei."])
    assert text == "Seite eins.\nSeite zwei."
    assert anfaenge == [0, 12]


def test_position_wird_der_seite_zugeordnet() -> None:
    anfaenge = [0, 12, 30]
    assert seite_bei(anfaenge, 0) == 1
    assert seite_bei(anfaenge, 11) == 1
    assert seite_bei(anfaenge, 12) == 2
    assert seite_bei(anfaenge, 29) == 2
    assert seite_bei(anfaenge, 30) == 3


def test_der_bericht_nennt_die_strategie_je_datei(tmp_path: Path) -> None:
    report = _pipeline(tmp_path).index()
    assert report.dateien
    bericht = report.dateien[0]
    assert bericht.quelle == "handbuch.txt"
    assert bericht.strategie == "struktur"
    assert bericht.kennzahlen is not None
    assert bericht.kennzahlen.koepfe_pro_1000 > 0
    assert bericht.chunks > 0


def test_jede_zeile_traegt_ihre_strategie(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline.index()
    rows = pipeline.store._table("k").to_arrow().to_pylist()
    assert rows
    assert {row["strategie"] for row in rows} == {"struktur"}


def test_uebersteuerte_strategie_steht_im_bericht(tmp_path: Path) -> None:
    report = _pipeline(tmp_path, chunk_strategie="fenster").index()
    bericht = report.dateien[0]
    assert bericht.strategie == "fenster"
    assert bericht.uebersteuert is True


def test_bericht_nennt_tabellen_und_zeilen_chunks(tmp_path: Path) -> None:
    text = (
        "1 Rechnung\n"
        "Die Rechnung nennt zwei Posten.\n\n"
        "| Art | Betrag |\n| --- | --- |\n"
        "| Miete | 940,00 |\n| Strom | 80,00 |\n| Gas | 60,00 |\n"
        "| Wasser | 40,00 |\n| Heizung | 120,00 |\n"
    )
    pipeline = _pipeline(tmp_path, text=text)
    report = pipeline.index()
    bericht = report.dateien[0]
    assert bericht.tabellen == 1
    assert bericht.zeilen_chunks == 2
    assert report.tabellen == 1
    assert report.zeilen_chunks == 2
    rows = pipeline.store._table("k").to_arrow().to_pylist()
    zeilen = [row for row in rows if "Zeile:" in row["text"]]
    assert len(zeilen) == 2
    assert all(row["strategie"] == bericht.strategie for row in rows)



def test_geaenderte_entscheider_schwelle_liest_die_datei_neu(tmp_path: Path) -> None:
    _pipeline(tmp_path).index()
    zweiter = _pipeline(tmp_path, xref_max=1).index()
    assert zweiter.unchanged == 0


def test_seitenuebergreifender_chunk_wird_auf_beiden_seiten_gefunden(tmp_path: Path) -> None:
    store = Store(tmp_path / "db", dim=2)
    uebergreifend = "Regel und Ausnahme in einem Abschnitt."
    nur_seite_eins = "Nur auf Seite eins."
    store.upsert(
        "k",
        [
            Record(
                id="a.pdf#1#1",
                text=nur_seite_eins,
                quelle="a.pdf",
                seite=1,
                seite_ende=1,
                hash=hashlib.sha256(nur_seite_eins.encode("utf-8")).hexdigest(),
                vector=[0.0, 1.0],
            ),
            Record(
                id="a.pdf#3#1",
                text=uebergreifend,
                quelle="a.pdf",
                seite=3,
                seite_ende=4,
                hash=hashlib.sha256(uebergreifend.encode("utf-8")).hexdigest(),
                vector=[1.0, 0.0],
            ),
        ],
    )
    assert uebergreifend in store.page_text("k", "a.pdf", 3)
    assert uebergreifend in store.page_text("k", "a.pdf", 4)
    # Seite 4 darf nicht ueber den Rueckfall auf das ganze Dokument gefunden werden.
    assert nur_seite_eins not in store.page_text("k", "a.pdf", 4)


def test_eingespeiste_qe_quelle_ergibt_qes_zeilen(tmp_path: Path) -> None:
    from arbiter.chunk import QePaar

    pipeline = _pipeline(
        tmp_path,
        "Der Hund bellt im Hof. Das Tor oeffnet um 17:00. Es schliesst um 22:00.",
        chunk_strategie="qes",
    )
    pipeline.qe_quelle = lambda _text: [
        QePaar(frage="Wann oeffnet das Tor?", belege=("Das Tor oeffnet um 17:00.",))
    ]
    report = pipeline.index()
    assert report.dateien[0].strategie == "qes"
    assert all(datei.strategie == "qes" for datei in report.dateien)


def test_strategie_wechsel_liest_die_datei_neu(tmp_path: Path) -> None:
    _pipeline(tmp_path).index()
    zweiter = _pipeline(tmp_path, chunk_strategie="fenster").index()
    assert zweiter.unchanged == 0, "eine andere Strategie aendert den Index"
