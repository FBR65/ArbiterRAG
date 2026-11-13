"""SPEC 11.16: Layout-Funde im Index (Tabellen-Chunks, Textentfernung, Bericht).

Der Detektor ist injiziert. Diese Tests belegen die Kette ab dem Fund; die
Erkennung selbst prüft tests/test_layout.py, das echte Modell der Messlauf.
"""

from __future__ import annotations

from pathlib import Path

from arbiter.config import Backends, Config, Rag
from arbiter.pipeline import Pipeline, Tabellenfund
from tests.test_loaders import _make_text_pdf

MARKDOWN = "| Art | Betrag |\n| --- | --- |\n| Miete | 940 |\n| Strom | 80 |"
ZEILEN = ("Kopf", "Miete 940", "Strom 80")


class FlatEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _pipeline(
    tmp_path: Path,
    zeilen: list[str],
    hook: object = None,
    *,
    layout: bool = True,
    **rag_aenderungen: object,
) -> Pipeline:
    korpus = tmp_path / "korpus"
    korpus.mkdir(exist_ok=True)
    # Eine Seite, mehrere Zeilen: der Text steht in einem Tj-Objekt mit Umbrüchen.
    _make_text_pdf(korpus / "a.pdf", ["\n".join(zeilen)])
    werte = {
        "name": "k",
        "description": "Test",
        "corpus_path": korpus,
        "extensions": (".pdf",),
        "rerank": False,
        "layout": layout,
        "tabellen_kopf_zeilen": 0,
        "chunk_chars": 400,
    }
    werte.update(rag_aenderungen)
    cfg = Config(
        db_path=tmp_path / "db",
        backends=Backends(
            base_url="http://test", embed_model="m", rerank_model="r", chat_model="c"
        ),
        rags={"k": Rag(**werte)},  # type: ignore[arg-type]
    )
    pipeline = Pipeline(cfg, cfg.rags["k"], embedder=FlatEmbedder())
    if hook is not None:
        pipeline.layout_quelle = hook  # type: ignore[assignment]
    return pipeline


SEITE = ["Einleitung", "Kopf", "Miete 940", "Strom 80", "Schluss"]


def _hook(funde: list[Tabellenfund]):
    def quelle(quelle: str, seite: int) -> list[Tabellenfund]:
        return funde if seite == 1 else []

    return quelle


def test_layout_fund_erzeugt_zeilen_chunks_und_entfernt_den_text(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        SEITE,
        _hook([Tabellenfund(kasten=(0.05, 0.3, 0.99, 0.6), markdown=MARKDOWN, zeilen=ZEILEN)]),
    )
    report = pipeline.index()

    bericht = report.dateien[0]
    assert bericht.tabellen == 1
    assert bericht.zeilen_chunks == 2
    assert bericht.tabellen_seiten == 1
    assert report.tabellen == 1 and report.zeilen_chunks == 2
    assert report.warnings == []

    rows = pipeline.store._table("k").to_arrow().to_pylist()
    zeilen = [row for row in rows if "Zeile:" in row["text"]]
    assert len(zeilen) == 2
    assert {row["seite"] for row in zeilen} == {1}
    # Der Tabellentext steht nicht mehr im Fließtext (W6, D44).
    prosatext = " ".join(row["text"] for row in rows if "Zeile:" not in row["text"])
    assert "Miete 940" not in prosatext
    assert "Einleitung" in prosatext and "Schluss" in prosatext


def test_fehlende_zeile_wird_gemeldet_statt_still_entfernt(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        SEITE,
        _hook(
            [
                Tabellenfund(
                    kasten=(0.05, 0.3, 0.99, 0.6),
                    markdown=MARKDOWN,
                    zeilen=("Kopf", "fehlt"),
                )
            ]
        ),
    )
    report = pipeline.index()
    assert any("nicht im Seitentext gefunden" in warnung for warnung in report.warnings)
    rows = pipeline.store._table("k").to_arrow().to_pylist()
    assert any("Miete 940" in row["text"] for row in rows)


def test_erkennung_fehler_haelt_den_ingest_nicht_an(tmp_path: Path) -> None:
    def kaputt(quelle: str, seite: int) -> list[Tabellenfund]:
        raise RuntimeError("Modell antwortet nicht")

    pipeline = _pipeline(tmp_path, SEITE, kaputt)
    report = pipeline.index()
    assert report.chunks > 0
    assert report.tabellen == 0
    assert any("Layout-Erkennung fehlgeschlagen" in warnung for warnung in report.warnings)


def test_ohne_layout_wird_der_haken_nicht_gefragt(tmp_path: Path) -> None:
    aufrufe: list[int] = []

    def zaehle(quelle: str, seite: int) -> list[Tabellenfund]:
        aufrufe.append(seite)
        return []

    pipeline = _pipeline(
        tmp_path,
        ["Nur Text"],
        zaehle,
        layout=False,
    )
    report = pipeline.index()
    assert aufrufe == []
    assert report.tabellen == 0
    assert report.chunks > 0


def test_layout_ohne_modell_ist_eine_warnung(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, ["Nur Text"], layout_modell="")
    report = pipeline.index()
    assert any("layout_modell" in warnung for warnung in report.warnings)
    assert report.chunks > 0


def test_tabellen_chunk_traegt_die_seite_der_tabelle(tmp_path: Path) -> None:
    # Tabelle auf Seite 2: der Chunk muss dorthin gehören, nicht auf Seite 1.
    korpus = tmp_path / "korpus"
    korpus.mkdir(exist_ok=True)
    _make_text_pdf(korpus / "a.pdf", ["Erste Seite", "Zweite Seite\nKopf\nMiete 940"])
    cfg = Config(
        db_path=tmp_path / "db",
        backends=Backends(
            base_url="http://test", embed_model="m", rerank_model="r", chat_model="c"
        ),
        rags={
            "k": Rag(
                name="k",
                description="Test",
                corpus_path=korpus,
                extensions=(".pdf",),
                rerank=False,
                layout=True,
                tabellen_kopf_zeilen=0,
            )
        },
    )
    pipeline = Pipeline(cfg, cfg.rags["k"], embedder=FlatEmbedder())

    def quelle(name: str, seite: int) -> list[Tabellenfund]:
        if seite != 2:
            return []
        return [
            Tabellenfund(
                kasten=(0.05, 0.3, 0.99, 0.6),
                markdown="| Art | Betrag |\n| --- | --- |\n| Miete | 940 |",
                zeilen=("Kopf", "Miete 940"),
            )
        ]

    pipeline.layout_quelle = quelle  # type: ignore[assignment]
    pipeline.index()
    rows = pipeline.store._table("k").to_arrow().to_pylist()
    zeilen = [row for row in rows if "Zeile:" in row["text"]]
    assert zeilen
    assert {row["seite"] for row in zeilen} == {2}
