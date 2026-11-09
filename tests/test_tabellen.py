"""Spec V: OTSL-Tabellen zu Markdown und zu Chunks (Paket 2, SPEC 11.14)."""

from __future__ import annotations

import pytest

from arbiter.tabellen import (
    Tabelle,
    markdown_zu_tabellen,
    maskiere_tabellen,
    otsl_zu_tabelle,
    tabelle_zu_chunks,
    tabelle_zu_markdown,
    tabellen_chunks,
)

BESCHEID = (
    "<fcel>Art<fcel>Betrag<lcel><fcel>Hinweis<nl>"
    "<fcel>Einkommen<fcel>1.234,56<lcel><fcel>Zeile 4<nl>"
    "<fcel>Abzug<fcel>234,56<lcel><ecel><nl>"
)


def test_otsl_ergibt_kopf_und_zeilen() -> None:
    tabelle = otsl_zu_tabelle(BESCHEID)
    assert tabelle is not None
    # <lcel> belegt einen Platz: die Zelle links spannt mit, ihr Wert wird wiederholt.
    assert tabelle.kopf == ("Art", "Betrag", "Betrag", "Hinweis")
    assert len(tabelle.zeilen) == 2
    assert tabelle.zeilen[0][0] == "Einkommen"


def test_verbundene_zelle_wird_rechts_wiederholt() -> None:
    # <lcel> heisst: an die Zelle links anschliessen. Der Wert gehoert zur Zeile.
    tabelle = otsl_zu_tabelle(BESCHEID)
    assert tabelle is not None
    assert tabelle.zeilen[0] == ("Einkommen", "1.234,56", "1.234,56", "Zeile 4")


def test_verbundene_zelle_wird_unten_wiederholt() -> None:
    text = "<fcel>A<fcel>B<nl><ucel><fcel>C<nl>"
    tabelle = otsl_zu_tabelle(text)
    assert tabelle is not None
    assert tabelle.zeilen[0] == ("A", "C")


def test_leere_zelle_bleibt_leer_und_erfindet_nichts() -> None:
    tabelle = otsl_zu_tabelle("<fcel>A<fcel>B<nl><fcel>C<ecel><nl>")
    assert tabelle is not None
    assert tabelle.zeilen[0] == ("C", "")


def test_ohne_token_gibt_es_keine_tabelle() -> None:
    assert otsl_zu_tabelle("Gewoehnlicher Text ohne Tabellenmarken.") is None


def test_unbekanntes_token_erzeugt_eine_warnung() -> None:
    tabelle = otsl_zu_tabelle("<fcel>A<bold>B<fcel>C<nl>")
    assert tabelle is not None
    assert tabelle.warnungen, "unbekannte Token duerfen nicht still verschwinden"


def test_tabelle_zu_markdown() -> None:
    tabelle = Tabelle(kopf=("Art", "Betrag"), zeilen=[("Miete", "940,00"), ("Strom", "80,00")])
    text = tabelle_zu_markdown(tabelle)
    assert text.splitlines()[0] == "| Art | Betrag |"
    assert "| Miete | 940,00 |" in text


def test_zellinhalt_mit_zeilenumbruch_bleibt_eine_zeile() -> None:
    # Ein Zellinhalt mit Umbruch zerlegte die Zeile in zwei Zeilen; die Tabelle waere
    # beim Wiedereinlesen kuerzer und die Zeile verloren.
    tabelle = Tabelle(
        kopf=("Art", "Hinweis"),
        zeilen=[("Miete", "erste Zeile\nzweite Zeile"), ("Strom", "80,00")],
    )
    text = tabelle_zu_markdown(tabelle)
    assert len(text.splitlines()) == 4
    zurueck = markdown_zu_tabellen(text)
    assert len(zurueck) == 1
    assert len(zurueck[0].zeilen) == 2


def test_markdown_tabelle_wird_gefunden() -> None:
    text = "Einleitung\n\n| Art | Betrag |\n| --- | --- |\n| Miete | 940,00 |\n\nDanach Text."
    tabellen = markdown_zu_tabellen(text)
    assert len(tabellen) == 1
    assert tabellen[0].kopf == ("Art", "Betrag")
    assert tabellen[0].zeilen == [("Miete", "940,00")]


def test_jede_zeile_ergibt_einen_chunk_mit_kopfzeile() -> None:
    tabelle = Tabelle(kopf=("Art", "Betrag"), zeilen=[("Miete", "940,00"), ("Strom", "80,00")])
    chunks = tabelle_zu_chunks(tabelle, pfad=("Rechnung 2026",), max_chars=200, kopf_zeilen=0)
    zeilen_chunks = [chunk for chunk in chunks if "Zeile:" in chunk.text]
    assert len(zeilen_chunks) == 2
    for chunk in zeilen_chunks:
        assert "Kopf: Art, Betrag" in chunk.text, "ohne Kopfzeile ist die Zeile wertlos"
        assert "Rechnung 2026" in chunk.text


def test_kopf_chunk_enthaelt_die_ersten_zeilen() -> None:
    tabelle = Tabelle(
        kopf=("Art", "Betrag"),
        zeilen=[("Miete", "940,00"), ("Strom", "80,00"), ("Gas", "60,00")],
    )
    chunks = tabelle_zu_chunks(tabelle, pfad=(), max_chars=200, kopf_zeilen=2)
    kopf_chunk = chunks[0]
    assert "Zeile:" not in kopf_chunk.text
    assert "940,00" in kopf_chunk.text and "80,00" in kopf_chunk.text
    assert "60,00" not in kopf_chunk.text


def test_breite_zeile_wird_mit_kopfzeile_geteilt() -> None:
    tabelle = Tabelle(
        kopf=("Art", "Betrag"),
        zeilen=[("Ein sehr langer Posten mit vielen Woertern", "940,00")],
    )
    chunks = tabelle_zu_chunks(tabelle, pfad=(), max_chars=60, kopf_zeilen=0)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert "Kopf: Art, Betrag" in chunk.text
        assert len(chunk.text) <= 60 * 2, "Kopfzeile und Praefix kommen hinzu"


def test_leere_zeilen_werden_uebersprungen() -> None:
    tabelle = Tabelle(kopf=("A",), zeilen=[("",), ("x",)])
    chunks = tabelle_zu_chunks(tabelle, pfad=(), max_chars=200, kopf_zeilen=0)
    assert [chunk for chunk in chunks if "Zeile:" in chunk.text] == [
        chunk for chunk in chunks if "x" in chunk.text
    ]


def test_ohne_kopfzeile_wird_das_gesagt() -> None:
    tabelle = Tabelle(kopf=(), zeilen=[("940,00",)])
    chunks = tabelle_zu_chunks(tabelle, pfad=(), max_chars=200, kopf_zeilen=0)
    assert "Kopf: unbekannt" in chunks[0].text


@pytest.mark.parametrize("zeilen", [1, 2, 5])
def test_jede_zeile_hat_gleich_viele_zellen(zeilen: int) -> None:
    text = "".join("<fcel>a<fcel>b<fcel>c<nl>" for _ in range(zeilen))
    tabelle = otsl_zu_tabelle(text)
    assert tabelle is not None
    assert all(len(zeile) == 3 for zeile in tabelle.zeilen)


MARKDOWN_MIT_TEXT = (
    "Vor der Tabelle steht ein Satz.\n\n"
    "| Art | Betrag |\n| --- | --- |\n| Miete | 940,00 |\n| Strom | 80,00 |\n\n"
    "Nach der Tabelle steht ein Satz.\n"
)


def test_tabellen_chunks_kennt_die_zeilen_und_ihre_position() -> None:
    ergebnis = tabellen_chunks(
        MARKDOWN_MIT_TEXT, max_chars=200, kopf_zeilen=0, max_zeilen=200
    )
    assert ergebnis.tabellen == 1
    assert ergebnis.zeilen_chunks == 2
    assert all("Kopf: Art, Betrag" in chunk.text for chunk in ergebnis.chunks)
    for chunk in ergebnis.chunks:
        assert chunk.start > 0, "ohne Position landet jede Tabelle auf Seite 1"
        assert chunk.ende > chunk.start
        assert MARKDOWN_MIT_TEXT[chunk.start : chunk.ende].startswith("| Art")


def test_tabellen_chunks_meldet_die_obergrenze() -> None:
    zeilen = "".join(f"| Posten {nummer} | {nummer},00 |\n" for nummer in range(5))
    text = "| Art | Betrag |\n| --- | --- |\n" + zeilen
    ergebnis = tabellen_chunks(text, max_chars=200, kopf_zeilen=0, max_zeilen=2)
    assert ergebnis.zeilen_chunks == 2
    assert any("Obergrenze" in warnung for warnung in ergebnis.warnungen)


def test_maskiere_tabellen_haelt_die_laenge_und_entfernt_nur_die_tabelle() -> None:
    maskiert = maskiere_tabellen(MARKDOWN_MIT_TEXT)
    assert len(maskiert) == len(MARKDOWN_MIT_TEXT)
    assert "|" not in maskiert
    assert "Vor der Tabelle steht ein Satz." in maskiert
    assert "Nach der Tabelle steht ein Satz." in maskiert
    assert markdown_zu_tabellen(maskiert) == []
