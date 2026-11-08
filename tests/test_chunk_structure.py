"""Spec K, L: strukturorientiertes Chunking (StrucChunk-Stil, Stufe 7.1)."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from arbiter.chunk import Chunk
from arbiter.chunk_structure import abschnitte, chunk_struktur, kennzahlen, kopfzeile

HANDBUCH = """\
1 Anreise
Der Besucher meldet sich am Tor an.

1.1 Oeffnungszeiten
Das untere Tor oeffnet um 17:00.
Es schliesst vor dem Beginn des dunklen Fensters nach Abschnitt 1.2.

1.2 Dunkles Fenster
Ein dunkles Fenster ist jede Zeitspanne zwischen dem Ende der Daemmerung und dem Beginn der Nacht.

2 Barrierefreiheit
Der Zugang zum Dom ist stufenlos.

2.1 Aufzug
Der Aufzug faesst acht Personen.
"""


def test_markdown_ueberschrift_traegt_die_ebene_der_raute() -> None:
    kopf = kopfzeile("### 4.3 Fotografie")
    assert kopf is not None
    assert kopf.ebene == 3
    assert kopf.titel == "4.3 Fotografie"


def test_nummerierte_zeile_ist_ueberschrift_mit_ebene_der_gruppen() -> None:
    kopf = kopfzeile("2.1 Aufzug")
    assert kopf is not None
    assert kopf.ebene == 2
    assert kopf.nummer == "2.1"
    assert kopf.titel == "2.1 Aufzug"


def test_fliesszeilige_saetze_sind_keine_ueberschrift() -> None:
    assert kopfzeile("Das untere Tor oeffnet um 17:00.") is None
    assert kopfzeile("Der Besucher meldet sich am Tor an, wenn er kommt.") is None


def test_grosszeile_ist_ueberschrift() -> None:
    kopf = kopfzeile("DUNKLES FENSTER")
    assert kopf is not None
    assert kopf.ebene == 2


def test_abschnitte_tragen_pfad_der_oberen_ebenen() -> None:
    teile = abschnitte(HANDBUCH)
    pfade = {teil.pfad[-1]: teil.pfad for teil in teile}
    assert pfade["1.2 Dunkles Fenster"] == (
        "1 Anreise",
        "1.2 Dunkles Fenster",
    )


def test_ein_chunk_enthaelt_nur_seinen_eigenen_rumpf() -> None:
    chunks = chunk_struktur(HANDBUCH, ebene=2, max_chars=600, overlap=0, xref_max=3, xref_chars=200)
    torzeiten = next(c for c in chunks if "Oeffnungszeiten" in c.text)
    assert "untere Tor oeffnet" in torzeiten.text
    assert "stufenlos" not in torzeiten.text, "der naechste Abschnitt gehoert nicht dazu"


def test_kurzer_abschnitt_bleibt_ganz() -> None:
    text = "1.1 Kurz\nEin Satz. Noch ein Satz.\n"
    chunks = chunk_struktur(text, ebene=2, max_chars=600, overlap=0, xref_max=3, xref_chars=200)
    assert len(chunks) == 1


def test_abschnitt_bis_eineinhalbfacher_laenge_bleibt_ganz() -> None:
    rumpf = "Ein Satz mit Inhalt. " * 30  # rund 620 Zeichen bei max_chars 500
    text = f"1.1 Lang\n{rumpf}\n"
    chunks = chunk_struktur(text, ebene=2, max_chars=500, overlap=0, xref_max=3, xref_chars=200)
    assert len(chunks) == 1, "bis 1,5 x L bleibt der Abschnitt ganz (logische Atomizitaet)"


def test_sehr_langer_abschnitt_wird_an_satzgrenzen_geteilt() -> None:
    rumpf = "Ein Satz mit Inhalt. " * 120
    text = f"1.1 Sehr lang\n{rumpf}\n"
    chunks = chunk_struktur(text, ebene=2, max_chars=300, overlap=0, xref_max=3, xref_chars=200)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 300
        assert chunk.text.rstrip().endswith(".")


def test_verweis_haengt_zieltext_an() -> None:
    chunks = chunk_struktur(HANDBUCH, ebene=2, max_chars=600, overlap=0, xref_max=3, xref_chars=200)
    torzeiten = next(c for c in chunks if "Oeffnungszeiten" in c.text)
    assert "Abschnitt 1.2" in torzeiten.text, "der eigene Rumpf bleibt unveraendert"
    assert "Ein dunkles Fenster ist jede Zeitspanne" in torzeiten.text


def test_verweis_haengt_hoechstens_die_angegebene_laenge_an() -> None:
    text = "1.1 Regel\nDas gilt nach Abschnitt 1.2.\n\n1.2 Ziel\n" + "Wort " * 200 + "\n"
    chunks = chunk_struktur(text, ebene=2, max_chars=2000, overlap=0, xref_max=3, xref_chars=40)
    regel = next(c for c in chunks if "Regel" in c.text)
    anhang = regel.text.split("↳", 1)[1]
    assert len(anhang) <= 60, "Anhang bleibt bei xref_chars"


def test_hoechstens_drei_verweise_je_chunk() -> None:
    text = (
        "1.1 Regel\nSiehe Abschnitt 1.2 und Abschnitt 1.3 und Abschnitt 1.4 und Abschnitt 1.5.\n\n"
        "1.2 A\nInhalt A.\n\n1.3 B\nInhalt B.\n\n1.4 C\nInhalt C.\n\n1.5 D\nInhalt D.\n"
    )
    chunks = chunk_struktur(text, ebene=2, max_chars=2000, overlap=0, xref_max=3, xref_chars=200)
    regel = next(c for c in chunks if "Regel" in c.text)
    assert regel.text.count("↳") == 3


def test_verweis_ohne_ziel_erzeugt_keinen_anhang() -> None:
    text = "1.1 Regel\nDas gilt nach Abschnitt 9.9.\n"
    chunks = chunk_struktur(text, ebene=2, max_chars=600, overlap=0, xref_max=3, xref_chars=200)
    assert "↳" not in chunks[0].text


def test_kein_verweis_auf_sich_selbst() -> None:
    text = "1.1 Regel\nDiese Regel gilt nach Abschnitt 1.1.\n"
    chunks = chunk_struktur(text, ebene=2, max_chars=600, overlap=0, xref_max=3, xref_chars=200)
    assert chunks[0].text.count("Regel") == 2, "kein doppelter Rumpf"


def test_kennzahlen_zaehlen_ueberschriften_und_verweise() -> None:
    kennzahlen_handbuch = kennzahlen(HANDBUCH, ebene=2)
    assert kennzahlen_handbuch.koepfe_pro_1000 > 0
    assert kennzahlen_handbuch.xrefs_pro_1000 > 0
    assert 0 < kennzahlen_handbuch.kopf_anteil <= 1.0


def test_kennzahlen_ohne_ueberschriften_sind_null() -> None:
    kennzahlen_text = kennzahlen("Ein Satz. Noch ein Satz. Und ein dritter Satz.", ebene=2)
    assert kennzahlen_text.kopf_anteil == 0.0
    assert kennzahlen_text.koepfe_pro_1000 == 0.0


@given(st.text(alphabet="ab .1\n", min_size=1, max_size=500))
def test_property_kein_textverlust(text: str) -> None:
    chunks = chunk_struktur(text, ebene=2, max_chars=200, overlap=0, xref_max=3, xref_chars=50)
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert chunk.start < chunk.ende
        assert text[chunk.start : chunk.ende].strip() in chunk.text
