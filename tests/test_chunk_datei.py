"""Spec M8, N5, K: Weiche je Datei (Stufe 7.1)."""

from __future__ import annotations

from arbiter.chunk import ChunkParameter, chunk_datei, chunk_fenster, chunk_text

TEXT = (
    "1 Anreise\nDer Besucher meldet sich am Tor an.\n\n"
    "1.1 Oeffnungszeiten\nDas untere Tor oeffnet um 17:00.\n"
)


def test_fenster_und_chunk_text_ergeben_denselben_text() -> None:
    text = "Absatz " * 400
    chunks = chunk_fenster(text, max_chars=600, overlap=100)
    assert [chunk.text for chunk in chunks] == chunk_text(text, 600, 100)


def test_uebersteuerte_strategie_fragt_den_entscheider_nicht() -> None:
    ergebnis = chunk_datei(
        TEXT, "handbuch.txt", ChunkParameter(strategie="fenster"), qe_quelle=None
    )
    assert ergebnis.strategie == "fenster"
    assert ergebnis.auswahl is None


def test_automatik_waehlt_bei_struktur_die_struktur() -> None:
    ergebnis = chunk_datei(TEXT, "handbuch.txt", ChunkParameter(strategie="auto"))
    assert ergebnis.strategie == "struktur"
    assert ergebnis.auswahl is not None
    assert ergebnis.auswahl.strategie == "struktur"


def test_qes_ohne_frage_beleg_paare_faellt_mit_warnung_auf_fenster() -> None:
    ergebnis = chunk_datei(TEXT, "handbuch.txt", ChunkParameter(strategie="qes"), qe_quelle=None)
    assert ergebnis.strategie == "fenster"
    assert any("Rückfall" in warnung for warnung in ergebnis.warnungen)


def test_leerer_text_ergibt_keine_chunks() -> None:
    ergebnis = chunk_datei("   \n\n ", "leer.txt", ChunkParameter(strategie="auto"))
    assert ergebnis.chunks == []


def test_struktur_chunks_tragen_ihren_pfad() -> None:
    ergebnis = chunk_datei(TEXT, "handbuch.txt", ChunkParameter(strategie="struktur"))
    assert ergebnis.chunks
    assert any(chunk.pfad for chunk in ergebnis.chunks)


def test_fenster_chunks_tragen_ihren_text_wortgleich() -> None:
    text = "Erster Absatz mit Inhalt. " * 60
    ergebnis = chunk_datei(text, "a.txt", ChunkParameter(strategie="fenster"))
    assert ergebnis.chunks
    for chunk in ergebnis.chunks:
        assert text[chunk.start : chunk.ende].strip() in chunk.text
        # Die Position zeigt auf das erste und das letzte sichtbare Zeichen, nicht
        # auf den Leerraum davor; sonst waere sie fuer die Seitenzuordnung wertlos.
        assert not text[chunk.start].isspace()
        assert not text[chunk.ende - 1].isspace()
