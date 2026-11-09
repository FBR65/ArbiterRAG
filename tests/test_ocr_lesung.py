"""Spec V1, V9: die structured Lesung wird indiziert (SPEC 11.14, Paket 3)."""

from __future__ import annotations

from arbiter.pipeline import ocr_lesung

OTSL = "<fcel>Art<fcel>Betrag<nl><fcel>Miete<fcel>940,00<nl>"


def test_otsl_wird_die_indizierte_lesung() -> None:
    text, warnungen = ocr_lesung("[dots] Art Betrag Miete 940,00", OTSL)
    assert text.startswith("| Art | Betrag |")
    assert "| Miete | 940,00 |" in text
    assert warnungen == []


def test_ohne_tabelle_bleibt_der_dots_text() -> None:
    # V9: ohne strukturierte Lesung bleibt alles wie heute.
    assert ocr_lesung("Erkannte Miete 940 Euro", "Fließtext ohne Tabellenmarken") == (
        "Erkannte Miete 940 Euro",
        [],
    )


def test_unbekanntes_token_geht_nicht_verloren() -> None:
    _, warnungen = ocr_lesung("dots", "<fcel>A<bold>B<nl>")
    assert any("bold" in warnung for warnung in warnungen)
