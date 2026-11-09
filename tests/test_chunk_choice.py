"""Spec N: der Entscheider waehlt die Chunk-Strategie je Datei (Stufe 7.1)."""

from __future__ import annotations

from arbiter.chunk_choice import waehle

STRUKTURTEXT = """\
1 Anreise
Der Besucher meldet sich am Tor an.

1.1 Oeffnungszeiten
Das untere Tor oeffnet um 17:00. Es schliesst nach Abschnitt 1.2.

1.2 Dunkles Fenster
Ein dunkles Fenster ist jede Zeitspanne zwischen Daemmerung und Nacht.

2 Barrierefreiheit
Der Zugang ist stufenlos nach Abschnitt 2.1.

2.1 Aufzug
Der Aufzug faesst acht Personen.
"""

FLIESSTEXT = (
    "Die Miete betraegt 940 Euro monatlich und wird am dritten Werktag faellig. "
    "Der Vermieter darf die Miete nur mit Begruendung erhoehen. "
    "Die Nebenkosten werden jaehrlich abgerechnet. "
) * 5


def test_strukturiertes_dokument_ergibt_struktur() -> None:
    auswahl = waehle(STRUKTURTEXT, ebene=2)
    assert auswahl.strategie == "struktur"
    assert auswahl.wert >= 0.60


def test_fliessender_text_ergibt_qes() -> None:
    auswahl = waehle(FLIESSTEXT, ebene=2)
    assert auswahl.strategie == "fenster"
    assert auswahl.wert < 0.60


def test_eine_riesige_kapitelueberschrift_ist_keine_struktur() -> None:
    # Gemessen 2026-11-08 an einem Blog-PDF: eine Ueberschrift ueber fast den ganzen
    # Text ergab Kopfanteil 0.60 und schickte Fliessetext in die Struktur-Strategie.
    text = "## Die Forschung\n" + "Ein langer Absatz ohne Gliederung. " * 400
    auswahl = waehle(text, ebene=2, max_chars=600)
    assert auswahl.strategie == "fenster", auswahl
    assert auswahl.kennzahlen.kopf_anteil == 0.0


def test_gleicher_text_ergibt_gleiche_auswahl() -> None:
    assert waehle(STRUKTURTEXT, ebene=2) == waehle(STRUKTURTEXT, ebene=2)
