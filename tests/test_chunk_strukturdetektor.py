"""Spec N (Revision 11.12): Detektor „hat dieses Dokument Überschriften?".

Die Frage ist binär. Der Detektor sagt nicht, welche Strategie mehr Belege holt — das
ist gemessen worden und bleibt eine Messung. Er sagt nur, ob der Text eine Gliederung
hat, an der der Struktur-Chunker sinnvoll schneiden kann.
"""

from __future__ import annotations

from arbiter.chunk_choice import hat_struktur, waehle
from arbiter.chunk_structure import abschnitte, median_abschnitt

HANDBUCH = """\
1 Anreise
Der Besucher meldet sich am Tor an.

1.1 Oeffnungszeiten
Das untere Tor oeffnet um 17:00. Es schliesst nach Abschnitt 1.2.

1.2 Dunkles Fenster
Ein dunkles Fenster ist jede Zeitspanne zwischen Daemmerung und Nacht.

2 Barrierefreiheit
Der Zugang ist zum Dom stufenlos.

2.1 Aufzug
Der Aufzug faesst acht Personen.
"""

MAIL = """\
Hallo Herr Meier,

anbei die Unterlagen zum Termin am Montag. Die Dateien liegen im Ordner Freigabe.
Bitte pruefen Sie bis Freitag, ob die Zahlen im Anhang stimmen.

Viele Gruesse
Anna Schmidt
"""

FLIESSTEXT = (
    "Die Miete betraegt 940 Euro monatlich und wird am dritten Werktag faellig. "
    "Der Vermieter darf die Miete nur mit Begruendung erhoehen. "
    "Die Nebenkosten werden jaehrlich abgerechnet. "
) * 40


def test_handbuch_hat_struktur() -> None:
    assert hat_struktur(HANDBUCH, max_chars=600)


def test_mail_hat_keine_struktur() -> None:
    assert not hat_struktur(MAIL, max_chars=600)


def test_fliessetext_hat_keine_struktur() -> None:
    assert not hat_struktur(FLIESSTEXT, max_chars=600)


def test_zwei_ueberschriften_sind_gliederung_auch_bei_langen_absaetzen() -> None:
    # Revision 11.13: gemessen war die Struktur-Strategie auch dort besser, wo die
    # Abschnitte lang sind. Sie schneidet sie an Satzgrenzen und traegt den Pfad.
    text = (
        "## Einleitung\n" + "Ein langer Absatz ohne Gliederung. " * 600 + "\n## Schluss\n" + "Ende."
    )
    assert hat_struktur(text, max_chars=600)


def test_eine_einzige_ueberschrift_reicht_nicht() -> None:
    assert not hat_struktur("1.1 Titel\nEin Satz.", max_chars=600)


def test_ueberschriften_mit_kurzen_abschnitten_sind_eine_gliederung() -> None:
    text = "".join(
        f"## Abschnitt {nummer}\nKurzer Text unter Abschnitt {nummer}.\n\n"
        for nummer in range(1, 6)
    )
    assert hat_struktur(text, max_chars=600)


def test_median_abschnitt_ist_die_mittere_laenge() -> None:
    teile = abschnitte(HANDBUCH)
    assert median_abschnitt(HANDBUCH) > 0
    assert median_abschnitt(HANDBUCH) <= max(len(HANDBUCH[t.start : t.ende]) for t in teile)


def test_median_abschnitt_ohne_abschnitte_ist_null() -> None:
    assert median_abschnitt("Nur ein Satz ohne Gliederung.") == 0.0


def test_waehle_nennt_handbuch_struktur_und_mail_fenster() -> None:
    assert waehle(HANDBUCH, ebene=2, max_chars=600).strategie == "struktur"
    assert waehle(MAIL, ebene=2, max_chars=600).strategie == "fenster"


def test_die_entscheidung_haengt_nicht_mehr_an_einer_schwelle() -> None:
    # Revision 11.12: die Punktzahl aus N3 ist ersetzt. Ein Wert 0 heisst jetzt
    # „keine Gliederung", nicht „unter der Schwelle".
    auswahl = waehle(FLIESSTEXT, ebene=2, max_chars=600)
    assert auswahl.wert == 0.0
    assert auswahl.begruendung == "keine Gliederung erkannt"
