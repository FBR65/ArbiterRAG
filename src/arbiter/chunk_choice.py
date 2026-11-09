"""The decider: does this document have a structure the chunker can cut at? (spec N)

Pure function of the text. No model call, no network, no threshold. The question is
binary — headings and numbered clauses, or not — because a score tried to predict which
strategy retrieves better and was measured wrong (SPEC 11.11). Whether structure
*retrieves* better stays a measurement, not a claim of this module.
"""

from __future__ import annotations

from dataclasses import dataclass

from arbiter.chunk_structure import Kennzahlen, abschnitte, kennzahlen, median_abschnitt


@dataclass(frozen=True)
class Auswahl:
    strategie: str
    wert: float
    kennzahlen: Kennzahlen
    begruendung: str = ""


def hat_struktur(text: str, *, max_chars: int = 600) -> bool:
    """True when the text has at least two headings (spec N3, revision 11.13).

    The earlier version also required a small median section length. Measured on the
    corpus: that excluded exactly the files where the structure chunker won — a long
    section is still cut at sentence boundaries, and its breadcrumb still helps. The
    detector now answers only the question it can answer: are there headings at all.
    """
    del max_chars  # kept in the signature: the caller passes its chunk size
    return len(_abschnitts_laengen(text)) >= 2


def _abschnitts_laengen(text: str) -> list[int]:
    return [teil.ende - teil.start for teil in abschnitte(text) if teil.ebene > 0]


def waehle(text: str, *, ebene: int, max_chars: int = 600) -> Auswahl:
    """``struktur`` when the document is structured, else ``fenster`` (spec N3, rev. 11.12)."""
    gemessen = kennzahlen(text, ebene=ebene, max_chars=max_chars)
    laenge = median_abschnitt(text)
    if hat_struktur(text, max_chars=max_chars):
        return Auswahl(
            strategie="struktur",
            wert=laenge,
            kennzahlen=gemessen,
            begruendung=f"Gliederung erkannt, Median-Abschnitt {laenge:.0f} Zeichen",
        )
    if laenge > 0.0:
        grund = f"Überschriften ohne nutzbare Gliederung, Median-Abschnitt {laenge:.0f} Zeichen"
    else:
        grund = "keine Gliederung erkannt"
    # Rückfall ist fenster, nicht qes: qes braucht erzeugte Fragen, kostet das 2,6-fache
    # an Index und war in der Messung (7.3, 7.3b) auf keiner Datei besser. qes bleibt
    # über --strategie qes erreichbar.
    return Auswahl(strategie="fenster", wert=0.0, kennzahlen=gemessen, begruendung=grund)
