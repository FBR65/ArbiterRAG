"""The decider: which chunk strategy fits this document (spec N).

Pure function of the text. No model call, no network, no configuration beyond the
heading level and the threshold, so the same PDF always gets the same strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

from arbiter.chunk_structure import Kennzahlen, kennzahlen


@dataclass(frozen=True)
class Auswahl:
    strategie: str
    wert: float
    kennzahlen: Kennzahlen


def wert(kennzahlen_wert: Kennzahlen) -> float:
    """Structure score in [0, 1] (spec N3)."""
    kopf = min(1.0, kennzahlen_wert.kopf_anteil / 0.60)
    koepfe = min(1.0, kennzahlen_wert.koepfe_pro_1000 / 2.0)
    xrefs = min(1.0, kennzahlen_wert.xrefs_pro_1000 / 1.0)
    return round(0.5 * kopf + 0.3 * koepfe + 0.2 * xrefs, 6)


def waehle(text: str, *, ebene: int, schwelle: float, max_chars: int = 600) -> Auswahl:
    """``struktur`` when the document carries enough structure, else ``qes`` (spec N3)."""
    gemessen = kennzahlen(text, ebene=ebene, max_chars=max_chars)
    punktzahl = wert(gemessen)
    return Auswahl(
        strategie="struktur" if punktzahl >= schwelle else "qes",
        wert=punktzahl,
        kennzahlen=gemessen,
    )
