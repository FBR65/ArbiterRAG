"""Question-evidence generation for the QES strategy (spec M3).

The chunker needs supervision before the index exists: for every generated question
the sentence that answers it, word for word. The model writes the pairs; a pair whose
evidence is not a sentence of the document is dropped, so the objective can only
reward text that is really there.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from arbiter.chunk import QePaar, satz_grenzen

Chat = Callable[[str, str], str]

SYSTEM = (
    "Du erstellst Prüffragen für ein Suchsystem. Antworte ausschließlich im "
    "vorgegebenen Format, ohne Einleitung und ohne Erklärung."
)
ANWEISUNG = """\
Lies den Text unten. Schreibe {anzahl} Fragen, die dieser Text beantwortet.
Zu jeder Frage gehören ein bis drei Sätze, die die Antwort wörtlich enthalten.
Schreibe die Sätze zeichengetreu ab, ohne Kürzung und ohne Umformulierung.
Format je Frage:
FRAGE: <Frage>
BELEG: <Satz aus dem Text>

Text:
{text}"""


@dataclass(frozen=True)
class QeErzeugung:
    paare: list[QePaar]
    warnungen: list[str]


def erzeuge_qe(
    text: str,
    *,
    chat: Chat,
    max_fragen: int = 60,
    fenster: int = 6000,
    fenster_max: int = 4,
) -> list[QePaar]:
    """Generate question-evidence pairs for one document (spec M3)."""
    saetze = [" ".join(text[start:ende].split()) for start, ende in satz_grenzen(text)]
    if not saetze or max_fragen <= 0 or fenster_max <= 0:
        return []

    abschnitte = _fenster(text, fenster, fenster_max)
    je_fenster = max(1, -(-max_fragen // len(abschnitte)))
    paare: list[QePaar] = []
    gesehen: set[str] = set()

    for abschnitt in abschnitte:
        try:
            antwort = chat(SYSTEM, ANWEISUNG.format(anzahl=je_fenster, text=abschnitt))
        except Exception:
            # A model that does not answer is not a reason to lose the document; the
            # caller falls back and reports it (spec M8).
            continue
        for paar in _parse(antwort):
            if paar.frage in gesehen:
                continue
            if not all(_steht_im_text(beleg, saetze) for beleg in paar.belege):
                continue
            gesehen.add(paar.frage)
            paare.append(paar)
            if len(paare) >= max_fragen:
                return paare
    return paare


def _fenster(text: str, groesse: int, anzahl_max: int) -> list[str]:
    """Windows of the document, on sentence boundaries where possible (spec M3)."""
    if groesse <= 0:
        return [text][:anzahl_max]
    abschnitte: list[str] = []
    for start in range(0, len(text), groesse):
        abschnitte.append(text[start : start + groesse])
        if len(abschnitte) >= anzahl_max:
            break
    return abschnitte


def _parse(antwort: str) -> list[QePaar]:
    """``FRAGE:`` with one or more following ``BELEG:`` lines."""
    paare: list[QePaar] = []
    frage: str | None = None
    belege: list[str] = []
    for zeile in antwort.splitlines():
        roh = zeile.strip()
        ober = roh.upper()
        if ober.startswith("FRAGE:"):
            if frage is not None and belege:
                paare.append(QePaar(frage=frage, belege=tuple(belege)))
            frage = roh[len("FRAGE:") :].strip()
            belege = []
        elif ober.startswith("BELEG:") and frage is not None:
            beleg = roh[len("BELEG:") :].strip()
            if beleg:
                belege.append(beleg)
    if frage is not None and belege:
        paare.append(QePaar(frage=frage, belege=tuple(belege)))
    return paare


def _steht_im_text(beleg: str, saetze: list[str]) -> bool:
    from arbiter.chunk_qes import beleg_bloecke

    return beleg_bloecke(beleg, saetze) is not None


def mache_qe_quelle(
    *,
    base_url: str,
    model: str,
    chat: Chat | None = None,
    max_fragen: int = 60,
    fenster: int = 6000,
    fenster_max: int = 4,
) -> Callable[[str], list[QePaar]]:
    """A generator for the pipeline: one chat model, the limits from the configuration."""

    def quelle(text: str) -> list[QePaar]:
        ruf = chat if chat is not None else _client(base_url, model)
        return erzeuge_qe(
            text,
            chat=ruf,
            max_fragen=max_fragen,
            fenster=fenster,
            fenster_max=fenster_max,
        )

    return quelle


def _client(base_url: str, model: str) -> Chat:
    from arbiter.chat import ChatClient

    client = ChatClient(base_url, model)
    return lambda system, user: client.complete(system, user)
