"""QES chunking: cost-benefit greedy with overlap (spec M, arXiv 2610.36557).

The paper's objective, in its own terms: a pair ``(question j, evidence block r)``
counts as covered when the chunk holds ``r`` **and** at least one block that the
retriever would actually surface for question ``j`` (the hit gate). Weights are
``w(j,r) = 1 / (m * |R_j|)``. On top sits a tiered structural term for the chunk's
right end: paragraph end 0.1, sentence end 0.005, clause end 0.001, else 0, all
multiplied by ``qes_lambda``.

Selection is the budgeted cost-benefit greedy with the singleton safeguard, followed
by the residual term ``f2`` (delta 0.05) when the coverage saturates. The paper's index
is not a complete index; ArbiterRAG needs one, so uncovered sentences are packed into
gap chunks afterwards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from arbiter.chunk import Chunk, ChunkParameter, EmbedderLike, QePaar, satz_grenzen

SATZENDE = (".", "!", "?")
KLAUSELENDE = (";", ":")


@dataclass(frozen=True)
class _Fenster:
    erster: int
    letzter: int
    start: int
    ende: int

    @property
    def kosten(self) -> int:
        return self.ende - self.start


def _normal(text: str) -> str:
    return " ".join(text.split())


MIN_BELEG = 10


def beleg_bloecke(beleg: str, saetze: list[str]) -> tuple[int, ...] | None:
    """The sentences a quoted evidence passage belongs to (spec M2).

    A whole sentence or a word-for-word fragment that occurs in exactly one sentence is
    accepted; the fragment is mapped to that sentence. A fragment that occurs in more
    than one sentence is ambiguous and therefore dropped, as is anything that does not
    stand word for word in the document.
    """
    gesucht = _normal(beleg)
    if len(gesucht) < MIN_BELEG:
        return None
    normalisiert = [_normal(satz) for satz in saetze]
    for nummer, satz in enumerate(normalisiert):
        if satz == gesucht:
            return (nummer,)
    treffer = [nummer for nummer, satz in enumerate(normalisiert) if gesucht in satz]
    if len(treffer) == 1:
        return (treffer[0],)
    if treffer:
        return None
    for laenge in (2, 3):
        for start in range(len(normalisiert) - laenge + 1):
            verbunden = " ".join(normalisiert[start : start + laenge])
            if gesucht in verbunden:
                return tuple(range(start, start + laenge))
    return None


def beleg_indizes(paare: list[QePaar], saetze: list[str]) -> list[tuple[int, tuple[int, ...]]]:
    """Usable pairs as ``(question index, sentence indices of its evidence)`` (spec M2).

    Every evidence passage must stand word for word in the document. A pair with a
    paraphrased or missing passage is dropped: the objective would otherwise reward
    text that is not there.
    """
    ergebnis: list[tuple[int, tuple[int, ...]]] = []
    for index, paar in enumerate(paare):
        if not paar.belege:
            continue
        bloecke: list[int] = []
        for beleg in paar.belege:
            gefunden = beleg_bloecke(beleg, saetze)
            if gefunden is None:
                bloecke = []
                break
            bloecke.extend(gefunden)
        if bloecke:
            ergebnis.append((index, tuple(dict.fromkeys(bloecke))))
    return ergebnis


def treffer_indizes(vektoren: list[list[float]], frage: list[float], k: int) -> set[int]:
    """The ``k`` blocks a first-stage retriever would surface (spec M5, hit gate)."""
    punkte = sorted(
        ((_kosinus(vektor, frage), nummer) for nummer, vektor in enumerate(vektoren)),
        key=lambda paar: (-paar[0], paar[1]),
    )
    return {nummer for _, nummer in punkte[: max(k, 0)]}


def _kosinus(links: list[float], rechts: list[float]) -> float:
    nennwert = math.sqrt(sum(w * w for w in links)) * math.sqrt(sum(w * w for w in rechts))
    if nennwert == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(links, rechts, strict=False)) / nennwert


def laengen_bereich(text: str, max_chars: int) -> tuple[int, int]:
    """Candidate length bounds: 10th..90th percentile of paragraph length (spec M6)."""
    absaetze = [teil for teil in text.split("\n\n") if teil.strip()]
    if len(absaetze) >= 4:
        laengen = sorted(len(teil.strip()) for teil in absaetze)
        kurz = max(1, laengen[int(0.10 * (len(laengen) - 1))])
        lang = max(kurz + 1, laengen[int(0.90 * (len(laengen) - 1))])
        return kurz, min(lang, max_chars)
    return max(1, max_chars // 2), max_chars


def chunke_qes(
    text: str,
    paare: list[QePaar],
    *,
    parameter: ChunkParameter,
    embedder: EmbedderLike | None,
) -> list[Chunk]:
    """QES chunks for one document; empty when the supervision is unusable (spec M)."""
    spannen = satz_grenzen(text)
    if not spannen or not paare or embedder is None:
        return []

    saetze = [text[start:ende] for start, ende in spannen]
    belege = beleg_indizes(paare, saetze)
    if not belege:
        return []

    fragen = [paare[nummer].frage for nummer, _ in belege]
    satz_vektoren = embedder.embed(saetze)
    frage_vektoren = embedder.embed(fragen)
    treffer = [
        treffer_indizes(satz_vektoren, vektor, parameter.qe_hits) for vektor in frage_vektoren
    ]

    kurz, lang = laengen_bereich(text, parameter.max_chars)
    kandidaten = _kandidaten(spannen, kurz, lang)
    if not kandidaten:
        return []

    gewaehlt = _greedy(text, spannen, kandidaten, belege, treffer, parameter)
    spannen_auswahl = _mit_luecken(spannen, gewaehlt, parameter.max_chars)
    return [
        Chunk(text=text[start:ende], start=start, ende=ende, strategie="qes")
        for start, ende in spannen_auswahl
    ]


def _kandidaten(spannen: list[tuple[int, int]], kurz: int, lang: int) -> list[_Fenster]:
    """Every sentence run whose own text lies inside the length bounds (spec M4)."""
    fenster: list[_Fenster] = []
    anzahl = len(spannen)
    for erster in range(anzahl):
        for letzter in range(erster, anzahl):
            laenge = spannen[letzter][1] - spannen[erster][0]
            if laenge > lang:
                break
            if laenge >= kurz:
                fenster.append(_Fenster(erster, letzter, spannen[erster][0], spannen[letzter][1]))
    return fenster


def _greedy(
    text: str,
    spannen: list[tuple[int, int]],
    kandidaten: list[_Fenster],
    belege: list[tuple[int, tuple[int, ...]]],
    treffer: list[set[int]],
    parameter: ChunkParameter,
) -> list[_Fenster]:
    budget = parameter.qes_budget_faktor * len(text)
    gewichte = [1.0 / (len(belege) * len(indizes)) if indizes else 0.0 for _, indizes in belege]
    abgedeckt: set[tuple[int, int]] = set()
    gewaehlt: list[_Fenster] = []
    verbraucht = 0.0

    while True:
        bester: _Fenster | None = None
        bester_wert = 0.0
        for kandidat in kandidaten:
            if kandidat in gewaehlt or verbraucht + kandidat.kosten > budget:
                continue
            nutzen = _nutzen(
                text, spannen, kandidat, belege, treffer, gewichte, abgedeckt, parameter
            )
            if nutzen <= 0.0:
                continue
            dichte = nutzen / kandidat.kosten
            if dichte > bester_wert + 1e-12 or (
                abs(dichte - bester_wert) <= 1e-12
                and bester is not None
                and (kandidat.start, kandidat.ende) < (bester.start, bester.ende)
            ):
                bester = kandidat
                bester_wert = dichte
        if bester is None:
            break
        gewaehlt.append(bester)
        verbraucht += bester.kosten
        abgedeckt |= _paare_von(bester, belege, treffer)

    return _mit_singleton_und_rest(
        text, spannen, kandidaten, belege, treffer, gewichte, gewaehlt, budget, parameter
    )


def _paare_von(
    fenster: _Fenster, belege: list[tuple[int, tuple[int, ...]]], treffer: list[set[int]]
) -> set[tuple[int, int]]:
    """Pairs a chunk covers: evidence inside it *and* one hit block inside it (spec M5)."""
    if not treffer:
        return set()
    paare: set[tuple[int, int]] = set()
    for nummer, (_, indizes) in enumerate(belege):
        if not any(fenster.erster <= block <= fenster.letzter for block in treffer[nummer]):
            continue
        for block in indizes:
            if fenster.erster <= block <= fenster.letzter:
                paare.add((nummer, block))
    return paare


def _nutzen(
    text: str,
    spannen: list[tuple[int, int]],
    fenster: _Fenster,
    belege: list[tuple[int, tuple[int, ...]]],
    treffer: list[set[int]],
    gewichte: list[float],
    abgedeckt: set[tuple[int, int]],
    parameter: ChunkParameter,
) -> float:
    neu = _paare_von(fenster, belege, treffer) - abgedeckt
    nutzen = sum(gewichte[nummer] for nummer, _ in neu)
    return nutzen + parameter.qes_lambda * _strukturwert(text, spannen, fenster.letzter)


def _strukturwert(text: str, spannen: list[tuple[int, int]], letzter: int) -> float:
    """Tiered prior on the chunk's right end (spec M5, paper section 3.2)."""
    ende = spannen[letzter][1]
    naechster = spannen[letzter + 1][0] if letzter + 1 < len(spannen) else len(text)
    absatzende = not text[ende:naechster].strip()
    satz = text[spannen[letzter][0] : ende].rstrip()
    if absatzende:
        return 0.1
    if satz.endswith(SATZENDE):
        return 0.005
    if satz.endswith(KLAUSELENDE):
        return 0.001
    return 0.0


def _mit_singleton_und_rest(
    text: str,
    spannen: list[tuple[int, int]],
    kandidaten: list[_Fenster],
    belege: list[tuple[int, tuple[int, ...]]],
    treffer: list[set[int]],
    gewichte: list[float],
    gewaehlt: list[_Fenster],
    budget: float,
    parameter: ChunkParameter,
) -> list[_Fenster]:
    """Singleton safeguard, then the residual term f2 (spec M6)."""

    def gesamt(auswahl: list[_Fenster]) -> float:
        abgedeckt: set[tuple[int, int]] = set()
        for fenster in auswahl:
            abgedeckt |= _paare_von(fenster, belege, treffer)
        return sum(gewichte[nummer] for nummer, _ in abgedeckt) + parameter.qes_lambda * sum(
            _strukturwert(text, spannen, fenster.letzter) for fenster in auswahl
        )

    bester_einzelner = max(
        (kandidat for kandidat in kandidaten if kandidat.kosten <= budget),
        key=lambda kandidat: (gesamt([kandidat]), -kandidat.start),
        default=None,
    )
    if bester_einzelner is not None and gesamt([bester_einzelner]) > gesamt(gewaehlt):
        gewaehlt = [bester_einzelner]

    verbraucht = sum(fenster.kosten for fenster in gewaehlt)
    while True:
        rest = _rest_kandidat(
            text, spannen, kandidaten, belege, treffer, gewaehlt, budget - verbraucht
        )
        if rest is None:
            break
        gewaehlt.append(rest)
        verbraucht += rest.kosten
    return gewaehlt


def _rest_kandidat(
    text: str,
    spannen: list[tuple[int, int]],
    kandidaten: list[_Fenster],
    belege: list[tuple[int, tuple[int, ...]]],
    treffer: list[set[int]],
    gewaehlt: list[_Fenster],
    restbudget: float,
) -> _Fenster | None:
    """The residual term f2: concentrate one question's evidence in one chunk (spec M6).

    f2(S) = (1/m) * sum over questions of the largest share of that question's evidence
    one retrievable chunk holds. It only matters when the coverage saturates and index
    budget is left; then it prefers a chunk that carries more of one question's answer.
    """
    if restbudget <= 0 or not belege:
        return None
    delta = 0.05
    m = len(belege)
    aktuell = _bestandsanteile(belege, treffer, gewaehlt)
    bester: _Fenster | None = None
    bester_wert = 0.0
    for kandidat in kandidaten:
        if kandidat in gewaehlt or kandidat.kosten > restbudget:
            continue
        gewinn = 0.0
        for nummer, (_, indizes) in enumerate(belege):
            if not any(kandidat.erster <= block <= kandidat.letzter for block in treffer[nummer]):
                continue
            gewinn += max(0.0, _anteil(kandidat, indizes) - aktuell[nummer])
        dichte = delta * gewinn / (m * kandidat.kosten)
        if dichte > bester_wert + 1e-12:
            bester = kandidat
            bester_wert = dichte
    return bester


def _bestandsanteile(
    belege: list[tuple[int, tuple[int, ...]]],
    treffer: list[set[int]],
    gewaehlt: list[_Fenster],
) -> list[float]:
    """For every question: the best evidence share any selected chunk already holds."""
    anteile: list[float] = []
    for nummer, (_, indizes) in enumerate(belege):
        beste = 0.0
        for fenster in gewaehlt:
            if not any(fenster.erster <= block <= fenster.letzter for block in treffer[nummer]):
                continue
            beste = max(beste, _anteil(fenster, indizes))
        anteile.append(beste)
    return anteile


def _anteil(fenster: _Fenster, indizes: tuple[int, ...]) -> float:
    if not indizes:
        return 0.0
    return sum(1 for block in indizes if fenster.erster <= block <= fenster.letzter) / len(indizes)


def _mit_luecken(
    spannen: list[tuple[int, int]], gewaehlt: list[_Fenster], max_chars: int
) -> list[tuple[int, int]]:
    """Selected windows plus chunks for everything they do not cover (spec M7, K3)."""
    abgedeckt = [False] * len(spannen)
    for fenster in gewaehlt:
        for block in range(fenster.erster, fenster.letzter + 1):
            abgedeckt[block] = True

    spannen_auswahl = [(fenster.start, fenster.ende) for fenster in gewaehlt]
    block = 0
    while block < len(spannen):
        if abgedeckt[block]:
            block += 1
            continue
        start = spannen[block][0]
        ende = spannen[block][1]
        block += 1
        while block < len(spannen) and not abgedeckt[block]:
            if spannen[block][1] - start > max_chars:
                break
            ende = spannen[block][1]
            block += 1
        spannen_auswahl.extend(_teile(start, ende, max_chars))

    return sorted(set(spannen_auswahl))


def _teile(start: int, ende: int, max_chars: int) -> list[tuple[int, int]]:
    """Hard split for a run that exceeds the window — one sentence can be that long (K5)."""
    stuecke: list[tuple[int, int]] = []
    while ende - start > max_chars:
        stuecke.append((start, start + max_chars))
        start += max_chars
    if start < ende:
        stuecke.append((start, ende))
    return stuecke
