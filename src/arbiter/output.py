"""Serialisation of results for CLI ``--json`` and the HTTP layer (spec H3, P3).

One shape for both clients, so the CLI and the API can never drift apart.
"""

from __future__ import annotations

from typing import Any

from arbiter.pipeline import Antwort, Passage, SearchResult


def passage_json(passage: Passage) -> dict[str, Any]:
    return {
        "id": passage.id,
        "quelle": passage.quelle,
        "seite": passage.seite,
        "rrf_rank": passage.rrf_rank,
        "rrf_score": passage.rrf_score,
        "rerank_score": passage.rerank_score,
        "text": passage.text,
    }


def suchergebnis_json(result: SearchResult) -> dict[str, Any]:
    return {
        "frage": result.question,
        "hyde_text": result.hyde_text,
        "rerank": result.rerank,
        "passagen": [passage_json(passage) for passage in result.passages],
        "warnungen": result.warnungen,
    }


def antwort_json(antwort: Antwort) -> dict[str, Any]:
    entscheidung = antwort.entscheidung
    return {
        "antwort": antwort.text,
        "grund": antwort.grund,
        "fehler": antwort.fehler,
        "quellen": [
            {
                "id": quelle.passage_id,
                "quelle": quelle.quelle,
                "seite": quelle.seite,
            }
            for quelle in antwort.quellen
        ],
        "entscheidung": None
        if entscheidung is None
        else {
            "beantwortbar": entscheidung.beantwortbar,
            "zulassungen": [
                {
                    "id": zulassung.passage.id,
                    "p": zulassung.p,
                    "deckung": zulassung.deckung,
                }
                for zulassung in entscheidung.zulassungen
            ],
            "auswahl": [zulassung.passage.id for zulassung in entscheidung.auswahl],
            "paare": [
                {
                    "a": befund.a,
                    "b": befund.b,
                    "aehnlichkeit": befund.aehnlichkeit,
                    "widerspruch_p": befund.widerspruch_p,
                    "widerspruch": befund.widerspruch,
                }
                for befund in entscheidung.paare
            ],
        },
        "warnungen": antwort.warnungen,
    }
