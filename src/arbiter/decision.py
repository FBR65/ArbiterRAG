"""The Laya decision layer (spec F).

One state carries the question and every candidate passage, marked as
``[1]``, ``[2]``, … One request asks a ``noul`` question per marker, so the
model sees all candidates at once and returns P(true) for each. A second
request grades the admissible passages with a fixed rubric; a third compares
them pairwise. Nothing here re-orders: it accepts, drops and ranks coverage.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx

# Laya's multilingual checkpoint has a 1024-token window. Each passage is cut
# to keep five candidates plus the question inside it; a cut is reported by the
# server as ``usage.truncated``.
PASSAGE_CHARS = 420
RETRY_DELAYS = (1, 2, 4)
RUBRIK_DECKUNG = [
    "keine Deckung: der Text beantwortet die Frage nicht",
    "randlich: der Text berührt das Thema, nennt aber keine Antwort",
    "teilweise: der Text enthält einen Teil der Antwort",
    "vollständig: der Text beantwortet die Frage direkt",
]
RUBRIK_PAAR = [
    "unabhängig: die beiden Auszüge haben nichts gemeinsam",
    "gleiches Thema, verschiedene Aussage",
    "gleiche Aussage in anderer Formulierung",
    "überwiegend gleich, kleine Abweichung",
    "Paraphrase: inhaltlich identisch",
]


class DecisionError(RuntimeError):
    """Raised when the decision backend cannot answer."""


class SystemOne:
    """Client for the System One route (``/v1/systemone``)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._model = model
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, transport=transport
        )
        self._sleep = sleep or time.sleep

    def ask(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        """Send state and typed questions; return the answers map."""
        payload = {"model": self._model, "state": state, "questions": questions}
        last_error = ""
        for delay in RETRY_DELAYS:
            try:
                response = self._client.post("/v1/systemone", json=payload)
            except httpx.HTTPError as exc:
                last_error = f"nicht erreichbar ({exc})"
            else:
                if response.status_code == 200:
                    return dict(response.json().get("answers", {}))
                last_error = f"HTTP {response.status_code} — {response.text[:200]}"
                if response.status_code not in (429, 503):
                    raise DecisionError(f"{self._model}: {last_error}")
            self._sleep(delay)
        raise DecisionError(f"{self._model}: nach {len(RETRY_DELAYS)} Versuchen {last_error}")

    def close(self) -> None:
        self._client.close()


def build_noul_state(question: str, passages: list[str]) -> str:
    """Question plus every candidate, each marked with its index."""
    parts = [f"Frage: {question}"]
    for number, text in enumerate(passages, start=1):
        cut = " ".join(text.split())[:PASSAGE_CHARS]
        parts.append(f"Auszug [{number}]:\n{cut}")
    return "\n\n".join(parts)


def build_questions(count: int) -> dict[str, Any]:
    """One ``noul`` question per passage marker."""
    return {
        f"p{number}": {
            "type": "noul",
            "instructions": (
                f"Beantwortet Auszug [{number}] die Frage? "
                "Antworte mit ja nur, wenn der Auszug die Antwort enthält."
            ),
        }
        for number in range(1, count + 1)
    }


def build_deckung_questions(count: int) -> dict[str, Any]:
    """One ``score`` question per passage marker over the coverage rubric."""
    return {
        f"d{number}": {
            "type": "score",
            "instructions": f"Wie vollständig beantwortet Auszug [{number}] die Frage?",
            "criteria": RUBRIK_DECKUNG,
        }
        for number in range(1, count + 1)
    }


def build_pair_questions(pairs: list[tuple[int, int]]) -> dict[str, Any]:
    """One ``score`` question per pair over the similarity rubric."""
    return {
        f"v{a}_{b}": {
            "type": "score",
            "instructions": f"Wie ähnlich sind Auszug [{a}] und Auszug [{b}] in der Aussage?",
            "criteria": RUBRIK_PAAR,
        }
        for a, b in pairs
    }


def build_widerspruch_questions(pairs: list[tuple[int, int]]) -> dict[str, Any]:
    """One ``noul`` question per pair: do the two passages contradict each other?"""
    return {
        f"w{a}_{b}": {
            "type": "noul",
            "instructions": (
                f"Widersprechen sich Auszug [{a}] und Auszug [{b}] in einer Aussage? "
                "Antworte mit ja nur bei einem echten inhaltlichen Widerspruch."
            ),
        }
        for a, b in pairs
    }


def build_stuetzen_state(passage: str, antwort: str) -> str:
    """Auszug plus the answer that claims to come from it (role: answer check)."""
    cut = " ".join(passage.split())[:PASSAGE_CHARS]
    return f"Auszug [1]:\n{cut}\n\nAntwort:\n{antwort.strip()}"


def build_stuetzen_questions() -> dict[str, Any]:
    """One ``noul`` question: does this passage support the answer?"""
    return {
        "s1": {
            "type": "noul",
            "instructions": (
                "Stützt Auszug [1] die Antwort? Antworte mit ja nur, wenn die "
                "Aussagen der Antwort im Auszug stehen. Eine Antwort, die nur "
                "zum Thema passt, stützt der Auszug nicht."
            ),
        }
    }


def build_choice_questions(count: int) -> dict[str, Any]:
    """One ``choice`` question over all markers plus "keine" (E2).

    A small model decides better between named options than on an absolute
    yes/no per passage: the softmax over options forces one winner, so a state
    that contains no answer has to spend its probability on "keine".
    """
    criteria = {
        f"p{number}": f"Auszug [{number}] beantwortet die Frage" for number in range(1, count + 1)
    }
    criteria["keine"] = "Keiner der Auszüge beantwortet die Frage"
    return {
        "wahl": {
            "type": "choice",
            "instructions": "Welcher Auszug beantwortet die Frage?",
            "criteria": criteria,
        }
    }


def choice_probabilities(answers: dict[str, Any], key: str = "wahl") -> dict[str, float]:
    """Probabilities per option of one ``choice`` answer (empty if missing)."""
    entry = answers.get(key)
    if not isinstance(entry, dict):
        return {}
    probabilities = entry.get("probabilities")
    if not isinstance(probabilities, dict):
        return {}
    return {str(option): float(value) for option, value in probabilities.items()}


def ja_wahrscheinlichkeit(verteilung: dict[str, float]) -> float | None:
    """Sum the probability of every token that means yes (role: logprob gate)."""
    if not verteilung:
        return None
    ja = 0.0
    nein = 0.0
    for token, wert in verteilung.items():
        wort = token.strip().strip(".,!;:").casefold()
        if wort.startswith("ja"):
            ja += wert
        elif wort.startswith("nein"):
            nein += wert
    if ja == 0.0 and nein == 0.0:
        return None
    return ja


def build_ja_nein_prompt(question: str, passage: str, variante: str = "streng") -> tuple[str, str]:
    """System and user message for a single-token yes/no decision.

    ``streng`` demands the whole answer in the text, ``teil`` accepts a partial
    clue, ``beleg`` asks whether the text contains the answer at all.
    """
    if variante == "teil":
        system = (
            "Du prüfst, ob ein Text zur Beantwortung einer Frage beiträgt. "
            "Antworte mit genau einem Wort: ja oder nein. Antworte mit ja, wenn "
            "der Text Angaben enthält, die für die Antwort nötig sind — auch wenn "
            "sie allein nicht ausreichen."
        )
    elif variante == "beleg":
        system = (
            "Enthält der Text die Information, nach der die Frage fragt? Antworte "
            "mit genau einem Wort: ja oder nein."
        )
    else:
        system = (
            "Du prüfst, ob ein Text eine Frage beantwortet. Antworte mit genau einem "
            "Wort: ja oder nein. Antworte mit ja nur, wenn die Antwort auf die Frage "
            "wirklich im Text steht."
        )
    return system, f"Frage: {question}\n\nText: {passage}"


def probability(answers: dict[str, Any], key: str) -> tuple[float, str | None]:
    """P(true) of one ``noul`` answer; a missing key is 0.0 with a note."""
    entry = answers.get(key)
    if not isinstance(entry, dict) or "noul" not in entry:
        return 0.0, f"keine Antwort für {key}"
    return float(entry["noul"]), None


def level(answers: dict[str, Any], key: str) -> float | None:
    """Rubric level 1..n of one ``score`` answer (0-based score + 1)."""
    entry = answers.get(key)
    if not isinstance(entry, dict) or "score" not in entry:
        return None
    return float(entry["score"]) + 1.0
