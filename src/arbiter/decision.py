"""The gate: one yes/no decision per candidate passage (spec F).

The decision is read from the logprobs of a single generated token. Measured
2026-11-07: with the prompt variant "teil" the distribution separates unanswerable
questions (maximum 0.09) from answerable ones (median 1.00), so the threshold works.

Earlier forms — a 322M multilingual model asked per passage, as one choice over all
candidates, and as a checker of the finished answer — were removed: they removed
good answers and no bad ones. See docs/EVIDENCE-verbesserungen.md.
"""

from __future__ import annotations


class DecisionError(Exception):
    """The gate could not be asked."""


GATE_GRAMMATIK = 'root ::= "ja" | "nein"'


def ja_wahrscheinlichkeit(verteilung: dict[str, float]) -> float | None:
    """Sum the probability of every token that means yes.

    ``None`` means "no usable answer" — the caller must not read that as a no.
    """
    if not verteilung:
        return None
    ja = 0.0
    nein = 0.0
    for token, wert in verteilung.items():
        wort = token.strip().strip(".,!;:").casefold()
        if wort.startswith("ja"):
            ja += wert
        elif wort.startswith(("nein", "ne")):
            # With the grammar only "ja" and "nein" are allowed, and "nein" is split
            # into "ne" + "in" by the tokenizer; the prefix still identifies it.
            nein += wert
    if ja == 0.0 and nein == 0.0:
        return None
    return ja


def build_ja_nein_prompt(question: str, passage: str, variante: str = "teil") -> tuple[str, str]:
    """System and user message for a single-token yes/no decision.

    ``teil`` accepts a partial clue, ``streng`` demands the whole answer in the
    text, ``beleg`` asks whether the text contains the answer at all. Measured:
    ``teil`` separates best, ``streng`` and ``beleg`` cost about a third of the
    answerable questions.
    """
    if variante == "streng":
        system = (
            "Du prüfst, ob ein Text eine Frage beantwortet. Antworte mit genau einem "
            "Wort: ja oder nein. Antworte mit ja nur, wenn die Antwort auf die Frage "
            "wirklich im Text steht."
        )
    elif variante == "beleg":
        system = (
            "Enthält der Text die Information, nach der die Frage fragt? Antworte "
            "mit genau einem Wort: ja oder nein."
        )
    else:
        system = (
            "Du prüfst, ob ein Text zur Beantwortung einer Frage beiträgt. "
            "Antworte mit genau einem Wort: ja oder nein. Antworte mit ja, wenn "
            "der Text Angaben enthält, die für die Antwort nötig sind — auch wenn "
            "sie allein nicht ausreichen. Frage und Text können in verschiedenen "
            "Sprachen stehen; das ist kein Grund für nein."
        )
    return system, f"Frage: {question}\n\nText: {passage}"
