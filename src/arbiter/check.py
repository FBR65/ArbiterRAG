"""Check every claim of a finished answer against the passage it cites.

The gate decides whether a passage may serve as evidence. It cannot stop the answer
model from writing something the passage does not say. For "no hallucination at all"
the answer itself has to be checked, sentence by sentence, against its own source —
with the same method as the gate: one token, read from the logprobs.
"""

from __future__ import annotations

import re

# A claim ends at a sentence end. The lookahead keeps "z. B." and "S. 3" in one piece.
_SATZ = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ])")

# Sentences about the sources instead of about the matter. They are no claim about the
# corpus and must not be checked (and never shown): an answer that talks about its own
# excerpts is a defect of the prompt, not content.
_META = (
    re.compile(r"\b(die|der|den|im|in\s+den)\s+ausz(ü|u)ge?n?\b", re.IGNORECASE),
    re.compile(r"\b(steht|stehen)\s+nicht\s+in\s+den\s+ausz", re.IGNORECASE),
    re.compile(r"\bes\s+wird\s+erw(ä|ae)hnt\b", re.IGNORECASE),
    re.compile(r"\b(die|diese)\s+quellen?\b.{0,40}\b(sagen|enthalten|belegen)\b", re.IGNORECASE),
    re.compile(r"\bkeine_antwort\b", re.IGNORECASE),
)


def ist_meta(satz: str) -> bool:
    """True for sentences that talk about the excerpts instead of about the question."""
    return any(muster.search(satz) for muster in _META)


_QUELLE = re.compile(r"\[([^\]]+),\s*S\.\s*(\d+)\]")


def split_claims(text: str) -> list[tuple[str, str | None, int | None]]:
    """Split an answer into claims with the source each one names.

    Returns ``(claim, quelle, seite)``. A sentence without a citation keeps ``None``
    for both — such a claim has no source and cannot be checked, which counts as
    unsupported.
    """
    claims: list[tuple[str, str | None, int | None]] = []
    for satz in _SATZ.split(text.strip()):
        satz = satz.strip()
        if not satz:
            continue
        treffer = _QUELLE.search(satz)
        if treffer is None:
            claims.append((satz, None, None))
        else:
            claims.append((satz, treffer.group(1).strip(), int(treffer.group(2))))
    return claims


def build_claim_prompt(claim: str, passage: str) -> tuple[str, str]:
    """Ask whether one sentence of the answer is stated in the cited passage."""
    system = (
        "Du prüfst eine Aussage gegen einen Textausschnitt. Antworte mit genau einem "
        "Wort: ja oder nein. Antworte mit ja, wenn die Aussage sinngemäß im "
        "Textausschnitt steht — auch in anderen Worten und in einer anderen Sprache. "
        "Prüfe den Inhalt, nicht die Wortwahl. Antworte mit nein, wenn der Ausschnitt "
        "die Aussage nicht hergibt oder ihr widerspricht."
    )
    return system, f"Aussage: {claim}\n\nTextausschnitt: {passage}"


def build_repair_prompt(quellen: list[tuple[str, int, str]], antwort: str) -> tuple[str, str]:
    """Ask the model to rewrite its answer using only what the passages state.

    Refusing is the last resort, not the first: an unsupported sentence usually
    disappears when the model is shown what it may use. The result is checked again.
    """
    system = (
        "Du überarbeitest eine Antwort. Behalte nur Aussagen, die in den Auszügen "
        "stehen. Streiche alles, was die Auszüge nicht hergeben. Nenne für jede "
        "Aussage die Quelle in eckigen Klammern, genau in der Form "
        "[Dateiname, S. Seite]. Antworte nur mit der überarbeiteten Antwort, ohne "
        "Vorrede."
    )
    auszuege = "\n\n".join(
        f"Auszug [{datei}, S. {seite}]:\n{text}" for datei, seite, text in quellen
    )
    user = f"AUSZÜGE:\n{auszuege}\n\nBISHERIGE ANTWORT:\n{antwort}"
    return system, user
