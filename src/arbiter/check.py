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
        "Wort: ja oder nein. Antworte mit ja nur, wenn der Textausschnitt die Aussage "
        "enthält. Eine Aussage, die der Text nur dem Thema nach berührt, ist nein."
    )
    return system, f"Aussage: {claim}\n\nTextausschnitt: {passage}"
