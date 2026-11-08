"""Answer generation from admissible passages (spec G).

Only passages the decision layer accepted reach this prompt. Every statement
must carry the source of the passage it came from, so a reader can check it.
"""

from __future__ import annotations

from dataclasses import dataclass

# A token, not a sentence: an instruction that spells out the refusal sentence ends up
# inside the answer ("Das steht nicht in den Auszügen, aber es wird erwähnt ...").
KEINE_ANTWORT = "KEINE_ANTWORT"

SYSTEM = (
    "Du beantwortest die Frage ausschließlich aus den gegebenen Auszügen. "
    "Jede Aussage trägt die Quelle in eckigen Klammern, zum Beispiel [a.pdf, S. 4]. "
    "Beantworten die Auszüge die Frage nur zum Teil, antworte mit dem, was sie hergeben, "
    "und nenne die Lücke nicht. "
    f"Nur wenn die Auszüge nichts zur Frage enthalten, gib allein das Wort {KEINE_ANTWORT} aus. "
    "Schreibe keinen Satz über die Auszüge selbst, keine Vorrede, keine Anmerkung zur "
    "eigenen Antwort und keine Erklärung, was fehlt. Erfinde keine Zahlen und keine Quellen."
)


@dataclass(frozen=True)
class Quelle:
    passage_id: str
    quelle: str
    seite: int | None
    text: str


def build_answer_prompt(question: str, quellen: list[Quelle]) -> tuple[str, str]:
    """System and user message for the answer step."""
    blocks = []
    for number, quelle in enumerate(quellen, start=1):
        seite = f", S. {quelle.seite}" if quelle.seite is not None else ""
        blocks.append(f"Auszug {number} [{quelle.quelle}{seite}]:\n{quelle.text}")
    user = f"Frage: {question}\n\n" + "\n\n".join(blocks)
    return SYSTEM, user
