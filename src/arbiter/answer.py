"""Answer generation from admissible passages (spec G).

Only passages the decision layer accepted reach this prompt. Every statement
must carry the source of the passage it came from, so a reader can check it.
"""

from __future__ import annotations

from dataclasses import dataclass

SYSTEM = (
    "Du beantwortest die Frage ausschließlich aus den gegebenen Auszügen. "
    "Jede Aussage trägt die Quelle in eckigen Klammern, zum Beispiel [a.pdf, S. 4]. "
    "Wenn die Auszüge die Frage nicht beantworten, antworte: "
    "Das steht nicht in den Auszügen. Erfinde keine Zahlen und keine Quellen."
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
