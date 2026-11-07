"""Spec E2: the gate as one `choice` over all candidates plus "keine"."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from arbiter.config import Backends, Config, Rag
from arbiter.decision import build_choice_questions, choice_probabilities
from arbiter.pipeline import Pipeline


class FlatEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def test_choice_question_has_one_option_per_passage_plus_keine() -> None:
    questions = build_choice_questions(3)
    criteria = questions["wahl"]["criteria"]
    assert sorted(criteria) == ["keine", "p1", "p2", "p3"]
    assert questions["wahl"]["type"] == "choice"


def test_choice_probabilities_are_read_from_the_payload() -> None:
    answers = {"wahl": {"type": "choice", "probabilities": {"p1": 0.7, "keine": 0.3}}}
    assert choice_probabilities(answers) == {"p1": 0.7, "keine": 0.3}
    assert choice_probabilities({}) == {}


def _pipeline(tmp_path: Path, antworten: dict[str, Any], schwelle: float = 0.5) -> Pipeline:
    corpus = tmp_path / "korpus"
    corpus.mkdir(exist_ok=True)
    (corpus / "a.txt").write_text("Die Miete betraegt 940 Euro.", encoding="utf-8")
    (corpus / "b.txt").write_text("Das Wetter ist sonnig.", encoding="utf-8")
    cfg = Config(
        db_path=tmp_path / "db",
        backends=Backends(
            base_url="http://test",
            embed_model="m",
            rerank_model="r",
            chat_model="c",
            laya_model="l",
        ),
        rags={
            "k": Rag(
                name="k",
                description="Test",
                corpus_path=corpus,
                extensions=(".txt",),
                hyde=False,
                rerank=False,
                fuse_top_k=5,
                laya_schwelle=schwelle,
                gate_modus="choice",
            )
        },
    )
    pipeline = Pipeline(cfg, cfg.rags["k"], embedder=FlatEmbedder())
    pipeline.index()
    pipeline.decision = lambda state, questions: antworten
    pipeline.answerer = lambda system, user: "Antwort."
    return pipeline


def test_choice_gate_picks_the_named_passage(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        {
            "wahl": {"type": "choice", "probabilities": {"p1": 0.1, "p2": 0.8, "keine": 0.1}},
            "d1": {"type": "score", "score": 2.0},
        },
    )
    entscheidung = pipeline.bewerte(pipeline.search("Wie hoch ist die Miete?"))
    assert entscheidung.beantwortbar is True
    assert len(entscheidung.auswahl) == 1


def test_choice_gate_answers_none_when_keine_wins(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        {"wahl": {"type": "choice", "probabilities": {"p1": 0.1, "p2": 0.1, "keine": 0.8}}},
    )
    antwort = pipeline.ask("Wie hoch ist die Miete?")
    assert antwort.text is None
    assert antwort.grund == "keine belegte Passage"


def test_choice_gate_needs_a_lead_over_keine(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        {"wahl": {"type": "choice", "probabilities": {"p1": 0.45, "keine": 0.55}}},
        schwelle=0.6,
    )
    antwort = pipeline.ask("Wie hoch ist die Miete?")
    assert antwort.text is None
