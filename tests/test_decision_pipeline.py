"""Spec F1-F7 and G: decision flow and answer generation.

The gate asks one passage at a time (measured 2026-11-07), so the fake decides
from the state text it receives rather than from a marker number.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from arbiter.config import Backends, Config, Rag
from arbiter.decision import DecisionError
from arbiter.pipeline import Pipeline


class FlatEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class LayaFake:
    """Answers from the state: P(true) and coverage per passage text."""

    def __init__(
        self,
        wahrscheinlichkeit: dict[str, float] | None = None,
        deckung: dict[str, float] | None = None,
        aehnlichkeit: float = 5.0,
        widerspruch: float = 0.05,
    ) -> None:
        self.wahrscheinlichkeit = wahrscheinlichkeit or {}
        self.deckung = deckung or {}
        self.aehnlichkeit = aehnlichkeit
        self.widerspruch = widerspruch
        self.aufrufe: list[dict[str, Any]] = []

    def _passage_text(self, state: str) -> str:
        """Only the passage part — the question itself must not match keywords."""
        _, _, rest = state.partition("Auszug [1]:")
        return rest.split("Auszug [2]:")[0]

    def _treffer(self, state: str, tabelle: dict[str, float], default: float) -> float:
        passage = self._passage_text(state)
        for marke, wert in tabelle.items():
            if marke in passage:
                return wert
        return default

    def __call__(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        self.aufrufe.append(questions)
        if "p1" in questions:
            return {
                "p1": {
                    "type": "noul",
                    "noul": self._treffer(state, self.wahrscheinlichkeit, 0.1),
                }
            }
        if "d1" in questions:
            stufe = self._treffer(state, self.deckung, 1.0)
            return {"d1": {"type": "score", "score": stufe - 1.0}}
        if "v1_2" in questions:
            return {
                "v1_2": {"type": "score", "score": self.aehnlichkeit - 1.0},
                "w1_2": {"type": "noul", "noul": self.widerspruch},
            }
        return {}


def _pipeline(
    tmp_path: Path,
    laya: Any,
    answer_text: str = "Die Miete betraegt 940 Euro.",
    schwelle: float = 0.5,
    redundanz: float | None = None,
) -> Pipeline:
    corpus = tmp_path / "korpus"
    corpus.mkdir(exist_ok=True)
    (corpus / "a.txt").write_text("Die Miete betraegt 940 Euro monatlich.", encoding="utf-8")
    (corpus / "b.txt").write_text("Das Wetter ist heute sonnig.", encoding="utf-8")
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
                redundanz_niveau=redundanz,
            )
        },
    )
    pipeline = Pipeline(cfg, cfg.rags["k"], embedder=FlatEmbedder())
    pipeline.index()
    pipeline.decision = laya
    pipeline.answerer = lambda system, user: answer_text
    return pipeline


def test_gate_accepts_only_passages_above_the_threshold(tmp_path: Path) -> None:
    laya = LayaFake(wahrscheinlichkeit={"Miete": 0.91}, deckung={"Miete": 3.0})
    pipeline = _pipeline(tmp_path, laya)
    entscheidung = pipeline.bewerte(pipeline.search("Wie hoch ist die Miete?"))
    assert entscheidung.beantwortbar is True
    assert [z.passage.quelle for z in entscheidung.auswahl] == ["a.txt"]
    assert entscheidung.auswahl[0].p == 0.91
    assert entscheidung.auswahl[0].deckung == 3.0


def test_every_passage_gets_its_own_gate_request(tmp_path: Path) -> None:
    laya = LayaFake(wahrscheinlichkeit={"Miete": 0.9}, deckung={"Miete": 3.0})
    pipeline = _pipeline(tmp_path, laya)
    pipeline.bewerte(pipeline.search("Wie hoch ist die Miete?"))
    gate_aufrufe = [aufruf for aufruf in laya.aufrufe if "p1" in aufruf]
    assert len(gate_aufrufe) == 2  # one per passage, not one for the whole set


def test_no_admissible_passage_gives_a_reason_not_an_answer(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, LayaFake())
    antwort = pipeline.ask("Wie hoch ist die Miete?")
    assert antwort.text is None
    assert antwort.grund == "keine belegte Passage"
    assert antwort.quellen == []


def test_missing_answer_key_counts_as_zero_with_a_note(tmp_path: Path) -> None:
    class Stumm(LayaFake):
        def __call__(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
            if "Miete" in self._passage_text(state):
                return {
                    "p1": {"type": "noul", "noul": 0.9},
                    "d1": {"type": "score", "score": 2.0},
                }
            return {}

    pipeline = _pipeline(tmp_path, Stumm())
    entscheidung = pipeline.bewerte(pipeline.search("Wie hoch ist die Miete?"))
    assert any("keine Antwort" in warning for warning in entscheidung.warnungen)
    assert [z.passage.quelle for z in entscheidung.auswahl] == ["a.txt"]


def test_decision_outage_blocks_the_answer(tmp_path: Path) -> None:
    def kaputt(state: str, questions: dict[str, Any]) -> dict[str, Any]:
        raise DecisionError("HTTP 503")

    pipeline = _pipeline(tmp_path, LayaFake())
    pipeline.decision = kaputt
    antwort = pipeline.ask("Wie hoch ist die Miete?")
    assert antwort.text is None
    assert antwort.grund == "Prüfschicht nicht erreichbar"
    assert antwort.fehler is True


def test_pairs_are_reported_but_not_filtered_without_thresholds(tmp_path: Path) -> None:
    laya = LayaFake(wahrscheinlichkeit={"Miete": 0.9, "Wetter": 0.8}, deckung={"Miete": 3.0})
    pipeline = _pipeline(tmp_path, laya)
    entscheidung = pipeline.bewerte(pipeline.search("Wie hoch ist die Miete?"))
    assert len(entscheidung.paare) == 1
    assert entscheidung.paare[0].aehnlichkeit == 5.0
    assert len(entscheidung.auswahl) == 2  # reported, nothing dropped


def test_redundancy_threshold_drops_the_weaker_passage(tmp_path: Path) -> None:
    laya = LayaFake(wahrscheinlichkeit={"Miete": 0.9, "Wetter": 0.8}, deckung={"Miete": 3.0})
    pipeline = _pipeline(tmp_path, laya, redundanz=4.0)
    entscheidung = pipeline.bewerte(pipeline.search("Wie hoch ist die Miete?"))
    assert [z.passage.quelle for z in entscheidung.auswahl] == ["a.txt"]


def test_coverage_decides_the_order_not_the_reranker(tmp_path: Path) -> None:
    laya = LayaFake(
        wahrscheinlichkeit={"Miete": 0.6, "Wetter": 0.7},
        deckung={"Miete": 1.0, "Wetter": 4.0},
    )
    pipeline = _pipeline(tmp_path, laya)
    entscheidung = pipeline.bewerte(pipeline.search("Wie hoch ist die Miete?"))
    assert entscheidung.auswahl[0].passage.quelle == "b.txt"


def test_answer_cites_its_sources(tmp_path: Path) -> None:
    laya = LayaFake(wahrscheinlichkeit={"Miete": 0.9}, deckung={"Miete": 3.0})
    pipeline = _pipeline(tmp_path, laya)
    antwort = pipeline.ask("Wie hoch ist die Miete?")
    assert antwort.text is not None
    assert "940" in antwort.text
    assert [quelle.quelle for quelle in antwort.quellen] == ["a.txt"]
    assert antwort.grund is None


def test_empty_answer_from_the_model_is_an_error(tmp_path: Path) -> None:
    laya = LayaFake(wahrscheinlichkeit={"Miete": 0.9}, deckung={"Miete": 3.0})
    pipeline = _pipeline(tmp_path, laya, answer_text="   ")
    antwort = pipeline.ask("Wie hoch ist die Miete?")
    assert antwort.text is None
    assert antwort.fehler is True


def test_dead_answer_model_is_reported_not_raised(tmp_path: Path) -> None:
    laya = LayaFake(wahrscheinlichkeit={"Miete": 0.9}, deckung={"Miete": 3.0})
    pipeline = _pipeline(tmp_path, laya)

    def kaputt(system: str, user: str) -> str:
        raise RuntimeError("Modell antwortet nicht")

    pipeline.answerer = kaputt
    antwort = pipeline.ask("Wie hoch ist die Miete?")
    assert antwort.text is None
    assert antwort.grund == "Antwortmodell nicht erreichbar"
    assert antwort.fehler is True
