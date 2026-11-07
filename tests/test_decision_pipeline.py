"""Spec F1-F7 and G: decision flow and answer generation."""

from __future__ import annotations

from pathlib import Path

from arbiter.config import Backends, Config, Rag
from arbiter.decision import DecisionError
from arbiter.pipeline import Pipeline


class FlatEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _pipeline(
    tmp_path: Path, antworten: dict, answer_text: str = "Die Miete betraegt 940 Euro."
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
                laya_schwelle=0.5,
            )
        },
    )
    pipeline = Pipeline(cfg, cfg.rags["k"], embedder=FlatEmbedder())
    pipeline.index()
    pipeline.decision = lambda state, questions: antworten
    pipeline.answerer = lambda system, user: answer_text
    return pipeline


def test_gate_accepts_only_passages_above_the_threshold(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        {
            "p1": {"type": "noul", "noul": 0.91},
            "p2": {"type": "noul", "noul": 0.12},
            "d1": {"type": "score", "score": 3.0},
            "d2": {"type": "score", "score": 0.0},
        },
    )
    result = pipeline.search("Wie hoch ist die Miete?")
    entscheidung = pipeline.bewerte(result)
    assert entscheidung.beantwortbar is True
    assert [z.passage.quelle for z in entscheidung.auswahl] == ["a.txt"]
    assert entscheidung.auswahl[0].p == 0.91


def test_no_admissible_passage_gives_a_reason_not_an_answer(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        {
            "p1": {"type": "noul", "noul": 0.2},
            "p2": {"type": "noul", "noul": 0.3},
        },
    )
    antwort = pipeline.ask("Wie hoch ist die Miete?")
    assert antwort.text is None
    assert antwort.grund == "keine belegte Passage"
    assert antwort.quellen == []


def test_missing_answer_key_counts_as_zero_with_a_note(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, {"p1": {"type": "noul", "noul": 0.9}})
    result = pipeline.search("Wie hoch ist die Miete?")
    entscheidung = pipeline.bewerte(result)
    assert any("p2" in warning for warning in entscheidung.warnungen)


def test_decision_outage_blocks_the_answer(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, {})

    def kaputt(state: str, questions: dict) -> dict:
        raise DecisionError("HTTP 503")

    pipeline.decision = kaputt
    antwort = pipeline.ask("Wie hoch ist die Miete?")
    assert antwort.text is None
    assert antwort.grund == "Prüfschicht nicht erreichbar"
    assert antwort.fehler is True


def test_pairs_are_reported_but_not_filtered_without_thresholds(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        {
            "p1": {"type": "noul", "noul": 0.9},
            "p2": {"type": "noul", "noul": 0.8},
            "d1": {"type": "score", "score": 2.0},
            "d2": {"type": "score", "score": 1.0},
            "v1_2": {"type": "score", "score": 4.0},
        },
    )
    result = pipeline.search("Wie hoch ist die Miete?")
    entscheidung = pipeline.bewerte(result)
    assert len(entscheidung.paare) == 1
    assert entscheidung.paare[0].aehnlichkeit == 5.0
    assert len(entscheidung.auswahl) == 2  # reported, nothing dropped


def test_coverage_decides_the_order_not_the_reranker(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        {
            "p1": {"type": "noul", "noul": 0.6},
            "p2": {"type": "noul", "noul": 0.7},
            "d1": {"type": "score", "score": 0.0},
            "d2": {"type": "score", "score": 3.0},
        },
    )
    result = pipeline.search("Wie hoch ist die Miete?")
    entscheidung = pipeline.bewerte(result)
    assert entscheidung.auswahl[0].passage.quelle == "b.txt"


def test_answer_cites_its_sources(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        {
            "p1": {"type": "noul", "noul": 0.9},
            "p2": {"type": "noul", "noul": 0.1},
            "d1": {"type": "score", "score": 3.0},
            "d2": {"type": "score", "score": 0.0},
        },
    )
    antwort = pipeline.ask("Wie hoch ist die Miete?")
    assert antwort.text is not None
    assert "940" in antwort.text
    assert [quelle.quelle for quelle in antwort.quellen] == ["a.txt"]
    assert antwort.grund is None


def test_empty_answer_from_the_model_is_an_error(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        {
            "p1": {"type": "noul", "noul": 0.9},
            "p2": {"type": "noul", "noul": 0.1},
            "d1": {"type": "score", "score": 3.0},
            "d2": {"type": "score", "score": 0.0},
        },
        answer_text="   ",
    )
    antwort = pipeline.ask("Wie hoch ist die Miete?")
    assert antwort.text is None
    assert antwort.fehler is True
