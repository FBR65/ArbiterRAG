"""Keine Spuren des Prüf-Prompts in der Antwort (G5)."""

from __future__ import annotations

from pathlib import Path

from arbiter.answer import KEINE_ANTWORT, build_answer_prompt
from arbiter.check import ist_meta, split_claims
from arbiter.config import Backends, Config, Rag
from arbiter.pipeline import Pipeline


class FlatEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def test_the_prompt_no_longer_spells_out_a_refusal_sentence() -> None:
    system, _ = build_answer_prompt("Frage?", [])
    assert KEINE_ANTWORT in system
    assert "Das steht nicht in den Auszügen" not in system
    assert "keine Anmerkung" in system


def test_meta_sentences_are_recognised() -> None:
    assert ist_meta("Das steht nicht in den Auszügen, aber es wird erwähnt, dass es so ist.")
    assert ist_meta("Die Auszüge enthalten keine Angabe zur Miete.")
    assert ist_meta("Das steht im Auszug.")
    assert not ist_meta("Die Miete betraegt 940 Euro [a.pdf, S. 3].")


def test_meta_sentences_are_dropped_from_the_answer() -> None:
    text = (
        "Die Miete betraegt 940 Euro [a.pdf, S. 3]. "
        "Das steht nicht in den Auszügen, aber es wird erwähnt, dass die Nebenkosten steigen."
    )
    claims = [satz for satz, _, _ in split_claims(text) if not ist_meta(satz)]
    assert claims == ["Die Miete betraegt 940 Euro [a.pdf, S. 3]."]


def _pipeline(tmp_path: Path, antwort: str) -> Pipeline:
    corpus = tmp_path / "korpus"
    corpus.mkdir(exist_ok=True)
    (corpus / "a.txt").write_text("Die Miete betraegt 940 Euro.", encoding="utf-8")
    cfg = Config(
        db_path=tmp_path / "db",
        backends=Backends(base_url="http://t", embed_model="m", rerank_model="r", chat_model="c"),
        rags={
            "k": Rag(
                name="k",
                description="T",
                corpus_path=corpus,
                extensions=(".txt",),
                hyde=False,
                rerank=False,
                gate_kandidaten=1,
                antwort_pruefung=True,
            )
        },
    )
    pipeline = Pipeline(cfg, cfg.rags["k"], embedder=FlatEmbedder())
    pipeline.index()
    pipeline.gate_logprobs = lambda system, user: {"Ja": 0.95, "Nein": 0.05}
    pipeline.answerer = lambda system, user: antwort
    return pipeline


def test_no_answer_token_becomes_a_refusal(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, KEINE_ANTWORT)
    antwort = pipeline.ask("Wie hoch ist die Miete?")
    assert antwort.text is None
    assert antwort.grund == "Die Auszüge beantworten die Frage nicht"


def test_meta_sentence_never_reaches_the_user(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        "Das steht nicht in den Auszügen, aber es wird erwähnt, dass die Miete 940 Euro betraegt.",
    )
    antwort = pipeline.ask("Wie hoch ist die Miete?")
    # The only sentence is meta, so nothing is left: the answer is the source quote or a
    # refusal — in no case the meta sentence itself.
    assert antwort.text is None or "steht nicht in den Auszügen" not in antwort.text
