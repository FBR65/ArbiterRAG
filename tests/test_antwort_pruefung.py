"""Spec: no answer without a checked source — every sentence is verified."""

from __future__ import annotations

from pathlib import Path

from arbiter.check import build_claim_prompt, split_claims
from arbiter.config import Backends, Config, Rag
from arbiter.pipeline import Pipeline


class FlatEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def test_claims_keep_the_source_they_name() -> None:
    text = "Die Miete betraegt 940 Euro [a.pdf, S. 3]. Sie ist monatlich faellig [a.pdf, S. 3]."
    claims = split_claims(text)
    assert len(claims) == 2
    assert claims[0][1] == "a.pdf"
    assert claims[0][2] == 3


def test_a_sentence_without_citation_has_no_source() -> None:
    claims = split_claims("Die Miete betraegt 940 Euro. Das steht im Vertrag [a.pdf, S. 3].")
    assert claims[0][1] is None
    assert claims[1][1] == "a.pdf"


def test_claim_prompt_names_both_parts() -> None:
    system, user = build_claim_prompt("Die Miete betraegt 940 Euro.", "Miete: 940 EUR monatlich")
    assert "ja oder nein" in system
    assert "940" in user


def _pipeline(
    tmp_path: Path, antwort: str, p_ja: float, p_pruefung: float | None = None
) -> Pipeline:
    corpus = tmp_path / "korpus"
    corpus.mkdir(exist_ok=True)
    (corpus / "a.txt").write_text("Die Miete betraegt 940 Euro.", encoding="utf-8")
    cfg = Config(
        db_path=tmp_path / "db",
        backends=Backends(
            base_url="http://test", embed_model="m", rerank_model="r", chat_model="c"
        ),
        rags={
            "k": Rag(
                name="k",
                description="Test",
                corpus_path=corpus,
                extensions=(".txt",),
                hyde=False,
                rerank=False,
                gate_kandidaten=1,
                antwort_pruefung=True,
                antwort_schwelle=0.5,
            )
        },
    )
    pipeline = Pipeline(cfg, cfg.rags["k"], embedder=FlatEmbedder())
    pipeline.index()

    def logprobs(system: str, user: str) -> dict[str, float]:
        # The same call serves the gate and the sentence check; the claim prompt names
        # "Aussage", so the tests can give both steps their own probability.
        treffer = p_ja if p_pruefung is None or "Aussage:" not in user else p_pruefung
        return {"Ja": treffer, "Nein": 1.0 - treffer}

    pipeline.gate_logprobs = logprobs
    pipeline.answerer = lambda system, user: antwort
    return pipeline


def test_answer_with_unsupported_sentence_is_refused(tmp_path: Path) -> None:
    # The gate accepts (p=0.9 everywhere); the claim check is the same call, so a
    # low probability rejects both — therefore the answer must be refused.
    pipeline = _pipeline(tmp_path, "Die Miete betraegt 940 Euro [a.txt, S. 1].", p_ja=0.2)
    antwort = pipeline.ask("Wie hoch ist die Miete?")
    assert antwort.text is None
    assert antwort.grund in {"keine belegte Passage", "Antwort nicht durch die Auszüge gedeckt"}


def test_answer_without_citation_is_allowed_when_a_passage_carries_it(tmp_path: Path) -> None:
    # A summary sentence without a citation is fine as long as an admitted passage
    # supports it; only unsupported sentences make the answer unusable.
    pipeline = _pipeline(tmp_path, "Die Miete betraegt 940 Euro.", p_ja=0.9)
    antwort = pipeline.ask("Wie hoch ist die Miete?")
    assert antwort.text is not None


def test_unsupported_sentence_is_dropped_not_the_whole_answer(tmp_path: Path) -> None:
    # A wrong sentence must not silence the right one, and the answer must never be
    # empty while a passage is admitted: the source is quoted instead.
    pipeline = _pipeline(
        tmp_path, "Die Miete betraegt 1.400 Euro [a.txt, S. 1].", p_ja=0.9, p_pruefung=0.1
    )
    antwort = pipeline.ask("Wie hoch ist die Miete?")
    assert antwort.text is not None
    assert antwort.text.startswith("Belegstelle:")
    assert any("Gestrichen" in warnung for warnung in antwort.warnungen)
