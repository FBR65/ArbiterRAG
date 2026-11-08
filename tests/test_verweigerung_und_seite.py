"""Eine Verweigerung nennt keine Quellen (G6) und prüft gegen die ganze Seite."""

from __future__ import annotations

from pathlib import Path

from arbiter.config import Backends, Config, Rag
from arbiter.pipeline import Pipeline


class FlatEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _pipeline(tmp_path: Path, satz: str = "Die Miete betraegt 940 Euro.") -> Pipeline:
    corpus = tmp_path / "korpus"
    corpus.mkdir(exist_ok=True)
    (corpus / "a.txt").write_text(satz, encoding="utf-8")
    cfg = Config(
        db_path=tmp_path / "db",
        backends=Backends(base_url="http://t", embed_model="m", rerank_model="r", chat_model="c"),
        rags={
            "k": Rag(
                name="k",
                description="T",
                corpus_path=corpus,
                extensions=(".txt",),
                rerank=False,
                gate_kandidaten=1,
                antwort_pruefung=True,
                chunk_chars=20,
                chunk_overlap=5,
            )
        },
    )
    pipeline = Pipeline(cfg, cfg.rags["k"], embedder=FlatEmbedder())
    pipeline.index()
    return pipeline


def test_a_refusal_lists_no_sources(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline.gate_logprobs = lambda system, user: {"Ja": 0.95, "Nein": 0.05}
    pipeline.answerer = lambda system, user: "KEINE_ANTWORT"
    antwort = pipeline.ask("Wie hoch ist die Miete?")
    assert antwort.text is None
    assert antwort.quellen == [], "eine Verweigerung darf keine Quellen anzeigen"
    assert any("geprüft, nicht ausreichend" in warnung for warnung in antwort.warnungen)


def test_the_check_reads_the_whole_page_not_only_the_chunk(tmp_path: Path) -> None:
    # Der Satz steht über die Seite verteilt; der abgerufene Abschnitt enthält nur den
    # Anfang. Geprüft wird gegen die ganze Seite, sonst gilt er als unbelegt.
    pipeline = _pipeline(
        tmp_path,
        "Die Miete betraegt 940 Euro monatlich und ist am dritten Werktag faellig.",
    )
    pipeline.gate_logprobs = lambda system, user: {"Ja": 0.95, "Nein": 0.05}
    pipeline.answerer = lambda system, user: "Die Miete betraegt 940 Euro monatlich [a.txt, S. 1]."
    seite = pipeline.store.page_text("k", "a.txt", 1)
    assert "940 Euro" in seite and "Werktag" in seite
    antwort = pipeline.ask("Wie hoch ist die Miete?")
    assert antwort.text is not None
