"""Spec H1/H2/I1/I2/I3: pipeline and CLI."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from arbiter.cli import main
from arbiter.config import Backends, Config, Rag
from arbiter.pipeline import Pipeline, QuestionTooLong


class FakeEmbedder:
    """Deterministic vectors, no network. Same text -> same vector."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            out.append([digest[i] / 255.0 for i in range(4)])
        return out


def _config(tmp_path: Path) -> Config:
    corpus = tmp_path / "korpus"
    corpus.mkdir()
    return Config(
        db_path=tmp_path / "db",
        backends=Backends(
            base_url="http://test",
            embed_model="m",
            rerank_model="r",
            chat_model="c",
            laya_model="l",
        ),
        rags={
            "miethaus": Rag(
                name="miethaus",
                description="Mietvertraege",
                corpus_path=corpus,
                extensions=(".txt",),
            )
        },
    )


def test_index_reports_files_and_chunks(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _config(tmp_path)
    (cfg.rags["miethaus"].corpus_path / "a.txt").write_text(
        "Die Miete betraegt 900 Euro monatlich.", encoding="utf-8"
    )
    (cfg.rags["miethaus"].corpus_path / "b.txt").write_text(
        "Die Nebenkosten werden jaehrlich abgerechnet.", encoding="utf-8"
    )
    exit_code = main(["index", "miethaus"], config=cfg, embedder=FakeEmbedder())
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "2 Dateien" in out
    assert "2 Abschnitte" in out


def test_search_prints_source_and_rank(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _config(tmp_path)
    (cfg.rags["miethaus"].corpus_path / "a.txt").write_text(
        "Die Miete betraegt 900 Euro monatlich.", encoding="utf-8"
    )
    embedder = FakeEmbedder()
    assert main(["index", "miethaus"], config=cfg, embedder=embedder) == 0
    capsys.readouterr()
    exit_code = main(["search", "miethaus", "Miete"], config=cfg, embedder=embedder)
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "a.txt" in out
    assert "RRF" in out


def test_unknown_rag_is_exit_2_and_lists_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _config(tmp_path)
    exit_code = main(["search", "gibtsnicht", "Frage"], config=cfg, embedder=FakeEmbedder())
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "miethaus" in captured.err


def test_empty_corpus_is_exit_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _config(tmp_path)
    exit_code = main(["index", "miethaus"], config=cfg, embedder=FakeEmbedder())
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Korpus leer" in captured.err


def test_overlong_question_is_refused_without_calling_the_embedder(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (cfg.rags["miethaus"].corpus_path / "a.txt").write_text("Inhalt", encoding="utf-8")
    embedder = FakeEmbedder()
    main(["index", "miethaus"], config=cfg, embedder=embedder)
    with pytest.raises(QuestionTooLong):
        Pipeline(cfg, cfg.rags["miethaus"], embedder=embedder).search("x" * 2001)


def test_reindex_does_not_duplicate(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (cfg.rags["miethaus"].corpus_path / "a.txt").write_text("Inhalt A", encoding="utf-8")
    embedder = FakeEmbedder()
    pipeline = Pipeline(cfg, cfg.rags["miethaus"], embedder=embedder)
    pipeline.index()
    report = pipeline.index()
    assert report.chunks == 1
    assert pipeline.store.count("miethaus") == 1
