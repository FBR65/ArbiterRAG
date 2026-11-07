"""Spec D2/D3: HyDE inside the search pipeline."""

from __future__ import annotations

from pathlib import Path

from arbiter.config import Backends, Config, Rag
from arbiter.pipeline import Pipeline


class KeywordEmbedder:
    """Vector per keyword, so a HyDE document can reach a chunk the question misses."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            lowered = text.lower()
            if "alpha" in lowered:
                out.append([1.0, 0.0, 0.0, 0.0])
            elif "beta" in lowered:
                out.append([0.0, 1.0, 0.0, 0.0])
            else:
                out.append([0.0, 0.0, 0.0, 1.0])
        return out


class StubHyde:
    def __init__(self, text: str = "alpha antwort", fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.calls = 0

    def __call__(self, question: str, description: str) -> str:
        self.calls += 1
        if self.fail:
            raise RuntimeError("HyDE-Modell antwortet 503")
        return self.text


def _config(tmp_path: Path) -> Config:
    corpus = tmp_path / "korpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Das Dokument alpha nennt eine Zahl.", encoding="utf-8")
    (corpus / "b.txt").write_text("Das Dokument beta nennt etwas anderes.", encoding="utf-8")
    return Config(
        db_path=tmp_path / "db",
        backends=Backends(
            base_url="http://test",
            embed_model="m",
            rerank_model="r",
            chat_model="c",
        ),
        rags={
            "k": Rag(
                name="k",
                description="Testkorpus",
                corpus_path=corpus,
                extensions=(".txt",),
                fuse_top_k=5,
                hyde=True,
                hyde_max_chars=400,
            )
        },
    )


def _pipeline(tmp_path: Path, hyde) -> Pipeline:
    cfg = _config(tmp_path)
    pipeline = Pipeline(cfg, cfg.rags["k"], embedder=KeywordEmbedder())
    pipeline.index()
    pipeline.hyde = hyde
    return pipeline


def test_hyde_document_moves_the_right_chunk_to_the_top(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, StubHyde("alpha antwort"))
    result = pipeline.search("gamma frage")
    assert result.hyde_text == "alpha antwort"
    assert result.passages[0].quelle == "a.txt"


def test_without_hyde_the_alpha_chunk_falls_behind(tmp_path: Path) -> None:
    """The dense half always returns every chunk in a tiny corpus; HyDE decides the order."""
    pipeline = _pipeline(tmp_path, StubHyde("alpha antwort"))
    result = pipeline.search("beta frage", hyde=False)
    assert result.hyde_text is None
    assert result.passages[0].quelle == "b.txt"


def test_failing_hyde_falls_back_to_the_question_and_warns(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, StubHyde(fail=True))
    result = pipeline.search("beta frage")
    assert result.hyde_text is None
    assert any("503" in warning for warning in result.warnungen)
    assert any(passage.quelle == "b.txt" for passage in result.passages)


def test_hyde_is_not_called_when_switched_off(tmp_path: Path) -> None:
    stub = StubHyde()
    pipeline = _pipeline(tmp_path, stub)
    pipeline.search("beta frage", hyde=False)
    assert stub.calls == 0
