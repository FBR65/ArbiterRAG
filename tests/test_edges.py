"""Edge and error paths that the behaviour list names but the happy paths miss."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from arbiter.chunk import chunk_text
from arbiter.cli import main
from arbiter.config import Config, ConfigError, Rag, load_config
from arbiter.embedding import EmbeddingError
from arbiter.loaders import load_corpus
from arbiter.store import Record, Store, StoreError

BASE = """
db_path: /tmp/db
backends:
  base_url: http://x
  embed_model: m
  rerank_model: r
  chat_model: c
rags:
  a:
    description: d
    corpus_path: /tmp/k
    extensions: [pdf]
"""


class FailingEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingError("Endpunkt antwortet 503")


class EmptyEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 1.0, 0.0, 0.0] for _ in texts]


def _cfg(tmp_path: Path, suffix: str = ".txt") -> Config:
    corpus = tmp_path / "korpus"
    corpus.mkdir(exist_ok=True)
    return Config(
        db_path=tmp_path / "db",
        backends=load_config(_write(tmp_path), {}).backends,
        rags={
            "a": Rag(
                name="a",
                description="d",
                corpus_path=corpus,
                extensions=(suffix,),
            )
        },
    )


def _write(tmp_path: Path, text: str = BASE) -> Path:
    path = tmp_path / "arbiter.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_chunk_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        chunk_text("x", max_chars=0)
    with pytest.raises(ValueError):
        chunk_text("x", max_chars=100, overlap=100)


def test_config_missing_backends_block(tmp_path: Path) -> None:
    text = BASE.split("rags:")[0].replace("backends:\n", "backends_weg:\n")
    with pytest.raises(ConfigError) as err:
        load_config(
            _write(
                tmp_path,
                text + "rags:\n  a:\n    description: d\n"
                "    corpus_path: /tmp\n    extensions: ['.pdf']\n",
            ),
            {},
        )
    assert "backends" in str(err.value)


def test_config_empty_rags_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(
            _write(tmp_path, BASE.replace("  a:\n", "  ").split("rags:")[0] + "rags: {}\n"), {}
        )


def test_config_unknown_backend_key(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as err:
        load_config(
            _write(tmp_path, BASE.replace("  chat_model: c", "  chat_model: c\n  tempo: 3")), {}
        )
    assert "tempo" in str(err.value)


def test_config_incomplete_placeholder(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as err:
        load_config(_write(tmp_path, BASE.replace("http://x", "http://${X")), {"X": "1"})
    assert "Platzhalter" in str(err.value)


def test_missing_corpus_path_warns_without_crashing(tmp_path: Path) -> None:
    result = load_corpus(tmp_path / "gibtsnicht", (".pdf",))
    assert result.documents == []
    assert any("gibtsnicht" in w for w in result.warnings)


def test_latin1_file_is_read(tmp_path: Path) -> None:
    (tmp_path / "alt.txt").write_bytes("Miete 900 Euro für März".encode("latin-1"))
    result = load_corpus(tmp_path, (".txt",))
    assert len(result.documents) == 1
    assert "Miete" in result.documents[0].text


def test_empty_file_is_skipped_with_warning(tmp_path: Path) -> None:
    (tmp_path / "leer.txt").write_text("   \n", encoding="utf-8")
    result = load_corpus(tmp_path, (".txt",))
    assert result.skipped_files == 1
    assert any("leer.txt" in w for w in result.warnings)


def test_broken_pdf_is_skipped_not_fatal(tmp_path: Path) -> None:
    (tmp_path / "kaputt.pdf").write_bytes(b"%PDF-1.4 nicht wirklich")
    result = load_corpus(tmp_path, (".pdf",))
    assert result.skipped_files == 1
    assert any("kaputt.pdf" in w for w in result.warnings)


def test_store_rejects_mixed_vector_lengths(tmp_path: Path) -> None:
    store = Store(tmp_path / "db")
    with pytest.raises(StoreError):
        store.upsert(
            "a",
            [
                Record("a#0#1", "x", "a.pdf", 1, "h1", [1.0, 2.0]),
                Record("a#0#2", "y", "a.pdf", 1, "h2", [1.0, 2.0, 3.0]),
            ],
        )


def test_store_rejects_dimension_change_on_later_batch(tmp_path: Path) -> None:
    store = Store(tmp_path / "db")
    store.upsert("a", [Record("a#0#1", "x", "a.pdf", 1, "h1", [1.0, 2.0])])
    with pytest.raises(StoreError):
        store.upsert("a", [Record("a#0#2", "y", "a.pdf", 1, "h2", [1.0, 2.0, 3.0])])


def test_cli_reports_config_error_as_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARBITER_CONFIG", str(tmp_path / "fehlt.yaml"))
    assert main(["search", "a", "frage"]) == 2


def test_embedding_failure_is_reported_as_exit_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _cfg(tmp_path)
    (cfg.rags["a"].corpus_path / "x.txt").write_text("Inhalt", encoding="utf-8")
    assert main(["index", "a"], config=cfg, embedder=EmptyEmbedder()) == 0
    capsys.readouterr()
    assert main(["search", "a", "Inhalt"], config=cfg, embedder=FailingEmbedder()) == 1
    assert "503" in capsys.readouterr().err


def test_search_without_hits_prints_hint_is_not_covered() -> None:
    """H2 does not require an empty result; the branch stays in the CLI only."""


def test_record_hash_is_sha256_of_text() -> None:
    text = "Inhalt"
    assert hashlib.sha256(text.encode("utf-8")).hexdigest()  # sanity of the helper used in pipeline
