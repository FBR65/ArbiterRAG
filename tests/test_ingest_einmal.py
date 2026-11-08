"""Der Ingest eines Dokuments findet einmal statt (B12)."""

from __future__ import annotations

from pathlib import Path

from arbiter.config import Backends, Config, Rag
from arbiter.pipeline import Pipeline


class FlatEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _pipeline(tmp_path: Path, **rag_aenderungen: object) -> Pipeline:
    corpus = tmp_path / "korpus"
    corpus.mkdir(exist_ok=True)
    (corpus / "a.txt").write_text("Die Miete betraegt 940 Euro.", encoding="utf-8")
    (corpus / "b.txt").write_text("Das Wetter ist sonnig.", encoding="utf-8")
    werte = {
        "name": "k",
        "description": "Test",
        "corpus_path": corpus,
        "extensions": (".txt",),
        "rerank": False,
    }
    werte.update(rag_aenderungen)
    cfg = Config(
        db_path=tmp_path / "db",
        backends=Backends(
            base_url="http://test", embed_model="m", rerank_model="r", chat_model="c"
        ),
        rags={"k": Rag(**werte)},  # type: ignore[arg-type]
    )
    return Pipeline(cfg, cfg.rags["k"], embedder=FlatEmbedder())


def test_a_second_run_reads_nothing(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    erster = pipeline.index()
    assert erster.files == 2
    assert erster.unchanged == 0

    zweiter = pipeline.index()
    assert zweiter.unchanged == 2, "unveraenderte Dateien muessen uebersprungen werden"
    assert zweiter.changed == 0
    assert zweiter.files == 0, "keine Datei wird erneut gelesen"


def test_a_changed_file_is_read_again(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline.index()
    (pipeline.rag.corpus_path / "a.txt").write_text(
        "Die Miete betraegt 980 Euro.", encoding="utf-8"
    )
    zweiter = pipeline.index()
    assert zweiter.files == 1, "nur die geaenderte Datei wird gelesen"
    assert zweiter.unchanged == 1


def test_skipping_keeps_the_rows_of_the_skipped_file(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    vorher = pipeline.index().chunks
    pipeline.index()
    assert pipeline.store.count("k") == vorher, (
        "uebersprungene Dateien duerfen nicht geloescht werden"
    )


def test_changed_settings_read_the_file_again(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline.index()
    geaendert = _pipeline(tmp_path, chunk_chars=120)
    zweiter = geaendert.index()
    assert zweiter.unchanged == 0, "andere Abschnittsgroesse heisst: neu lesen"
    assert zweiter.files == 2
