"""Spec P1-P9: the HTTP layer."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from arbiter.api import create_app
from arbiter.config import Backends, Config, Rag
from arbiter.embedding import EmbeddingError
from arbiter.pipeline import QuestionTooLong
from arbiter.store import StoreError


class FakePipeline:
    def __init__(self, rag: Rag, verhalten: str = "ok") -> None:
        self.rag = rag
        self.verhalten = verhalten
        self.gesucht: list[str] = []
        self.store = _Counter(7)

    def search(self, question: str):
        if self.verhalten == "zu_lang":
            raise QuestionTooLong("Frage zu lang")
        if self.verhalten == "kein_index":
            raise StoreError("Kein Index für 'k'")
        if self.verhalten == "backend_weg":
            raise EmbeddingError("Embedding-Endpunkt antwortet 503")
        from arbiter.pipeline import Passage, SearchResult

        self.gesucht.append(question)
        return SearchResult(
            question=question,
            passages=[
                Passage(
                    id="a.txt#0#1",
                    text="Die Miete betraegt 940 Euro.",
                    quelle="a.txt",
                    seite=None,
                    rrf_rank=1,
                    rrf_score=0.5,
                )
            ],
            rerank="aus",
        )

    def index(self):
        from arbiter.pipeline import IndexReport

        return IndexReport(files=2, chunks=3, changed=3)


class _Counter:
    def __init__(self, wert: int) -> None:
        self._wert = wert

    def count(self, _name: str) -> int:
        return self._wert


def _app(tmp_path: Path, verhalten: str = "ok", factory=None):
    corpus = tmp_path / "korpus"
    corpus.mkdir(exist_ok=True)
    (corpus / "a.txt").write_text("Inhalt", encoding="utf-8")
    config = Config(
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
                description="Testkorpus",
                corpus_path=corpus,
                extensions=(".txt",),
            )
        },
    )
    builder = factory or (lambda _cfg, rag: FakePipeline(rag, verhalten))
    return create_app(config, factory=builder), config


def test_health_answers_without_a_model_call(tmp_path: Path) -> None:
    app, _ = _app(tmp_path)
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "version" in response.json()


def test_rags_lists_name_and_count(tmp_path: Path) -> None:
    app, _ = _app(tmp_path)
    with TestClient(app) as client:
        payload = client.get("/rags").json()
    assert payload == [{"name": "k", "description": "Testkorpus", "abschnitte": 7}]


def test_rags_reports_zero_without_an_index(tmp_path: Path) -> None:
    class Kaputt:
        def count(self, _name: str) -> int:
            raise StoreError("kein Index")

    def factory(_cfg, rag):
        pipeline = FakePipeline(rag)
        pipeline.store = Kaputt()  # type: ignore[assignment]
        return pipeline

    app, _ = _app(tmp_path, factory=factory)
    with TestClient(app) as client:
        assert client.get("/rags").json()[0]["abschnitte"] == 0


def test_search_returns_the_passage_list(tmp_path: Path) -> None:
    app, _ = _app(tmp_path)
    with TestClient(app) as client:
        response = client.post("/search", json={"rag": "k", "frage": "Miete?"})
    assert response.status_code == 200
    body = response.json()
    assert body["frage"] == "Miete?"
    assert body["passagen"][0]["quelle"] == "a.txt"
    assert body["passagen"][0]["id"] == "a.txt#0#1"


def test_unknown_rag_is_404_with_the_known_names(tmp_path: Path) -> None:
    app, _ = _app(tmp_path)
    with TestClient(app) as client:
        response = client.post("/search", json={"rag": "gibtsnicht", "frage": "x"})
    assert response.status_code == 404
    assert "k" in response.json()["detail"]


def test_missing_field_is_422(tmp_path: Path) -> None:
    app, _ = _app(tmp_path)
    with TestClient(app) as client:
        assert client.post("/search", json={"rag": "k"}).status_code == 422


@pytest.mark.parametrize(
    ("verhalten", "status"),
    [("zu_lang", 400), ("kein_index", 409), ("backend_weg", 503)],
)
def test_errors_map_to_status_codes(tmp_path: Path, verhalten: str, status: int) -> None:
    app, _ = _app(tmp_path, verhalten=verhalten)
    with TestClient(app) as client:
        response = client.post("/search", json={"rag": "k", "frage": "x"})
    assert response.status_code == status
    assert "detail" in response.json()


def test_index_starts_a_job_and_reports_the_numbers(tmp_path: Path) -> None:
    app, _ = _app(tmp_path)
    with TestClient(app) as client:
        accepted = client.post("/index", json={"rag": "k"})
        assert accepted.status_code == 202
        job_id = accepted.json()["job_id"]

        for _ in range(50):
            status = client.get(f"/index/{job_id}").json()
            if status["status"] != "laeuft":
                break
            import time

            time.sleep(0.05)

    assert status["status"] == "fertig"
    assert status["bericht"]["dateien"] == 2
    assert status["bericht"]["abschnitte"] == 3


def test_unknown_job_is_404(tmp_path: Path) -> None:
    app, _ = _app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/index/ffffffff").status_code == 404


def test_second_index_run_for_the_same_rag_is_409(tmp_path: Path) -> None:
    import threading

    gate = threading.Event()

    def factory(_cfg, rag):
        pipeline = FakePipeline(rag)

        def langsam():
            gate.wait(5)
            return FakePipeline(rag).index()

        pipeline.index = langsam  # type: ignore[method-assign]
        return pipeline

    app, _ = _app(tmp_path, factory=factory)
    with TestClient(app) as client:
        assert client.post("/index", json={"rag": "k"}).status_code == 202
        zweiter = client.post("/index", json={"rag": "k"})
        assert zweiter.status_code == 409
        assert "läuft bereits" in zweiter.json()["detail"]
        gate.set()
