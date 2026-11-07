"""Gate over Qwen logprobs (role: typed decision via a single-token answer)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from arbiter.config import Backends, Config, Rag
from arbiter.decision import ja_wahrscheinlichkeit
from arbiter.hyde import ChatClient
from arbiter.pipeline import Pipeline


def _logprobs_payload(tokens: list[tuple[str, float]]) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {"content": tokens[0][0]},
                "logprobs": {
                    "content": [
                        {
                            "token": tokens[0][0],
                            "logprob": tokens[0][1],
                            "top_logprobs": [
                                {"token": token, "logprob": logprob} for token, logprob in tokens
                            ],
                        }
                    ]
                },
            }
        ]
    }


def test_client_returns_normalised_probabilities() -> None:
    payload = _logprobs_payload([("Ja", -0.1), ("ja", -2.4), ("Nein", -4.0)])

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert body["max_tokens"] == 1
        assert body["logprobs"] is True
        assert body["top_logprobs"] == 20
        return httpx.Response(200, json=payload)

    client = ChatClient(
        base_url="http://test", model="qwen", transport=httpx.MockTransport(handler)
    )
    verteilung = client.probabilities("system", "user")
    assert verteilung["Ja"] > verteilung["ja"] > verteilung["Nein"]
    assert sum(verteilung.values()) == pytest.approx(1.0, abs=1e-6)


def test_ja_probability_adds_up_the_yes_tokens() -> None:
    verteilung = {"Ja": 0.6, "ja": 0.2, "Nein": 0.1, "nein": 0.1}
    assert ja_wahrscheinlichkeit(verteilung) == pytest.approx(0.8)


def test_ja_probability_is_none_without_yes_or_no() -> None:
    assert ja_wahrscheinlichkeit({"vielleicht": 1.0}) is None


def test_ja_probability_is_zero_when_the_answer_is_no() -> None:
    # 0.0 and None mean different things: "no" is a decision, "unknown" is not.
    assert ja_wahrscheinlichkeit({"Nein": 0.95, "nein": 0.05}) == 0.0


def test_client_reports_missing_logprobs() -> None:
    client = ChatClient(
        base_url="http://test",
        model="m",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "Ja"}}]})
        ),
    )
    assert client.probabilities("s", "u") == {}


class FlatEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _pipeline(tmp_path: Path, wahrscheinlichkeiten: list[float]) -> Pipeline:
    corpus = tmp_path / "korpus"
    corpus.mkdir(exist_ok=True)
    (corpus / "a.txt").write_text("Die Miete betraegt 940 Euro.", encoding="utf-8")
    (corpus / "b.txt").write_text("Das Wetter ist sonnig.", encoding="utf-8")
    cfg = Config(
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
                description="Test",
                corpus_path=corpus,
                extensions=(".txt",),
                hyde=False,
                rerank=False,
                fuse_top_k=5,
                gate_schwelle=0.5,
            )
        },
    )
    pipeline = Pipeline(cfg, cfg.rags["k"], embedder=FlatEmbedder())
    pipeline.index()
    reihe = iter(wahrscheinlichkeiten)

    def gate(system: str, user: str) -> dict[str, float]:
        p = next(reihe)
        return {"Ja": p, "Nein": 1.0 - p}

    pipeline.gate_logprobs = gate
    pipeline.answerer = lambda system, user: "Antwort."
    return pipeline


def test_logprob_gate_admits_only_passages_above_the_threshold(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, [0.91, 0.12])
    entscheidung = pipeline.bewerte(pipeline.search("Wie hoch ist die Miete?"))
    assert entscheidung.beantwortbar is True
    assert len(entscheidung.auswahl) == 1
    assert entscheidung.auswahl[0].p == 0.91


def test_logprob_gate_refuses_when_all_stay_below(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, [0.2, 0.3])
    antwort = pipeline.ask("Wie hoch ist die Miete?")
    assert antwort.text is None
    assert antwort.grund == "keine belegte Passage"
