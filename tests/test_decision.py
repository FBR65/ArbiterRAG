"""Spec F: the Laya decision layer over /v1/systemone."""

from __future__ import annotations

import json

import httpx
import pytest

from arbiter.decision import DecisionError, SystemOne, build_noul_state, build_questions


def _client(handler, sleeps: list[float] | None = None) -> SystemOne:
    return SystemOne(
        base_url="http://test",
        model="laya-multilingual",
        transport=httpx.MockTransport(handler),
        sleep=(sleeps.append if sleeps is not None else lambda _s: None),
    )


def test_ask_sends_state_and_questions_and_returns_answers() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        seen["path"] = request.url.path
        seen["model"] = body["model"]
        seen["state"] = body["state"]
        seen["questions"] = body["questions"]
        return httpx.Response(200, json={"answers": {"p1": {"type": "noul", "noul": 0.81}}})

    client = _client(handler)
    answers = client.ask(
        "Frage? Auszug [1]: Text",
        {"p1": {"type": "noul", "instructions": "Steht die Antwort in Auszug [1]?"}},
    )
    assert seen["path"] == "/v1/systemone"
    assert seen["model"] == "laya-multilingual"
    assert "Auszug [1]" in str(seen["state"])
    assert answers["p1"]["noul"] == pytest.approx(0.81)


def test_retries_on_503_and_then_succeeds() -> None:
    calls: list[int] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"answers": {}})

    assert _client(handler, sleeps).ask("s", {}) == {}
    assert len(calls) == 3
    assert sleeps == [1, 2]


def test_gives_up_after_three_attempts() -> None:
    sleeps: list[float] = []
    handler = lambda request: httpx.Response(503, text="busy")  # noqa: E731
    with pytest.raises(DecisionError) as err:
        _client(handler, sleeps).ask("s", {})
    assert "503" in str(err.value)
    assert sleeps == [1, 2, 4]


def test_non_retryable_error_fails_at_once() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, text="schlechter Aufbau")

    with pytest.raises(DecisionError):
        _client(handler).ask("s", {})
    assert len(calls) == 1


def test_noul_state_marks_every_passage() -> None:
    state = build_noul_state("Wie hoch ist die Miete?", ["Miete 940 Euro", "Wetter"])
    assert "Auszug [1]" in state
    assert "Auszug [2]" in state
    assert "Miete 940 Euro" in state


def test_questions_ask_about_each_passage_marker() -> None:
    questions = build_questions(3)
    assert sorted(questions) == ["p1", "p2", "p3"]
    assert questions["p2"]["type"] == "noul"
    assert "[2]" in questions["p2"]["instructions"]
