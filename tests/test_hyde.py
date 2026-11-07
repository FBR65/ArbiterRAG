"""Spec D: HyDE (hypothetical document embeddings)."""

from __future__ import annotations

import json

import httpx
import pytest

from arbiter.hyde import ChatClient, ChatError, build_prompt, truncate


def test_chat_client_sends_messages_and_returns_text() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        seen["path"] = request.url.path
        seen["model"] = body["model"]
        seen["roles"] = [m["role"] for m in body["messages"]]
        seen["temperature"] = body["temperature"]
        return httpx.Response(200, json={"choices": [{"message": {"content": "  Antwort  "}}]})

    client = ChatClient(
        base_url="http://test", model="gemma-4-12B", transport=httpx.MockTransport(handler)
    )
    assert client.complete("system", "user") == "Antwort"
    assert seen["path"] == "/v1/chat/completions"
    assert seen["model"] == "gemma-4-12B"
    assert seen["roles"] == ["system", "user"]
    assert seen["temperature"] == 0.0


def test_chat_client_reports_http_error() -> None:
    client = ChatClient(
        base_url="http://test",
        model="m",
        transport=httpx.MockTransport(lambda request: httpx.Response(503, text="busy")),
    )
    with pytest.raises(ChatError) as err:
        client.complete("s", "u")
    assert "503" in str(err.value)


def test_prompt_contains_question_and_description() -> None:
    system, user = build_prompt("Wie hoch ist die Miete?", "Mietvertraege und Nebenkosten")
    assert "Mietvertraege und Nebenkosten" in system
    assert "Wie hoch ist die Miete?" in user


def test_truncate_cuts_at_a_word_boundary() -> None:
    text = "wort " * 400
    cut = truncate(text, 120)
    assert len(cut) <= 120
    assert not cut.endswith("wor")


def test_truncate_keeps_short_text_untouched() -> None:
    assert truncate("kurz", 120) == "kurz"
