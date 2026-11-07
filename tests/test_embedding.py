"""Spec C1: embedding client (HTTP, OpenAI-compatible)."""

from __future__ import annotations

import json

import httpx
import pytest

from arbiter.embedding import EmbeddingError, HttpEmbedder


def _client(payload: dict, status: int = 200) -> HttpEmbedder:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert request.url.path == "/v1/embeddings"
        assert body["model"] == "bge-m3-gguf"
        return httpx.Response(status, json=payload)

    transport = httpx.MockTransport(handler)
    return HttpEmbedder(base_url="http://test", model="bge-m3-gguf", transport=transport)


def test_returns_vectors_in_input_order() -> None:
    payload = {
        "data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]
    }
    embedder = _client(payload)
    vectors = embedder.embed(["erster", "zweiter"])
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]


def test_http_error_carries_status_and_message() -> None:
    embedder = _client({"error": "kaputt"}, status=500)
    with pytest.raises(EmbeddingError) as err:
        embedder.embed(["x"])
    assert "500" in str(err.value)


def test_empty_input_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("kein Aufruf erwartet")

    embedder = HttpEmbedder(
        base_url="http://test", model="m", transport=httpx.MockTransport(handler)
    )
    assert embedder.embed([]) == []
