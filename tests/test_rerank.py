"""Spec E: cross-encoder reranking over /v1/rerank."""

from __future__ import annotations

import json

import httpx
import pytest

from arbiter.rerank import RerankClient, RerankError


def _client(handler) -> RerankClient:
    return RerankClient(
        base_url="http://test", model="bge-reranker-v2-m3", transport=httpx.MockTransport(handler)
    )


def test_scores_are_returned_in_input_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert request.url.path == "/v1/rerank"
        assert body["query"] == "Frage"
        assert body["documents"] == ["erst", "zweit", "dritt"]
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": 9.0},
                    {"index": 0, "relevance_score": 3.0},
                    {"index": 1, "relevance_score": 1.0},
                ]
            },
        )

    assert _client(handler).scores("Frage", ["erst", "zweit", "dritt"]) == [3.0, 1.0, 9.0]


def test_http_error_is_reported() -> None:
    client = _client(lambda request: httpx.Response(500, text="kaputt"))
    with pytest.raises(RerankError) as err:
        client.scores("f", ["a"])
    assert "500" in str(err.value)


def test_empty_documents_make_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("kein Aufruf erwartet")

    assert _client(handler).scores("f", []) == []


def test_missing_index_is_an_error_not_a_zero() -> None:
    client = _client(
        lambda request: httpx.Response(200, json={"results": [{"relevance_score": 1.0}]})
    )
    with pytest.raises(RerankError):
        client.scores("f", ["a"])
