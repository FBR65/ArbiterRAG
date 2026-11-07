"""Embedding client (spec C1, F9).

Talks the OpenAI-compatible ``/v1/embeddings`` route, so the same client works
against llama-swap and any other compatible server.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx


class EmbeddingError(RuntimeError):
    """Raised when the embedding backend cannot answer."""


class HttpEmbedder:
    """Dense embeddings over HTTP. One call per batch, order preserved."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._model = model
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, transport=transport
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self._client.post(
                "/v1/embeddings", json={"model": self._model, "input": list(texts)}
            )
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Embedding-Endpunkt nicht erreichbar: {exc}") from exc

        if response.status_code != 200:
            raise EmbeddingError(
                f"Embedding-Endpunkt antwortet {response.status_code}: {response.text[:300]}"
            )

        payload = response.json()
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise EmbeddingError(f"Embedding-Antwort unbrauchbar: {str(payload)[:300]}")

        ordered = sorted(data, key=lambda item: item.get("index", 0))
        return [[float(value) for value in item["embedding"]] for item in ordered]

    def close(self) -> None:
        self._client.close()
