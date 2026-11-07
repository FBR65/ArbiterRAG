"""Cross-encoder reranking over ``/v1/rerank`` (spec E).

The scores this client returns are logits, not probabilities. They are only
ever used for ordering — never as a threshold (spec E4).
"""

from __future__ import annotations

import httpx


class RerankError(RuntimeError):
    """Raised when the reranker cannot answer."""


class RerankClient:
    """Scores ``(query, document)`` pairs through llama-server's rerank route."""

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

    def scores(self, query: str, documents: list[str]) -> list[float]:
        """One score per document, in the order the documents were given."""
        if not documents:
            return []
        payload = {"model": self._model, "query": query, "documents": documents}
        try:
            response = self._client.post("/v1/rerank", json=payload)
        except httpx.HTTPError as exc:
            raise RerankError(f"{self._model}: nicht erreichbar ({exc})") from exc
        if response.status_code != 200:
            raise RerankError(f"{self._model}: HTTP {response.status_code} — {response.text[:200]}")

        results = response.json().get("results")
        if not isinstance(results, list) or len(results) != len(documents):
            raise RerankError(
                f"{self._model}: {len(documents)} Dokumente, aber "
                f"{len(results) if isinstance(results, list) else 'keine'} Bewertungen"
            )

        scores = [0.0] * len(documents)
        for entry in results:
            index = entry.get("index")
            if not isinstance(index, int) or not 0 <= index < len(documents):
                raise RerankError(f"{self._model}: Bewertung ohne brauchbaren Index")
            scores[index] = float(entry["relevance_score"])
        return scores

    def close(self) -> None:
        self._client.close()
