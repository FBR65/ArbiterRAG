"""HyDE: a hypothetical answer document as a second query (spec D).

The chat backend is reached through the OpenAI-compatible
``/v1/chat/completions`` route, so the same client works against llama-swap.
The document is deliberately kept short: it is a query, not an answer.
"""

from __future__ import annotations

import math

import httpx

SYSTEM_TEMPLATE = (
    "Du hilfst beim Suchen in einem Dokumentenarchiv. "
    "Der Bestand enthält: {description}. "
    "Antworte auf Deutsch. Nenne keine ganzen Sätze, sondern nur eine Liste von "
    "Fachbegriffen, Eigennamen, Zahlen und Wortverbindungen, die in dem Abschnitt "
    "stehen müssten, der diese Frage beantwortet. Höchstens zwölf Einträge, "
    "durch Komma getrennt. Erfinde keine Zahlen."
)

USER_TEMPLATE = (
    "Frage: {question}\n\nWelche Begriffe und Zahlen müsste der antwortende Abschnitt enthalten?"
)


class ChatError(RuntimeError):
    """Raised when the chat backend cannot answer."""


class ChatClient:
    """Minimal client for ``/v1/chat/completions``."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 180.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._model = model
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, transport=transport
        )

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        payload = {
            "model": self._model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            response = self._client.post("/v1/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise ChatError(f"{self._model}: nicht erreichbar ({exc})") from exc
        if response.status_code != 200:
            raise ChatError(f"{self._model}: HTTP {response.status_code} — {response.text[:200]}")
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise ChatError(f"{self._model}: Antwort ohne Text") from exc
        return str(content).strip()

    def probabilities(
        self, system: str, user: str, top_n: int = 20, grammar: str | None = None
    ) -> dict[str, float]:
        """Probability per candidate first token (spec: gate over logprobs).

        The model answers with one token. ``top_logprobs`` gives the alternatives;
        a softmax over those alternatives is the probability of the answer. An
        answer without logprobs yields an empty map — the caller must not invent
        a probability in that case.
        """
        payload = {
            "model": self._model,
            "temperature": 0.0,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": top_n,
            "grammar": grammar,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            response = self._client.post("/v1/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise ChatError(f"{self._model}: nicht erreichbar ({exc})") from exc
        if response.status_code != 200:
            raise ChatError(f"{self._model}: HTTP {response.status_code} — {response.text[:200]}")
        try:
            content = response.json()["choices"][0]["logprobs"]["content"][0]
            alternatives = content["top_logprobs"]
        except (KeyError, IndexError, TypeError):
            return {}

        logprobs = {str(entry["token"]): float(entry["logprob"]) for entry in alternatives}
        if not logprobs:
            return {}
        hoechster = max(logprobs.values())
        gewichte = {token: math.exp(wert - hoechster) for token, wert in logprobs.items()}
        summe = sum(gewichte.values())
        return {token: gewicht / summe for token, gewicht in gewichte.items()}

    def close(self) -> None:
        self._client.close()


def build_prompt(question: str, description: str) -> tuple[str, str]:
    """System and user message for one HyDE document."""
    return (
        SYSTEM_TEMPLATE.format(description=description),
        USER_TEMPLATE.format(question=question),
    )


def truncate(text: str, max_chars: int) -> str:
    """Cut to ``max_chars`` without ending inside a word."""
    if len(text) <= max_chars:
        return text.strip()
    cut = text[:max_chars]
    space = cut.rfind(" ")
    if space > max_chars // 2:
        cut = cut[:space]
    return cut.strip()
