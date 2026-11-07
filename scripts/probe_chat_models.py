"""Measure one chat call per model: prompt tokens, wall time, output.

Run: uv run python scripts/probe_chat_models.py gemma-4-12B LFM3-1.7B deepseek-v4-pro-qwen3.5-9b
"""

from __future__ import annotations

import sys
import time

import httpx

PROMPT = (
    "Du bist ein Suchassistent für ein Dokumentenarchiv. Der Bestand enthält: "
    "Heruntergeladene Fachartikel zu KI, RAG und Inferenz.\n\n"
    "Schreibe den Textabschnitt, der diese Frage beantworten würde.\n\n"
    "Frage: Welche Rolle spielt das Model Context Protocol zwischen Sprachmodell "
    "und BI-System?"
)


def main() -> None:
    for model in sys.argv[1:]:
        started = time.monotonic()
        try:
            response = httpx.post(
                "http://127.0.0.1:8080/v1/chat/completions",
                json={
                    "model": model,
                    "temperature": 0.0,
                    "max_tokens": 160,
                    "messages": [{"role": "user", "content": PROMPT}],
                },
                timeout=900.0,
            )
        except httpx.HTTPError as exc:
            print(f"{model}: FEHLER nach {time.monotonic() - started:.1f}s — {exc}")
            continue
        took = time.monotonic() - started
        if response.status_code != 200:
            print(f"{model}: HTTP {response.status_code} nach {took:.1f}s — {response.text[:120]}")
            continue
        body = response.json()
        text = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        print(
            f"{model}: {took:6.1f}s, prompt {usage.get('prompt_tokens')} Tokens, "
            f"Antwort {usage.get('completion_tokens')} Tokens"
        )
        print("   " + " ".join(text.split())[:220])


if __name__ == "__main__":
    main()
