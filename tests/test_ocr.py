"""Spec B8: OCR fallback for pages without a text layer."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from arbiter.ocr import PROMPT_DOTS, VisionOcr, dots_to_text, render_png, word_overlap


def test_dots_json_becomes_reading_order_text() -> None:
    raw = json.dumps(
        [
            {"category": "Title", "bbox": [0, 0, 10, 10], "text": "Mietvertrag"},
            {"category": "Text", "bbox": [0, 20, 10, 30], "text": "Miete 940 Euro"},
            {"category": "Picture", "bbox": [0, 40, 10, 50]},
            {
                "category": "Table",
                "bbox": [0, 60, 10, 70],
                "text": "<table><tr><td>210</td></tr></table>",
            },
        ]
    )
    assert dots_to_text(raw) == (
        "Mietvertrag\n\nMiete 940 Euro\n\n<table><tr><td>210</td></tr></table>"
    )


def test_dots_json_wrapped_in_code_fence_is_read() -> None:
    raw = (
        "```json\n"
        + json.dumps([{"category": "Text", "bbox": [0, 0, 1, 1], "text": "Hallo"}])
        + "\n```"
    )
    assert dots_to_text(raw) == "Hallo"


def test_non_json_answer_is_passed_through_unchanged() -> None:
    assert dots_to_text("reiner Text ohne JSON") == "reiner Text ohne JSON"


def test_word_overlap_bounds() -> None:
    assert word_overlap("Miete 940 Euro", "Miete 940 Euro") == 1.0
    assert word_overlap("Miete 940 Euro", "Wetter sonnig") == 0.0
    assert 0.0 < word_overlap("Miete 940 Euro", "Miete 950 Euro") < 1.0


def test_render_produces_png(tmp_path: Path) -> None:
    from tests.test_loaders import _make_text_pdf

    pdf = _make_text_pdf(tmp_path / "seite.pdf", ["Eine Seite mit Text."])
    png = render_png(pdf, 1, dpi=150)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_rejects_unknown_page(tmp_path: Path) -> None:
    from tests.test_loaders import _make_text_pdf

    pdf = _make_text_pdf(tmp_path / "seite.pdf", ["Eine Seite."])
    with pytest.raises(IndexError):
        render_png(pdf, 2, dpi=150)


def test_vision_client_sends_data_uri_and_reads_answer() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        seen["model"] = body["model"]
        content = body["messages"][0]["content"]
        seen["types"] = [part["type"] for part in content]
        seen["image"] = content[0]["image_url"]["url"][:22]
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '[{"text": "Hallo"}]'}}]}
        )

    client = VisionOcr(
        base_url="http://test", model="dots-mocr", transport=httpx.MockTransport(handler)
    )
    assert client.text_from_png(b"\x89PNG fake") == '[{"text": "Hallo"}]'
    assert seen["model"] == "dots-mocr"
    assert seen["types"] == ["image_url", "text"]
    assert str(seen["image"]).startswith("data:image/png;base64,")


def test_vision_client_reports_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="busy")

    client = VisionOcr(base_url="http://test", model="m", transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError) as err:
        client.text_from_png(b"x")
    assert "503" in str(err.value)


def test_prompt_is_the_documented_one() -> None:
    assert "layout information" in PROMPT_DOTS
    assert "single JSON object" in PROMPT_DOTS
