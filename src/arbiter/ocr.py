"""OCR fallback for scanned pages (spec B8, D10/D13).

A page without a text layer is rendered to PNG and sent to two OCR models.
``dots-mocr`` supplies the text that goes into the index — its answer is a
JSON layout, which is reduced to reading-order text. ``paddleocr-vl`` runs on
the same image; its answer is kept beside the page as a second reading and the
word overlap between the two is reported. The two texts are never merged.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import httpx
import pypdfium2 as pdfium

# Official prompt of dots-studio/dots.mocr (model card). Do not paraphrase:
# the layout categories and the "single JSON object" rule are part of the
# contract this code parses.
PROMPT_DOTS = (
    "\n".join(
        [
            "Please output the layout information from the PDF image, including each layout "
            "element's bbox, its category, and the corresponding text content within the bbox.",
            "",
            "1. Bbox format: [x1, y1, x2, y2]",
            "",
            "2.",
            "Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', "
            "'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', "
            "'Text', 'Title'].",
            "",
            "3. Text Extraction & Formatting Rules:",
            "    - Picture: For the 'Picture' category, the text field should be omitted.",
            "    - Formula: Format its text as LaTeX.",
            "- Table: Format its text as HTML.",
            "    - All Others (Text, Title, etc.): Format their text as Markdown.",
            "",
            "4. Constraints:",
            "    - The output text must be the original text from the image, with no translation.",
            "    - All layout elements must be sorted according to human reading order.",
            "",
            "5. Final Output: The entire output must be a single JSON object.",
        ]
    )
    + "\n"
)


class OcrError(RuntimeError):
    """Raised when an OCR model cannot answer."""


def render_png(pdf_path: Path, page_number: int, dpi: int = 300) -> bytes:
    """Render one page (1-based) to PNG bytes."""
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        if page_number < 1 or page_number > len(document):
            raise IndexError(
                f"Seite {page_number} gibt es nicht ({len(document)} Seiten in {pdf_path})"
            )
        page = document[page_number - 1]
        bitmap = page.render(scale=dpi / 72)
        buffer = io.BytesIO()
        bitmap.to_pil().save(buffer, format="PNG")
        return buffer.getvalue()
    finally:
        document.close()


def dots_to_text(raw: str) -> str:
    """Reduce a dots-mocr answer to reading-order text.

    Anything that is not the documented JSON layout is passed through: a model
    that answers in prose has still read the page, and dropping that text would
    lose a page for the sake of a format.
    """
    payload = raw.strip()
    if payload.startswith("```"):
        payload = payload.split("\n", 1)[-1] if "\n" in payload else ""
        if payload.rstrip().endswith("```"):
            payload = payload.rstrip()[:-3]
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return raw.strip()

    elements = parsed
    if isinstance(parsed, dict):
        elements = next((value for value in parsed.values() if isinstance(value, list)), [])
    if not isinstance(elements, list):
        return raw.strip()

    parts = [
        str(element.get("text", "")).strip()
        for element in elements
        if isinstance(element, dict) and str(element.get("text", "")).strip()
    ]
    return "\n\n".join(parts) if parts else raw.strip()


def word_overlap(left: str, right: str) -> float:
    """Share of words the two readings have in common, 0.0 to 1.0."""
    words_left = set(left.casefold().split())
    words_right = set(right.casefold().split())
    if not words_left or not words_right:
        return 0.0
    return len(words_left & words_right) / max(len(words_left), len(words_right))


class VisionOcr:
    """One OCR model behind the OpenAI-compatible chat route."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._model = model
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, transport=transport
        )

    def text_from_png(self, png: bytes, prompt: str = PROMPT_DOTS) -> str:
        import base64

        payload = {
            "model": self._model,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + base64.b64encode(png).decode("ascii")
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        try:
            response = self._client.post("/v1/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise OcrError(f"{self._model}: nicht erreichbar ({exc})") from exc
        if response.status_code != 200:
            raise OcrError(f"{self._model}: HTTP {response.status_code} — {response.text[:200]}")
        body = response.json()
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise OcrError(f"{self._model}: Antwort ohne Text ({str(body)[:200]})") from exc
        return str(content).strip()

    def close(self) -> None:
        self._client.close()
