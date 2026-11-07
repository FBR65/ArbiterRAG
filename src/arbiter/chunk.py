"""Text chunking (spec B6).

Paragraph boundaries are preferred; a window is only shortened to a boundary
when that wastes no more than a quarter of it. Chunks overlap by ``overlap``
characters so a sentence cut at a boundary is still complete in one chunk.
"""

from __future__ import annotations


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 100) -> list[str]:
    """Split ``text`` into chunks of at most ``max_chars`` characters."""
    if max_chars <= 0:
        raise ValueError("max_chars muss positiv sein")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap muss kleiner als max_chars sein")

    if not text.strip():
        return []

    min_snap = max_chars // 4
    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            snap = text.rfind("\n\n", start, end)
            if snap != -1 and snap - start >= min_snap:
                end = snap
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - overlap, start + 1)

    return chunks
