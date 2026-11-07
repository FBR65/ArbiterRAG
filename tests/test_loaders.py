"""Spec B1/B2/B5: corpus loading, including PDFs without a text layer."""

from __future__ import annotations

from pathlib import Path

import pypdf

from arbiter.loaders import load_corpus


def test_reads_configured_extensions_recursively(tmp_path: Path) -> None:
    (tmp_path / "unter").mkdir()
    (tmp_path / "a.txt").write_text("erste Datei", encoding="utf-8")
    (tmp_path / "unter" / "b.md").write_text("zweite Datei", encoding="utf-8")
    (tmp_path / "ignoriert.csv").write_text("x", encoding="utf-8")
    result = load_corpus(tmp_path, (".txt", ".md"))
    quellen = sorted(d.quelle for d in result.documents)
    assert quellen == ["a.txt", "unter/b.md"]
    assert result.skipped_files == 0


def test_pdf_pages_become_separate_documents(tmp_path: Path) -> None:
    text = "Seite eins mit Text.\n" + "Inhalt " * 40
    _pdf = _make_text_pdf(tmp_path / "vertrag.pdf", [text, "Seite zwei."])
    result = load_corpus(tmp_path, (".pdf",))
    assert [d.seite for d in result.documents] == [1, 2]
    assert result.documents[0].quelle == "vertrag.pdf"
    assert "Seite eins" in result.documents[0].text
    assert result.textless_pages == 0


def test_pdf_page_without_text_layer_is_counted_not_fatal(tmp_path: Path) -> None:
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(tmp_path / "scan.pdf", "wb") as fh:
        writer.write(fh)

    result = load_corpus(tmp_path, (".pdf",))
    assert result.documents == []
    assert result.textless_pages == 1
    assert result.skipped_files == 1
    assert any("scan.pdf" in w for w in result.warnings)


def test_empty_corpus_reports_zero(tmp_path: Path) -> None:
    result = load_corpus(tmp_path, (".pdf",))
    assert result.documents == []
    assert result.skipped_files == 0
    assert result.textless_pages == 0


def _make_text_pdf(path: Path, pages: list[str]) -> Path:
    """Smallest PDF that pypdf reads back with extract_text() per page."""
    objects: list[str] = []
    kids: list[str] = []
    contents: list[str] = []
    for _ in pages:
        contents.append("")

    stream_objs = []
    for page_text in pages:
        escaped = page_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 40 700 Td ({escaped}) Tj ET"
        stream_objs.append(stream)

    n_pages = len(pages)
    page_ids = [4 + i for i in range(n_pages)]
    content_ids = [4 + n_pages + i for i in range(n_pages)]
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append("<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for _pid, cid in zip(page_ids, content_ids, strict=True):
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {cid} 0 R >>"
        )
    for stream in stream_objs:
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")

    out = "%PDF-1.4\n"
    offsets: list[int] = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{obj}\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n"
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
    path.write_bytes(out.encode("latin-1"))
    return path
