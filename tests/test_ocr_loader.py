"""Spec B8 and B3: OCR hook in the loader, comparison fields in the store."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pypdf
import pytest

from arbiter.loaders import OcrPage, load_corpus
from arbiter.store import Record, Store


def _blank_pdf(path: Path) -> Path:
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as fh:
        writer.write(fh)
    return path


def test_textless_page_goes_through_the_ocr_hook(tmp_path: Path) -> None:
    _blank_pdf(tmp_path / "scan.pdf")
    calls: list[tuple[str, int]] = []

    def hook(path: Path, page: int) -> OcrPage:
        calls.append((path.name, page))
        return OcrPage(
            text="Erkannte Miete 940 Euro", vergleich="Miete 940 Euro", ueberlappung=0.75
        )

    result = load_corpus(tmp_path, (".pdf",), ocr=hook)
    assert calls == [("scan.pdf", 1)]
    assert len(result.documents) == 1
    document = result.documents[0]
    assert document.text == "Erkannte Miete 940 Euro"
    assert document.ocr is True
    assert document.ocr_paddle == "Miete 940 Euro"
    assert document.ocr_ueberlappung == pytest.approx(0.75)
    assert result.textless_pages == 0


def test_without_hook_the_page_stays_skipped(tmp_path: Path) -> None:
    _blank_pdf(tmp_path / "scan.pdf")
    result = load_corpus(tmp_path, (".pdf",))
    assert result.documents == []
    assert result.textless_pages == 1
    assert result.skipped_files == 1


def test_failing_ocr_is_a_warning_not_a_crash(tmp_path: Path) -> None:
    _blank_pdf(tmp_path / "scan.pdf")

    def hook(path: Path, page: int) -> OcrPage:
        raise RuntimeError("Modell antwortet 503")

    result = load_corpus(tmp_path, (".pdf",), ocr=hook)
    assert result.documents == []
    assert result.textless_pages == 1
    assert any("503" in warning for warning in result.warnings)


def test_pages_with_text_layer_do_not_use_the_hook(tmp_path: Path) -> None:
    from tests.test_loaders import _make_text_pdf

    _make_text_pdf(tmp_path / "digital.pdf", ["Seite mit Textebene."])
    calls: list[int] = []

    def hook(path: Path, page: int) -> OcrPage:
        calls.append(page)
        return OcrPage(text="x", vergleich="x", ueberlappung=1.0)

    load_corpus(tmp_path, (".pdf",), ocr=hook)
    assert calls == []


def test_parser_warnungen_der_ocr_seite_stehen_im_bericht(tmp_path: Path) -> None:
    _blank_pdf(tmp_path / "scan.pdf")

    def hook(path: Path, page: int) -> OcrPage:
        return OcrPage(
            text="[scan, S. 1] Kopf: A | Zeile: 1",
            vergleich="A 1",
            ueberlappung=0.5,
            warnungen=("Tabelle ohne erkennbare Kopfzeile",),
        )

    result = load_corpus(tmp_path, (".pdf",), ocr=hook)
    assert any(
        "scan.pdf" in warning and "Kopfzeile" in warning for warning in result.warnings
    )



def test_store_migrates_a_table_without_ocr_columns(tmp_path: Path) -> None:
    import lancedb
    import pyarrow as pa

    db = lancedb.connect(str(tmp_path / "db"))
    legacy = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("quelle", pa.string()),
            pa.field("seite", pa.int32()),
            pa.field("hash", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 2)),
        ]
    )
    table = db.create_table("a", schema=legacy)
    table.add(
        [
            {
                "id": "alt#1#1",
                "text": "Alt",
                "quelle": "alt.pdf",
                "seite": 1,
                "hash": "x",
                "vector": [1.0, 0.0],
            }
        ]
    )

    store = Store(tmp_path / "db")
    text = "Neu"
    store.upsert(
        "a",
        [
            Record(
                id="neu.pdf#1#1",
                text=text,
                quelle="neu.pdf",
                seite=1,
                hash=hashlib.sha256(text.encode()).hexdigest(),
                vector=[0.0, 1.0],
                ocr=True,
                ocr_paddle="Neu",
                ocr_ueberlappung=1.0,
            )
        ],
    )
    hits = store.search_dense("a", [0.0, 1.0], k=2)
    assert hits[0].id == "neu.pdf#1#1"
    assert hits[0].ocr is True
    legacy_hit = next(hit for hit in hits if hit.id == "alt#1#1")
    assert legacy_hit.ocr is False


def test_store_keeps_the_ocr_fields(tmp_path: Path) -> None:
    store = Store(tmp_path / "db")
    text = "Erkannte Miete"
    store.upsert(
        "a",
        [
            Record(
                id="scan.pdf#1#1",
                text=text,
                quelle="scan.pdf",
                seite=1,
                hash=hashlib.sha256(text.encode()).hexdigest(),
                vector=[1.0, 0.0],
                ocr=True,
                ocr_paddle="Miete erkannt",
                ocr_ueberlappung=0.5,
            )
        ],
    )
    hit = store.search_dense("a", [1.0, 0.0], k=1)[0]
    assert hit.ocr is True
    assert hit.ocr_paddle == "Miete erkannt"
    assert hit.ocr_ueberlappung == pytest.approx(0.5)
