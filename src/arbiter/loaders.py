"""Corpus loading (spec B1, B2, B5, B8).

PDF pages without a text layer are offered to an optional OCR hook. The hook
returns the text to index plus the second model's reading; a hook that raises
is reported and the page stays skipped. One broken document never stops a run.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pypdf

# pypdf warns about fonts it cannot fully decode (one block per font object,
# hundreds of lines on a print PDF). The text still comes out; the noise hides
# the run's own report.
logging.getLogger("pypdf").setLevel(logging.ERROR)


@dataclass(frozen=True)
class OcrPage:
    """One OCR result: the indexed text plus the second model's reading."""

    text: str
    vergleich: str
    ueberlappung: float


OcrHook = Callable[[Path, int], OcrPage]


@dataclass(frozen=True)
class Document:
    quelle: str
    seite: int | None
    text: str
    ocr: bool = False
    ocr_paddle: str = ""
    ocr_ueberlappung: float = 0.0


@dataclass
class LoadResult:
    documents: list[Document] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped_files: int = 0
    textless_pages: int = 0
    duplicates: int = 0


def load_corpus(
    root: Path,
    extensions: tuple[str, ...],
    ocr: OcrHook | None = None,
) -> LoadResult:
    """Read every configured file under ``root``, recursively."""
    result = LoadResult()
    if not root.is_dir():
        result.warnings.append(f"Korpuspfad fehlt: {root}")
        return result

    wanted = {ext.lower() for ext in extensions}
    seen: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in wanted:
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() == ".pdf":
            before = len(result.documents)
            _load_pdf(path, relative, result, ocr)
            _drop_duplicate(path, relative, result, before, seen)
        else:
            before = len(result.documents)
            _load_text(path, relative, result)
            _drop_duplicate(path, relative, result, before, seen)
    return result


def _drop_duplicate(
    path: Path,
    relative: str,
    result: LoadResult,
    before: int,
    seen: dict[str, str],
) -> None:
    """Skip a file whose text was already indexed from another path (spec B3).

    The same document often sits twice in a download folder (once at the top
    level, once in a subfolder). Indexing both puts the same text in two rows
    and steals ranks from real results.
    """
    added = result.documents[before:]
    if not added:
        return
    digest = hashlib.sha256("".join(doc.text for doc in added).encode("utf-8")).hexdigest()
    earlier = seen.get(digest)
    if earlier is not None:
        del result.documents[before:]
        result.duplicates += 1
        result.warnings.append(f"{relative}: Dublette von {earlier}, übersprungen")
        return
    seen[digest] = relative


def _load_pdf(path: Path, relative: str, result: LoadResult, ocr: OcrHook | None) -> None:
    try:
        reader = pypdf.PdfReader(str(path))
        pages_before = len(result.documents)
        for number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                result.warnings.append(f"{relative}, Seite {number}: nicht lesbar ({exc})")
                text = ""
            if not text.strip():
                document = _ocr_page(path, relative, number, result, ocr)
                if document is None:
                    result.textless_pages += 1
                    continue
                result.documents.append(document)
                continue
            result.documents.append(Document(quelle=relative, seite=number, text=text))
        if len(result.documents) == pages_before:
            result.skipped_files += 1
            if ocr is None:
                result.warnings.append(f"{relative}: keine Textebene, übersprungen")
    except Exception as exc:
        result.skipped_files += 1
        result.warnings.append(f"{relative}: PDF nicht lesbar ({exc})")


def _ocr_page(
    path: Path,
    relative: str,
    number: int,
    result: LoadResult,
    ocr: OcrHook | None,
) -> Document | None:
    if ocr is None:
        return None
    try:
        page = ocr(path, number)
    except Exception as exc:
        result.warnings.append(f"{relative}, Seite {number}: OCR fehlgeschlagen ({exc})")
        return None
    if not page.text.strip():
        result.warnings.append(f"{relative}, Seite {number}: OCR lieferte keinen Text")
        return None
    return Document(
        quelle=relative,
        seite=number,
        text=page.text,
        ocr=True,
        ocr_paddle=page.vergleich,
        ocr_ueberlappung=page.ueberlappung,
    )


def _load_text(path: Path, relative: str, result: LoadResult) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="latin-1")
        except Exception as exc:
            result.skipped_files += 1
            result.warnings.append(f"{relative}: nicht lesbar ({exc})")
            return
    except Exception as exc:
        result.skipped_files += 1
        result.warnings.append(f"{relative}: nicht lesbar ({exc})")
        return

    if not text.strip():
        result.skipped_files += 1
        result.warnings.append(f"{relative}: leer, übersprungen")
        return
    result.documents.append(Document(quelle=relative, seite=None, text=text))
