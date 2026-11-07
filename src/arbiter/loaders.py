"""Corpus loading (spec B1, B2, B5).

PDF pages without a text layer are counted and reported, never fatal. A file
that yields no text at all is skipped with a warning so one broken document
cannot stop an index run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pypdf


@dataclass(frozen=True)
class Document:
    quelle: str
    seite: int | None
    text: str


@dataclass
class LoadResult:
    documents: list[Document] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped_files: int = 0
    textless_pages: int = 0


def load_corpus(root: Path, extensions: tuple[str, ...]) -> LoadResult:
    """Read every configured file under ``root``, recursively."""
    result = LoadResult()
    if not root.is_dir():
        result.warnings.append(f"Korpuspfad fehlt: {root}")
        return result

    wanted = {ext.lower() for ext in extensions}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in wanted:
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() == ".pdf":
            _load_pdf(path, relative, result)
        else:
            _load_text(path, relative, result)
    return result


def _load_pdf(path: Path, relative: str, result: LoadResult) -> None:
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
                result.textless_pages += 1
                continue
            result.documents.append(Document(quelle=relative, seite=number, text=text))
        if len(result.documents) == pages_before:
            result.skipped_files += 1
            result.warnings.append(f"{relative}: keine Textebene, übersprungen")
    except Exception as exc:
        result.skipped_files += 1
        result.warnings.append(f"{relative}: PDF nicht lesbar ({exc})")


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
