"""Command line (spec H1, H2, I1-I3). Thin client of the library.

Model calls go through the OpenAI-compatible endpoints in the configuration;
nothing is hardcoded here.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from arbiter.config import Config, ConfigError, Rag, load_config
from arbiter.embedding import EmbeddingError, HttpEmbedder
from arbiter.pipeline import Pipeline, QuestionTooLong
from arbiter.store import StoreError

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2


def main(
    argv: Sequence[str] | None = None,
    *,
    config: Config | None = None,
    embedder: HttpEmbedder | None = None,
) -> int:
    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        cfg = config if config is not None else _load_default_config()
    except ConfigError as exc:
        print(f"Fehler in der Konfiguration: {exc}", file=sys.stderr)
        return EXIT_USAGE

    rag = cfg.rags.get(args.rag)
    if rag is None:
        known = ", ".join(sorted(cfg.rags)) or "keine"
        print(
            f"Unbekannter RAG '{args.rag}'. Bekannt: {known}",
            file=sys.stderr,
        )
        return EXIT_USAGE

    try:
        if args.command == "index":
            return _index(cfg, rag, embedder)
        return _search(cfg, rag, args.question, embedder)
    except QuestionTooLong as exc:
        print(f"Frage abgelehnt: {exc}", file=sys.stderr)
        return EXIT_FAIL
    except (StoreError, EmbeddingError) as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return EXIT_FAIL


def _index(cfg: Config, rag: Rag, embedder: HttpEmbedder | None) -> int:
    pipeline = Pipeline(cfg, rag, embedder=embedder)
    report = pipeline.index()

    if report.files == 0 and report.skipped_files == 0 and not report.warnings:
        print(f"Korpus leer: {rag.corpus_path}", file=sys.stderr)
        return EXIT_FAIL

    print(
        f"{report.files} Dateien, {report.chunks} Abschnitte, "
        f"{report.changed} neu oder geändert, {report.skipped_files} übersprungen, "
        f"{report.textless_pages} Seiten ohne Text, {report.ocr_pages} Seiten per OCR"
    )
    for warning in report.warnings:
        print(f"  Hinweis: {warning}", file=sys.stderr)
    return EXIT_OK


def _search(cfg: Config, rag: Rag, question: str, embedder: HttpEmbedder | None) -> int:
    pipeline = Pipeline(cfg, rag, embedder=embedder)
    result = pipeline.search(question)

    if not result.passages:
        print("Keine Treffer.")
        return EXIT_OK

    print(f"Rang  {'Quelle':<32} {'Seite':>5}  {'RRF':>7}  Auszug")
    for passage in result.passages:
        page = "-" if passage.seite is None else str(passage.seite)
        snippet = " ".join(passage.text.split())[:70]
        print(
            f"{passage.rrf_rank:>4}  {passage.quelle:<32.32} {page:>5}  "
            f"{passage.rrf_score:>7.4f}  {snippet}"
        )
    return EXIT_OK


def _load_default_config() -> Config:
    path = os.environ.get("ARBITER_CONFIG", "arbiter.yaml")
    return load_config(Path(path), os.environ)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arbiter", description="ArbiterRAG")
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="Korpus indizieren")
    index.add_argument("rag")

    search = sub.add_parser("search", help="Passagen suchen")
    search.add_argument("rag")
    search.add_argument("question")

    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
