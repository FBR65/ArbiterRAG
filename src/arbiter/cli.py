"""Command line (spec H1, H2, I1-I3). Thin client of the library.

Model calls go through the OpenAI-compatible endpoints in the configuration;
nothing is hardcoded here.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from arbiter.config import Config, ConfigError, Rag, load_config
from arbiter.embedding import EmbeddingError, HttpEmbedder
from arbiter.eval import format_table, load_goldset, run
from arbiter.output import antwort_json
from arbiter.pipeline import DateiBericht, Pipeline, QuestionTooLong, neuer_pipeline
from arbiter.store import StoreError

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2


def main(
    argv: Sequence[str] | None = None,
    *,
    config: Config | None = None,
    embedder: HttpEmbedder | None = None,
    factory: Callable[[Config, Rag], Pipeline] | None = None,
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

    if args.command == "index" and args.strategie is not None:
        # The override belongs to this run, not to the corpus (spec N5).
        rag = replace(rag, chunk_strategie=args.strategie)

    def build() -> Pipeline:
        if factory is not None:
            return factory(cfg, rag)
        return neuer_pipeline(cfg, rag, embedder=embedder)

    try:
        if args.command == "index":
            return _index(build())
        if args.command == "ask":
            return _ask(build(), args.question, args.json)
        if args.command == "eval":
            return _eval(build(), Path(args.goldset), args.limit)
        return _search(build(), args.question)
    except QuestionTooLong as exc:
        print(f"Frage abgelehnt: {exc}", file=sys.stderr)
        return EXIT_FAIL
    except (StoreError, EmbeddingError) as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return EXIT_FAIL


def _index(pipeline: Pipeline) -> int:
    report = pipeline.index()
    rag = pipeline.rag

    if (
        report.files == 0
        and report.skipped_files == 0
        and report.unchanged == 0
        and not report.warnings
    ):
        print(f"Korpus leer: {rag.corpus_path}", file=sys.stderr)
        return EXIT_FAIL

    print(
        f"{report.files} Dateien gelesen, {report.unchanged} unverändert übersprungen, "
        f"{report.chunks} Abschnitte, {report.changed} neu oder geändert, "
        f"{report.skipped_files} ohne Text, {report.textless_pages} Seiten ohne Text, "
        f"{report.ocr_pages} Seiten per OCR, {report.duplicates} Dubletten, "
        f"{report.entfernt} Dateien aus dem Index entfernt, "
        f"{report.tabellen} Tabellen auf {report.tabellen_seiten} Seiten mit "
        f"{report.zeilen_chunks} Zeilen-Chunks"
    )
    for datei in report.dateien:
        print(_datei_zeile(datei))
    for warning in report.warnings:
        print(f"  Hinweis: {warning}", file=sys.stderr)
    return EXIT_OK


def _datei_zeile(datei: DateiBericht) -> str:
    """One line per file: which strategy was chosen, and why (spec N4, V8)."""
    herkunft = "übersteuert" if datei.uebersteuert else datei.begruendung or "ohne Begründung"
    zeile = f"  {datei.quelle}: {datei.strategie} ({herkunft}), {datei.chunks} Abschnitte"
    if datei.tabellen:
        zeile += (
            f", {datei.tabellen} Tabellen auf {datei.tabellen_seiten} Seiten, "
            f"{datei.zeilen_chunks} Zeilen-Chunks"
        )
    return zeile


def _search(pipeline: Pipeline, question: str) -> int:
    result = pipeline.search(question)

    if not result.passages:
        print("Keine Treffer.")
        return EXIT_OK

    for warning in result.warnungen:
        print(f"  Hinweis: {warning}", file=sys.stderr)

    print(f"Rang  {'Quelle':<32} {'Seite':>5}  {'RRF':>7}  Auszug")
    for passage in result.passages:
        page = "-" if passage.seite is None else str(passage.seite)
        snippet = " ".join(passage.text.split())[:70]
        print(
            f"{passage.rrf_rank:>4}  {passage.quelle:<32.32} {page:>5}  "
            f"{passage.rrf_score:>7.4f}  {snippet}"
        )
    return EXIT_OK


def _ask(pipeline: Pipeline, question: str, as_json: bool) -> int:
    antwort = pipeline.ask(question)
    for warning in antwort.warnungen:
        print(f"  Hinweis: {warning}", file=sys.stderr)

    if as_json:
        print(json.dumps(antwort_json(antwort), ensure_ascii=False, indent=2))
    elif antwort.text is not None:
        print(antwort.text)
        print()
        for quelle in antwort.quellen:
            seite = f", S. {quelle.seite}" if quelle.seite is not None else ""
            print(f"  Quelle: {quelle.quelle}{seite}")
    else:
        print(f"Keine Antwort: {antwort.grund}", file=sys.stderr)

    if antwort.fehler:
        return EXIT_FAIL
    return EXIT_OK


def _eval(pipeline: Pipeline, goldset_path: Path, limit: int) -> int:
    goldset = load_goldset(goldset_path)
    if limit:
        goldset = goldset[:limit]
    if not goldset:
        print(f"Goldset leer: {goldset_path}", file=sys.stderr)
        return EXIT_FAIL

    stufen = run(pipeline, goldset)
    print(format_table(stufen))
    print()
    print(f"Fragen: {len(goldset)} ({sum(1 for g in goldset if g.beantwortbar)} beantwortbar)")
    for stufe in stufen:
        print(
            f"  {stufe.name}: {stufe.beantwortet} Antworten, "
            f"{stufe.beleg_getroffen} mit richtigem Beleg, {stufe.falschantwort} falsch"
        )
    return EXIT_OK


def _load_default_config() -> Config:
    from arbiter.runtime import lade_umgebung

    umgebung = lade_umgebung()
    path = umgebung.get("ARBITER_CONFIG", "arbiter.yaml")
    return load_config(Path(path), umgebung)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arbiter", description="ArbiterRAG")
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="Korpus indizieren")
    index.add_argument("rag")
    index.add_argument(
        "--strategie",
        choices=["auto", "struktur", "qes", "fenster"],
        default=None,
        help="Chunk-Strategie erzwingen (Standard: auto, der Entscheider wählt)",
    )

    search = sub.add_parser("search", help="Passagen suchen")
    search.add_argument("rag")
    search.add_argument("question")

    ask = sub.add_parser("ask", help="Antwort aus belegten Passagen")
    ask.add_argument("rag")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true", help="maschinenlesbare Ausgabe")

    evaluate = sub.add_parser("eval", help="Goldset messen")
    evaluate.add_argument("rag")
    evaluate.add_argument("--goldset", default="goldset/downloads.yaml")
    evaluate.add_argument("--limit", type=int, default=0, help="nur die ersten N Fragen")

    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
