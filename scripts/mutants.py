"""Manual mutation run (gauntlet layer).

Applies one real bug at a time to the source, runs the suite, and records
whether the suite noticed. Every mutant must be killed. The file is restored
from git after each run, so a survivor is visible instead of silent.

Run:  uv run python scripts/mutants.py
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Mutant:
    name: str
    path: str
    old: str
    new: str


MUTANTS = [
    Mutant(
        "chunk-window-off-by-one",
        "src/arbiter/chunk.py",
        "ende = min(start + max_chars, length)",
        "ende = min(start + max_chars + 1, length)",
    ),
    Mutant(
        "fusion-rank-off-by-one",
        "src/arbiter/fusion.py",
        "for rank, doc_id in enumerate(ranked, start=1):",
        "for rank, doc_id in enumerate(ranked, start=0):",
    ),
    Mutant(
        "fusion-tie-break-dropped",
        "src/arbiter/fusion.py",
        "fused = sorted(scores.items(), key=lambda item: (-item[1], item[0]))",
        "fused = sorted(scores.items(), key=lambda item: -item[1])",
    ),
    Mutant(
        "store-forgets-to-delete-changed",
        "src/arbiter/store.py",
        "self._delete(table, sorted(set(stale) | {r.id for r in changed}))",
        "self._delete(table, sorted(set(stale)))",
    ),
    Mutant(
        "config-accepts-empty-variable",
        "src/arbiter/config.py",
        "if not environ[name]:",
        "if False:",
    ),
    Mutant(
        "cli-unknown-rag-succeeds",
        "src/arbiter/cli.py",
        '        return EXIT_USAGE\n\n    if args.command == "index"',
        '        return EXIT_OK\n\n    if args.command == "index"',
    ),
    Mutant(
        "pipeline-skips-question-length-check",
        "src/arbiter/pipeline.py",
        "if len(question) > MAX_FRAGE_ZEICHEN:",
        "if len(question) > MAX_FRAGE_ZEICHEN * 10:",
    ),
    Mutant(
        "ocr-overlap-divides-by-the-smaller-set",
        "src/arbiter/ocr.py",
        "return len(words_left & words_right) / max(len(words_left), len(words_right))",
        "return len(words_left & words_right) / min(len(words_left), len(words_right))",
    ),
    Mutant(
        "loaders-never-calls-ocr",
        "src/arbiter/loaders.py",
        "    if ocr is None:\n        return None\n    try:",
        "    if True:\n        return None\n    try:",
    ),
    Mutant(
        "store-skips-column-migration",
        "src/arbiter/store.py",
        "        if missing:\n            table.add_columns(missing)",
        "        if False:\n            table.add_columns(missing)",
    ),
    Mutant(
        "rerank-scores-land-on-the-first-slot",
        "src/arbiter/rerank.py",
        '            scores[index] = float(entry["relevance_score"])',
        '            scores[0] = float(entry["relevance_score"])',
    ),
    Mutant(
        "api-allows-a-second-index-run",
        "src/arbiter/api.py",
        "            if request.rag in running:",
        "            if False:",
    ),
    Mutant(
        "chunks-lose-the-document-header",
        "src/arbiter/pipeline.py",
        'kennung = f"[{Path(quelle).stem}, S. {seite or 0}"',
        'kennung = f""',
    ),
    Mutant(
        "duplicate-detection-off",
        "src/arbiter/loaders.py",
        "    if earlier is not None:",
        "    if False:",
    ),
    Mutant(
        "logprob-gate-ignores-the-threshold",
        "src/arbiter/pipeline.py",
        "            if p >= self.rag.gate_schwelle:",
        "            if True:",
    ),
    Mutant(
        "struktur-verweis-laenge-ignoriert",
        "src/arbiter/chunk_structure.py",
        "[:xref_chars].strip()",
        "[:].strip()",
    ),
    Mutant(
        "struktur-abschnittsgrenze-aufgehoben",
        "src/arbiter/chunk_structure.py",
        "        ende = koepfe[index + 1][0] if index + 1 < len(koepfe) else len(text)",
        "        ende = len(text)",
    ),
    Mutant(
        "struktur-haengt-verweis-auf-sich-an",
        "src/arbiter/chunk_structure.py",
        "        if not nummer or nummer == teil.nummer or nummer in gesehen:",
        "        if not nummer or nummer in gesehen:",
    ),
    Mutant(
        "entscheider-schwelle-ignoriert",
        "src/arbiter/chunk_choice.py",
        '        strategie="struktur" if punktzahl >= schwelle else "qes",',
        '        strategie="struktur",',
    ),
    Mutant(
        "entscheider-zaehlt-riesenabschnitte",
        "src/arbiter/chunk_structure.py",
        "        if teil.ebene >= ebene and teil.ende - teil.start <= max_chars * 1.5",
        "        if teil.ebene >= ebene",
    ),
    Mutant(
        "entscheider-ignoriert-die-uebersteuerung",
        "src/arbiter/chunk.py",
        '    if strategie == "auto":',
        "    if True:",
    ),
    Mutant(
        "fenster-chunk-verliert-seine-position",
        "src/arbiter/chunk.py",
        "                    start=links,\n                    ende=rechts,",
        "                    start=start,\n                    ende=rechts,",
    ),
    Mutant(
        "seitenzuordnung-immer-seite-eins",
        "src/arbiter/pipeline.py",
        "    return bisect_right(list(anfaenge), position)",
        "    return 1",
    ),
    Mutant(
        "seitenbereich-ignoriert-das-ende",
        "src/arbiter/store.py",
        "    return start <= seite <= max(start, ende)",
        "    return start == seite",
    ),
    Mutant(
        "strategie-fehlt-in-der-zeile",
        "src/arbiter/store.py",
        '        "strategie": record.strategie,',
        '        "strategie": "",',
    ),
    Mutant(
        "fingerabdruck-ignoriert-die-strategie",
        "src/arbiter/pipeline.py",
        "                rag.chunk_strategie,",
        '                "",',
    ),
    Mutant(
        "qes-hit-gate-abgeschaltet",
        "src/arbiter/chunk_qes.py",
        "        if not any(fenster.erster <= block <= fenster.letzter\n"
        "                for block in treffer[nummer]):\n            continue",
        "        if False:\n            continue",
    ),
    Mutant(
        "qes-singleton-schutz-aus",
        "src/arbiter/chunk_qes.py",
        "    if bester_einzelner is not None and gesamt([bester_einzelner]) > gesamt(gewaehlt):",
        "    if False:",
    ),
    Mutant(
        "qes-nimmt-mehrdeutige-fragmente",
        "src/arbiter/chunk_qes.py",
        "    if len(treffer) == 1:",
        "    if treffer:",
    ),
    Mutant(
        "qes-laesst-luecken",
        "src/arbiter/chunk_qes.py",
        "        spannen_auswahl.extend(_teile(start, ende, max_chars))",
        "        spannen_auswahl.extend([])",
    ),
    Mutant(
        "qes-fensterlaenge-ueber-dem-limit",
        "src/arbiter/chunk_qes.py",
        "        return kurz, min(lang, max_chars)",
        "        return kurz, lang",
    ),
    Mutant(
        "qe-obergrenze-ignoriert",
        "src/arbiter/chunk_qe.py",
        "            if len(paare) >= max_fragen:",
        "            if False:",
    ),
    Mutant(
        "qe-doppelte-fragen-bleiben",
        "src/arbiter/chunk_qe.py",
        "            if paar.frage in gesehen:\n                continue",
        "            if False:\n                continue",
    ),
    Mutant(
        "qe-nimmt-nicht-woertliche-belege",
        "src/arbiter/chunk_qe.py",
        "            if not all(_steht_im_text(beleg, saetze) for beleg in paar.belege):",
        "            if False:",
    ),
]


def run(cmd: list[str]) -> int:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).returncode


def main() -> int:
    if run(["git", "diff", "--quiet"]) != 0:
        print("Arbeitsbaum ist nicht sauber — erst committen, dann mutieren.")
        return 2

    survivors: list[str] = []
    for mutant in MUTANTS:
        target = ROOT / mutant.path
        original = target.read_text(encoding="utf-8")
        if mutant.old not in original:
            print(f"FEHLER {mutant.name}: Suchtext nicht gefunden in {mutant.path}")
            survivors.append(mutant.name + " (nicht anwendbar)")
            continue
        target.write_text(original.replace(mutant.old, mutant.new, 1), encoding="utf-8")
        code = subprocess.run(
            ["uv", "run", "pytest", "-q", "-x"], cwd=ROOT, capture_output=True, text=True
        ).returncode
        target.write_text(original, encoding="utf-8")
        killed = code != 0
        print(f"{'getötet ' if killed else 'ÜBERLEBT'}  {mutant.name}")
        if not killed:
            survivors.append(mutant.name)

    print()
    print(f"{len(MUTANTS) - len(survivors)}/{len(MUTANTS)} Mutanten getötet")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
