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
        "end = min(start + max_chars, length)",
        "end = min(start + max_chars + 1, length)",
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
        "        return EXIT_USAGE\n\n    def build() -> Pipeline:",
        "        return EXIT_OK\n\n    def build() -> Pipeline:",
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
        "hyde-does-not-truncate",
        "src/arbiter/hyde.py",
        "    if len(text) <= max_chars:\n        return text.strip()",
        "    if True:\n        return text.strip()",
    ),
    Mutant(
        "rerank-scores-land-on-the-first-slot",
        "src/arbiter/rerank.py",
        '            scores[index] = float(entry["relevance_score"])',
        '            scores[0] = float(entry["relevance_score"])',
    ),
    Mutant(
        "laya-accepts-every-passage",
        "src/arbiter/pipeline.py",
        "            if prob >= self.rag.laya_schwelle:",
        "            if prob >= 0.0:",
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
        'kopf = f"[{Path(document.quelle).stem}, S. {page}] "',
        'kopf = ""',
    ),
    Mutant(
        "duplicate-detection-off",
        "src/arbiter/loaders.py",
        "    if earlier is not None:",
        "    if False:",
    ),
    Mutant(
        "choice-gate-ignores-keine",
        "src/arbiter/pipeline.py",
        '        if bester == "keine" or probabilities[bester] < keine + marge:',
        "        if False:",
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
