"""Measurement on a gold set (spec H4, stage 5).

Four stages are measured separately — fusion, with HyDE, with reranking, with
the decision layer — because an improvement that only shows up in the total
says nothing about which stage produced it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from arbiter.pipeline import Passage, Pipeline


@dataclass(frozen=True)
class GoldFrage:
    frage: str
    beantwortbar: bool
    erwartet: list[tuple[str, int]]


@dataclass
class Stufe:
    name: str
    recall: float = 0.0
    ndcg: float = 0.0
    beantwortet: int = 0
    beleg_getroffen: int = 0
    falschantwort: int = 0
    fragen: int = 0
    warnungen: list[str] = field(default_factory=list)


def load_goldset(path: Path) -> list[GoldFrage]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    fragen = []
    for entry in raw.get("fragen", []):
        erwartet = [
            (str(item["quelle"]), int(item["seite"])) for item in entry.get("erwartet") or []
        ]
        fragen.append(
            GoldFrage(
                frage=str(entry["frage"]),
                beantwortbar=bool(entry["beantwortbar"]),
                erwartet=erwartet,
            )
        )
    return fragen


def ist_beleg(passage: Passage, erwartet: list[tuple[str, int]]) -> bool:
    return any(quelle in passage.quelle and passage.seite == seite for quelle, seite in erwartet)


def recall_at(passages: list[Passage], gold: GoldFrage, k: int) -> float:
    if not gold.erwartet:
        return 0.0
    return 1.0 if any(ist_beleg(p, gold.erwartet) for p in passages[:k]) else 0.0


def ndcg_at(passages: list[Passage], gold: GoldFrage, k: int) -> float:
    if not gold.erwartet:
        return 0.0
    hits = [ist_beleg(p, gold.erwartet) for p in passages[:k]]
    if not any(hits):
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, hit in enumerate(hits) if hit)
    return dcg  # the ideal DCG for one relevant document at rank 1 is 1.0


def _stage(pipeline: Pipeline, goldset: list[GoldFrage], name: str, mit_antwort: bool) -> Stufe:
    stufe = Stufe(name=name, fragen=len(goldset))
    recall_sum = 0.0
    ndcg_sum = 0.0
    for gold in goldset:
        antwort = pipeline.ask(gold.frage) if mit_antwort else None
        result = pipeline.search(gold.frage)
        recall_sum += recall_at(result.passages, gold, 10)
        ndcg_sum += ndcg_at(result.passages, gold, 10)
        stufe.warnungen.extend(result.warnungen)

        if antwort is None:
            continue
        if antwort.text is not None:
            stufe.beantwortet += 1
            if gold.erwartet and any(
                quelle in antwort_quelle.quelle and antwort_quelle.seite == seite
                for antwort_quelle in antwort.quellen
                for quelle, seite in gold.erwartet
            ):
                stufe.beleg_getroffen += 1
            elif not gold.erwartet:
                stufe.falschantwort += 1  # answered although the corpus has no answer
        elif gold.beantwortbar:
            stufe.falschantwort += 1  # refused although the corpus has the answer

    stufe.recall = recall_sum / len(goldset) if goldset else 0.0
    stufe.ndcg = ndcg_sum / len(goldset) if goldset else 0.0
    return stufe


def run(pipeline: Pipeline, goldset: list[GoldFrage]) -> list[Stufe]:
    """Measure the four stages on the same questions."""
    nur_fusion = replace(pipeline.rag, hyde=False, rerank=False)
    mit_hyde = replace(pipeline.rag, hyde=True, rerank=False)
    mit_rerank = replace(pipeline.rag, hyde=True, rerank=True)

    def stage_pipeline(rag: object) -> Pipeline:
        twin = Pipeline(pipeline.config, rag, embedder=pipeline.embedder)  # type: ignore[arg-type]
        twin.store = pipeline.store
        twin.hyde = pipeline.hyde
        twin.reranker = pipeline.reranker
        twin.decision = pipeline.decision
        twin.answerer = pipeline.answerer
        return twin

    return [
        _stage(stage_pipeline(nur_fusion), goldset, "nur Fusion", mit_antwort=False),
        _stage(stage_pipeline(mit_hyde), goldset, "mit HyDE", mit_antwort=False),
        _stage(stage_pipeline(mit_rerank), goldset, "mit Reranker", mit_antwort=False),
        _stage(stage_pipeline(mit_rerank), goldset, "mit Laya", mit_antwort=True),
    ]


def format_table(stufen: list[Stufe]) -> str:
    zeilen = [
        f"{'Stufe':<14} {'Recall@10':>10} {'nDCG@10':>8} {'Antworten':>10} "
        f"{'Beleg ok':>9} {'falsch':>7}"
    ]
    for stufe in stufen:
        zeilen.append(
            f"{stufe.name:<14} {stufe.recall:>10.2f} {stufe.ndcg:>8.2f} "
            f"{stufe.beantwortet:>10} {stufe.beleg_getroffen:>9} {stufe.falschantwort:>7}"
        )
    return "\n".join(zeilen)
