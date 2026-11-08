"""Measure the two search switches: German stemming and hybrid vs dense only (B2, B3)."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from arbiter.config import load_config
from arbiter.eval import load_goldset, ndcg_at, recall_at
from arbiter.pipeline import Pipeline

cfg = load_config(Path("arbiter.yaml"), os.environ)
rag = cfg.rags["downloads"]
goldset = load_goldset(Path("goldset/downloads.yaml"))
pipeline = Pipeline(cfg, rag)
nur_suche = replace(rag, rerank=False, fuse_top_k=10, rerank_top_n=10)
twin = Pipeline(cfg, nur_suche, embedder=pipeline.embedder)
twin.store = pipeline.store

print(f"{'Variante':<28} {'Recall@10':>10} {'nDCG@10':>8}")
for stem in (False, True):
    pipeline.store.ensure_fts(rag.name, stem=stem)
    for hybrid in (True, False):
        twin.rag = replace(nur_suche, hybrid=hybrid)
        recall = ndcg = 0.0
        for gold in goldset:
            result = twin.search(gold.frage)
            recall += recall_at(result.passages, gold, 10)
            ndcg += ndcg_at(result.passages, gold, 10)
        print(
            f"stem={stem!s:<5} hybrid={hybrid!s:<5} "
            f"{recall / len(goldset):>10.2f} {ndcg / len(goldset):>8.2f}",
            flush=True,
        )
