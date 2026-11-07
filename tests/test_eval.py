"""Spec H4: metrics on a gold set."""

from __future__ import annotations

from pathlib import Path

import pytest

from arbiter.eval import GoldFrage, format_table, ist_beleg, load_goldset, ndcg_at, recall_at
from arbiter.pipeline import Passage

GOLD = GoldFrage(frage="Frage?", beantwortbar=True, erwartet=[("a.pdf", 3)])


def _passage(quelle: str, seite: int, rang: int) -> Passage:
    return Passage(
        id=f"{quelle}#{seite}#1",
        text="Text",
        quelle=quelle,
        seite=seite,
        rrf_rank=rang,
        rrf_score=0.0,
    )


def test_beleg_needs_both_document_and_page() -> None:
    assert ist_beleg(_passage("a.pdf", 3, 1), GOLD.erwartet)
    assert not ist_beleg(_passage("a.pdf", 4, 1), GOLD.erwartet)
    assert not ist_beleg(_passage("b.pdf", 3, 1), GOLD.erwartet)


def test_recall_at_k_respects_the_cut() -> None:
    passages = [_passage("x.pdf", 1, 1), _passage("a.pdf", 3, 2)]
    assert recall_at(passages, GOLD, 10) == 1.0
    assert recall_at(passages, GOLD, 1) == 0.0


def test_ndcg_rewards_a_higher_rank() -> None:
    oben = ndcg_at([_passage("a.pdf", 3, 1)], GOLD, 10)
    unten = ndcg_at([_passage("x.pdf", 1, 1), _passage("a.pdf", 3, 2)], GOLD, 10)
    assert oben == pytest.approx(1.0)
    assert 0.0 < unten < oben


def test_ndcg_is_zero_without_a_hit() -> None:
    assert ndcg_at([_passage("x.pdf", 1, 1)], GOLD, 10) == 0.0


def test_unanswerable_questions_are_not_scored_in_recall() -> None:
    gold = GoldFrage(frage="Frage?", beantwortbar=False, erwartet=[])
    assert recall_at([_passage("x.pdf", 1, 1)], gold, 10) == 0.0


def test_goldset_file_parses() -> None:
    fragen = load_goldset(Path("goldset/downloads.yaml"))
    assert len(fragen) >= 20
    assert any(frage.beantwortbar for frage in fragen)
    assert any(not frage.beantwortbar for frage in fragen)


def test_table_has_one_row_per_stage() -> None:
    from arbiter.eval import Stufe

    text = format_table([Stufe(name="nur Fusion", recall=0.5, ndcg=0.4, fragen=2)])
    assert "nur Fusion" in text
    assert text.splitlines()[0].startswith("Stufe")
