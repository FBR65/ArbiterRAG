"""Die TUI fragt, zeigt Antwort und Quellen und schaltet den Suchmodus."""

from __future__ import annotations

import pytest
from textual.widgets import Input, Label, Static

from arbiter.pipeline import Antwort, Passage, Quelle, SearchResult
from arbiter.tui import ArbiterTui, quellen_text, treffer_text


class FakeRag:
    name = "k"
    hyde = False
    rerank = True
    gate_prompt = "teil"
    gate_kandidaten = 5


class FakeStore:
    def count(self, rag: str) -> int:
        return 42


class FakePipeline:
    """Ersetzt die Pipeline: die Oberfläche darf keine Modellaufrufe machen."""

    rag = FakeRag()
    store = FakeStore()

    def ask(self, frage: str) -> Antwort:
        return Antwort(
            text="Die Miete betraegt 940 Euro [a.pdf, S. 3].",
            grund=None,
            quellen=[Quelle(passage_id="a#3#1", quelle="a.pdf", seite=3, text="Miete 940 Euro")],
            warnungen=["Hinweis zur Prüfung"],
        )

    def search(self, frage: str) -> SearchResult:
        return SearchResult(
            question=frage,
            passages=[
                Passage(
                    id="a#3#1",
                    text="Miete 940 Euro",
                    quelle="a.pdf",
                    seite=3,
                    rrf_rank=1,
                    rrf_score=0.5,
                    rerank_score=-0.97,
                )
            ],
        )


def test_quellen_text_has_file_and_page() -> None:
    antwort = FakePipeline().ask("Wie hoch ist die Miete?")
    text = quellen_text(antwort)
    assert "a.pdf" in text and "S. 3" in text


def test_treffer_text_shows_score_and_rank() -> None:
    text = treffer_text(FakePipeline().search("Miete"))
    assert "1." in text
    assert "-0.97" in text


@pytest.mark.asyncio
async def test_question_is_shown_with_answer_and_source() -> None:
    app = ArbiterTui(FakePipeline())  # type: ignore[arg-type]
    async with app.run_test() as pilot:
        await pilot.press("enter")  # leeres Feld: nichts passiert
        app.query_one("#eingabe", Input).value = "Wie hoch ist die Miete?"
        await pilot.press("enter")
        await pilot.pause()
        for _ in range(50):
            if "940 Euro" in str([w.content for w in app.query(Label)]):
                break
            await pilot.pause(0.05)
        texte = [str(w.content) for w in app.query(Label)]
        assert any("Wie hoch ist die Miete?" in t for t in texte)
        assert any("940 Euro" in t and "a.pdf" in t for t in texte)


@pytest.mark.asyncio
async def test_search_mode_switches() -> None:
    app = ArbiterTui(FakePipeline())  # type: ignore[arg-type]
    async with app.run_test() as pilot:
        await pilot.press("ctrl+s")
        assert app.suchmodus is True
        await pilot.press("ctrl+s")
        assert app.suchmodus is False


@pytest.mark.asyncio
async def test_status_line_names_the_rag() -> None:
    app = ArbiterTui(FakePipeline())  # type: ignore[arg-type]
    async with app.run_test():
        stand = app.query_one("#stand", Static)
        assert "k" in str(stand.content)
        assert "42" in str(stand.content)
