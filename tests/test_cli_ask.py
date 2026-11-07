"""Spec H3: `arbiter ask` output and exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arbiter.answer import Quelle
from arbiter.cli import main
from arbiter.config import Config, Rag, load_config
from arbiter.pipeline import Antwort, Entscheidung
from tests.test_config import MINIMAL, _env, _write


class FakePipeline:
    def __init__(self, rag: Rag, antwort: Antwort) -> None:
        self.rag = rag
        self._antwort = antwort

    def ask(self, question: str) -> Antwort:
        return self._antwort


def _config(tmp_path: Path) -> Config:
    return load_config(_write(tmp_path, MINIMAL), _env())


def _antwort(text: str | None, grund: str | None = None, fehler: bool = False) -> Antwort:
    quellen = (
        [Quelle(passage_id="a.pdf#3#1", quelle="a.pdf", seite=3, text="Miete 940 Euro")]
        if text
        else []
    )
    return Antwort(
        text=text,
        grund=grund,
        quellen=quellen,
        entscheidung=Entscheidung(
            beantwortbar=text is not None,
            grund=grund,
            zulassungen=[],
            auswahl=[],
            paare=[],
        ),
        fehler=fehler,
    )


def test_answer_is_printed_with_its_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _config(tmp_path)
    factory = lambda _cfg, rag: FakePipeline(rag, _antwort("Die Miete betraegt 940 Euro."))  # noqa: E731
    assert main(["ask", "miethaus", "Miete?"], config=cfg, factory=factory) == 0
    out = capsys.readouterr().out
    assert "940 Euro" in out
    assert "Quelle: a.pdf, S. 3" in out


def test_json_output_is_parseable(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _config(tmp_path)
    factory = lambda _cfg, rag: FakePipeline(rag, _antwort("Antwort."))  # noqa: E731
    assert main(["ask", "miethaus", "Miete?", "--json"], config=cfg, factory=factory) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["antwort"] == "Antwort."
    assert payload["quellen"][0]["quelle"] == "a.pdf"
    assert payload["entscheidung"]["beantwortbar"] is True


def test_missing_evidence_exits_zero_and_explains(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _config(tmp_path)
    factory = lambda _cfg, rag: FakePipeline(rag, _antwort(None, "keine belegte Passage"))  # noqa: E731
    assert main(["ask", "miethaus", "Miete?"], config=cfg, factory=factory) == 0
    assert "keine belegte Passage" in capsys.readouterr().err


def test_failed_run_exits_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _config(tmp_path)
    factory = lambda _cfg, rag: FakePipeline(  # noqa: E731
        rag, _antwort(None, "Prüfschicht nicht erreichbar", fehler=True)
    )
    assert main(["ask", "miethaus", "Miete?"], config=cfg, factory=factory) == 1
    assert "Prüfschicht" in capsys.readouterr().err
