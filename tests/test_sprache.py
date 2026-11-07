"""Spec: the corpus holds documents in the wanted language only."""

from __future__ import annotations

from pathlib import Path

from arbiter.loaders import detect_language, load_corpus


def test_german_and_english_are_told_apart() -> None:
    deutsch = "Der Verarbeitungslauf wird mit einem Zeitplan gestartet, die Datei ist nicht lesbar."
    englisch = "The processing run is started by a schedule and the file is not readable."
    assert detect_language(deutsch) == "de"
    assert detect_language(englisch) == "en"
    assert detect_language("1234 ---") == "unbekannt"


def test_foreign_documents_are_skipped_with_a_warning(tmp_path: Path) -> None:
    (tmp_path / "deutsch.txt").write_text(
        "Der Bericht beschreibt die Miete und die Nebenkosten, die nicht enthalten sind.",
        encoding="utf-8",
    )
    (tmp_path / "english.txt").write_text(
        "The report describes the rent and the additional costs which are not included.",
        encoding="utf-8",
    )
    geladen = load_corpus(tmp_path, (".txt",), sprache="de")
    assert [doc.quelle for doc in geladen.documents] == ["deutsch.txt"]
    assert any("fremdsprachig" in warnung for warnung in geladen.warnings)


def test_without_a_language_both_stay(tmp_path: Path) -> None:
    (tmp_path / "deutsch.txt").write_text("Die Miete ist nicht enthalten.", encoding="utf-8")
    (tmp_path / "english.txt").write_text("The rent is not included.", encoding="utf-8")
    geladen = load_corpus(tmp_path, (".txt",))
    assert len(geladen.documents) == 2
