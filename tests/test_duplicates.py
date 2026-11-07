"""Spec B3: a document that sits twice in the folder is indexed once."""

from __future__ import annotations

from pathlib import Path

from arbiter.loaders import load_corpus


def test_identical_file_in_a_subfolder_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "unter").mkdir()
    inhalt = "Derselbe Text steht zweimal im Ordner.\n\n" * 20
    (tmp_path / "artikel.txt").write_text(inhalt, encoding="utf-8")
    (tmp_path / "unter" / "artikel.txt").write_text(inhalt, encoding="utf-8")

    result = load_corpus(tmp_path, (".txt",))
    assert len(result.documents) == 1
    assert result.duplicates == 1
    assert any("Dublette" in warning for warning in result.warnings)


def test_different_files_are_both_kept(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("Der erste Text.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Der zweite Text.", encoding="utf-8")
    result = load_corpus(tmp_path, (".txt",))
    assert len(result.documents) == 2
    assert result.duplicates == 0
