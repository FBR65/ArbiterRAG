"""Ohne exportierte Variablen startet das Werkzeug trotzdem (arbiter.env)."""

from __future__ import annotations

import os
from pathlib import Path

from arbiter.runtime import lade_umgebung


def test_values_come_from_the_file(tmp_path: Path, monkeypatch) -> None:
    env_datei = tmp_path / "arbiter.env"
    env_datei.write_text(
        "# Kommentar\nARBITER_DB=/tmp/db\nARBITER_BASE_URL=http://127.0.0.1:8080\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("ARBITER_DB", raising=False)
    monkeypatch.delenv("ARBITER_BASE_URL", raising=False)
    monkeypatch.delenv("ARBITER_CONFIG", raising=False)
    monkeypatch.setenv("ARBITER_ENV", str(env_datei))

    umgebung = lade_umgebung()
    assert umgebung["ARBITER_DB"] == "/tmp/db"
    assert umgebung["ARBITER_BASE_URL"] == "http://127.0.0.1:8080"
    assert umgebung["ARBITER_CONFIG"] == str(tmp_path / "arbiter.yaml")


def test_an_exported_value_wins(tmp_path: Path, monkeypatch) -> None:
    env_datei = tmp_path / "arbiter.env"
    env_datei.write_text("ARBITER_DB=/tmp/aus-der-datei\n", encoding="utf-8")
    monkeypatch.setenv("ARBITER_ENV", str(env_datei))
    monkeypatch.setenv("ARBITER_DB", "/tmp/aus-der-shell")

    assert lade_umgebung()["ARBITER_DB"] == "/tmp/aus-der-shell"


def test_a_missing_file_is_not_an_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ARBITER_ENV", str(tmp_path / "gibt-es-nicht.env"))
    monkeypatch.setenv("ARBITER_DB", "/tmp/db")
    assert lade_umgebung()["ARBITER_DB"] == "/tmp/db"
    assert os.environ["ARBITER_DB"] == "/tmp/db"
