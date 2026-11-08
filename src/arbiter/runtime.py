"""Umgebung für die Befehle: ``arbiter.env`` lesen, wenn nichts exportiert ist.

Der Dienst bekommt seine Variablen von systemd. Wer ``arbiter`` oder ``arbiter-tui``
in einer frischen Shell startet, hat sie nicht — und bekommt eine Fehlermeldung statt
einer Oberfläche. Diese Datei schließt die Lücke: fehlende Variablen kommen aus
``arbiter.env`` (Pfad über ``ARBITER_ENV`` änderbar), vorhandene gewinnen immer.
"""

from __future__ import annotations

import os
from pathlib import Path

STANDARD_ENV = "arbiter.env"


def lade_umgebung(start: Path | None = None) -> dict[str, str]:
    """Return the environment with values from ``arbiter.env`` filled in.

    Already exported variables stay untouched: an explicit ``ARBITER_DB=…`` on the
    command line must win over the file.
    """
    umgebung = dict(os.environ)
    pfad = Path(umgebung.get("ARBITER_ENV") or STANDARD_ENV)
    if not pfad.is_absolute():
        basis = start or Path.cwd()
        pfad = basis / pfad
        if not pfad.is_file():
            pfad = Path(__file__).resolve().parents[2] / STANDARD_ENV
    if not pfad.is_file():
        return umgebung

    for zeile in pfad.read_text(encoding="utf-8").splitlines():
        zeile = zeile.strip()
        if not zeile or zeile.startswith("#") or "=" not in zeile:
            continue
        name, wert = zeile.split("=", 1)
        name = name.strip()
        wert = wert.strip().strip('"').strip("'")
        if name and name not in umgebung:
            umgebung[name] = wert
    umgebung.setdefault("ARBITER_CONFIG", str(pfad.parent / "arbiter.yaml"))
    os.environ.setdefault("ARBITER_ENV", str(pfad))
    return umgebung
