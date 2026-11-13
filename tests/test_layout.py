"""SPEC 11.16: Layout-Erkennung (Dekodierung, Briefkasten, Textentfernung).

Die Tests brauchen kein Modell. Sie arbeiten mit festen Vorhersagen und
synthetischem Text; das echte Modell prüft der Messlauf.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from arbiter.layout import (
    Letterbox,
    Region,
    dekodiere,
    eindeutige_tabellen,
    entferne_tabellenzeilen,
    letterbox,
    tabellen_regionen,
)


def _bild(breite: int, hoehe: int, farbe: tuple[int, int, int] = (255, 0, 0)) -> Image.Image:
    return Image.new("RGB", (breite, hoehe), farbe)


def test_letterbox_haelt_das_seitenverhaeltnis() -> None:
    batch, box = letterbox(_bild(200, 100), 100)
    assert batch.shape == (1, 3, 100, 100)
    assert box.gain == pytest.approx(0.5)
    assert box.rand_x == 0
    assert box.rand_y == 25


def test_letterbox_fuellt_mit_grau_und_dreht_auf_bgr() -> None:
    batch, _ = letterbox(_bild(200, 100, (255, 0, 0)), 100)
    # Oben ist Rand (114/255 in allen Kanälen), in der Mitte steht das rote Bild als BGR.
    assert batch[0, :, 0, 50].tolist() == pytest.approx([114 / 255] * 3)
    assert batch[0, :, 60, 50].tolist() == pytest.approx([0.0, 0.0, 1.0])


def _vorhersage(*boxen: tuple[float, float, float, float, float, float]) -> np.ndarray:
    return np.array([list(boxen)], dtype=np.float32)


def test_dekodiere_rechnet_auf_seitenanteile_zurueck() -> None:
    # Seite 200x100, Briefkasten 100, gain 0.5, Rand oben 25.
    box = Letterbox(groesse=100, gain=0.5, rand_x=0, rand_y=25)
    (region,) = dekodiere(
        _vorhersage((50, 50, 100, 75, 0.9, 5)),
        seite=(200, 100),
        box=box,
        schwelle=0.25,
        min_flaeche=0.01,
    )
    assert region.klasse == "table"
    assert region.konfidenz == pytest.approx(0.9)
    assert region.kasten == pytest.approx((0.5, 0.5, 1.0, 1.0))


def test_dekodiere_verwirft_geringe_konfidenz() -> None:
    box = Letterbox(groesse=100, gain=1.0, rand_x=0, rand_y=0)
    assert (
        dekodiere(
            _vorhersage((10, 10, 50, 50, 0.2, 5)),
            seite=(100, 100),
            box=box,
            schwelle=0.25,
            min_flaeche=0.0,
        )
        == []
    )


def test_dekodiere_verwirft_kleine_flaechen() -> None:
    box = Letterbox(groesse=100, gain=1.0, rand_x=0, rand_y=0)
    assert (
        dekodiere(
            _vorhersage((10, 10, 12, 12, 0.9, 5)),
            seite=(100, 100),
            box=box,
            schwelle=0.25,
            min_flaeche=0.01,
        )
        == []
    )


def test_dekodiere_begrenzt_kaesten_auf_die_seite() -> None:
    box = Letterbox(groesse=100, gain=1.0, rand_x=0, rand_y=0)
    (region,) = dekodiere(
        _vorhersage((-20, -20, 50, 50, 0.9, 5)),
        seite=(100, 100),
        box=box,
        schwelle=0.25,
        min_flaeche=0.0,
    )
    assert region.kasten == (0.0, 0.0, 0.5, 0.5)


def test_dekodiere_kennt_alle_klassen_und_filtert_tabellen() -> None:
    box = Letterbox(groesse=100, gain=1.0, rand_x=0, rand_y=0)
    regionen = dekodiere(
        _vorhersage((10, 10, 50, 50, 0.9, 1), (50, 50, 90, 90, 0.8, 5)),
        seite=(100, 100),
        box=box,
        schwelle=0.25,
        min_flaeche=0.0,
    )
    assert [region.klasse for region in regionen] == ["plain text", "table"]
    assert [region.klasse for region in tabellen_regionen(regionen)] == ["table"]


def test_leere_vorhersage_ergibt_keine_region() -> None:
    box = Letterbox(groesse=100, gain=1.0, rand_x=0, rand_y=0)
    assert dekodiere(np.zeros((1, 0, 6)), (100, 100), box, 0.25, 0.0) == []


def test_ueberlappende_tabellen_werden_zusammengefasst() -> None:
    # Dieselbe Tabelle zweimal gefunden: nur die sichere Box bleibt.
    sicher = Region(klasse="table", konfidenz=0.9, kasten=(0.1, 0.1, 0.9, 0.5))
    doppelt = Region(klasse="table", konfidenz=0.4, kasten=(0.1, 0.2, 0.9, 0.5))
    assert eindeutige_tabellen([sicher, doppelt]) == [sicher]


def test_getrennte_tabellen_bleiben_beide() -> None:
    oben = Region(klasse="table", konfidenz=0.9, kasten=(0.1, 0.1, 0.9, 0.4))
    unten = Region(klasse="table", konfidenz=0.8, kasten=(0.1, 0.6, 0.9, 0.9))
    assert eindeutige_tabellen([oben, unten]) == [oben, unten]


TEXT = "Kopf\nMiete 940 Euro\nStrom 80 Euro\nRest des Textes.\n"


def test_entferne_tabellenzeilen_trifft_nur_die_tabelle() -> None:
    befund = entferne_tabellenzeilen(TEXT, ["Kopf", "Miete 940 Euro", "Strom 80 Euro"])
    assert befund.text == "Rest des Textes.\n"
    assert befund.entfernt == 3
    assert befund.fehlend == 0
    assert befund.position == 0


def test_entferne_tabellenzeilen_zaehlt_fehlende_zeilen() -> None:
    befund = entferne_tabellenzeilen(TEXT, ["Kopf", "fehlt"])
    assert befund.text == "Miete 940 Euro\nStrom 80 Euro\nRest des Textes.\n"
    assert befund.entfernt == 1
    assert befund.fehlend == 1


def test_entferne_tabellenzeilen_ohne_treffer_aendert_nichts() -> None:
    befund = entferne_tabellenzeilen(TEXT, ["nicht vorhanden"])
    assert befund.text == TEXT
    assert befund.entfernt == 0
    assert befund.fehlend == 1
    assert befund.position == -1


def test_entferne_tabellenzeilen_zaehlt_gleiche_zeilen_einzeln() -> None:
    text = "a\nb\na\n"
    befund = entferne_tabellenzeilen(text, ["a", "a"])
    assert befund.text == "b\n"
    assert befund.entfernt == 2
    assert befund.fehlend == 0


def test_position_zeigt_auf_die_stelle_im_neuen_text() -> None:
    text = "Einleitung\nKopf\nMiete 940\nSchluss\n"
    befund = entferne_tabellenzeilen(text, ["Kopf", "Miete 940"])
    assert befund.text == "Einleitung\nSchluss\n"
    assert befund.position == len("Einleitung\n")


def test_entferne_tabellenzeilen_ignoriert_leerraum_und_groesse() -> None:
    befund = entferne_tabellenzeilen("  Miete   940 Euro  \nRest\n", ["miete 940 euro"])
    assert befund.text == "Rest\n"
    assert befund.entfernt == 1


def test_text_im_kasten_liest_die_seite(tmp_path: Path) -> None:
    import pypdfium2 as pdfium

    from arbiter.loaders import text_im_kasten
    from tests.test_loaders import _make_text_pdf

    pfad = _make_text_pdf(tmp_path / "a.pdf", ["Einleitung\nMiete 940 Euro\nSchluss"])
    pdf = pdfium.PdfDocument(str(pfad))
    try:
        text = text_im_kasten(pdf, 1, (0.05, 0.13, 0.99, 0.20))
    finally:
        pdf.close()
    assert "Miete 940 Euro" in text
    assert "Einleitung" in text


def test_datei_sha256_nennt_die_modellversion(tmp_path: Path) -> None:
    from arbiter.pipeline import datei_sha256

    datei = tmp_path / "modell.onnx"
    datei.write_bytes(b"abc")
    assert datei_sha256(datei) == hashlib.sha256(b"abc").hexdigest()
