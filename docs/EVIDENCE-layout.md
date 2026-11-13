# EVIDENCE — Layout-Erkennung mit DocLayout-YOLO (SPEC 11.16)

**Stand:** 2026-11-09
**SPEC:** Abschnitt 11.16 (Freigabe durch Frank am 2026-11-09: „Freigabe“, nach der
Entscheidung D44 „Qualität ist die Maßgabe“)
**Arbeitsverzeichnis:** `/home/speedy/Dokumente/dev/ArbiterRAG`

Dieser Bericht belegt je SPEC-Punkt den Test oder die Messung. Zahlen stammen aus einem
Lauf nach der letzten Codeänderung.

## 1. Was gebaut wurde

| Datei | Inhalt |
|---|---|
| `src/arbiter/layout.py` | Letterbox, Dekodierung der Modellausgabe, Seitenanteile, Schwelle, Mindestfläche, Zusammenfassen überlappender Tabellen, Textentfernung je Zeile, ONNX-Sitzung |
| `src/arbiter/loaders.py` | `text_im_kasten`: Text im Kasten über `get_text_bounded`, Umrechnung Seitenanteile → PDF-Punkte |
| `src/arbiter/pipeline.py` | `Tabellenfund`, Haken je Seite, Ausschnitt, `paddleocr-vl`, `tabellen_chunks`, Seitenzuordnung, Textentfernung, Bericht je Datei, Fingerabdruck |
| `src/arbiter/config.py` | sieben Schlüssel `layout*` |
| `src/arbiter/cli.py` | Berichtszeile nennt Tabellen, Seiten und Zeilen-Chunks |
| `scripts/hole_layout_modell.py` | Download der Apache-2.0-Gewichte mit Größe und SHA-256 |
| `scripts/pruefe_zeilenkaesten.py` | Paket 1: welche Textkoordinaten `pypdfium2` liefert |
| `scripts/messe_layout.py` | Zensus und Recall@10 mit Layout aus/an |
| `scripts/mutants.py` | neun Mutanten für Layout und Pipeline |

## 2. Ergebnis Paket 1

Eine Seite aus `ix.26.13.pdf` (S. 145), Kasten auf dem Anzeigenkasten:

| Größe | Wert |
|---|---|
| Zeichen / Rechtecke | 4525 / 553 |
| Y-Bänder (Zeilen) | 64 |
| Text im Kasten | 257 Zeichen, 7 Zeilen |
| davon im Seitentext gefunden | **7 von 7** |

`pypdfium2` 5.15.0 liefert `get_text_bounded`, `get_rect` (Wortkästen) und
`get_charbox`. Der Text im Kasten steht wortgleich im Seitentext. Damit ist der Weg aus
D44 festgelegt: Kasteninhalt lesen, Zeilen im Seitentext entfernen. Die Wahl zwischen
Zeilen- und Zeichenkästen entfällt; die SPEC hält die Revision fest.

## 3. Abnahme je SPEC-Punkt

| SPEC | Kriterium | Beleg |
|---|---|---|
| W1 | `dekodiere` ist rein und liefert Seitenanteile in [0, 1] | `test_layout.py::test_dekodiere_rechnet_auf_seitenanteile_zurueck`, `test_dekodiere_begrenzt_kaesten_auf_die_seite` |
| W2 | Apache-2.0-Gewichte, Datei außerhalb des Repos, SHA-256 | `scripts/hole_layout_modell.py`; SHA-256 `fece9af0…5b21`, 75 324 598 Bytes |
| W3 | nur `table`, Schwelle, Mindestfläche, keine Doppelboxen | `test_dekodiere_verwirft_geringe_konfidenz`, `test_dekodiere_verwirft_kleine_flaechen`, `test_dekodiere_kennt_alle_klassen_und_filtert_tabellen`, `test_ueberlappende_tabellen_werden_zusammengefasst` |
| W4 | Ausschnitt = Kasten plus Rand, 300 dpi | `test_layout_fund_erzeugt_zeilen_chunks_und_entfernt_den_text` (Kasten im Fund), Echtlauf |
| W5 | Ausschnitt → OTSL → Zeilen-Chunks mit Seite; sonst Warnung | `test_layout_fund_erzeugt_zeilen_chunks_und_entfernt_den_text`; Fehlerfall in `_tabelle_aus_region` |
| W6 | Tabellentext fällt aus dem Seitentext, fehlende Zeile wird gezählt | `test_entferne_tabellenzeilen_*` (6 Tests), `test_fehlende_zeile_wird_gemeldet_statt_still_entfernt` |
| W7 | Bericht nennt Seiten, Tabellen, Zeilen-Chunks | `test_layout_fund_erzeugt_zeilen_chunks_und_entfernt_den_text`, `test_layout_pipeline.py` |
| W8 | Seite ohne Textebene: dots bleibt, kein Ganzseiten-Prompt | Zensus: 0 Tabellen auf dem Trennblatt (vorher 1 Fehlalarm) |
| W9 | `layout: false` → alles wie heute | `test_ohne_layout_wird_der_haken_nicht_gefragt` |
| W10 | Schlüssel, Fingerabdruck, Fail-soft | `test_layout_schluessel_werden_gelesen`, `test_layout_ohne_modell_ist_eine_warnung`, `test_jedes_feld_der_rag_klasse_ist_ein_erlaubter_schluessel` |
| D47 | Erkennungsfehler hält den Ingest nicht an | `test_erkennung_fehler_haelt_den_ingest_nicht_an` |

## 4. Gauntlet

| Ebene | Befehl | Ergebnis |
|---|---|---|
| Testsuite | `uv run pytest` | 278 Tests, alle grün |
| Lint | `uv run ruff check .` | keine Meldung |
| Typen | `uv run mypy` | keine Meldung in 26 Dateien |
| Mutation | `uv run python scripts/mutants.py` | 48/48 Mutanten getötet |

Neue Mutanten: Schwelle ignoriert, Fläche ignoriert, Kasten nicht begrenzt, Bildkanal
nicht gedreht, Zeilen nicht entfernt, Position immer null, Chunk ohne Seite,
Erkennungsfehler ignoriert, Tabellentext nicht entfernt.

## 5. Echte Läufe

### 5.1 Zensus über den Korpus

Alle 292 Seiten der fünf indizierten Dateien, Erkennung bei 200 dpi:

| Datei | Seiten | Seiten mit Tabelle | Tabellenregionen | ms/Seite |
|---|---|---|---|---|
| `PATCHT-Trennblatt.pdf` | 1 | 0 | 0 | 703 |
| `ix.26.13.pdf` | 156 | 16 | 19 | 905 |
| `ppt/…BI-Entwickler…` | 10 | 0 | 0 | 855 |
| `ppt/…nicht in den Betrieb…` | 9 | 0 | 0 | 842 |
| `ppt/Workshop_KI-Inferenz` | 116 | 11 | 12 | 853 |
| **Summe** | **292** | **27** | **30** | |

Roh erkannte der Detektor 32 Regionen; zwei überlappten eine bessere Box und fallen
zusammen (S. 57 und S. 105 der Zeitschrift).

### 5.2 Stichprobe von Hand

Drei Seiten visuell geprüft:

| Seite | Befund |
|---|---|
| `ix.26.13.pdf` S. 58 | echte Tabelle „Testkalieng von Riechstoffen und Ihre Teststufe“, 6 Spalten × 6 Zeilen |
| `Workshop_KI-Inferenz` S. 26 | echte Tabelle „Quantisierung“, 6 Spalten × 6 Zeilen (Format, Bytes/Par., Qualität, …) |
| `ix.26.13.pdf` S. 105 | echte Tabelle „Verschiedene Benutzerkonfigurationen“, 4 Spalten × 3 Zeilen, doppelt erkannt (0,41 und 0,25) |

Kein Fehlalarm in der Stichprobe. Der Fehlalarm aus 11.14 (Trennblatt) ist weg: dort
werden jetzt 0 Tabellen erkannt.

### 5.3 Index und Recall

`scripts/messe_layout.py --recall`, dichte Suche, gleiches Goldset:

| Layout | Abschnitte | Index (Zeichen) | Recall@10 |
|---|---|---|---|
| aus | 1643 | 860 844 | **0,692** |
| an | 1767 | 891 785 | **0,692** |

Layout an erzeugt **124 zusätzliche Chunks** (Kopf- und Zeilen-Chunks aus 30 Tabellen)
und 3,6 % mehr Index. Der Recall bleibt unverändert; die Goldfragen zielen auf Fließtext,
nicht auf Tabellen. Der Gewinn liegt in der Genauigkeit der Belegstelle, nicht in dieser
Kennzahl. Eine Recall-Zahl für Tabellenzeilen gibt es nicht, weil das Goldset keine
Tabellenfragen enthält.

### 5.4 Dauer

| Lauf | Dauer |
|---|---|
| Index ohne Layout | rund 7 min |
| Index mit Layout | rund 27 min |

Die Differenz entsteht bei den 30 Tabellenausschnitten: Ausschnitte aus Foliensätzen
sind bei 300 dpi mehrere Tausend Pixel breit, und `paddleocr-vl` braucht dafür deutlich
länger als für eine Textseite. Die Erkennung selbst kostet 0,85–0,91 s je Seite.

## 6. Grenzen

- **Keine Messung der Erkennungsgüte über den ganzen Korpus.** Die Stichprobe umfasst
  drei Seiten. Fehlalarme außerhalb der Stichprobe sind möglich und nicht beziffert.
- **Gedrehte oder verzerrte Seiten** sind nicht abgedeckt.
- **Keine Lesereihenfolge aus Kästen.** Der Text kommt weiter aus der Textebene.
- **Der Ingest wird teurer.** Rund 27 statt 7 Minuten für 292 Seiten, vor allem durch
  große Ausschnitte bei 300 dpi. Eine Obergrenze für die Ausschnittgröße wäre der
  nächste Schritt.
- **Eine Tabelle über mehrere Seiten** bleibt eine Einheit; alle Zeilen tragen die
  Startseite.
- **Der Standard bleibt `false`** (D48). Ohne Messung auf einem weiteren Korpus ist
  `true` eine Behauptung.

## 7. Reproduktion

```bash
uv run pytest
uv run ruff check .
uv run mypy
uv run python scripts/mutants.py

uv run python scripts/hole_layout_modell.py
set -a && . ./arbiter.env && set +a
uv run python scripts/messe_layout.py
uv run python scripts/messe_layout.py --recall
```
