# EVIDENCE — Pakete 3–5: strukturierte Lesung, Tabellen-Chunks, Prüfschicht

**Stand:** 2026-11-09
**SPEC:** Abschnitte 11.14 und 11.15 (Auftrag: „Führe das zuende“, nach Paket 1–2)
**Arbeitsverzeichnis:** `/home/speedy/Dokumente/dev/ArbiterRAG`

Dieser Bericht belegt je SPEC-Punkt den Test oder die Messung. Zahlen stammen aus einem
Lauf nach der letzten Codeänderung.

## 1. Was gebaut wurde

| Datei | Inhalt |
|---|---|
| `src/arbiter/pipeline.py` | `ocr_lesung` (reine Funktion), OCR-Haken indiziert die strukturierte Lesung, Bericht je Datei mit Tabellenzahlen, Fingerabdruck mit den Tabellenschlüsseln |
| `src/arbiter/loaders.py` | `OcrPage.warnungen`; Parser-Warnungen stehen im Ladeprotokoll |
| `src/arbiter/tabellen.py` | `tabellen_bloecke` (Positionen), `tabellen_chunks`, `maskiere_tabellen`; Zellumbruch im Markdown wird zu einem Leerzeichen |
| `src/arbiter/chunk.py` | Tabellen werden vor dem Prosachunker maskiert; Tabellen-Chunks tragen `start`/`ende`; Maskenreste werden entfernt |
| `src/arbiter/check.py` | `ist_tabellenzeile`; jede Tabellenzeile ist eine eigene Aussage, die Quelle der ersten Zeile gilt weiter |
| `src/arbiter/config.py` | `tabellen_kopf_zeilen`, `tabellen_max_zeilen`, `ocr_paddle_prompt` |
| `src/arbiter/store.py` | Upsert vergleicht auch die Strategie; gleicher Text mit alter leerer Strategie wird neu geschrieben |
| `src/arbiter/cli.py` | Berichtszeile nennt Tabellen und Zeilen-Chunks |
| `scripts/pruefe_tabellenpfad.py` | der Tabellenpfad mit den echten Modellen, ohne Zellinhalte auszugeben |
| `scripts/mutants.py` | ein nicht mehr anwendbarer Mutant ersetzt, sechs Mutanten für den Tabellenpfad |

## 2. Abnahme je SPEC-Punkt

| SPEC | Kriterium | Beleg |
|---|---|---|
| V1 | strukturierte Lesung wird indiziert, dots bleibt Vergleich | `tests/test_ocr_lesung.py::test_otsl_wird_die_indizierte_lesung`, `tests/test_ocr_lesung.py::test_ohne_tabelle_bleibt_der_dots_text` |
| V2 | OTSL-Form, unbekanntes Token erzeugt Warnung, kein stiller Verlust | `test_tabellen.py::test_unbekanntes_token_erzeugt_eine_warnung`, `test_ocr_lesung.py::test_unbekanntes_token_geht_nicht_verloren`, `test_ocr_loader.py::test_parser_warnungen_der_ocr_seite_stehen_im_bericht` |
| V3 | Kopf-Chunk plus eine Zeile je Chunk, mit Pfad und Kopfzeile | `test_tabellen.py::test_jede_zeile_ergibt_einen_chunk_mit_kopfzeile`, `test_tabellen.py::test_tabellen_chunks_kennt_die_zeilen_und_ihre_position` |
| V4 | kein Zeilen-Chunk ohne Kopfzeile | `test_tabellen.py::test_ohne_kopfzeile_wird_das_gesagt`, `test_tabellen.py::test_jede_zeile_ergibt_einen_chunk_mit_kopfzeile` |
| V5 | Zeilen-Chunk höchstens `chunk_chars`, breite Zeile geteilt | `test_tabellen.py::test_breite_zeile_wird_mit_kopfzeile_geteilt`; echter Lauf: längster Chunk 571 Zeichen |
| V6 | kein Schnitt in eine Tabelle | `test_chunk_datei.py::test_struktur_chunks_schneiden_nicht_in_eine_tabelle`, `test_chunk_datei.py::test_prosachunk_traegt_keine_maskenreste` |
| V7 | Tabellenzeile gilt als Aussage | `test_antwort_pruefung.py::test_tabellenzeile_ist_eine_eigene_aussage`, `test_antwort_pruefung.py::test_quelle_der_ersten_zeile_gilt_fuer_die_folgenden_zeilen` |
| V8 | Bericht nennt Tabellen und Zeilen-Chunks | `test_index_je_datei.py::test_bericht_nennt_tabellen_und_zeilen_chunks`, `test_chunk_datei.py::test_der_bericht_zaehlt_tabellen_und_zeilen_chunks` |
| V9 | ohne Paddle-Ausgabe bleibt alles wie heute | `test_ocr_lesung.py::test_ohne_tabelle_bleibt_der_dots_text`, `test_ocr_loader.py::test_without_hook_the_page_stays_skipped` |
| D39 | eine Zeile je Chunk, Kopfzeile wiederholt | `test_tabellen.py::test_jede_zeile_ergibt_einen_chunk_mit_kopfzeile` |
| D40 | verbundene Zelle wiederholt den Wert, erfindet nichts | `test_tabellen.py::test_verbundene_zelle_wird_rechts_wiederholt`, `test_tabellen.py::test_leere_zelle_bleibt_leer_und_erfindet_nichts` |
| D41 | Format am Text erkannt, nicht an der Konfiguration | `test_tabellen.py::test_ohne_token_gibt_es_keine_tabelle`, `test_tabellen.py::test_otsl_ergibt_kopf_und_zeilen` |

Zusätzlich: `test_tabellen.py::test_zellinhalt_mit_zeilenumbruch_bleibt_eine_zeile` belegt
einen Fehler, den erst der echte Lauf zeigte — ein Zellinhalt mit Umbruch zerlegte seine
Zeile beim Schreiben des Markdowns.

## 3. Gauntlet

| Ebene | Befehl | Ergebnis |
|---|---|---|
| Testsuite | `uv run pytest` | 252 Tests, alle grün |
| Lint | `uv run ruff check .` | keine Meldung |
| Typen | `uv run mypy` | keine Meldung in 25 Dateien |
| Abdeckung | `uv run pytest --cov=arbiter --cov-report=term-missing` | gesamt 87 %, `tabellen.py` 96 %, `chunk.py` 99 %, `check.py` 88 %, `store.py` 92 % |
| Mutation | `uv run python scripts/mutants.py` | 39/39 Mutanten getötet |

Die sechs neuen Mutanten greifen: Zeile ohne Kopfzeile, Chunk ohne Position,
Obergrenze ignoriert, Tabelle nicht aus dem Prosatext entfernt, OCR-Lesung ignoriert
die Tabelle, Aussagen verschmelzen Tabellenzeilen. Der frühere Mutant
`entscheider-schwelle-ignoriert` zeigte auf Code, den Abschnitt 11.12 entfernt hatte;
er ist durch `entscheider-erkennt-eine-ueberschrift-als-struktur` ersetzt.

## 4. Echte Läufe

### 4.1 Tabellenpfad mit den echten Modellen

`scripts/pruefe_tabellenpfad.py` auf dem Bescheid-Bild, 38,6 s:

| Größe | Wert |
|---|---|
| OTSL-Zeichen | 1859 |
| Spalten × Zeilen | 5 × 12 |
| indizierte Lesung | Markdown, 3939 Zeichen |
| dots-Vergleichslesung | 1873 Zeichen |
| Kopf-Chunks | 1 (Kopfzeile plus 3 Zeilen) |
| Zeilen-Chunks | 12 (drei breite Zeilen an Zellgrenzen geteilt) |
| Zeilen-Chunk-Längen | 227 bis 571 Zeichen |
| über `chunk_chars` (600) | 0 |
| Warnungen | 0 |

Damit ist die Kette Prompt → OTSL → Tabelle → Zeilen-Chunks mit dem laufenden
`paddleocr-vl` belegt, nicht nur mit Testdaten.

### 4.2 Index neu gebaut

`uv run arbiter index downloads`, 6 min 54 s (eine textlose Seite, deren OCR-Aufruf in
einen Zeitüberschreitung lief):

```
5 Dateien gelesen, 1643 Abschnitte, 1642 neu oder geändert, 1 Seiten ohne Text,
1 Seiten per OCR, 3 Dateien aus dem Index entfernt, 1 Tabellen mit 0 Zeilen-Chunks
  PATCHT-Trennblatt.pdf: fenster (...), 1 Abschnitte, 1 Tabellen, 0 Zeilen-Chunks
  ix.26.13.pdf: struktur (Median-Abschnitt 3735 Zeichen), 1309 Abschnitte
  ppt/...BI-Entwickler...: struktur (766 Zeichen), 51 Abschnitte
  ppt/...nicht in den Betrieb...: struktur (1377 Zeichen), 51 Abschnitte
  ppt/Workshop_KI-Inferenz: struktur (250 Zeichen), 231 Abschnitte
```

Der echte Index trägt jetzt 1643 Zeilen: 1641 `struktur`, 1 `fenster`, 1 Zeile des
Trennblatts (Tabellen-Kopf-Chunk aus der OCR-Lesung). Vorher waren es 1610 Zeilen aus
dem alten Chunking.

### 4.3 Messung Recall@10

`scripts/messe_chunking.py`, dichte Suche, gleiche Goldfragen, drei Strategien:

| Strategie | Dateien | Abschnitte | Index | Recall@10 |
|---|---|---|---|---|
| fenster | fenster:5 | 1511 | 943 919 | 0,538 |
| struktur | struktur:5 | 1643 | 860 844 | **0,692** |
| auto | fenster:1, struktur:4 | 1643 | 860 844 | **0,692** |

Der Detektor wählt dieselbe Aufteilung wie in Abschnitt 11.12. Der Wert 0,692 gilt
damit auch für den echten Index, nicht nur für die Messung. Die Indexgröße weicht um
5 Zeichen von der früheren Zeile ab; Ursache ist der Tabellen-Kopf-Chunk des
Trennblatts.

## 5. Grenzen

- **Der Korpus misst den Tabellenpfad nicht.** Nur das Trennblatt hat keine Textebene,
  und es trägt keine Datentabelle. Die Zahlen aus 4.1 kommen von einem Bild außerhalb
  des Korpus. Eine Recall-Zahl für Tabellenzeilen gibt es deshalb nicht.
- **Das Tabellenmodell antwortet unbedingt.** Auf dem Trennblatt ohne Tabelle lieferte
  `Table Recognition:` eine Tabelle mit Kopfzeile und null Zeilen. Die Seite wird als
  ein Kopf-Chunk indiziert. D38 (Tabellenerkennung) bleibt offen.
- **Eine Tabelle über mehrere Seiten ist eine Einheit.** Alle Zeilen-Chunks tragen die
  Seite des Tabellenanfangs. Bei einem Seitenwechsel kann die zitierte Seite um eins
  abweichen. Im Korpus kommt der Fall nicht vor.
- **`qes` bleibt unverändert.** Die Strategie maskiert Tabellen nicht und ist nur über
  `--strategie qes` erreichbar; `auto` wählt sie nie.
- **Digitale Tabellen werden nicht zerlegt.** Ein PDF mit Textebene behält seinen Text;
  Zeilen-Chunks entstehen nur aus der OCR-Lesung.

## 6. Reproduktion

```bash
uv run pytest
uv run ruff check .
uv run mypy
uv run python scripts/mutants.py

set -a && . ./arbiter.env && set +a
uv run arbiter index downloads
uv run python scripts/messe_chunking.py --db "$TMPDIR/arbiter-messung" \
    --strategien fenster,struktur,auto
uv run python scripts/pruefe_tabellenpfad.py /pfad/zu/seite.png
```
