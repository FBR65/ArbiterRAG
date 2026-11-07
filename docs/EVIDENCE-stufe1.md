# ArbiterRAG — Stufe 1: Nachweis

Datum: 2026-11-07 · Stand: `be28e83` plus Doku-Commit · Tier 2 (normales Projekt)

Freigabe: SPEC rev. 2, erteilt 2026-11-07 mit den Antworten D1–D9.

## 1. Abnahmekriterien, einzeln abgehakt

| Kriterium | Test | Ergebnis |
|---|---|---|
| A1 Pflichtfelder, Meldung mit Feld und Datei | `test_config.py::test_missing_required_field_names_field_and_file` | grün |
| A2 `${VAR}` fehlt oder leer → Abbruch | `test_unset_variable_aborts_with_its_name`, `test_empty_variable_is_an_error_not_a_default` | grün |
| A3 unbekannter Schlüssel ist ein Fehler | `test_unknown_key_is_rejected`, `test_config_unknown_backend_key` (test_edges) | grün |
| A4 Modell-IDs aus der Konfiguration | `test_model_ids_come_from_config_not_code` | grün |
| B1 Endungen, rekursiv | `test_reads_configured_extensions_recursively` | grün |
| B2 PDF seitenweise, leere Seite gezählt | `test_pdf_pages_become_separate_documents`, `test_pdf_page_without_text_layer_is_counted_not_fatal` | grün |
| B3 Felder `id/text/quelle/seite/hash/vector` | `test_upsert_and_count`, `test_dense_search_orders_by_similarity` | grün |
| B4 unverändert nicht doppelt, geändert ersetzt | `test_resame_id_replaces_instead_of_duplicating`, `test_reindex_does_not_duplicate` | grün |
| B5 Warnung je Datei, Lauf geht weiter | `test_empty_file_is_skipped_with_warning`, `test_broken_pdf_is_skipped_not_fatal` | grün |
| B6 Abschnitte ≤ 1200 Zeichen, 100 Überlappung | `test_chunk.py` (6 Tests + Property) | grün |
| B7 Suche ohne Volltextindex ist ein Fehler | `test_search_without_fts_index_is_an_error` | grün |
| C1 dichte Suche, Kosinus, absteigend | `test_dense_search_orders_by_similarity` | grün |
| C2 Volltextsuche nach BM25 | `test_full_text_search_finds_german_term` | grün |
| C3 RRF k=60, Gleichstand nach id | `test_fusion.py` (4 Tests + Property) | grün |
| C4 Abschnitt auf `fuse_top_k` | `test_top_k_truncates_after_sorting` | grün |
| H1 `index` meldet Zahlen | `test_index_reports_files_and_chunks` | grün |
| H2 `search` druckt Quelle, Seite, RRF | `test_search_prints_source_and_rank` | grün |
| I1 unbekannter RAG → Exit 2 | `test_unknown_rag_is_exit_2_and_lists_names` | grün |
| I2 leerer Korpus → Exit 1 | `test_empty_corpus_is_exit_1` | grün |
| I3 zu lange Frage → Abbruch ohne Modellaufruf | `test_overlong_question_is_refused_without_calling_the_embedder` | grün |
| N5 kein Netz in Unit-Tests | alle 55 Tests laufen ohne Endpunkt | grün |
| N7 keine echten Daten in Fixtures | Platzhaltertexte, eigener Probe-Korpus unter `/tmp` | grün |

Nicht in Stufe 1 und deshalb ohne Test: D (HyDE), E (Reranker), F (Laya), G (Antwort), H3/H4.
Der Code hat dafür keine stillen Ausweichpfade — `SearchResult.rerank` steht fest auf `"aus"`.

## 2. Gauntlet

Ein frischer Lauf nach der letzten Codeänderung:

```
$ uv run pytest -q                                  55 passed
$ uv run pytest -q --cov=arbiter --cov-report=term   92 % (500 Zeilen, 39 nicht ausgeführt)
$ uv run ruff check .                                All checks passed
$ uv run ruff format --check .                       clean
$ uv run mypy                                        Success: no issues found in 9 source files
$ uv run python scripts/mutants.py                   7/7 Mutanten getötet
```

Abdeckung je Modul: `chunk` 100 %, `fusion` 100 %, `pipeline` 100 %, `cli` 93 %,
`store` 93 %, `config` 89 %, `loaders` 84 %, `embedding` 84 %.

Nicht abgedeckte Zeilen sind Fehlerpfade, die ein echter Endpunkt auslösen müsste
(Netzwerkabbruch mitten im Request, YAML-Syntaxfehler) und der Zweig
`Keine Treffer` in der CLI. Alle drei stehen hier, statt als grün gezählt zu werden.

Mutanten (jeder einzeln eingebaut, Suite laufen lassen, aus Git zurückgeholt):

| Mutant | getötet von |
|---|---|
| Abschnittsfenster ein Zeichen zu groß | Property-Test in `test_chunk.py` |
| RRF-Rang bei 0 beginnend | `test_documents_in_both_lists_wins` |
| Gleichstand nicht nach id sortiert | `test_equal_scores_sort_by_id_ascending` |
| geänderte Abschnitte nicht gelöscht | `test_resame_id_replaces_instead_of_duplicating` |
| leere Umgebungsvariable akzeptiert | `test_empty_variable_is_an_error_not_a_default` |
| unbekannter RAG endet mit 0 | `test_unknown_rag_is_exit_2_and_lists_names` |
| Längenprüfung der Frage abgeschaltet | `test_overlong_question_is_refused_without_calling_the_embedder` |

Wiederholbar aus dem Repo: `scripts/mutants.py` (setzt einen sauberen Arbeitsbaum voraus).

## 3. Echter Lauf

Ausgeführt gegen den laufenden llama-swap (`bge-m3-gguf`, echte Embeddings) und einen
synthetischen PDF-Korpus unter `/tmp/arbiter-probe` mit drei Dateien, fünf Seiten.

```
$ uv run arbiter index probe
3 Dateien, 5 Abschnitte, 5 neu oder geändert, 0 übersprungen, 0 Seiten ohne Text

$ uv run arbiter search probe "Wie hoch ist die Miete?"
Rang  Quelle                           Seite      RRF  Auszug
   1  mietvertrag.pdf                      1   0.0328  Mietvertrag Haus Lindenweg 4. Die monatliche Miete betraegt 940 Euro z
   2  mietvertrag.pdf                      2   0.0323  Die Kaution betraegt drei Nettokaltmieten und ist in drei Raten faelli
   3  nebenkosten.pdf                      1   0.0159  Nebenkostenabrechnung 2025. Heizung und Warmwasser 612 Euro, Muellgebu
   4  nebenkosten.pdf                      2   0.0156  Der Umlageschluessel folgt der Wohnflaeche. Nachzahlung 183 Euro, fael
   5  protokoll.pdf                      1   0.0154  Eigentuemerversammlung Protokoll. Beschluss: Das Dach des Hauses Linde
```

Damit ist die Kette PDF → Abschnitte → dichte Vektoren → LanceDB → Volltextsuche → RRF
mit echten Modellantworten belegt.

**Was dieser Lauf nicht belegt:** die Qualität auf Franks Dokumenten. Der Korpus war
synthetisch, weil D12 (Pfad) noch offen ist. Recall und Rangfolge auf echten Miet-
und Steuerunterlagen sind ungemessen.

## 4. Abweichungen und Befunde

1. **`pypdf` als Abhängigkeit** — Folge von D2 = PDF. Steht in der SPEC, Abschnitt 6.
2. **`types-PyYAML` als Entwicklungsabhängigkeit** — mypy verlangt Stubs. Kein Laufzeitpaket.
3. **Vektorlänge wird beim ersten Schreibvorgang festgelegt**, nicht in der Konfiguration
   (zuerst vorgesehen als Konstante 1024). Grund: die Länge kommt vom Modell; eine
   Konstante hätte bei einem anderen Embedder den ersten Lauf mit einem Cast-Fehler beendet.
4. **`create_fts_index` ist in LanceDB 0.39 veraltet** (`create_index(config=FTS())` ist der
   Nachfolger). Es funktioniert und ist getestet; die Warnung bleibt sichtbar, statt sie
   zu unterdrücken.
5. **Volltextindex muss nach jedem Schreibvorgang neu gebaut werden.** Neue Zeilen sind
   vorher nicht auffindbar. Die Pipeline macht das einmal am Ende von `index`; der Test
   hält diese Reihenfolge fest.
6. **Ein Test wurde präzisiert, nicht abgeschwächt:** der Test für „geändert ersetzt" baute
   seinen Hash aus der ID statt aus dem Text und konnte die Ersetzung deshalb nicht
   auslösen. Jetzt ist der Hash ein SHA-256 des Textes — das ist die Zusage aus B4.

## 5. Nachtrag Stufe 1b — OCR für Seiten ohne Textebene (D10=b, D13=a, D14=a)

Freigabe 2026-11-07. Neue Abhängigkeiten: `pypdfium2` (BSD-3/Apache-2.0, rendert die
Seite) und `pillow` (HPND, kodiert PNG).

**Achtung, D14 nicht wie geplant erfüllt:** `~/Dokumente/dev/U-OCR_test/` existiert
nicht mehr. Der Prompt stammt deshalb nicht aus deinem Projekt, sondern aus der
offiziellen Model Card von `dots-studio/dots.mocr` — dort steht derselbe
JSON-Auftrag mit Kategorien und Lesereihenfolge. Er steht wörtlich in
`src/arbiter/ocr.py` und ist nicht umformuliert.

| Kriterium | Test | Ergebnis |
|---|---|---|
| B8 Seite ohne Text → beide Modelle, dots-Text indiziert | `test_textless_page_goes_through_the_ocr_hook` | grün |
| B8 Layout-JSON wird Lesereihenfolge, Bilder ohne Text | `test_dots_json_becomes_reading_order_text`, `test_dots_json_wrapped_in_code_fence_is_read` | grün |
| B8 Antwort ohne JSON wird durchgereicht, nicht verworfen | `test_non_json_answer_is_passed_through_unchanged` | grün |
| B8 Wortüberlappung | `test_word_overlap_bounds`, `test_one_shared_word_out_of_four_is_a_quarter_not_a_match` | grün |
| B8 Rendern ergibt PNG, unbekannte Seite wird abgelehnt | `test_render_produces_png`, `test_render_rejects_unknown_page` | grün |
| B8 Aufruf ist OpenAI-kompatibel (Bild als data-URI, Text danach) | `test_vision_client_sends_data_uri_and_reads_answer` | grün |
| B9 OCR-Fehler → Warnung, Seite bleibt übersprungen | `test_failing_ocr_is_a_warning_not_a_crash`, `test_vision_client_reports_http_error` | grün |
| B8 Seiten **mit** Textebene lösen kein OCR aus | `test_pages_with_text_layer_do_not_use_the_hook` | grün |
| B10 alte Tabelle wird migriert | `test_store_migrates_a_table_without_ocr_columns` | grün |

Echter Lauf (derselbe Korpus, `ocr: true`):

```
10 Dateien, 1054 Abschnitte, 1 neu oder geändert, 0 übersprungen, 0 Seiten ohne Text, 1 Seiten per OCR
```

Die Seite `PATCHT-Trennblatt.pdf` ist damit im Index. Beide Modelle lesen sie
gleich; Überlappung 1.0:

```
dots  : TRENNBLATT Dieses Blatt trennt Dokumente. Nach diesem Blatt beginnt ein neues Dokument.
paddle: TRENNBLATT Dieses Blatt trennt Dokumente. Nach diesem Blatt beginnt ein neues Dokument.
```

**Gemessene Grenze:** `dots-mocr` liest bei 100, 150 und 200 dpi identisch; bei 250
und 300 dpi antwortet der Server `HTTP 500 failed to process mtmd chunk`. Default
ist deshalb 200, nicht 300. Messskript: `scripts/probe_ocr_dpi.py`.

**Zwei Befunde aus dem Lauf:**

1. Der erste Versuch bei 300 dpi scheiterte an dieser Grenze — die OCR-Grenze war
   nicht die Auflösung, sondern das Bildmaß. Ohne Messung wäre 300 dpi im Code
   gelandet.
2. Der Schreibvorgang brach mit `'ocr' does not exist in table schema` ab, weil die
   Tabelle vor den neuen Spalten entstand. Behoben durch `add_columns` beim Öffnen
   (B10) statt durch Neuaufbau des Index.

**Gauntlet nach dem Nachtrag** (frischer Lauf, `6d1a5c2`):

```
uv run pytest -q                                    71 passed
uv run pytest -q --cov=arbiter                      90 % (630 Zeilen, 60 nicht ausgeführt)
uv run ruff check . / format --check                clean / clean
uv run mypy                                         keine Befunde in 10 Dateien
uv run python scripts/mutants.py                    10/10 Mutanten getötet
```

Der Mutantenlauf hat eine echte Lücke gefunden: `word_overlap` mit `min()` im Nenner
überlebte, weil beide Testpaare gleich viele Wörter hatten. Erst der zusätzliche
Fall „ein gemeinsames Wort von vier" tötet ihn. Der Test ist damit schärfer, nicht
weicher.

## 6. Offen

| ID | Frage |
|---|---|
| D11 | erledigt: Reranker läuft (`/v1/rerank`, Logits, nicht kalibriert) |
| D12 | erledigt: `/home/speedy/Downloads`, 10 Dateien, 1054 Abschnitte |
| D14 | Abweichung: Quellprojekt weg, Prompt aus der Model Card — bestätigen oder korrigieren |
