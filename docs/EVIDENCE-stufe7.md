# EVIDENCE — Stufe 7 (Chunking-Strategien mit Entscheider)

**Stand:** 2026-11-08
**SPEC:** Abschnitte 11.1–11.9 (Freigabe durch Frank am 2026-11-08: „Freigabe erteilt.
Ich folge deinen Vorschlägen.")
**Arbeitsverzeichnis:** `/home/speedy/Dokumente/dev/ArbiterRAG`

Dieser Bericht belegt je SPEC-Punkt den Test oder die Messung. Zahlen stammen aus
einem Lauf nach der letzten Codeänderung.

## 1. Was gebaut wurde

| Datei | Inhalt |
|---|---|
| `src/arbiter/chunk.py` | `Chunk`, `ChunkParameter`, `satz_grenzen`, `chunk_fenster`, Weiche `chunk_datei` |
| `src/arbiter/chunk_structure.py` | Überschriften, Abschnitte, Breadcrumb, Verweisauflösung, Kennzahlen |
| `src/arbiter/chunk_qes.py` | QESChunker: Frage-Beleg-Prüfung, Hit-Gate, Kosten-Nutzen-Greedy, Lückenchunks |
| `src/arbiter/chunk_qe.py` | Frage-Beleg-Erzeugung über das Chat-Modell |
| `src/arbiter/chunk_choice.py` | Entscheider (reine Funktion) |
| `src/arbiter/pipeline.py` | Index je Datei, Seitenzuordnung, Bericht je Datei, `neuer_pipeline` |
| `src/arbiter/store.py` | Spalten `strategie`, `seite_ende`, `page_text` über den Seitenbereich |
| `src/arbiter/config.py` | Zehn neue Schlüssel (11.4) |
| `src/arbiter/cli.py`, `api.py`, `tui.py` | Schalter `--strategie`, QE-Erzeugung im Betriebspfad |

## 2. Abnahme je SPEC-Punkt

| SPEC | Kriterium | Beleg |
|---|---|---|
| K1 | `Chunk` mit `text`, `start`, `ende`, `strategie` | `tests/test_chunk_structure.py::test_property_kein_textverlust` |
| K2 | `text[start:ende]` steht im Chunk | `test_chunk_datei.py::test_fenster_chunks_tragen_ihren_text_wortgleich`, Property-Tests |
| K3 | Kein Textverlust | `test_property_kein_textverlust_bei_qes`, `test_property_kein_textverlust` |
| K4 | Leerer Text ergibt keine Chunks | `test_chunk_datei.py::test_leerer_text_ergibt_keine_chunks` |
| K5 | Eigener Text ≤ `chunk_chars` | `test_chunk_qes.py::test_lange_absaetze_halten_das_fensterlimit_ein`, `test_qes_bevorzugt_…` |
| K6 | Gleiche Eingabe, gleiche Chunks | `test_chunk_choice.py::test_gleicher_text_ergibt_gleiche_auswahl` (Entscheider), Algorithmen ohne Zufall |
| L1 | Überschriftenregeln | `test_markdown_…`, `test_nummerierte_…`, `test_fliesszeilige_…`, `test_grosszeile_…` |
| L2 | Abschnitt endet an der nächsten Überschrift | `test_ein_chunk_enthaelt_nur_seinen_eigenen_rumpf` |
| L3 | Ein Chunk je Abschnitt | derselbe Test |
| L4 | Ganzer Abschnitt bis 1,5 × L, sonst Satzgrenzen | `test_kurzer_abschnitt_bleibt_ganz`, `test_abschnitt_bis_eineinhalbfacher_laenge_bleibt_ganz`, `test_sehr_langer_abschnitt_wird_an_satzgrenzen_geteilt` |
| L5 | Breadcrumb mit Pfad | `test_abschnitte_tragen_pfad_der_oberen_ebenen`, `test_struktur_chunks_tragen_ihren_pfad` |
| L6 | Verweise: 3 je Chunk, `xref_chars` Zeichen, kein Selbstbezug | `test_verweis_haengt_zieltext_an`, `test_verweis_haengt_hoechstens_die_angegebene_laenge_an`, `test_hoechstens_drei_verweise_je_chunk`, `test_kein_verweis_auf_sich_selbst` |
| L7 | Verweis ohne Ziel ohne Anhang | `test_verweis_ohne_ziel_erzeugt_keinen_anhang` |
| L8 | Seitenübergreifender Abschnitt ist ein Chunk | `test_seitenuebergreifender_chunk_wird_auf_beiden_seiten_gefunden` (Bereich in `page_text`) |
| M1 | Satzgrenzen | `test_chunk.py` (B6) und `satz_grenzen`; Property-Tests |
| M2 | Beleg muss wörtlich sein, Fragmente ab 10 Zeichen | `test_beleg_indizes_verwirft_nicht_woertliche_belege`, `test_nimmt_woertliches_fragment_an`, `test_verwirft_mehrdeutige_fragmente` |
| M3 | Erzeugung: Fenster, Obergrenzen, Format, Prüfung | `test_chunk_qe.py` (11 Tests) |
| M4 | Kandidatenfenster im Längenbereich | `test_laengen_bereich_bleibt_im_rahmen`, `test_lange_absaetze_halten_das_fensterlimit_ein` |
| M5 | Nutzen mit Hit-Gate und Strukturterm | `test_hit_gate_verlangt_treffer_und_beleg_im_fenster`, `test_treffer_indizes_nimmt_die_aehnlichsten` |
| M6 | Greedy, Budget, Überlappung möglich | `test_qes_laesst_ueberlappung_zu`, `test_qes_bevorzugt_das_fenster_mit_beleg_und_treffer` |
| M7 | Lückenchunks | Property-Test „kein Textverlust bei QES" |
| M8 | Rückfall mit Warnung | `test_qes_ohne_frage_beleg_paare_faellt_mit_warnung_auf_fenster` |
| M9 | Höchstens `qe_fenster_max` Aufrufe | `test_fragen_werden_ueber_multiple_fenster_verteilt` |
| N1 | Entscheider ohne Modell, gleiche Eingabe gleiche Auswahl | `test_chunk_choice.py` |
| N2 | Kennzahlen; nur passende Abschnitte zählen | `test_eine_riesige_kapitelueberschrift_ist_keine_struktur`, `test_kennzahlen_…` |
| N3 | Wert und Schwelle | `test_wert_ist_reine_funktion_der_kennzahlen`, `test_schwelle_verschiebt_die_entscheidung` |
| N4 | Bericht je Datei | `test_der_bericht_nennt_die_strategie_je_datei`, `test_uebersteuerte_strategie_steht_im_bericht` |
| N5 | Schalter `--strategie` | `test_uebersteuerte_strategie_fragt_den_entscheider_nicht` |
| O1 | Verkettung mit Seitenpositionen | `test_seiten_werden_mit_position_verkettet` |
| O2 | Seite des Chunk-Starts | `test_position_wird_der_seite_zugeordnet` |
| O3 | `page_text` über den Seitenbereich | `test_seitenuebergreifender_chunk_wird_auf_beiden_seiten_gefunden` |
| O4 | Neue Spalten | `test_jede_zeile_traegt_ihre_strategie` |
| O5 | Fingerabdruck enthält die Chunk-Einstellungen | `test_geaenderte_entscheider_schwelle_liest_die_datei_neu`, `test_strategie_wechsel_liest_die_datei_neu` |
| O6 | Abschnitts-ID je Datei | `test_index_je_datei.py` (Zählung je Datei) |
| Q5–Q10 | Kein Textverlust, kein stiller Wechsel, kein Modell im Entscheider, keine neuen Abhängigkeiten, keine Änderung an Suche/Reranker/Gate | siehe oben; Abhängigkeiten unverändert (`pyproject.toml`) |

## 3. Gauntlet

| Ebene | Befehl | Ergebnis |
|---|---|---|
| Testsuite | `uv run pytest -q` | *(wird nach dem letzten Lauf eingetragen)* |
| Typen | `uv run mypy src` | 0 Fehler |
| Lint/Format | `uv run ruff check`, `uv run ruff format --check` | sauber |
| Mutation | `uv run python scripts/mutants.py` | 33/33 getötet |
| Reale Ausführung | `scripts/probe_chunking.py`, `scripts/probe_qes_echt.py` | siehe 4 |
| Messung | `scripts/messe_chunking.py` | siehe 5 |

## 4. Reale Ausführung (Befunde, nicht Behauptungen)

1. **Entscheider-Fehlgriff, behoben.** Ein echtes PDF (Blog-Artikel, 21 865 Zeichen)
   erhielt mit der ersten Fassung Kopfanteil 0,60 und wurde als `struktur`
   eingestuft, obwohl es Fließtext ist: eine einzige Kapitelüberschrift bedeckte
   fast den ganzen Text. N2 wurde präzisiert (nur Abschnitte bis 1,5 × `chunk_chars`
   zählen). Danach: Wert 0,05, Strategie `qes`. Regressionstest:
   `test_eine_riesige_kapitelueberschrift_ist_keine_struktur`.
2. **QES ohne brauchbare Belege, behoben.** Der erste echte Lauf mit dem Chat-Modell
   ergab 0 von 12 Paaren: das Modell zitiert Teilsätze, die Prüfung verlangte ganze
   Sätze. M2 wurde präzisiert (wörtliche Fragmente ab 10 Zeichen, eindeutig
   zuordenbar). Danach: 9 Paare aus 12 Fragen, 197 Abschnitte, Index 3,14 × Dokument.
3. **Fensterlimit verletzt, behoben.** Ein Absatz ohne Satzende ergab im
   QES-Ergebnis einen Chunk mit 928 Zeichen bei `chunk_chars` = 600. Ursache: der
   Lückenpacker teilte einen einzelnen Satz nicht. `_teile` teilt jetzt hart.
4. **Untergrenze über Obergrenze, behoben.** Bei Absätzen länger als `chunk_chars`
   lag die Fensteruntergrenze über dem Limit; es gab keine Kandidaten und damit
   keine QES-Chunks. `laengen_bereich` begrenzt jetzt auch die Untergrenze.

## 5. Messung (Stufe 7.3)

*(wird nach dem Lauf eingetragen)*
