# Verbesserungen A–E: was gemessen wurde und was es gebracht hat

Datum: 2026-11-07 · Korpus: `/home/speedy/Downloads`, 7 Dateien nach Dublettenprüfung
· Goldset: 31 Fragen, davon 23 beantwortbar

Jede Zeile ist ein echter Lauf gegen die laufenden Modelle. Alle Zahlen sind
Recall@10 / nDCG@10 über das ganze Goldset; „Beleg ok" heißt: die Antwort nennt
die Quelle, in der die Antwort tatsächlich steht.

## A — Messung zuerst

| Schritt | Änderung | Wirkung |
|---|---|---|
| A1 | Gold-Etiketten gegen den Text geprüft | **4 von 23 Etiketten waren falsch**: die Dokumentkennung lautete `BI im KI-Zeitalter`, im Dateinamen steht `Business Intelligence im KI-Zeitalter`. Diese vier Fragen konnten nie treffen. Jetzt lösen alle 23 auf eine Seite mit der Antwort auf. |
| A2 | Alle Stufen mit derselben Ausgabelänge (10) | Vorher wurde der Reranker auf fünf Passagen gemessen, die Fusion auf zehn. Sein Nachteil war zum Teil Messfehler. |
| A3 | Dubletten übersprungen, verschwundene Dateien aus dem Index entfernt | Drei PDFs lagen zusätzlich unter `ppt/`. Der Index fiel von 1054 auf 878 Zeilen; die Kopien belegten vorher doppelt Ränge. |

Recall@10 nach A: **0,48 → 0,58**.

## B — Suche

| Schritt | Änderung | Recall@10 | nDCG@10 |
|---|---|---|---|
| vorher | 1200 Zeichen, kein Kopf, hybrid | 0,58 | 0,55 |
| B1 | 600 Zeichen, Kopf `[Datei, S. n]` | 0,65 | 0,61 |
| B2+B3 | Volltext aus, Stemming an | **0,71** | **0,65** |

Der Volltextteil schadet: dichte Suche allein 0,71, mit Volltext 0,65. Ursache ist
die Fußzeile `SVA System Vertrieb Alexander GmbH` auf jeder Folie des
Workshop-Foliensatzes — sie trifft bei fast jeder Frage und verdrängt echte
Treffer. Stemming bringt im Hybridbetrieb +0,08 nDCG, ohne Volltext nichts.

Konfiguration: `chunk_chars: 600`, `hybrid: false`, `fts_stem: true`.

## C — HyDE

| Variante | Recall@10 | nDCG@10 |
|---|---|---|
| ohne HyDE | 0,71 | 0,65 |
| Pseudo-Absatz (alt) | 0,39 | 0,39 |
| Begriffskatalog, Gewicht 0,5 | 0,71 | 0,58 |
| Begriffskatalog, Gewicht 1,0 | 0,61 | 0,37 |

Der umgebaute Prompt (C1: „nenne Fachbegriffe, Zahlen, Eigennamen" statt „schreibe
den Abschnitt") nimmt HyDE den Schaden am Recall, bringt aber keinen Gewinn: die
Ordnung wird schlechter, weil die zusätzlichen Listen die Reihenfolge verwässern.
Herabgewichten (C2) half gegen den Recall-Verlust, nicht gegen den nDCG-Verlust.

**Entscheidung:** HyDE bleibt im Code und in der Konfiguration abschaltbar, ist
für diesen Korpus aber aus (`hyde: false`). Ein Nutzen war nicht messbar.

## E — Entscheidungsschicht

| Gate | Antworten | Beleg ok | falsch | Anmerkung |
|---|---|---|---|---|
| `aus` (nur Reranker) | 31 | **19** | 8 | Vergleichsgröße ohne Modellaufruf |
| `noul` je Passage, Schwelle 0,5 | 29 | 10 | 8 | sättigt: jede Passage bekommt bis 1,000 |
| `choice` über alle Kandidaten | 10 | 2 | 21 | lehnt fast alles ab |

Kalibrierung über alle 31 Fragen, `noul`: die Gold-Passage erreicht hohe Werte,
aber jede andere Passage auch. Der Schwellen-Sweep von 0,3 bis 0,9 lehnt **0 von 8**
nicht beantwortbaren Fragen ab — es gibt keine trennende Schwelle.

`choice` mit absoluter Schwelle lehnt alles ab (elf Optionen verteilen die
Wahrscheinlichkeit); mit Vorsprung vor `keine` (`gate_marge`) werden 10 Fragen
beantwortet, davon 2 richtig.

**Ergebnis:** Die Entscheidungsschicht in dieser Form entfernt gute Antworten und
keine schlechten. `gate_modus: aus` ist deshalb der Default; `noul` und `choice`
bleiben getestet im Code, damit die Messung wiederholbar ist.

## E2 — andere Rolle: Prüfer der fertigen Antwort

Getestet am 2026-11-07: Laya bekommt nicht die Frage mit einem Kandidaten, sondern
den **Auszug mit der fertigen Antwort** und beantwortet „Stützt Auszug [1] die
Antwort?".

Verteilung über alle 31 Fragen:

| Gruppe | n | min | Median | max |
|---|---|---|---|---|
| Antwort mit richtigem Beleg | 20 | 0,604 | **0,987** | 1,000 |
| Antwort ohne Beleg / auf nicht beantwortbare Frage | 11 | 0,532 | **0,874** | 0,991 |

Beide Gruppen liegen fast übereinander. Der Schwellen-Sweep zeigt den Tausch:

| Schwelle | richtige Antworten behalten | falsche abgelehnt |
|---|---|---|
| 0,60 | 20/20 | 2/11 |
| 0,90 | 16/20 | 6/11 |
| 0,95 | 12/20 | 8/11 |

Im Goldset-Lauf mit `pruefung: true`, Schwelle 0,9: **16 richtige Belege (statt 19),
9 falsch (statt 8)**. Die Prüfung nimmt also drei gute Antworten weg und keine
schlechte. Der Tausch aus dem Sweep überträgt sich nicht auf die Kette.

**Ergebnis:** Die Rolle „Prüfer der Antwort" trennt besser als das Tor, aber nicht
gut genug: die Verteilungen überlappen, und im Betrieb kostet sie richtige
Antworten. `pruefung: false` bleibt der Default; der Code und die Messung bleiben
erhalten.

## E3 — drittes Modell: Qwen über Logprobs (gefunden)

`qwen2.5-7b-instruct` antwortet mit **einem** Token auf „Beantwortet der Text die
Frage?"; `top_logprobs` liefert die Alternativen, daraus wird die Wahrscheinlichkeit
für `ja` gebildet (`ja_wahrscheinlichkeit`). Ein Chat-Modell ist nicht kalibriert —
deshalb der Prompt-Vergleich, alle drei auf demselben Goldset:

| Prompt | P(ja) beantwortbar (Median) | P(ja) nicht beantwortbar (max) | beantwortet bei Schwelle 0,9 |
|---|---|---|---|
| „streng" (ganze Antwort muss im Text stehen) | 0,997 | 0,000 | 14/23, 0/8 falsch |
| **„teil" (Angaben, die zur Antwort nötig sind)** | **1,000** | **0,091** | **21/23, 0/8 falsch** |
| „beleg" (enthält die Information) | 0,986 | 0,001 | 14/23, 0/8 falsch |

Der Prompt entscheidet, nicht das Modell. Mit „teil" trennt die Verteilung sauber:
nicht beantwortbare Fragen erreichen höchstens 0,091, beantwortbare im Median 1,000.
Der Schwellenwert ist zwischen 0,3 und 0,9 beliebig — es gibt einen echten Graben.

Kette mit `gate_modus: logprob`, `gate_prompt: teil`, Schwelle 0,5:

| Gate | Antworten | Beleg ok | falsch |
|---|---|---|---|
| `aus` | 31 | 19 | **8** |
| `noul` (Laya) | 29 | 10 | 8 |
| `choice` (Laya) | 10 | 2 | 21 |
| `logprob`, Prompt „streng" | 15 | 10 | 8 |
| **`logprob`, Prompt „teil"** | **22** | **14** | **1** |

**Ergebnis:** Das ist der erste Gate-Modus, der die gestellte Aufgabe löst. Er
kostet fünf richtige Antworten (19 → 14), nimmt aber sieben von acht Falschantworten
weg (8 → 1). Für ein System, das nicht lügen soll, ist das der bessere Tausch —
und es ist der einzige, den die Messung stützt. Default ist deshalb `logprob`.

## E4 — Kombination Qwen-Gate und Laya (gemessen, verworfen)

Zwei Modelle auf **verschiedene** Rollen verteilt: Qwen entscheidet über die
Zulassung, Laya vergleicht die zugelassenen Passagen paarweise (Redundanz,
Widerspruch). `kombi: true`, Gate bleibt `logprob`.

| Einstellung | Antworten | Beleg ok | falsch |
|---|---|---|---|
| nur Qwen-Gate | 22 | **15** | 1 |
| + Laya, Redundanz ab 3,0 | 22 | 10 | 1 |
| + Laya, nur Widerspruch ab 0,8 | 22 | 11 | 1 |

Beide Kombinationen kosten richtige Antworten, keine Falschantwort geht zusätzlich
weg. Laya verwirft Passagen als redundant oder widersprüchlich, die den Beleg
tragen. `kombi: false` bleibt.

Nebenbefund: Zwei Läufe derselben Konfiguration ergaben 14 und 15 richtige Belege.
Die Kette ist also auf ±1 nicht reproduzierbar; Unterschiede dieser Größe sind
nicht belegbar.

## Endstand der Kette

| Stufe | Recall@10 | nDCG@10 | Beleg ok (von 23) |
|---|---|---|---|
| nur Fusion | 0,71 | 0,65 | – |
| mit HyDE | 0,71 | 0,58 | – |
| mit Reranker | 0,71 | 0,66 | – |
| mit Laya (`aus`) | 0,71 | 0,66 | 19 |
| mit Laya (`noul`) | 0,71 | 0,66 | 10 |

## Gauntlet (frischer Lauf)

```
uv run pytest -q        140 passed
uv run ruff check .     All checks passed
uv run mypy             keine Befunde in 17 Dateien
```

Mutation siehe `scripts/mutants.py`; die neuen Mutanten für Chunk-Kopf,
Dublettenprüfung und Gate-Marge stehen dort.

## Was offen bleibt

1. **Kein trennendes Gate.** Mit laya-multilingual ist die Entscheidungsschicht
   nicht tragfähig. Entweder ein anderes Entscheidungsmodell, oder die Rolle
   ändern (Laya nur für Redundanz und Widerspruch, nicht für die Annahme).
2. **Boilerplate im Korpus.** Die Fußzeile der Folien ist der Grund, warum die
   Volltextsuche schadet. Eine Wiederholungserkennung je Dokument würde den
   Hybridbetrieb wieder sinnvoll machen.
3. **Goldset zu klein.** 31 Fragen, ±0,09 Unsicherheit. Unterschiede unter 0,1
   sind nicht belegbar.
4. **Antworten ohne Gate zitieren zu 83 % die richtige Quelle, aber beantworten
   auch die 8 nicht beantwortbaren Fragen.** Das ist der eigentliche offene
   Punkt — und der einzige, für den die Entscheidungsschicht gedacht war.
