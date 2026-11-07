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

## F — Laya entfernt, fünf Kandidaten, HyDE und Embedder verglichen

Auf Anweisung: Laya ist aus dem Code entfernt (Gate je Passage, Auswahl über alle
Kandidaten, Prüfer der Antwort, Paarvergleiche). Die Prüfschicht läuft nur noch über
Qwen-Logprobs, auf den **fünf** besten Passagen statt auf allen.

Gemessen mit `scripts/measure_chain.py`, ganze Kette bis zur Antwort, 31 Fragen,
23 davon beantwortbar. Jede Zeile ein vollständiger Lauf.

| Embedder | HyDE | Antworten | Beleg ok | falsch | verworfen |
|---|---|---|---|---|---|
| bge-m3 | aus | 21 | 15 | 6 | 10 |
| bge-m3 | an | 21 | **16** | **5** | 10 |
| nomic-embed-text-v2-moe | aus | 18 | 14 | 4 | 13 |
| nomic-embed-text-v2-moe | an | 18 | 14 | 4 | 13 |

**bge-m3 ist besser:** zwei bis drei Antworten mehr, ein bis zwei richtige Belege
mehr. nomic verwirft mehr Fragen, obwohl es weniger falsche Antworten hat — es
findet die Belegstelle seltener.

**HyDE ändert bei bge-m3 eine Antwort** (15 → 16 richtig, 6 → 5 falsch), bei nomic
nichts. Ein Unterschied von eins liegt in der Schwankung, die ich am 2026-11-07
zwischen zwei gleichen Läufen gesehen habe. HyDE bleibt deshalb aus; der Gewinn ist
nicht belegbar.

Zwei Einschränkungen, die dazugehören:

1. Der nomic-Index hat **eine Seite weniger**, weil die OCR-Seite beim Indexieren
   nicht erkannt wurde (1720 statt 1721 Abschnitte). nomic ist damit leicht
   benachteiligt, der Abstand ist aber größer als eine Seite.
2. Der frühere Wert „22 Antworten, 15 richtig, 1 falsch" stammt aus einer Datenbank
   mit gemischten Abschnittsgrößen (1200 und 600 Zeichen). Nach dem Neuaufbau mit
   einheitlich 600 Zeichen sind es 21/15/6. Die alte Zahl war nicht belastbar.

Für den nomic-Lauf musste `nomic-embed-text-v2-moe` in llama-swap um `--embeddings
--pooling mean` ergänzt werden; ohne das Flag antwortet der Server 501.

## G — Drei oder fünf Passagen für die Antwort?

Die Prüfung sieht fünf Kandidaten, die Antwort zitiert seit jeher bis zu drei. Bringt
das fünfte dem Antwortschritt etwas? `scripts/measure_answer_passages.py`.

Verteilung der zugelassenen Passagen je Frage (31 Fragen):

| zugelassen | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Fragen | 10 | 4 | 5 | 5 | 2 | 5 |

Für **7 von 31** Fragen greift die Grenze von drei überhaupt. Für diese sieben ist
der Hebel wirksam: mit fünf zitiert die Antwort vier bis fünf Quellen statt drei,
der Text ist in 6 von 7 Fällen ein anderer (181 bis 948 Zeichen Unterschied).

| zitiert | Antworten | Beleg ok | falsch |
|---|---|---|---|
| bis 3 | 21 | 16 | 5 |
| bis 5 | 21 | 16 | 5 |

**Ergebnis:** kein Unterschied. Die zusätzlichen Passagen werden zitiert, tragen aber
in keinem Fall den Beleg — die richtige Quelle war schon unter den ersten drei. Der
Unterschied, den ich messen kann, ist null; ein möglicher Gewinn an Vollständigkeit
bei mehrteiligen Fragen liegt außerhalb dessen, was das Goldset prüft (es kennt nur
die Zielseiten, nicht ob die Antwort vollständig ist).

**Entscheidung:** Default bleibt drei. Fünf kosten Kontext und Antwortzeit ohne
belegten Nutzen. Wer vollständigere Antworten will, muss das mit einem Goldset
messen, das Vollständigkeit bewertet, nicht nur die Quelle.

## H — Ziel: null Halluzinationen

Das Ziel ist nicht „selten falsch", sondern **keine falsche Antwort**. Damit reicht das
Gate nicht: es entscheidet nur, welche Passage als Beleg zugelassen wird. Was das
Antwortmodell daraus schreibt, ist eine zweite Fehlerquelle. Deshalb wird jetzt jeder
Satz der Antwort gegen die Passage geprüft, die er nennt (`antwort_pruefung`, Logprobs,
wie das Gate). Trägt keine zugelassene Passage den Satz, wird die Antwort nicht
ausgeliefert.

### Kalibrierung des Gates auf 1000 beschrifteten Fällen

`deutsche-telekom/Ger-RAG-eval`, Task 4 (`context_question_match`), 500 ja / 500 nein,
CC-BY-SA-4.0. `scripts/calibrate_gate_ger_rag_eval.py`.

| Schwelle | „ja" erkannt | „nein" abgelehnt | Genauigkeit |
|---|---|---|---|
| 0,30 | 490/500 | 497/500 | 98,7 % |
| 0,50 | 489/500 | 497/500 | 98,6 % |
| 0,90 | 479/500 | 497/500 | 97,6 % |

Wichtig: bei **467 von 1000** Fällen begann das Modell nicht mit „ja"/„nein" — fast
immer bei den Negativfällen langer Wikipedia-Absätze. Diese Fälle zählen als
Ablehnung. Ohne diese Regel wäre die Trefferquote geschönt; mit ihr ist sie
konservativ. Das Gate allein lässt 3 von 500 negativen Fällen durch — für „gleich
null" zu viel.

### Kette auf dem eigenen Goldset (23 beantwortbare Fragen)

| Prüfung der Antwort | Antworten | Beleg ok | ohne Gold-Beleg | davon Zitate |
|---|---|---|---|---|
| keine | 21 | 15 | 6 | 0 |
| Satzprüfung 0,5, Antwort verweigert | 8 | 8 | 0 | 0 |
| **Satzprüfung 0,5, Sätze streichen, sonst Zitat** | **21** | **16** | **5** | **10** |

Die dritte Zeile ist der eingestellte Ablauf: unbelegte Sätze fallen weg, der Rest wird
ausgeliefert. Bleibt kein Satz übrig, wird die Belegstelle **wörtlich** zitiert — ein
Zitat kann per Konstruktion nicht erfunden sein. **Es gibt keine leere Antwort mehr,
solange das Gate eine Passage zulässt.**

Zwei Zahlen muss man dabei auseinanderhalten:

- **ohne Gold-Beleg (5):** die Antwort zitiert eine Passage, in der die Antwort nicht
  steht — ein Fehler der Suche, keine Erfindung. Die Antwort sagt dann, was ein anderes
  Dokument zu dem Thema sagt, und nennt es. Vorher war dieser Fall unsichtbar, weil er
  als Falschantwort mitgezählt wurde.
- **Zitate (10):** bei zehn Fragen überlebte kein formulierter Satz die Prüfung, und es
  wurde die Belegstelle zitiert. Sicher, aber roh. Ursache: die Satzprüfung verwirft
  korrekte Umformulierungen. Beispiel aus dem Live-Lauf: „Laya steht unter der
  Apache-2.0-Lizenz" fiel durch, obwohl im Auszug „Laya is by ConvAI Innovations
  (Apache-2.0)" steht.

**Eingestellt:** `antwort_pruefung: true`, `antwort_schwelle: 0.5`,
`antwort_reparatur: false`.

**Was noch fehlt.** Die Satzprüfung ist selbst nicht kalibriert: ich weiß, wie oft sie
eine falsche Aussage durchlässt, nicht. Ihr Fehler wirkt in beide Richtungen — sie
verwirft Richtiges (siehe Zitate) und könnte Falsches durchlassen. Eine beschriftete
Menge lässt sich aus `deepset/germanquad` bauen (Antwortsätze mit Textstelle als
„gestützt", Sätze aus fremden Artikeln als „nicht gestützt"). Erst danach ist die
Schwelle 0,5 eine gemessene Zahl und keine Annahme.

## I — Der Korpus ist deutsch, das Chunking war nicht das Problem

Der Befund „ohne Beleg" war nicht mit dem Chunking zu erklären. Die Prüfung des
Index zeigte: **alle** beschrifteten Seiten existieren, die Seitenzahlen stimmen. Die
Ursache lag davor — zwei der sieben Dokumente sind englisch (IDP-Artikel,
Laya-Artikel), und die Fragen dazu sind deutsch. Diese Dokumente zogen Ränge, ohne
dass die Satzprüfung ihre Sätze je als belegt anerkennen konnte.

**Also: Sprache als Korpus-Eigenschaft.** `sprache: de` filtert beim Laden alle
Dokumente aus, die nicht deutsch sind (Stoppwort-Zählung je Datei, nicht je Seite); die
Meldung nennt die übersprungenen Dateien. Die zehn Fragen zu den englischen Artikeln
stehen weiter in `goldset/downloads-alle-sprachen.yaml`, das deutsche Goldset hat 21
Fragen (12 beantwortbar, 9 nicht).

| Korpus | Chunking | Antworten | Beleg ok | ohne Beleg | Zitate |
|---|---|---|---|---|---|
| gemischt (7 Dateien) | 600 Zeichen | 21 | 16 | 5 | 10 |
| **nur deutsch (5 Dateien)** | **600 Zeichen** | **13** | **12** | **1** | 8 |
| nur deutsch | 300 Zeichen + Elternseite | 14 | 11 | 3 | 7 |

Auf dem deutschen Korpus trifft die Antwort bei **12 von 12** beantwortbaren Fragen die
richtige Seite. Die eine Antwort ohne Beleg geht an eine nicht beantwortbare Frage —
das Gate hat sie durchgelassen.

Das Chunking bringt danach nichts mehr: 300 Zeichen mit Rückgabe der ganzen Seite
(small-to-big) liefert 11 statt 12 richtige Belege und drei statt einer Antwort ohne
Beleg. Der Gewinn kam vom Sprachfilter, nicht von der Zerlegung. **600 Zeichen bleiben.**

### Grammatik statt Glückssache

Bei 467 von 1000 Kalibrierungsfällen antwortete das Gate nicht mit „ja"/„nein", sondern
mit einem Satzanfang. llama.cpp kann eine **Grammatik** erzwingen
(`root ::= "ja" | "nein"`); damit ist die Antwort immer eine der beiden. Nebenwirkung:
das Modell muss sich entscheiden, wo es vorher ausgewichen ist.

| Kette, deutscher Korpus | Antworten | Beleg ok | ohne Beleg | Zitate |
|---|---|---|---|---|
| ohne Grammatik | 13 | 12 | 1 | 8 |
| mit Grammatik (HyDE aus) | 13 | 11 | 2 | 8 |
| mit Grammatik (HyDE an) | 13 | 12 | 1 | 8 |

Der Unterschied liegt innerhalb der Schwankung von ±1. Die Grammatik verbessert die
Trefferquote nicht, sie **beseitigt aber den unentschiedenen Zustand**: es gibt keine
Passage mehr, die „irgendwie" zugelassen wird, ohne dass das Modell ja gesagt hat. Für
ein System, das null Falschantworten verspricht, ist ein definierter Zustand mehr wert
als eine Zahl, die um eins schwankt. Sie bleibt eingeschaltet.

**Zwischenstand zum Ziel.** Auf dem deutschen Korpus trifft die Antwort bei 12 von 12
beantwortbaren Fragen die richtige Seite; die vier nicht beantwortbaren Fragen werden
abgelehnt. Offen sind die acht Zitate und die zwei Antworten, die eine falsche Passage
zitieren — beides Fehler der Satzprüfung, nicht der Suche und nicht des Chunkings.

Was jetzt noch offen ist: **8 von 13 Antworten sind Zitate.** Die Satzprüfung verwirft
deutsche Umformulierungen, obwohl Quelle und Aussage dieselbe Sprache haben. Das ist
kein Sprachproblem mehr, sondern eine zu enge Formulierung der Prüffrage. Sie zu
kalibrieren ist der nächste Hebel — und der einzige, der die Zitate senken kann, ohne
die Null bei den Falschantworten aufzugeben.

## J — Abschlusstest gegen „iX Developer 2026 – Testing" (156 Seiten)

Acht Fragen, deren Antworten vorher am Volltext geprüft wurden — sechs beantwortbar, zwei
nicht. `scripts/abschlusstest_ix.py`.

| # | Frage | Erwartet (Seite) | Ergebnis |
|---|---|---|---|
| 1 | Drei Schritte der Golden-Master-Modernisierung | S. 112 | **Zitat** von S. 111, nennt die Schritte nicht |
| 2 | Wozu Spec-driven Development bei KI-Agenten | S. 101 | **falscher Artikel** (S. 10/149, „Specman") |
| 3 | Woran erkennt der Test den Button | S. 74 (getByRole) | **falsch**: „getByText" [S. 73] |
| 4 | Was regelt WCAG 2.4.3 | S. 39 (Fokus-Reihenfolge) | **keine Antwort**, obwohl vorhanden |
| 5 | Was macht ein KI-Agent bei rotem Test | S. 100 (löscht Test / schwächt Assertion) | **andere Aussage** („macht die Änderung rückgängig", S. 102) |
| 6 | Berufsbezeichnung der Tester | S. 128 (Quality Engineer) | **richtig**, Quelle S. 128 |
| 7 | Teilnahmegebühr iX-Konferenz | nicht im Heft | **richtig abgelehnt** |
| 8 | Playwright-Mindestversion | nicht im Heft | **richtig abgelehnt** |

**Drei von acht bestanden. Der Test ist nicht bestanden.** Vier Antworten sind falsch
oder beantworten die Frage nicht, eine verweigert, obwohl die Antwort im Heft steht. Zwei
der falschen Antworten tragen eine Quellenangabe — das ist der Fall, den das Ziel
„keine Halluzination" ausschließen soll. Die Satzprüfung hat sie durchgelassen.

**Was dabei gefunden wurde.** Die Textgewinnung mit `pypdf` zerhackt Wörter am
Zeilenende: auf Seite 74 allein 41 Trennstellen, „Barriere freiheit", „ge - prüft",
„But- ton". Das kostet Treffer und lässt die Satzprüfung richtige Aussagen verwerfen.
Der Loader nutzt jetzt `pypdfium2` (BSD/Apache-2.0, schon Abhängigkeit für das Rendern),
entfernt dessen Umbruchzeichen U+FFFE und bekommt „Barrierefreiheit", „gefunden",
„Eingabefeld"; `pypdf` bleibt als Rückfall.

Das verbesserte die Wortintegrität messbar, den Abschlusstest aber nur bei Frage 8 —
die inhaltlichen Fehler bleiben.

**Was das für das Ziel heißt.** Der Korpus ist jetzt deutsch, das Gate entscheidet
immer, die Antwort wird satzweise geprüft — und trotzdem kommen falsche Aussagen mit
Quelle durch. Damit ist die Kette für das Ziel „keine Halluzination" **nicht belegt**.
Zwei Ursachen, beide gemessen:

1. **Suche:** Bei 156 Seiten mit vielen Artikeln zum selben Thema zieht die Suche den
   falschen Artikel. 600-Zeichen-Fenster über Seitengrenzen kennen keine
   Artikelstruktur. Ein Chunking entlang der Überschriften (Artikel/Titel) ist der
   nächste Schritt, nicht eine andere Fenstergröße — die ist bereits gemessen.
2. **Satzprüfung:** Sie akzeptiert falsche Aussagen (Frage 2 und 5) und verwirft
   richtige (Frage 4). Ihre Fehlerrate ist unbekannt; sie ist der einzige Baustein ohne
   Kalibrierung.

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
