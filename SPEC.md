# ArbiterRAG — SPEC

**Status:** freigegeben 2026-11-07 (rev. 2, Antworten D1–D9 eingearbeitet). Offen: D10–D12.
**Zweck:** RAG-System mit Reranker und einer Prüfschicht über Logprobs.
**Arbeitsverzeichnis:** `/home/speedy/Dokumente/dev/ArbiterRAG`

Diese SPEC ist der Vertrag. Jede Änderung wird sichtbar ergänzt, nie still
überschrieben.

---

## 1. Was das System tut

Drei Stufen, jede mit einer eigenen Aufgabe:

| Stufe | Aufgabe | Wirkung |
|---|---|---|
| Reranker | bewertet jedes Paar `(Frage, Passage)` und ordnet | verändert die **Reihenfolge** |
| Prüfschicht | beantwortet je Passage eine Ja/Nein-Frage über Logprobs | verändert **Annahme**, nicht die Reihenfolge |

Der Kern: Der Reranker kann nicht sagen „keine dieser Passagen belegt die
Antwort". Die Prüfschicht kann es. ArbiterRAG trennt Ordnen von Urteilen.

## 2. Festlegungen

| # | Festlegung | Begründung |
|---|---|---|
| F1 | Python 3.12, `uv` für alles (`uv sync`, `uv run`) | Hausregel |
| F2 | Vektorstore LanceDB, eine Tabelle pro RAG | FTS und Vektor in einem Store |
| F3 | Embeddings `bge-m3` (1024 Dimensionen) über llama-swap `/v1/embeddings` | Modell liegt bereit, MIT-Lizenz |
| F4 | Hybrid = dichte Suche **plus** LanceDB-Volltextsuche, Fusion über RRF (k=60) | der Embedding-Endpunkt liefert nur dichte Vektoren, keine lexikalischen Gewichte; FTS ersetzt den sparse-Teil |
| F5 | Alle Backends über HTTP, Basis-URL per Umgebungsvariable | kein Modell im Prozess, testbar |
| F6 | Backends hinter Protokollen, Unit-Tests laufen mit Fakes | kein Netz in Unit-Tests |
| F7 | Ein neuer Korpus ist ein neuer YAML-Eintrag, kein neuer Code | Hausregel |
| F8 | Oberfläche und Meldungen auf Deutsch | Hausregel |
| F9 | **Jeder Modellaufruf ist OpenAI-kompatibel**: Chat über `/v1/chat/completions` (mit `logprobs`), Embeddings über `/v1/embeddings`, Reranking über `/v1/rerank`. | Vorgabe Frank 2026-11-07 |

## 3. Entscheidungen

### Beantwortet (2026-11-07)

| ID | Frage | Antwort | Folge |
|---|---|---|---|
| D1 | `GenericRAG` lesen? | **b** — nur lesen, Code neu | Konventionen übernehmen, kein Code kopieren |
| D2 | Korpus | **PDF** | PDF-Extraktor nötig (Abhängigkeit `pypdf`), Verhalten B2 |
| D3 | HyDE | **entfernt 2026-11-07** — gemessen ohne Wirkung: bei bge-m3 eine Antwort Unterschied, bei nomic keine. Anordnung Frank: raus, was nichts bringt | Abschnitt D der SPEC gestrichen |
| D4 | HyDE-Modell | **entfällt** mit D3 | — |
| D5 | Reranker | **a** — HTTP `/v1/rerank` | Eintrag `bge-reranker-v2-m3` ist defekt, siehe D11 |
| D6 | Prüfschicht-Umfang | **c** — zuerst Gate, Auswahl und Widerspruch/Mengenprüfung geplant; Widerspruch/Mengen verworfen, weil gemessen ohne Wirkung (F8) | Messung 2026-11-07 |
| D7 | Prüfschicht-Ausfall | **b** — keine Antwort | Verhalten F6, fail-closed |
| D8 | Antwort erzeugen | **a** — ja, abschaltbar | Verhalten G |
| D9 | Goldset | **b** — eigener Korpus, 30–40 originale deutsche Fragen | Stufe 5 |

### Offen

| ID | Frage | Optionen | Empfehlung |
|---|---|---|---|
| D10 | OCR | **b** — OCR über llama-swap, Modelle `dots-mocr` **und** `paddleocr-vl`, auf **jeder** PDF-Seite (Vorgabe Frank 2026-11-07) | Verhalten B8; `ocr_alle_seiten` schaltet es ein |
| D11 | Reranker-Eintrag ist defekt: `-m` zeigt auf ein safetensors-Verzeichnis, llama-server kann das nicht laden (`upstream command exited prematurely`). Kein GGUF vorhanden. | **reparieren** | erledigt 2026-11-07: GGUF `Q8_0` (gpustack, Apache-2.0) geladen, Eintrag mit `--reranking` versehen, `/v1/rerank` antwortet |
| D12 | Pfad des PDF-Korpus für den ersten echten Lauf | **`/home/speedy/Downloads`** | erledigt: 9 Dateien, 1053 Abschnitte, 1 Seite ohne Textebene |
| D13 | Wie werden die **zwei** OCR-Ausgaben kombiniert? | **a** | umgesetzt: dots-Text indiziert, paddle als Vergleichsfeld, Überlappung berichtet |
| D14 | Darf ich den OCR-Prompt aus `~/Dokumente/dev/U-OCR_test/` übernehmen (liegt außerhalb des Arbeitsverzeichnisses, N6) | **a** | Quelle existiert nicht mehr; Prompt aus der Model Card, siehe EVIDENCE |
| D15 | Umfang der Endpunkte | **b** (plus `/ask` nach Stufe 5) | umgesetzt: `health`, `rags`, `search`, `ask`, `index` |
| D16 | Indexierung als Auftrag | **b** | umgesetzt: `202` + `job_id`, Status über `GET /index/{job_id}` |
| D17 | Zugriffsschutz | **a** | umgesetzt: Bindung über `ARBITER_BIND`, Standard `127.0.0.1` |
| D18 | Betrieb | **b** | umgesetzt: systemd-User-Service `arbiter-api.service` |
| D19 | Fehlerformat | **a** | umgesetzt: FastAPIs `{"detail": ...}`, Meldung deutsch |

## 4. Verhalten (Abnahmekriterien)

Jedes Verhalten wird als Test geschrieben, rot gesehen, dann grün gemacht.

### A. Konfiguration

- **A1** Eine YAML-Datei beschreibt je RAG: Name, Beschreibung, Korpuspfad,
  Dateiendungen, `top_k` je Stufe, Schwellen, Modell-IDs, Basis-URLs. Fehlt ein
  Pflichtfeld, bricht der Start mit einer Meldung ab, die Feld und Datei nennt.
- **A2** `${VAR}` wird aus der Umgebung ersetzt. Ist die Variable leer, bricht
  der Start ab. Ein leerer Standardwert ist unzulässig.
- **A3** Ein unbekannter Schlüssel ist ein Fehler, keine stille Auslassung.
- **A4** Modell-IDs und URLs sind Konfigurationswerte. Kein Modellname steht im
  Code.

### B. Indexierung

- **B1** `arbiter index <rag>` liest alle Dateien der konfigurierten Endungen
  unter dem Korpuspfad, rekursiv.
- **B2** PDF werden über `pypdf` gelesen, seitenweise. Hat eine Seite keinen
  Text, wird sie gezählt, nicht als Fehler behandelt (D10).
- **B3** Jeder Datensatz hat `id`, `text`, `quelle`, `seite` (bei PDF), `hash`
  (SHA-256 des Textes) und `vector`.
- **B4** Eine erneute Indexierung fügt unveränderte Dateien nicht doppelt ein;
  geänderte ersetzen ihren alten Datensatz. Maßgeblich ist der Hash.
- **B12** Der Ingest einer Datei findet **einmal** statt. Vor dem Lesen wird eine
  Kennung geprüft: SHA-256 über den Dateiinhalt plus die Einstellungen, die den
  Text bestimmen (Abschnittsgröße, Überlappung, OCR an/aus und alle Seiten, beide
  DPI-Werte, Sprache, Modellnamen). Stimmt sie, wird die Datei **nicht gelesen** —
  kein OCR, keine Einbettung —, und ihre Zeilen bleiben unangetastet. Die Kennung
  steht in der Tabelle `<rag>_ingest`.
- **B5** Eine Datei ohne lesbaren Text erzeugt eine Warnung mit Pfad; die
  Indexierung läuft weiter. Am Ende stehen die Zahl der übersprungenen Dateien
  und der textlosen Seiten.
- **B6** Text wird in Abschnitte von höchstens `chunk_chars` Zeichen geteilt
  (Default 600, Überlappung `chunk_overlap`, Default 100), an Absatzgrenzen.
  Jeder Abschnitt beginnt mit `<Dateiname ohne Endung>, S. <Seite>` — der Kopf
  hilft der dichten und der lexikalischen Suche. Abschnitts-ID ist
  `<quelle>#<seite>#<laufende Nummer>`.
  *Geändert 2026-11-07: vorher 1200 Zeichen ohne Kopf. Messung: Recall@10
  0,58 → 0,65.*
- **B7** Nach der Indexierung existiert ein Volltextindex auf `text`, deutsch,
  Stemming über `fts_stem` schaltbar.
- **B11** Eine Datei, deren Text bereits aus einem anderen Pfad indiziert wurde,
  wird als Dublette übersprungen (Meldung mit beiden Pfaden). Zeilen von Dateien,
  die nicht mehr im Korpus liegen, werden beim Indexieren entfernt.
  *Grund: im Testkorpus lagen drei PDFs zusätzlich unter `ppt/` und belegten
  doppelt Ränge.*
- **B8** Bei `ocr: true` geht **jede** PDF-Seite durch **beide** OCR-Modelle —
  unabhängig davon, ob eine Textebene vorhanden ist. Jedes Modell bekommt ein
  eigenes Rendering: `dots-mocr` mit `ocr_dpi` (Default 200) und dem Layout-Prompt
  der Model Card, `paddleocr-vl` mit `ocr_dpi_paddle` (Default 300). `dots-mocr`
  liefert den Text, der indiziert wird: seine Layout-JSON wird in Lesereihenfolge
  zu Text reduziert, ohne Übersetzung, Bilder ohne Text. Die Ausgabe von
  `paddleocr-vl` steht als `ocr_paddle` im Datensatz, die Wortüberlappung beider
  Lesungen als `ocr_ueberlappung`. Die beiden Texte werden **nicht** gemischt
  (D13). Ist `ocr_alle_seiten: false`, greift OCR nur auf Seiten ohne Textebene.
- **B9** Scheitert ein OCR-Modell, bleibt die Seite übersprungen und der Grund
  steht als Warnung im Bericht. Der Lauf bricht nicht ab.
- **B10** Eine Tabelle aus einer älteren Version wird beim Schreiben um fehlende
  Spalten ergänzt, statt den Lauf mit einem Schema-Fehler zu beenden.

> Gemessene Grenze (2026-11-07, `PATCHT-Trennblatt.pdf`): `dots-mocr` liest bei
> 100, 150 und 200 dpi identisch (88 Zeichen), bei 250 und 300 dpi antwortet der
> Server `HTTP 500 failed to process mtmd chunk`. Deshalb Default 200, nicht 300.

### C. Suche und Fusion

- **C1** Dichte Suche liefert die `n` nächsten Nachbarn nach Kosinus, absteigend
  sortiert.
- **C2** Volltextsuche liefert die `n` besten Treffer nach BM25, absteigend.
- **C3** RRF fusioniert mit `k=60`: `score = Σ 1/(60 + rang)`. Gleicher Score
  bedeutet Sortierung nach `id` aufsteigend — die Ausgabe ist bei gleicher
  Eingabe gleich.
- **C4** Die Fusion schneidet auf `fuse_top_k` (Default 20) ab.
- **C6** `hybrid: false` lässt nur die dichte Suche laufen. Gemessen 2026-11-07:
  dichte Suche allein Recall 0,71, mit Volltextteil 0,65 — die wiederholte
  Kopfzeile der Foliensätze belegt sonst viele Plätze.

### D. HyDE — entfernt

HyDE (hypothetisches Antwortdokument als zweite Suche) ist am 2026-11-07 entfernt
worden. Messung: mit HyDE 15 richtige Belege, ohne 16 (bge-m3), bei nomic-embed-text
in beiden Fällen 14 — der Unterschied liegt in der Schwankung. Der Schalter, das
Dokument und die Gewichtung in der Fusion sind gelöscht; die Suche läuft mit der
Frage. Messwerte: `docs/EVIDENCE-verbesserungen.md`, Abschnitte C und F.

### E. Reranking

- **E1** Die `fuse_top_k` Kandidaten werden über `/v1/rerank` gegen die Frage
  bewertet; die Ausgabe enthält `rerank_top_n` (Default 5) Passagen.
- **E2** Die Reihenfolge folgt dem Reranker-Score absteigend. Bei Gleichstand
  gilt die vorherige Reihenfolge (stabile Sortierung).
- **E3** Fällt der Reranker aus, wird die RRF-Reihenfolge ausgegeben, mit einer
  Warnung und dem Feld `rerank: "ausgefallen"`.
- **E4** Der Score jeder Passage steht im Ergebnis. Er ist **keine**
  Wahrscheinlichkeit und wird nie als Schwelle benutzt.

### F. Prüfschicht (Gate)

- **F1** Die Prüfschicht prüft die **fünf besten** Passagen des Rerankers
  (`gate_kandidaten`, Default 5). Je Passage **ein eigener** Request: die Frage
  und diese eine Passage. Die Antwort zitiert bis zu drei zugelassene Passagen.
- **F2** Die Entscheidung ist die Wahrscheinlichkeit eines einzelnen Tokens:
  `qwen2.5-7b-instruct` antwortet auf „Beantwortet der Text die Frage?" mit `ja`
  oder `nein`; `top_logprobs` liefert die Alternativen, die Summe über alle
  `ja`-Token ist P(ja). Prompt-Variante über `gate_prompt`.
- **F3** Eine Passage gilt als zulässig, wenn `P(ja) >= gate_schwelle`
  (Default 0.50).
- **F4** Ist keine Passage zulässig, liefert das System keine Antwort, sondern
  den Grund `keine belegte Passage` und die Warnungen. Der Antwortschritt wird
  nicht aufgerufen.
- **F5** Fehlen die Logprobs, zählt keine Wahrscheinlichkeit: die Passage wird
  nicht zugelassen, eine Warnung steht im Ergebnis. Kein stilles Durchreichen,
  kein erfundener Wert.
- **F6** Bei einem Fehler des Aufrufs gilt die Passage als nicht geprüft; ist
  danach keine Passage zulässig, lautet der Grund `keine belegte Passage`.
- **F7** *Gemessen 2026-11-07, Goldset 31 Fragen:* Prompt `teil` trennt
  (nicht beantwortbare Fragen höchstens 0,091, beantwortbare im Median 1,000),
  der Schwellenwert ist zwischen 0,3 und 0,9 beliebig. In der Kette bleiben bei
  23 beantwortbaren Fragen 15 richtige Belege und 1 Falschantwort; ohne Gate
  waren es 19 richtige und 8 falsche.
- **F8** Die früheren Bauformen sind **entfernt**: Laya als Gate je Passage
  (`noul`), als eine Auswahl über alle Kandidaten (`choice`) und als Prüfer der
  fertigen Antwort. Alle drei entfernten richtige Antworten und keine falschen.
  Messwerte in `docs/EVIDENCE-verbesserungen.md`, Abschnitte E, E2, E4.

### G. Antwort

- **G1** Die Antwort wird nur aus den zulässigen Passagen über
  `/v1/chat/completions` gebildet. Jede Aussage trägt die `quelle` der Passage.
- **G2** Die Antwort nennt die verwendeten Quellen am Ende.
- **G5** Die Antwort enthält **keine Aussagen über die Auszüge selbst**. Der Prompt
  nennt keine Ablehnungsformel mehr im Klartext (sie erschien sonst in der Antwort:
  „Das steht nicht in den Auszügen, aber es wird erwähnt …"), sondern ein Token
  `KEINE_ANTWORT`. Enthält die Antwort dieses Token, lautet das Ergebnis „Die Auszüge
  beantworten die Frage nicht". Sätze, die über die Auszüge sprechen, werden vor der
  Prüfung verworfen und nie ausgeliefert.
- **G3** Liefert das Antwortmodell leeren Text, gilt der Lauf als fehlgeschlagen
  (Exit-Code 1, Meldung), nicht als leere Antwort.

### H. Schnittstelle

- **H1** `arbiter index <rag>` meldet Zahl der Dateien, Abschnitte,
  übersprungene Dateien und textlose Seiten.
- **H2** `arbiter search <rag> "<frage>"` gibt die Passagen mit Quelle, Seite,
  RRF-Rang und Reranker-Score als Tabelle aus.
- **H3** `arbiter ask <rag> "<frage>"` gibt Antwort plus Quellen aus; mit
  `--json` als maschinenlesbares Objekt.
- **H4** `arbiter eval <rag>` fährt das Goldset und berichtet Recall@10,
  nDCG@10 und Antwortrichtigkeit, je Stufe getrennt: nur Fusion, mit Reranker, mit
  Prüfschicht.
- **H5** Die CLI ist ein dünner Client der Bibliothek.

### I. Fehlerfälle

- **I1** Unbekannter RAG-Name: Meldung mit Liste der bekannten Namen, Exit 2.
- **I2** Leerer Korpus: Meldung `Korpus leer` mit Pfad, Exit 1.
- **I3** Frage länger als `max_frage_zeichen` (Default 2000): Abbruch mit
  Meldung, kein stilles Kürzen.

## 5. Was das System nicht tun darf

- **N1** Eine Passage ohne Quellenangabe in die Antwort aufnehmen.
- **N2** Einen Reranker-Score als Wahrscheinlichkeit ausgeben oder als Schwelle
  verwenden.
- **N3** Bei fehlender Belegstelle trotzdem antworten.
- **N4** Im Fehlerfall stillschweigend auf eine andere Stufe ausweichen, ohne
  Warnung im Ergebnis.
- **N5** Modellgewichte oder Netzzugriffe in Unit-Tests laden. Netztests nur
  hinter `-m integration`.
- **N6** Fremde Verzeichnisse außerhalb des Arbeitsverzeichnisses lesen, ohne
  dass D1 es erlaubt.
- **N7** Persönliche Daten aus deinen Dokumenten in Code, Tests, Logs oder
  Commits schreiben. Testdaten sind Platzhalter gleicher Form.
- **N8** Eine ungeprüfte Schwelle als Filter einsetzen (siehe F7).

## 6. Setup-Plan

Angenommen mit dieser SPEC:

| Punkt | Inhalt |
|---|---|
| Projekt | `uv init`, `pyproject.toml`, Python 3.12 |
| Versionierung | `git init`, Commit bei Freigabe und je Grün/Refactor-Stufe |
| Test | `pytest`, `pytest-cov`, `hypothesis`; Marker `integration` standardmäßig ausgeschlossen |
| Statik | `ruff` (Lint + Format), `mypy` |
| Laufzeit-Abhängigkeiten | `lancedb` (Store), `pyarrow` (von LanceDB verlangt), `numpy` (Vektorrechnung), `httpx` (HTTP-Backends), `pyyaml` (Konfiguration), `pypdf` (PDF-Textebene, BSD-3), `pypdfium2` (Seiten rendern, BSD-3/Apache-2.0), `pillow` (PNG-Kodierung für die OCR, HPND) |
| Test-Abhängigkeiten | `pytest`, `pytest-cov`, `hypothesis`, `ruff`, `mypy` |
| Modell-Abhängigkeiten | `bge-m3` (MIT), `bge-reranker-v2-m3` (Apache-2.0) — Lizenzen am 2026-11-07 über die Registry geprüft. HyDE- und Antwortmodell aus llama-swap (D4/D8) |
| Nicht installiert | kein torch, kein FlagEmbedding im Client |

## 7. Stufen mit Stopp-Punkten

Jede Stufe endet mit einem Bericht. Nach jeder Stufe darfst du anhalten; die
nächste braucht eine neue Freigabe. Aufwände sind Schätzungen.

| Stufe | Inhalt | Verhalten | Aufwand |
|---|---|---|---|
| 1 | Gerüst, Konfiguration, PDF-Index, Hybrid, RRF, CLI `index`/`search` | A, B, C, H1–H2, I | ~1 Tag |
| 2 | HyDE | D | ~0.5 Tag |
| 3 | Reranker über `/v1/rerank` (setzt D11 voraus) | E | ~0.5 Tag |
| 4 | Prüfschicht: Gate über Logprobs; Laya-Formen gemessen und entfernt | F | ~1 Tag |
| 5 | Antwort, Goldset, Schwellen aus der Messung | G, H4, F7 als Filter | ~1 Tag |

## 8. Messung (Stufe 5)

Eigener Korpus, eigene deutsche Fragen — keine Übersetzung eines fremden
Fragensatzes.

| Kennzahl | Was sie zeigt |
|---|---|
| Recall@10 | ob die richtige Passage im Kandidatenpool liegt (Wirkung von HyDE) |
| nDCG@10 | ob die Reihenfolge stimmt (Wirkung des Rerankers) |
| Antwortrichtigkeit | ob die Antwort stimmt (Wirkung der Prüfschicht) |
| Falschantwortquote bei nicht beantwortbaren Fragen | der eigentliche Gewinn der Entscheidungsschicht |

Berichtet wird je Stufe getrennt. Ein Ergebnis, das nur insgesamt besser ist,
sagt nichts.

---

## 9. Stand

Freigabe erteilt 2026-11-07. Stufe 1 beginnt mit RED.
D10, D11, D12 sind offen; D11 und D12 blockieren erst Stufe 3 bzw. den ersten
echten Lauf.

---

## 10. Stufe 6 — HTTP-Schnittstelle (Entwurf, wartet auf Freigabe)

Auftrag 2026-11-07: „Erweitere das System um eine FastAPI". Die Bibliothek
bleibt die Wahrheit; die API ist ein weiterer dünner Client wie die CLI (H5).

### Offene Entscheidungen

| ID | Frage | Optionen | Empfehlung |
|---|---|---|---|
| D15 | Umfang der Endpunkte | (a) nur `health`, `rags`, `search` (b) zusätzlich `index` als Hintergrundauftrag mit Status (c) zusätzlich `ask` (Antwort) | **(b)** — (c) gibt es erst mit Stufe 5; ein Endpunkt, der nichts kann, ist Ballast |
| D16 | Indexierung im Request oder als Auftrag? | (a) blockierend, Antwort nach Ende (b) 202 mit `job_id`, Status über `GET /index/{job_id}` | **(b)** — der echte Lauf braucht heute 36 s für 10 PDFs; bei 500 PDFs mit OCR sind es Stunden. Ein blockierender Request läuft in jedes Gateway-Timeout |
| D17 | Zugriffsschutz | (a) keiner, Bindung an `127.0.0.1` (b) Bearer-Token aus Umgebungsvariable, Pflicht (c) optional | **(a)** für den lokalen Betrieb, mit `ARBITER_BIND` als Konfiguration; (b) erst wenn der Dienst von außen erreichbar ist |
| D18 | Betrieb | (a) `uv run arbiter-api` im Vordergrund (b) systemd-User-Service (c) Podman-Container | **(b)** — Hausregel für lange laufende Prozesse, und der Dienst soll nach dem Reboot da sein. Container wäre die zweite Sandbox um dieselbe Sache |
| D19 | Fehlerformat | (a) FastAPIs `{"detail": ...}` (b) eigenes `{"fehler": {"code": ..., "meldung": ...}}` | **(a)** — Standard bleibt Standard; die Meldung darin ist deutsch |

### Verhalten (Abnahmekriterien)

- **P1** `GET /health` antwortet `200 {"status": "ok", "version": "<Paketversion>"}`
  ohne einen Modellaufruf.
- **P2** `GET /rags` antwortet je RAG `{name, description, abschnitte}`. Die Zahl
  kommt aus dem Store; ein RAG ohne Index meldet `abschnitte: 0` und keinen Fehler.
- **P3** `POST /search {rag, frage}` antwortet `200` mit derselben Passagenliste wie
  die CLI: `id, quelle, seite, rrf_rank, rrf_score, text`, dazu `ocr` und
  `ocr_ueberlappung`.
- **P4** Unbekannter RAG → `404` mit der Liste der bekannten Namen. Fehlendes Feld →
  `422` (FastAPI-Standard). Frage über `max_frage_zeichen` → `400` mit deutscher
  Meldung.
- **P5** `POST /index {rag}` antwortet `202 {"job_id": ...}`. `GET /index/{job_id}`
  antwortet `{"status": "laeuft"|"fertig"|"fehler", "bericht": {...}}`, wobei
  `bericht` die Zahlen aus H1 trägt. Unbekannte `job_id` → `404`.
- **P6** Ein zweiter Index-Lauf für denselben RAG, solange der erste läuft →
  `409` mit Meldung, kein zweiter Prozess.
- **P7** Ein Modell-Backend, das nicht antwortet, ergibt `503` mit deutscher
  Meldung. Kein Stacktrace im Antwortkörper.
- **P8** Ungültige Konfiguration → der Dienst startet nicht, Meldung auf stderr,
  Exit-Code 1. Kein Start mit halber Konfiguration.
- **P9** Alle API-Tests laufen über `httpx.ASGITransport` gegen den ASGI-App, ohne
  Server und ohne Netz; Embedder und OCR sind Attrappen.
- **P10** `POST /ask {rag, frage}` antwortet `200` mit Antwort, Quellen und
  Entscheidung — auch dann, wenn es keine Antwort gibt: `antwort: null` plus
  `grund` und `fehler: true|false`. Ein fehlendes Modell ist ein Fehler, keine
  leere Antwort.

### Was die API nicht tun darf

- **Q1** Eine zweite Logik neben der Bibliothek enthalten. Jeder Endpunkt ruft
  `Pipeline` oder `Store`, nichts anderes.
- **Q2** Modellnamen, Pfade oder URLs fest verdrahten (A4 gilt weiter).
- **Q3** Einen Stacktrace oder einen Dateipfad aus dem Korpus im Antwortkörper
  ausgeben.
- **Q4** Einen Index-Lauf ohne Auftragskennung starten, den niemand mehr abfragen
  kann.

### Setup

| Punkt | Inhalt |
|---|---|
| Neue Abhängigkeiten | `fastapi`, `uvicorn[standard]` (beide MIT/BSD-3) — nur für die API-Ebene |
| Einstiegspunkt | `arbiter-api = "arbiter.api:main"` neben `arbiter = "arbiter.cli:main"` |
| Tests | `httpx.ASGITransport`, keine laufende Instanz nötig |
| Betrieb | systemd-User-Service nach D18, Bindung über `ARBITER_BIND` |

Aufwand: rund ein halber Tag, mit Tests und Nachweis.

Freigabe für D15–D19 und diesen Abschnitt: danach beginnt die Stufe mit RED.

---

## 11. Stufe 7 — Zwei Chunking-Strategien mit Entscheider (Entwurf, wartet auf Freigabe)

Auftrag 2026-11-08, wörtlich: „1. QESChunker greedy + Overlap und Structure-aware
2. Implementiere das in ArbiterRAG 3. Beim ingest der Daten soll ein ‚Entscheider'
welche Chunk strategie die beste für das aufzunehmende PDF ist."

Quelle der beiden Verfahren: Vishal Mysore, „Best RAG Chunking Strategy in 2026",
Nov 2026, sowie die dort genannten Papiere arXiv 2610.36557 (QESChunker) und
arXiv 2610.26313 (StrucChunk).

Der Artikel ist ein Toy-Benchmark: ein Dokument, 25 Fragen, nur BM25, eine Frage
entspricht 7,7 Prozentpunkten. Er belegt Mechanismen, keine Rangfolge. Deshalb
enthält diese Stufe eine Messung (7.3) und keine zugesagte Verbesserung.

### 11.1 Was sich ändert, was bleibt

| Punkt | Vorher | Nachher |
|---|---|---|
| Chunk-Grenze | je Seite, `chunk_text` (B6) | je Datei (D20), Strategie je Datei |
| Strategien | eine | `struktur`, `qes`, `fenster` (Rückfall) |
| Entscheider | keiner | deterministisch, ohne Modell (D21) |
| Datensatz | `id, text, quelle, seite, hash, vector, …` | zusätzlich `strategie`, `seite_ende` |
| Suche, Reranker, Gate, Antwort | — | unverändert |

`chunk_text` bleibt bestehen und wird zur Strategie `fenster` (B6 gilt weiter).

### 11.2 Offene Entscheidungen

| ID | Frage | Optionen | Empfehlung |
|---|---|---|---|
| D20 | Chunk-Grenze: pro Seite oder pro Datei? | (a) pro Datei, Seiten über Zeichenposition zugeordnet (b) pro Seite wie heute | **(a)** — Struktur läuft über Seitengrenzen; pro Seite zerbricht jeder Abschnitt an jeder Seitengrenze. Kosten (b): strukturtreues Chunking verliert seinen Kern |
| D21 | Wie entscheidet der Entscheider? | (a) deterministische Heuristik über Überschriften und Querverweise (b) Chat-Modell klassifiziert (c) Heuristik, Modell nur im Grenzband | **(a)** — prüfbar, erklärbar, kein Modellaufruf je Datei, gleiche Datei ergibt gleiche Strategie. Kosten (b): ein Aufruf je Datei, nicht deterministisch, Test nur mit Attrappe |
| D22 | Woher kommen die Frage-Beleg-Paare für QESChunker? | (a) Chat-Modell erzeugt sie je Datei (b) keine Erzeugung, QES fällt aus (c) Modell, mit definiertem Rückfall | **(c)** — wie im Papier; ohne QE-Paare hat der Greedy keinen Nutzen und wird durch den Rückfall ersetzt. Kosten: ein bis vier Modellaufrufe je **neuer** Datei (der Fingerabdruck B12 verhindert Wiederholung) |
| D23 | Querverweise beim Indexieren auflösen? | (a) nur Überschriftpfad voranstellen (b) Pfad plus aufgelöste Verweise (c) nichts | **(b)** — im Artikel war die Verweisauflösung der Grund, warum der Graph 62 % → 69 % brachte; Regel und Definition werden ein Treffer |
| D24 | Wo steht die gewählte Strategie? | (a) nur im Indexbericht (b) zusätzlich als Spalte `strategie` je Zeile (c) zusätzlich eigene Tabelle | **(b)** — ohne Spalte ist keine spätere Messung je Strategie möglich; eine Tabelle wäre doppelte Buchführung |
| D25 | Strategie übersteuerbar? | (a) nur automatisch (b) `--strategie auto\|struktur\|qes\|fenster` | **(b)** — ohne Übersteuerung ist die Messung in 7.3 nicht möglich |
| D26 | Wie wird gemessen? | (a) Seiten-Recall@10 vor/nach auf dem bestehenden Goldset, je Strategie (b) zusätzlich Complete-Evidence-Recall, dafür `belege` (wörtliche Sätze) je Frage eintragen (c) keine Messung | **(a)** jetzt; (b) nur, wenn du die 31 Fragen annotierst (etwa 2 h Handarbeit) |
| D27 | Darf ein aufgelöster Querverweis `chunk_chars` überschreiten? | (a) ja, Anhang kommt zum eigenen Text hinzu (b) nein, eigener Text schrumpft | **(a)** — der Anhang ist Beiwerk zur Suche, kein Inhalt des Abschnitts; (b) würde kurze Abschnitte weiter zerstückeln |
| D28 | Auch die exakte DP-Variante aus dem Papier? | (a) nein (b) ja, als vierte Strategie | **(a)** — im Artikel war sie die schlechteste (15 % gehalten); Kosten (b) etwa 0,3 Tag plus Test |
| D29 | Auch Chunk-Graph und Wortbudget-Assembler (SpectRAG) auf der Suchseite? | (a) nein (b) ja | **(a)** — nicht Teil des Auftrags; Kosten (b) etwa 1 Tag, ändert C und E |

Unbeantwortete IDs werden nicht stillschweigend angenommen. Ohne deine Antwort
beginnt kein Code.

### 11.3 Verhalten (Abnahmekriterien)

#### K. Gemeinsame Form

- **K1** Eine Datei ergibt eine Liste `Chunk`. Ein `Chunk` hat `text` (der
  indizierte Text), `start`, `ende` (Zeichenpositionen des **eigenen** Textes im
  Dokumenttext) und `strategie`.
- **K2** `text[start:ende]` steht nach Leerraum-Normalisierung wörtlich in
  `chunk.text`. Kopfzeile (B6) und Verweis-Anhang stehen davor bzw. dahinter.
- **K3** Jedes Zeichen des Dokumenttextes außer Leerraum liegt in mindestens
  einem Chunk. Kein Textverlust.
- **K4** Leerer Text ergibt keine Chunks.
- **K5** Der eigene Text eines Chunks ist höchstens `chunk_chars` Zeichen lang.
  Kopfzeile und Verweis-Anhang kommen hinzu (D27).
- **K6** Gleiche Eingabe und gleiche Frage-Beleg-Paare ergeben gleiche Chunks.

#### L. Strategie `struktur` (StrucChunk-Stil)

- **L1** Eine Zeile gilt als Überschrift, wenn eine Regel greift:

  | Regel | Muster | Ebene |
  |---|---|---|
  | Markdown | `^\s{0,3}#{1,6}\s+\S` | Anzahl `#` |
  | Nummerierung | `^\s*(\d+\.)+\d*\s+\S` | Anzahl Zahlengruppen |
  | Paragraph | `^\s*§\s*\d+` | 2 |
  | Wortkopf | `^\s*(Kapitel\|Abschnitt\|Anhang\|Teil)\s+[0-9IVXLC]+\b` | 2 |
  | Großzeile | 3–80 Zeichen, ≥ 2 Wörter, nur Großbuchstaben, kein `. ! ? :` | 2 |

- **L2** Ein Abschnitt ist eine Überschrift der Ebene ≥ `struktur_ebene` (Default
  2) plus alle Zeilen bis zur nächsten Überschrift gleicher oder höherer Ebene.
  Text vor der ersten Überschrift ist ein Abschnitt ohne Pfad.
  *Präzisiert 2026-11-08 vor der Umsetzung:* Ein Abschnitt endet an der nächsten
  Überschrift **jeder** Ebene. Unterabschnitte werden eigene Chunks; der
  Elternabschnitt trägt nur seinen eigenen Rumpf. Damit steht kein Text doppelt
  im Index, und trotzdem bleibt kein Text ungefunden. Der Pfad nennt alle
  übergeordneten Überschriften.
- **L3** Ein Chunk ist ein Abschnitt. Er enthält nie Rumpf aus zwei Abschnitten.
- **L4** Ein Rumpf länger als `chunk_chars` wird an Satzgrenzen geteilt, in
  Lesereihenfolge, ohne Überlappung.
- **L5** Der Chunk-Text beginnt mit `[<Pfad>] `, Pfad = Überschriften von der
  obersten erkannten Ebene bis zur eigenen, getrennt durch ` › `, zum Beispiel
  `[2 Anreise › 2.1 Torzeiten] `. Ohne Pfad kein Präfix.
- **L6** Ein Verweis im Rumpf wird aufgelöst, wenn er auf einen vorhandenen
  Abschnitt zeigt. Erkannt werden `siehe Abschnitt 1.3`, `nach Ziffer 4.2`,
  `gemäß § 5`, `vgl. 2.1`, `Section 4.2`, `Kapitel 3`. Angehängt wird
  `\n↳ <Nummer> <Titel>: ` plus bis zu `xref_chars` Zeichen des Ziel-Rumpfes.
  Höchstens `xref_max` Verweise je Chunk, jeder Abschnitt höchstens einmal,
  nie auf sich selbst.
- **L7** Ein Verweis ohne Ziel wird übergangen. Er erzeugt keinen Anhang und
  keine Warnung.
- **L8** Seitenübergreifend: Ein Abschnitt, der auf zwei Seiten liegt, ist **ein**
  Chunk.

#### M. Strategie `qes` (QESChunker, Kosten-Nutzen-Greedy mit Überlappung)

- **M1** Einheit ist der Satz. Satzgrenzen sind `.`, `!`, `?` mit folgendem
  Leerraum; `1.2` und `1.2-meter` zerfallen nicht.
- **M2** Ein Frage-Beleg-Paar ist eine Frage plus ein bis drei Belegstellen. Eine
  Belegstelle ist ein Satz, der nach Leerraum-Normalisierung wörtlich im Dokument
  steht. Ein Paar mit einer nicht wörtlichen Belegstelle wird verworfen.
  *Präzisiert 2026-11-08 nach einem echten Lauf:* Ein Beleg darf auch ein **Teil**
  eines Satzes sein, wenn er wörtlich im Dokument steht und mindestens 20 Zeichen
  lang ist; er wird dann dem Satz zugeordnet, der ihn enthält. Grund: Das Modell
  zitierte im echten Lauf 0 von 12 Belegen als ganzen Satz, aber 12 von 12 als
  wörtliches Fragment. Umgangssprachliche oder gekürzte Fassungen ohne wörtlichen
  Wortlaut bleiben verworfen.
- **M3** Erzeugung durch das Chat-Modell (`qe_model`, leer = `chat_model`), ein
  Aufruf je Fenster von `qe_fenster` Zeichen (Default 6000), höchstens
  `qe_fenster_max` Fenster (Default 4) je Datei, höchstens `qe_fragen` Fragen
  (Default 6) je Fenster. Antwortformat je Paar zwei Zeilen: `FRAGE: …` und
  `BELEG: …`. Mehrere Belege einer Frage stehen in mehreren `BELEG:`-Zeilen.
- **M4** Kandidatenfenster sind alle Satzfolgen mit eigenem Text bis
  `chunk_chars` Zeichen und mindestens `chunk_chars`/2 Zeichen.
- **M5** Der Nutzen eines Fensters ist: Zahl der neu überdeckten Belegstellen
  plus Zahl der neu vollständigen Fragen (alle ihre Belege im Fenster) plus
  α je neu überdecktem Zeichen, α = 0,02.
- **M6** Auswahl: solange ein Kandidat in das Budget passt, wird der Kandidat mit
  dem größten Nutzen je Zeichen genommen. Budget ist
  `qes_budget_faktor` × Dokumentlänge (Default 1,35). Stop, wenn kein Kandidat
  passt oder der Nutzen 0 ist.
- **M7** Sätze außerhalb jedes gewählten Fensters werden in Chunks bis
  `chunk_chars` gepackt und mit ihren Positionen einsortiert (K3).
- **M8** Ohne verwertbares Frage-Beleg-Paar oder bei Ausfall des Modells gilt
  `chunk_text` (Strategie `fenster`), und der Bericht trägt die Warnung
  `QE-Erzeugung ohne Ergebnis, Rückfall auf Fenster` mit Dateinamen. Kein
  stiller Wechsel (N4).
- **M9** Die Zahl der Modellaufrufe je Datei ist höchstens `qe_fenster_max`.

#### N. Entscheider

- **N1** `strategie(dokument) -> Entscheidung` ist eine reine Funktion: gleicher
  Text ergibt gleiche Strategie, ohne Modell, ohne Netz.
- **N2** Kennzahlen: `kopf_anteil` = Anteil der Nicht-Leerraum-Zeichen, die unter
  einer Überschrift der Ebene ≥ `struktur_ebene` liegen; `koepfe_pro_1000` =
  erkannte Überschriften je 1000 Zeichen; `xrefs_pro_1000` = erkannte Verweise je
  1000 Zeichen.
  *Präzisiert 2026-11-08 nach einem Fehlgriff im echten Lauf:* In `kopf_anteil`
  zählen nur Abschnitte, deren eigener Text höchstens 1,5 × `chunk_chars` Zeichen
  lang ist. Grund: Ein PDF mit einer einzigen Kapitelüberschrift über fast den
  ganzen Text erreichte 0,60 Kopfanteil und wurde als strukturiert eingestuft,
  obwohl es Fließtext ist. Ein Abschnitt, der nicht in einen Chunk passt, ist
  keine Struktur, die dem Chunking hilft.
- **N3** Wert =
  `0.5 · min(1, kopf_anteil/0.60) + 0.3 · min(1, koepfe_pro_1000/2.0) + 0.2 · min(1, xrefs_pro_1000/1.0)`.
  Ergebnis `struktur`, wenn Wert ≥ `struktur_schwelle` (Default 0.60), sonst `qes`.
- **N4** Der Bericht führt je Datei eine Zeile mit `strategie`, Wert, den drei
  Kennzahlen und der Zahl der Chunks. Bei Übersteuerung nennt er
  `übersteuert` statt des Werts.
- **N5** `arbiter index <rag> --strategie auto|struktur|qes|fenster`; `auto` ist
  der Standard und folgt N3. Bei `fenster` wird kein Entscheider befragt.

#### O. Indexierung

- **O1** Gechunkt wird je Datei (D20). Die Seiten einer Datei werden in
  Lesereihenfolge mit `\n` verkettet; die Seitenanfänge werden als Positionen
  mitgeführt.
- **O2** `seite` ist die Seite, in der `start` liegt; `seite_ende` die Seite, in
  der `ende` liegt. `eltern` (bei `eltern_seite: true`) ist der Text aller Seiten
  von `seite` bis `seite_ende`.
- **O3** `Store.page_text(rag, quelle, seite)` liefert die Zeilen, deren Bereich
  die Seite einschließt (`seite <= s <= seite_ende`). Ohne diese Änderung würde
  ein seitenübergreifender Chunk bei der Satzprüfung nicht gefunden.
- **O4** Die Tabelle erhält die Spalten `strategie` und `seite_ende`. Fehlen sie
  in einer Tabelle einer älteren Version, werden sie nachgerüstet (B10).
- **O5** Der Fingerabdruck (B12) enthält zusätzlich `chunk_strategie`,
  `struktur_ebene`, `struktur_schwelle`, `xref_max`, `xref_chars`,
  `qes_budget_faktor`, `qe_fenster`, `qe_fenster_max`, `qe_fragen`, `qe_model`.
- **O6** Abschnitts-ID bleibt `<quelle>#<seite>#<laufende Nummer>`, die Nummer
  zählt je Datei.

### 11.4 Neue Konfigurationsschlüssel

| Schlüssel | Default | Bedeutung |
|---|---|---|
| `chunk_strategie` | `auto` | `auto`, `struktur`, `qes`, `fenster` |
| `struktur_ebene` | 2 | ab dieser Überschriftenebene beginnt ein Abschnitt |
| `struktur_schwelle` | 0.60 | Wert, ab dem der Entscheider `struktur` wählt |
| `xref_max` | 2 | Verweise je Chunk |
| `xref_chars` | 320 | Zeichen des Ziel-Rumpfes im Anhang |
| `qes_budget_faktor` | 1.35 | Indexbudget als Vielfaches der Dokumentlänge |
| `qe_fenster` | 6000 | Zeichen je QE-Anfrage |
| `qe_fenster_max` | 4 | Anfragen je Datei |
| `qe_fragen` | 6 | Fragen je Fenster |
| `qe_model` | `""` | leer = `backends.chat_model` |

A3 gilt: Ein unbekannter Schlüssel bleibt ein Fehler, die neuen Schlüssel werden
in `_RAG_KEYS` aufgenommen.

### 11.5 Was die Stufe nicht tun darf

- **Q5** Text verlieren. Jedes Zeichen steht in einem Chunk (K3).
- **Q6** Eine Strategie stillschweigend tauschen. Jeder Rückfall steht mit Grund
  im Bericht (M8, N4).
- **Q7** Den Entscheider ein Modell aufrufen lassen (N1).
- **Q8** Neue Laufzeit-Abhängigkeiten einführen. Die Stufe nutzt die
  Standardbibliothek und vorhandene Pakete.
- **Q9** Einem Verweis einen Anhang geben, der länger als `xref_chars` ist, oder
  denselben Abschnitt zweimal anhängen.
- **Q10** Die bestehende Suche, den Reranker, das Gate oder den Antwortschritt
  ändern.

### 11.6 Setup

| Punkt | Inhalt |
|---|---|
| Neue Abhängigkeiten | keine |
| Neue Dateien | `src/arbiter/chunk_structure.py`, `src/arbiter/chunk_qes.py`, `src/arbiter/chunk_choice.py`, Tests je Datei |
| Geänderte Dateien | `src/arbiter/chunk.py` (Satzzerlegung, Weiche), `pipeline.py` (Index je Datei), `store.py` (zwei Spalten, `page_text`), `config.py` (zehn Schlüssel), `cli.py` (`--strategie`) |
| Versionierung | Commit je Stufe (SPEC, 7.1, 7.2, 7.3), je Grün und Refactor |
| Test | `uv run pytest`, `uv run ruff check`, `uv run mypy src` |
| Mutanten | `scripts/mutants.py` um die neuen Funktionen erweitern |
| Netztests | QE-Erzeugung nur hinter `-m integration`; Unit-Tests mit Attrappe (N5) |

### 11.7 Stufen mit Stopp-Punkten

| Stufe | Inhalt | Verhalten | Aufwand |
|---|---|---|---|
| 7.1 | Chunking je Datei, Strategie `struktur`, Entscheider, Bericht, Spalten, CLI-Schalter | K, L, N, O, Q5–Q10 | ~0,5 Tag |
| 7.2 | Strategie `qes` mit QE-Erzeugung über das Chat-Modell und Rückfall | M | ~1 Tag |
| 7.3 | Messung auf dem echten Korpus: Recall@10 vorher/nachher, je Strategie, plus Strategie je Datei | D26 (a) | ~0,5 Tag |

Nach jeder Stufe folgt ein Bericht. Du darfst nach jeder Stufe anhalten.

### 11.8 Messung in 7.3

Die Zahlen aus dem Artikel sind **nicht** übertragbar: ein fiktives Dokument,
BM25, ein Viertel der Fragen als Trainingssatz. Gemessen wird deshalb am eigenen
Korpus mit dem eigenen Goldset.

| Kennzahl | Was sie zeigt |
|---|---|
| Recall@10 je Strategie | ob die Belegseite im Kandidatenpool liegt |
| Strategie je Datei mit Kennzahlen | ob der Entscheider plausibel trennt |
| Chunks je Strategie, Indexgröße | Preis der Strategie |
| Complete-Evidence-Recall | nur mit D26 (b), sonst offen |

Ein Ergebnis, das nur insgesamt besser ist, sagt nichts. Bleibt der Recall
gleich, steht das im Bericht als „ohne Wirkung", wie bei HyDE (D3).

**Status:** Entwurf 2026-11-08. Wartet auf Freigabe. Kein Code vor der Freigabe.

---

### 11.9 Revision nach Lesen der Papiere (2026-11-08)

Beide Papiere sind abrufbar: arXiv 2610.36557 (QESChunker, 27.10.2026) und
arXiv 2610.26313 (StrucChunk, 19.10.2026). Der erste Entwurf dieses Abschnitts
stützte sich nur auf den Blog. Der Blog vereinfacht beide Verfahren und erfindet
Parameter. Die folgenden Punkte ersetzen die genannten Kriterien. Die alten
Punkte bleiben lesbar stehen und sind mit „überholt" markiert.

#### Abweichungen QESChunker

| Punkt | Blog | Papier | Folge |
|---|---|---|---|
| Nutzen | Belegstelle überdeckt | gewichtetes Paar `(Frage, Beleg)` mit `w(j,r) = 1/(m·|R_j|)` | M5 überholt |
| Auffindbarkeit | nicht modelliert | **Hit-Gate**: ein Chunk zählt nur, wenn er zusätzlich einen Treffer-Block `H_j` enthält — die `κ=5` ähnlichsten Blöcke der Frage-Embedding | M5 neu: braucht den Embedder beim Ingest |
| Strukturterm | α = 0,02 je Zeichen (frei erfunden) | `λ(b)`: Absatzende 0,1 · λ, Satzende 0,005 · λ, Klauselende 0,001 · λ, sonst 0; λ = 10⁻⁴ bei Greedy | M5 überholt |
| Budget | 1,35 × Zeichenlänge | 3 × Dokumentlänge in Blöcken | M6 überholt |
| Fensterlängen | `chunk_chars`/2 … `chunk_chars` | 10. bis 90. Perzentil der Absatzlänge | M6 überholt |
| Auswahl | Nutzen/Kosten, Stop bei Nutzen 0 | Kosten-Nutzen-Greedy mit **Singleton-Schutz** (bester Einzelchunk als Vergleich), Lazy-Evaluation, bei Sättigung Zusatzterm `f₂` (δ = 0,05) | M6 neu |
| Frage-Beleg-Paare | 6 je Fenster | `n/5` Fragen je Dokument (n = Blöcke), Belege zu 1:1 kurz und kapitelübergreifend, mit automatischen Prüfungen | M3 überholt, neue Grenze D30 |

Das Papier nennt seine eigene Grenze so: Die Methode passt für Dokumente, die
wiederholt befragt werden und für die Überwachung vorliegt. Gemeint ist die Zahl
der **Fragen je Dokument**, nicht die Zahl der Lesevorgänge. In einem Chatsystem
wird ein Dokument bei jeder Frage erneut durchsucht; die Kosten für die
Frage-Beleg-Paare fallen dagegen **einmal** beim Aufnehmen an und werden durch
den Fingerabdruck (B12) nicht wiederholt. Für ein Chatsystem ist die Rechnung
deshalb günstig. Das eigentliche Risiko ist ein anderes: Die Fragen erzeugt ein
Modell aus demselben Text. Passen sie nicht zu den Fragen, die Nutzer wirklich
stellen, dann passt der Index auf erfundene Fragen. Genau das prüft 7.3 mit den
Fragen aus `goldset/downloads.yaml` — die sind von dir geschrieben, nicht erzeugt.

#### Abweichungen StrucChunk

| Punkt | Blog / Entwurf | Papier | Folge |
|---|---|---|---|
| Teilung | Rumpf über `chunk_chars` wird an Satzgrenzen geteilt | **logische Atomizität**: Abschnitt bleibt ganz bis 1,5 × L, erst darüber Teilung; dann Absatz-Packen mit Überlappung ρ | L4 überholt |
| Präfix | Überschriftpfad | Breadcrumb **mit Dokumenttitel**: `[CrPC 1973 > Chapter XII > Section 167]`, Trenner ` > ` | L5 überholt |
| Verweise | 2 Verweise × 320 Zeichen | `r_max = 3`, 200 Zeichen Zusammenfassung des Zielabschnitts | L6 überholt |
| Länge | Zeichen | Tokens, geschätzt als Zeichen/4 | L4 neu: Grenze 1,5 × `chunk_chars` |

StrucChunk wurde auf Rechtstexten mit sauberem PDF-Layout gemessen (pdfplumber,
Abschnitts-DAG, 100 % Grenzpräzision). ArbiterRAG zieht Text mit pypdfium2 ohne
Layout; Überschriften werden aus Zeilenmustern erkannt. Das ist schwächer, und
der Bericht muss es zeigen, nicht behaupten.

#### Was beide Papiere gemeinsam voraussetzen

Beide messen mit **hybridem Retriever** (bge-m3 dicht + BM25, RRF k=60) und
Abschnittstitel-Präfixen. Die aktuelle ArbiterRAG-Konfiguration fährt
`hybrid: false`, weil dicht allein auf diesem Korpus besser war (C6). Die
Gewinne der Papiere sind deshalb **nicht** übertragbar; 7.3 misst es.

Der Entscheider (N) steht in **keinem** der beiden Papiere. Er ist eine Zutat
von ArbiterRAG, begründet nur durch die Heuristik selbst, nicht durch Messung.

#### Neue offene Entscheidungen

| ID | Frage | Optionen | Empfehlung |
|---|---|---|---|
| D30 | Umfang der Frage-Beleg-Erzeugung je PDF (Papier: `n/5` Fragen, bei 1.500 Blöcken also 300 Fragen je Datei) | (a) wie Papier, ohne Obergrenze (b) Obergrenze `qe_fragen_max` (Default 60) je Datei, gleichmäßig über das Dokument verteilt (c) feste Zahl 20 je Datei | **(b)** — die Kosten fallen einmal beim Aufnehmen an, nicht je Frage; der Deckel begrenzt nur den ersten Lauf über ein großes Archiv. Der Wert ist eine Einstellung, kein Gesetz: bei häufig befragten Dokumenten höher setzen |
| D31 | Strukturterm ohne verlässliche Absatzgrenzen (PDF-Text hat oft keine Leerzeile) | (a) Papier treu, Absatz = Leerzeile, ohne Leerzeilen nur Satzstufe (b) nur Satzstufe 0,005 (c) ohne Strukturterm | **(a)** — deterministisch, fällt weich zurück, keine erfundene Absatzerkennung |

#### Geänderte Verhaltenskriterien

- **M3 (neu)** Die Zahl der Fragen ist `min(ceil(n/5), qe_fragen_max)` je Datei,
  n = Zahl der Sätze (D30). Die Fragen werden über das Dokument verteilt, nicht
  je Fenster erzeugt. Belege stehen wörtlich im Text, sonst wird das Paar
  verworfen.
- **M5 (neu)** Nutzen eines Chunks ist die Summe `w(j,r)` über alle Paare, deren
  Beleg `r` im Chunk liegt **und** deren Frage mindestens einen Treffer-Block
  `H_j` im Chunk hat; `H_j` sind die `κ=5` Blöcke mit der größten Kosinus-Ähnlichkeit
  zur Frage-Embedding, berechnet mit dem vorhandenen Embedder. Dazu der
  Strukturterm aus 11.9.
- **M6 (neu)** Budget 3 × Dokumentlänge (Zeichen), Kandidatenlängen vom 10. bis
  90. Perzentil der Absatzlängen, ohne Absätze `chunk_chars`/2 bis `chunk_chars`.
  Auswahl: Kosten-Nutzen-Greedy, danach Vergleich mit dem besten Einzelchunk
  (Singleton-Schutz); ist die Belegabdeckung gesättigt und Budget übrig, gilt
  der Zusatzterm `f₂` mit δ = 0,05.
- **M10 (neu)** Der Greedy-Index ist **kein** Vollindex. Deshalb bleiben M7
  (Lückenchunks) und K3 (kein Textverlust) bestehen — als Zutat von ArbiterRAG,
  nicht als Teil des Papiers.
- **L4 (neu)** Ein Abschnitt bleibt ganz, solange sein Text höchstens
  1,5 × `chunk_chars` Zeichen hat. Erst darüber wird er an Absatzgrenzen, dann
  an Satzgrenzen gepackt, mit Überlappung `chunk_overlap`.
- **L5 (neu)** Breadcrumb `[<Datei> > <Kapitel> > <Abschnitt>]`, Trenner ` > `.
- **L6 (neu)** Höchstens 3 Verweise je Chunk, je 200 Zeichen des Ziel-Rumpfes.

#### Geänderte Konfiguration

| Schlüssel | Default | Änderung |
|---|---|---|
| `qes_budget_faktor` | 3.0 | war 1.35 (Blog) |
| `xref_max` | 3 | war 2 |
| `xref_chars` | 200 | war 320 |
| `qe_fragen_max` | 60 | neu (D30) |
| `qe_fenster` | 6000 | Zeichen je Anfrage zur Frage-Erzeugung (M3) |
| `qe_fenster_max` | 4 | Anfragen je Datei (M3) |
| `qe_hits` | 5 | neu: κ des Hit-Gates |
| `qes_lambda` | 0.0001 | neu: Gewicht des Strukturterms |

`qes_budget_faktor` zählt Zeichen, das Papier Blöcke. Bei Sätzen als Blöcken ist
das Verhältnis nahezu gleich; die Abweichung steht im Bericht.

**Status:** Revision 2026-11-08 nach Lesen beider Papiere. Wartet auf Freigabe
der Abschnitte 11.1–11.9 und der Entscheidungen D20–D31.

---

### 11.10 Umsetzungsbefund nach der ersten Messung (2026-11-08)

**Lazy-Evaluation ist Pflicht, nicht Feinschliff.** Der naive Greedy prüft bei jeder
Auswahl alle Kandidatenfenster. Auf `ix.26.13.pdf` (626 326 Zeichen, 1309
Struktur-Abschnitte) lief er über zwei Stunden ohne Ergebnis und wurde abgebrochen.
Mit Lazy-Evaluation über einen Heap, wie im Papier (Algorithmus 2): **47 Sekunden**.
Das Verfahren selbst ist unverändert, der Nutzen eines einmal berechneten Kandidaten
ist eine obere Schranke, weil die Abdeckung nur wachsen kann.

**Grenze der Methode auf großen Dokumenten.** Dasselbe PDF ergibt 5322 QES-Abschnitte
und einen Index vom 3,12-fachen der Dokumentlänge — genau das Budget aus dem Papier.
Der Ingest ist damit vor allem ein Embedding-Aufwand: für den Hit-Gate wird jeder
Satz eines Dokuments eingebettet. Bei 500 PDFs sind das Stunden, nicht Minuten.

**`f₂` und Singleton-Schutz sind auf unseren Korpora nicht beobachtbar.** Bei
normalisierten Gewichten (`w = 1/(m·|R_j|)`, Summe 1) hat das Fenster, das mehr
Belege einer Frage trägt, immer die höhere Dichte als mehrere kleine Fenster. Der
Greedy deckt die Belege deshalb schon optimal ab, bevor `f₂` greifen kann. Beide
Bauformen sind implementiert und stehen im Code; kein Test kann sie von einer
Variante ohne sie unterscheiden. Ein Test, der das behauptet, wäre Theater — es gibt
keinen. Ein entsprechender Mutant ist in `scripts/mutants.py` mit Begründung
ausgelassen.

**Erste Messung (Fusion ohne Reranker, 31 Goldfragen, eigener Korpus):**

| Strategie | Abschnitte | Index (Zeichen) | Recall@10 |
|---|---|---|---|
| fenster (bisheriger Stand) | 1511 | 943 914 | 0,538 |
| struktur | 1643 | 860 839 | **0,692** |
| qes | 6180 | 2 469 004 | 0,538 |
| auto | 6180 | 2 469 004 | 0,538 |

**Der Entscheider verliert.** Er wählte für alle fünf Dateien `qes` (Werte 0,00 bis
0,48, alle unter der Schwelle 0,60) — und damit die Strategie mit dem schlechtesten
Ergebnis und dem 2,6-fachen Index. `struktur` war in der Messung um 15,4
Prozentpunkte besser, obwohl der Entscheider diesen Dateien kaum Struktur zubilligt.
Der Zusammenhang zwischen meinem Maß und dem Messergebnis ist auf diesem Korpus
nicht gegeben. Die Formel aus N3 ist damit eine unbelegte Annahme, keine Begründung.

---

### 11.11 Entscheider, zweiter Entwurf (2026-11-08, wartet auf Freigabe)

Auftrag: „Das System soll als plugable System laufen, deshalb geht
`chunk_strategie: struktur in arbiter.yaml` nicht. Denke erneut über einen Entscheider
nach! vlt ist ein Decision Modell hilfreich, oder eine Decision Model Abfrage an ein
LLM".

Festlegung aus dem Auftrag: **die Strategie ist keine Konfiguration je Korpus.** Eine
neue Strategie im System muss ohne Änderung an `arbiter.yaml` mitspielen.

#### Warum die drei naheliegenden Wege nicht reichen

| Bauform | Was sie leistet | Warum sie hier nicht genügt |
|---|---|---|
| Heuristik über Überschriften (N3, gebaut) | kostenlos, deterministisch | gemessen falsch: sagte `qes` für alle Dateien voraus, `struktur` gewann |
| LLM gefragt „welcher Chunker passt?" | billig, erklärbar | das Modell kennt das Messergebnis nicht; es beurteilt die Form des Textes, nicht die Auffindbarkeit. Auf diesem Korpus hätte es `struktur` kaum erkannt, weil die Überschriften schwach sind |
| gelerntes Entscheidungsmodell | schnell, wenn trainiert | es gibt 5 Dateien und 31 Fragen. Ein Modell darauf ist nicht unterscheidbar von Raten |

Alle drei schätzen. Das Messergebnis ist aber direkt herstellbar: für ein Dokument
lassen sich die Kandidaten **messen**, bevor der Index geschrieben wird.

#### Vorschlag: der Entscheider misst statt zu schätzen

Für eine neue Datei, vor dem Schreiben:

1. **Prüffragen erzeugen** mit der vorhandenen Maschine aus `chunk_qe` — dieselbe, die
   `qes` als Überwachung nutzt. Die Fragen werden **zweigeteilt**: Hälfte A zum
   Anpassen, Hälfte B zum Prüfen.
2. **Jede Kandidatenstrategie bauen**, nur im Speicher: `struktur`, `fenster`, `qes`.
   `qes` sieht ausschließlich Hälfte A (sonst optimiert es auf die Prüffragen und
   gewinnt per Konstruktion).
3. **Prüfen auf Hälfte B**: für jede Frage die Einbettung der Frage gegen die
   Chunks der Strategie, die besten `fuse_top_k` nehmen, und zählen, ob alle
   Belegsätze der Frage in einem dieser Chunks stehen — Complete-Evidence-Recall,
   die Kennzahl aus dem Artikel.
4. **Wählen**: höchster Wert gewinnt. Bei Gleichstand gewinnt der kleinere Index,
   danach die feste Reihenfolge `struktur` > `fenster` > `qes`.
5. **Bericht**: je Datei, je Strategie der gemessene Wert und die Indexgröße. Diese
   Zeile ist der Beleg des Entscheiders — nachprüfbar, nicht behauptet.

Damit ist der Entscheider pluggable: Jede Strategie, die sich in die Weiche einträgt,
konkurriert automatisch. Kein YAML-Schlüssel, keine Schwelle, kein Trainingssatz.

#### Kosten, ehrlich

Der Entscheider bettet die Chunks jeder Kandidatenstrategie einmal ein — das ist der
teuerste Schritt des Ingests, vervielfacht mit der Zahl der Kandidaten. Gemessen am
heutigen Korpus: `struktur` 860 839 Zeichen, `fenster` 943 914, `qes` 2 469 004. Drei
Kandidaten heißen also rund das Vierfache an Einbettungen gegenüber heute. Für ein
einmal aufgenommenes PDF ist das vertretbar; für 500 PDFs sind es Stunden. Deshalb D35.

#### Offene Entscheidungen

| ID | Frage | Optionen | Empfehlung |
|---|---|---|---|
| D32 | Wie entscheidet der Entscheider? | (a) messen je Datei mit eigenen Prüffragen (b) LLM-Klassifikation über Logprobs (c) gelerntes Modell (d) Heuristik behalten | **(a)** — (b) und (c) schätzen, und die Schätzung ist auf diesem Korpus bereits widerlegt. (b) kostet einen Modellaufruf, liefert aber keine Begründung, die man nachprüfen kann |
| D33 | Woher kommen die Prüffragen? | (a) dieselbe Erzeugung wie `qes`, zweigeteilt (Fit/Prüf) (b) feste Fragen je Korpus in der YAML (c) nur Merkmale, keine Fragen | **(a)** — (b) wäre wieder Konfiguration je Korpus, genau das, was du ausschließt |
| D34 | Was, wenn kein Modell erreichbar ist? | (a) Rückfall auf die Heuristik N3, im Bericht als „geschätzt" markiert (b) Abbruch (c) immer `fenster` | **(a)** — der Ingest darf nicht an einem Modell scheitern, aber die Zeile muss sagen, dass geschätzt wurde |
| D35 | Wie viele Kandidaten darf der Entscheider kosten? | (a) alle registrierten Strategien (b) höchstens drei, `qes` nur wenn sein Index höchstens 3 × Dokument ist (c) höchstens zwei | **(b)** — `qes` ist auf diesem Korpus 2,6-fach und ohne Gewinn; die Messung selbst soll ihn aber nicht ausschließen, wenn er billig ist |
| D36 | Prüft der Entscheider mit dichtem Retriever oder mit dem echten Hybrid? | (a) wie der RAG konfiguriert ist (`hybrid`, `fts`), also mit dem echten Retriever (b) immer dicht, weil billiger | **(a)** — der Entscheider soll messen, was das System tut, nicht ein Modell davon |

#### Verhalten (Abnahmekriterien)

- **R1** `strategie_waehlen(text, kandidaten, prueffragen, embedder, parameter)`
  liefert je Kandidat einen Wert in [0, 1] und die gewählte Strategie. Reine Funktion
  ohne Dateisystem und ohne Store.
- **R2** Die Prüffragen werden zweigeteilt: gerade Indizes an die Anpassung, ungerade
  an die Prüfung. `qes` erhält nur die Anpassungshälfte. Ist eine Hälfte leer, gilt
  die Heuristik (D34).
- **R3** Ein Beleg zählt als gefunden, wenn **alle** seine Belegsätze in einem der
  `fuse_top_k` besten Chunks stehen (Complete-Evidence-Recall, wie im Artikel).
  Teilweise gefundene Belege zählen nicht.
- **R4** Bei Gleichstand gewinnt der kleinere Index (Summe der Chunklängen), danach
  die feste Reihenfolge `struktur` > `fenster` > `qes` > weitere in Registrierreihenfolge.
- **R5** Der Bericht je Datei nennt je Kandidat den Wert und die Indexgröße und
  markiert, ob gemessen oder geschätzt wurde.
- **R6** Der Index selbst enthält nur die Chunks der gewählten Strategie.
- **R7** Ohne Prüffragen oder ohne Embedder gilt die Heuristik N3; der Bericht trägt
  `geschätzt` und den Grund.
- **R8** Der Entscheider schreibt nichts in den Store und ruft kein Chat-Modell
  außer über die vorhandene Frage-Erzeugung.

#### Setup

| Punkt | Inhalt |
|---|---|
| Neue Abhängigkeiten | keine |
| Neue Datei | `src/arbiter/chunk_decide.py`, `tests/test_chunk_decide.py` |
| Geändert | `pipeline.py` (Weiche ruft den Entscheider, Bericht), `chunk.py` (Registry der Strategien statt fester Verzweigung) |
| Aufwand | ~1 Tag samt Tests und erneuter Messung |
| Messung | gleiche Prozedur wie 7.3; zusätzlich: Wie oft wählt der Entscheider die gemessen beste Strategie? (Trefferquote) |

**Status:** Entwurf 2026-11-08. Wartet auf Freigabe von D32–D36 und R1–R8.
Kein Code vor der Freigabe.

---

### 11.12 Entscheider, dritter Entwurf: binärer Strukturdetektor (2026-11-08, umgesetzt)

Auftrag: „Also willst du mir erzählen, dass du nicht in der Lage bist einen Entscheider
zu implementieren, der Unterscheiden kann, ob ein Ingest Überschriften und Struktur
hat oder eine Mail oder ein Text ohne Struktur?" — Antwort: doch, das ist möglich, und
es ist gebaut. Die Punktzahl aus N3 ist **ersetzt**.

#### Was der Detektor fragt

Nicht „welche Strategie holt mehr Belege" — das ist eine Messung, keine Eigenschaft des
Textes. Er fragt nur: **hat dieser Text eine Gliederung, an der der Chunker schneiden
kann?**

- **Ja** (`struktur`), wenn mindestens **zwei** Überschriften erkannt werden **und** der
  Median der Abschnittslängen höchstens **1,5 × `chunk_chars`** beträgt.
- **Nein** (`fenster`), sonst. Eine Mail, ein Chatprotokoll oder ein Fließtext ohne
  Überschriften ergibt null Überschriften. Ein Artikel mit zwei Kapitelüberschriften
  über 20 000 Zeichen hat Überschriften, aber keine nutzbaren Einheiten — der Median
  entscheidet, nicht die Anzahl allein.

`qes` ist nicht mehr Rückfall. Grund: gemessen (7.3 und 7.3b) war `qes` auf keiner Datei
besser als `fenster`, kostet aber das 2,6-fache an Index und Modellaufrufe. `qes` bleibt
über `--strategie qes` erreichbar.

#### Was sich am Vertrag ändert

| Punkt | vorher | nachher |
|---|---|---|
| N3 | Punktzahl aus Kopfanteil, Überschriftendichte, Verweisdichte; Schwelle 0,60 | binärer Detektor, keine Schwelle |
| Rückfall | `qes` | `fenster` |
| `struktur_schwelle` | Konfigurationsschlüssel | **entfernt**; wer ihn setzt, bekommt den Fehler aus A3 |
| `wert` im Bericht | Punktzahl | Median-Abschnittslänge in Zeichen |
| Berichtszeile | Zahl plus Kennzahlen | Klartext: „Gliederung erkannt, Median-Abschnitt 766 Zeichen" |

#### Beleg (Korpusebene)

*Berichtigt 2026-11-08:* Eine erste Fassung dieses Abschnitts enthielt eine Tabelle
„Recall je Datei". Diese Zahlen waren falsch: das Hilfsskript prüfte nur die Seitenzahl
und nicht die Quelle, also galt jeder Treffer auf Seite 5 als Beleg. Das Skript prüft
jetzt beide Felder. Eine belastbare Pro-Datei-Zahl braucht mehr Fragen je Datei, als das
Goldset hergibt (4 bis 6). Gemessen wird deshalb auf Korpusebene.

| Strategie | Abschnitte | Index | Recall@10 |
|---|---|---|---|
| fenster | 1511 | 943 914 | 0,538 |
| struktur | 1643 | 860 839 | 0,692 |
| qes | 6180 | 2 469 004 | 0,538 |
| **auto (Detektor)** | **1643** | **860 839** | **0,692** |

Der Detektor wählt `struktur` für vier Dateien (Median-Abschnitte 250 bis 3735 Zeichen)
und `fenster` für das eine Trennblatt ohne Überschriften. Ergebnis: genau der Wert der
besten gemessenen Strategie, ohne einen einzigen Konfigurationsschlüssel, mit kleinerem
Index als der bisherige Stand.

**Grenze, die bleibt:** Welche Strategie beim *Suchen über den ganzen Korpus* gewinnt,
hängt auch davon ab, wie viele konkurrierende Chunks eine Datei beisteuert. Diese Zahl
misst der Detektor nicht. Er entscheidet über die Form des Dokuments, nicht über das
Retrieval-Ergebnis. Auf diesem Korpus fällt beides zusammen — das ist ein Messergebnis,
keine Garantie.

---

### 11.14 PaddleOCR-VL als Textquelle mit Tabellen-Chunking (2026-11-08, Entwurf)

Auftrag: „Setze B um inkl des chunking für Tabellen" und „Paddle nutzt otl oder
ähnlich".

Gemeint ist **OTSL** (Open Table Structure Language): Paddle gibt Tabellen als
Tokenfolge aus, zum Beispiel `<fcel>Betrag<fcel>Datum<fcel>Posten<nl><fcel>940,00…`.
Zellen werden durch `<fcel>` (Inhalt), `<ecel>` (leer), `<nl>` (Zeilenende) markiert;
verbundene Zellen durch `<lcel>`, `<ucel>`, `<xcel>`. Welche Teilmenge das laufende
Modell liefert, ist **nicht bestätigt**: der Probelauf lief in ein Timeout beim
Modellwechsel. Das erste Arbeitspaket ist deshalb die Formatprüfung an einer Seite.

#### Was heute passiert

| Stelle | heute | Folge |
|---|---|---|
| `ocr.py: dots_to_text` | reduziert dots-Layout-JSON zu Fließtext | Überschriften und Tabellen sind weg |
| `pipeline` | indiziert den dots-Text | Paddle-Ausgabe (`ocr_paddle`) wird nie indiziert |
| `loaders` | OCR nur bei fehlender Textebene | bei deinem Korpus läuft OCR fast nie |

#### Offene Entscheidungen

| ID | Frage | Optionen | Empfehlung |
|---|---|---|---|
| D37 | Wann läuft Paddle als Textquelle? | (a) nur bei fehlender Textebene wie heute (b) auf jeder Seite, wenn `ocr_alle_seiten: true` (c) nur auf Seiten mit erkanntem Tabellengitter | **(b)** — (a) ändert an deinem Korpus nichts, weil alle Seiten eine Textebene haben. (c) ist ein Billigfilter, aber ungetestet; er kann später ergänzt werden |
| D38 | Tabellenerkennung ohne Voll-OCR? | (a) nein (b) ja, über Linien und Rechtecke aus dem PDF (pypdfium2 liefert Seitenobjekte) (c) über Spaltenabstände im extrahierten Text | **(a)** jetzt: (c) ist gemessen wertlos (0 von 8 PDFs mit Spaltenabständen), (b) ist ein eigenes Arbeitspaket |
| D39 | Wie fein werden Tabellen gechunkt? | (a) eine Zeile je Chunk mit wiederholter Kopfzeile (b) bis zu fünf Zeilen je Chunk | **(a)** — eine Zeile ist die kleinste Aussage, die man prüfen kann; Bündelung verwässert den Beleg |
| D40 | Verbundene Zellen | (a) Wert in jede abgedeckte Zeile wiederholen (b) leere Zelle als `⟂` markieren (c) Zeile verwerfen | **(a)**, wo der Verbund eindeutig ist, sonst (b) mit Warnung. Nie einen Wert erfinden |
| D41 | Welches Ausgabeformat erwartet der Parser? | (a) OTSL (b) HTML (c) Markdown (d) alle drei | **(d)** — das Modell wird über den Prompt gesteuert, der Probelauf entscheidet, welche Form zurückkommt. Der Parser erkennt die Form am Text, nicht an der Konfiguration |

#### Verhalten (Abnahmekriterien)

- **V1** Die OCR-Strecke liefert zusätzlich zur Textlesung die **strukturierte** Lesung
  (Markdown oder OTSL/HTML). Indiziert wird die strukturierte Lesung, wenn sie
  vorhanden ist; die dots-Lesung bleibt als `ocr_paddle`-Vergleich erhalten.
- **V2** `otsl_zu_tabelle(text)` erkennt die Form und liefert `Tabelle(kopf: tuple[str,…],
  zeilen: list[tuple[str,…]])`. Unbekannte Token werden als Zellgrenze behandelt und
  erzeugen eine Warnung, nie einen stillen Verlust.
- **V3** `tabellen_chunks(markdown)` findet Tabellenblöcke und liefert je Tabelle:
  einen Kopf-Chunk mit Kopfzeile und bis zu `tabellen_kopf_zeilen` Zeilen (Default 3),
  danach je Zeile einen Chunk der Form
  `<Pfad> | Kopf: Betrag, Datum, Posten | Zeile: 940,00, 01.04.2026, Miete`.
- **V4** Kein Zeilen-Chunk ohne seine Kopfzeile. Ist die Kopfzeile im Dokument nicht
  erkennbar, trägt der Chunk `Kopf: unbekannt` und der Bericht eine Warnung.
- **V5** Ein Zeilen-Chunk ist höchstens `chunk_chars` Zeichen lang; breitere Zeilen
  werden an Zellgrenzen geteilt, mit wiederholter Kopfzeile.
- **V6** Der Struktur-Chunker schneidet nicht in eine Tabelle: ein Tabellenblock ist
  eine Einheit, auch wenn er über mehrere Seiten läuft.
- **V7** Die Satzprüfung verwirft Tabellenzeilen nicht mehr als „kein Satz". Ein
  Zeilen-Chunk gilt als eine Aussage.
- **V8** Der Indexbericht nennt je Datei Zahl der Tabellen und Zeilen-Chunks.
- **V9** Ohne Paddle-Ausgabe bleibt alles wie heute; kein Absturz, Warnung im Bericht.

#### Neue Konfiguration

| Schlüssel | Default | Bedeutung |
|---|---|---|
| `tabellen_kopf_zeilen` | 3 | Zeilen im Kopf-Chunk |
| `tabellen_max_zeilen` | 200 | Obergrenze je Tabelle, darüber Warnung |
| `ocr_paddle_prompt` | `OCR:` | Prompt; der Probelauf legt den Wert für strukturierte Ausgabe fest |

#### Arbeitspakete

| Paket | Inhalt | Aufwand |
|---|---|---|
| 1 | Probelauf: eine Seite aus `ix.26.13.pdf` und eine Rechnung, drei Prompts, Ausgabeformat feststellen | 0,5 h |
| 2 | `otsl_zu_tabelle` + `tabellen_chunks` mit Tests (OTSL, HTML, Markdown) | 3 h |
| 3 | Loader: strukturierte Lesung indizieren, `ocr_paddle` behalten | 1 h |
| 4 | Prüfschicht: Tabellenzeile als Aussage | 1 h |
| 5 | Messung: Recall@10 auf dem Korpus, vorher/nachher | 1 h |

**Status:** Freigegeben 2026-11-09 (Auftrag „Führe das zuende“). Pakete 1–2 sind
umgesetzt (Commit `528dd52`); Pakete 3–5 folgen in Abschnitt 11.15.

#### Ergebnis Paket 1 (2026-11-09, llama-swap `paddleocr-vl`)

Drei Seiten, zwei Prompts, Modell über llama-swap (kein Paddle-Paket):

| Fall | `OCR:` | `Table Recognition:` |
|---|---|---|
| Zeitungsseite mit Code (S. 17) | 35 833 Zeichen, treu, keine Struktur | 8 511 Zeichen, OTSL — modelliert die **Seitenspalten** als Tabelle |
| Zeitungsseite mit Anzeigenkasten (S. 145) | — | OTSL — Anzeigenkasten als Zelle, Tabelle selbst nicht erkannt |
| **Steuerbescheid (Bild, eine Seite)** | 50,7 s, 1 475 Zeichen, 80 Zeilen | **3,4 s, 78 OTSL-Token, 13 Zeilen × 5 Spalten**, 23 × `lcel`, 3 × `ucel`, 0 × `xcel` |
| Markdown-Prompt | 104 s, **keine** Pipes, Text verschlechtert (`postgix.de`) | — |

Prüfsumme der Bescheid-Tabelle ohne Ausgabe von Werten: 34 gefüllte Zellen, davon 6 im
deutschen Betragsformat, 1 Datum, 7 reine Textfelder, ein Textfeld mit 362 Zeichen.

**Folgerungen für den Vertrag:**

1. **Format ist OTSL**, bestätigt: `<fcel>`, `<ecel>`, `<lcel>`, `<ucel>`, `<nl>`. `<xcel>`
   trat nicht auf, wird aber nach OTSL-Spezifikation unterstützt.
2. **Das Modell erkennt Dokumenttabellen, keine Layout-Kästen.** Auf dem Steuerbescheid
   liefert es ein konsistentes 5-Spalten-Raster; auf der Zeitungsseite modelliert es
   die Seitenspalten. Der Prompt darf deshalb nicht blind auf jede Seite — D38
   (Tabellenerkennung) bleibt offen, ist aber für Papierdokumente unkritisch, weil dort
   fast jede Seite eine Tabelle hat.
3. **`OCR:` ist der richtige Prompt für Text**, `Table Recognition:` für Tabellen. Beide
   in einem Durchgang zu verlangen, hat nicht funktioniert (Markdown-Prompt).
4. **Geschwindigkeit reicht**: 3,4 s je Seite warm. `ix.26.13.pdf` mit 156 Seiten wäre
   damit in rund neun Minuten durch, nicht in zehn Stunden wie mit PP-StructureV3 auf CPU.

---

### 11.15 Pakete 3–5: Umsetzung (2026-11-09)

Auftrag: „Führe das zuende“ nach Paket 1–2. Dieses Kapitel hält fest, wie V1–V9
umgesetzt werden und welche Vertragspunkte sich dabei sichtbar ändern.

#### Lesung und Index (V1, V9, Paket 3)

- Die OCR-Strecke ruft `paddleocr-vl` mit dem Prompt `Table Recognition:` auf. Der
  Probelauf (11.14) zeigt: dieser Prompt liefert OTSL, `OCR:` liefert Fließtext.
- `ocr_lesung(dots_text, zweitmeinung)` ist eine reine Funktion. Erkennt
  `otsl_zu_tabelle` eine Tabelle, ist die indizierte Lesung das Markdown dieser
  Tabelle. Sonst ist die indizierte Lesung der dots-Text — das ist V9.
- Die dots-Lesung steht weiter im Feld `ocr_paddle` und dient dem Vergleich. Die
  Überlappung wird zwischen indizierter Lesung und dots-Lesung gerechnet.
- Warnungen des Parsers (unbekanntes Token, fehlende Kopfzeile) gehen in den
  Ladewarnungen auf; keine Warnung geht verloren (V2).

#### Tabellen-Chunks (V3–V6, V8)

- `tabellen_chunks(text, *, pfad, max_chars, kopf_zeilen, max_zeilen)` findet alle
  Markdown-Tabellen und liefert ihre Chunks **mit Position im Dokumenttext**. Ohne
  Position würde jede Tabelle auf Seite 1 liegen.
- `tabellen_max_zeilen` (Default 200) schneidet eine längere Tabelle ab und schreibt
  eine Warnung. Kein stilles Abschneiden.
- Vor dem Chunking der Strategie `struktur` und `fenster` wird jeder Tabellenblock
  durch Leerraum gleicher Länge ersetzt. Damit kann kein Fenster und kein Abschnitt
  in eine Tabelle schneiden (V6), und der Tabellentext steht nur einmal im Index.
  `qes` bleibt unverändert; die Strategie ist nur über `--strategie` erreichbar.
- Der Bericht führt je Datei `tabellen` und `zeilen_chunks`; die CLI-Zeile nennt
  beide Zahlen, wenn sie nicht null sind (V8).

#### Prüfschicht (V7, Paket 4)

- Eine Tabellenzeile ist eine **eigene Aussage**. `split_claims` trennt eine
  Tabellenzeile nicht an Satzzeichen, sondern an der Zeile. Eine Zeile ohne
  Satzende bleibt eine Aussage und wird nicht verworfen.
- Nennt die erste Zeile eines Tabellenblocks eine Quelle, gilt diese Quelle für die
  folgenden Zeilen desselben Blocks.
- Folge: Eine falsche Zeile wird allein gestrichen; die belegten Zeilen bleiben.

#### Vertragsänderung

| Punkt | vorher | nachher |
|---|---|---|
| `ocr_paddle_prompt` | Default `OCR:` | Default `Table Recognition:` — der Probelauf hat den Prompt für die strukturierte Lesung festgelegt |
| `tabellen_kopf_zeilen` | nicht vorhanden | Default 3 |
| `tabellen_max_zeilen` | nicht vorhanden | Default 200, darüber Warnung |
| V3 | Funktion `tabellen_chunks` geplant | `tabellen_chunks` liefert Chunks mit `start`/`ende` im Dokumenttext |
| Fingerabdruck | ohne Tabellen-Schlüssel | enthält beide Tabellen-Schlüssel und `ocr_paddle_prompt` |

#### Messung (Paket 5)

Gemessen mit `scripts/messe_chunking.py` auf dem echten Korpus (fünf deutsche Dateien,
zwei weitere PDFs fallen unter den Sprachfilter), dichte Suche, Recall@10:

| Strategie | Abschnitte | Index | Recall@10 |
|---|---|---|---|
| fenster | 1511 | 943 919 | 0,538 |
| struktur | 1643 | 860 844 | **0,692** |
| auto (Detektor) | 1643 | 860 844 | **0,692** |

Der Detektor wählt `struktur` für vier Dateien und `fenster` für das Trennblatt. Damit
gilt der Wert 0,692 jetzt auch für den echten Index; vorher standen dort 1610 Zeilen
aus dem alten Chunking. Die Abweichung zur früheren Zeile (943 914 zu 943 919 Zeichen)
kommt vom Tabellen-Kopf-Chunk des Trennblatts.

Der Tabellenpfad selbst wird durch den Korpus nicht gemessen: nur das Trennblatt hat
keine Textebene, und es trägt keine Datentabelle. Die Zahl für den Pfad kommt aus dem
Bescheid-Bild über `scripts/pruefe_tabellenpfad.py`: 5 Spalten, 12 Zeilen, daraus ein
Kopf-Chunk und 12 Zeilen-Chunks (drei breite Zeilen an Zellgrenzen geteilt), längster
Chunk 571 Zeichen, keine Warnung.

---

### 11.16 Layout-Erkennung mit einem YOLO-Modell (2026-11-09, Entwurf)

Auftrag: „Finde einen Artikel über ein yolo model, dass segmente in Dokumenten erkennt
und prüfe ob wir das einsetzen können“ und „Deutsche Steuerbescheide <- löse dich davon.
Schreibe SPEC“.

Der Steuerbescheid war die Probe, nicht der Gegenstand. Diese Stufe gilt dem
konfigurierten Korpus: fünf deutsche PDFs mit Textebene — ein 156-Seiten-Magazin, drei
Fachartikel beziehungsweise Foliensätze und ein Trennblatt ohne Textebene. Tabellen in
technischen Dokumenten, Anzeigenkästen und Codeblöcke sind die Fälle, die zählen.
Steuerformulare kommen im Korpus nicht vor.

#### Ausgangslage

| Punkt | Stand heute |
|---|---|
| Tabellen mit Textebene | Der Text steht als Fließtext im Fenster-Chunk. Es gibt keine Zeilen-Chunks, weil der Tabellenpfad nur auf OCR-Seiten läuft |
| Seiten ohne Textebene | Genau eine (Trennblatt). `Table Recognition:` über die ganze Seite lieferte dort eine Tabelle mit Kopfzeile und null Zeilen — ein Fehlalarm |
| Tabellen im Korpus | Die Zahl ist **nicht bekannt**. Ein Zensus ist Teil der Messung |
| Erkennung von Regionen | Es gibt keine. Der Struktur-Chunker liest Zeilen und Überschriften, keine Kästen |

Probe mit dem echten Modell (CPU, nur Klassen, Kästen und Zeiten ausgegeben):

| Fall | Ergebnis |
|---|---|
| Bescheid, 720 × 1022 | 567 ms, 15 Kästen, **eine Tabellenregion** (conf 0,55) genau auf dem Betragsblock |
| ix.26.13, Seiten 5, 17, 40 (Textseiten) | 565–698 ms, je **0 Tabellen** |
| ix.26.13, Seite 145 (Anzeigenkasten) | 698 ms, 1 Tabelle + 1 Table Footnote, conf 0,63 — das ist **keine** Datentabelle |
| Ausschnitt der erkannten Tabelle → `paddleocr-vl` | 2 Spalten × 4 Zeilen, 1 Zeilen-Chunk. Das Bild hat 87 dpi; bei 300 dpi wäre der Ausschnitt rund 560 × 465 px |
| `Table Recognition:` über die **ganze** Seite | 5 Spalten × 12 Zeilen: das Modell las die Seitenspalten als Tabelle. Der Ausschnitt ist die bessere Einheit |

#### Das Modell

DocLayout-YOLO (arXiv 2410.12628, Oktober 2024, opendatalab). YOLOv10-basiert. Das Paper
nennt mAP 70,3 (D4LA), 79,7 (DocLayNet) und 78,8 (DocStructBench) bei 85,5 FPS auf GPU.
Das Modell erkennt zehn Klassen: Title, Plain Text, Abandoned Text, Figure, Figure
Caption, **Table**, Table Caption, Table Footnote, Isolated Formula, Formula Caption.

| Artefakt | Lizenz | Verwendung |
|---|---|---|
| Code `opendatalab/DocLayout-YOLO` (baut auf ultralytics) | AGPL-3.0 | **nicht** übernehmen |
| Gewichte `DocLayout-YOLO-DocStructBench`, imgsz 1024, 40,7 MB | Apache-2.0 | zulässig |
| ONNX-Export derselben Gewichte, 75 MB | Apache-2.0 | zulässig |
| Gewichte `imgsz1280-2501` | AGPL-3.0 | **nicht** nehmen |
| `onnxruntime` | MIT | zulässig |

Der Weg ist damit festgelegt: Apache-2.0-Gewichte als ONNX, Ausführung über
`onnxruntime`, Vor- und Nachverarbeitung in eigenem Code. Kein AGPL-Code im Repository,
kein Torch, kein `ultralytics`.

#### Offene Entscheidungen

| ID | Frage | Optionen | Empfehlung |
|---|---|---|---|
| D42 | Auf welchen Seiten läuft der Detektor? | (a) jede Seite (b) nur Seiten mit Textebene (c) nur Seiten ohne Textebene | **(a)** — 0,7 s je Seite; Tabellen stehen in beiden Fällen, und der Trennblatt-Fehlalarm verschwindet nur, wenn die Erkennung dort läuft |
| D43 | Welche Klassen werden verwendet? | (a) nur `table` (b) `table` plus `table_caption` und `table_footnote` (c) alle Klassen | **(a)** — der Tabellenleser erwartet das Gitter. Beschriftung und Fußnote bleiben Fließtext |
| D44 | Was passiert mit dem Tabellentext der Textebene? | (a) bleibt stehen, Duplikat (b) Zeilen, deren Mitte im Kasten liegt, fallen weg (c) Zeichen im Kasten fallen weg | **(b)** — Zeilenkästen sind billiger als Zeichenkästen, und Fließtext und Tabellenzeile teilen normalerweise keine Grundlinie. Rückfall (a) mit Warnung, wenn die Koordinate fehlt |
| D45 | Welche Auflösungen? | (a) Erkennung 200 dpi, Ausschnitt 300 dpi (b) beides 300 (c) Erkennung 150, Ausschnitt 300 | **(a)** — 200 dpi reicht für Kästen und kostet 0,7 s; kleine Beträge brauchen 300 dpi im Ausschnitt |
| D46 | Schwelle und Mindestfläche? | (a) Konfidenz 0,25, Fläche 1 % der Seite (b) 0,5 und 1 % (c) 0,25 und 0,5 % | **(a)** — konservativ bei der Fläche; die Messung entscheidet, ob 0,25 Fehlalarme kostet. Der Anzeigenkasten auf S. 145 hatte 0,63 |
| D47 | Was, wenn Modell oder Laufzeit fehlt? | (a) Abbruch (b) Warnung, Layout aus, Ingest läuft (c) Abbruch nur bei `layout: true` | **(b)** — ein optionaler Detektor darf den Ingest nicht anhalten, wie bei der OCR (V9) |
| D48 | Standardwert | (a) `false` bis zur Messung (b) sofort `true` (c) `true` nur in der Messkonfiguration | **(a)** — ohne Messung ist `true` eine Behauptung. Nach positiver Messung wird der Standard sichtbar geändert |

#### Verhalten (Abnahmekriterien)

- **W1** `erkenne(bild)` ist eine reine Funktion und liefert je Region Klasse,
  Konfidenz und Kasten als Seitenanteile in [0, 1]. Netz nur beim Laden des Modells.
- **W2** Verwendet wird der ONNX-Export mit Apache-2.0-Gewichten. Die Datei liegt
  außerhalb des Repositories; der Pfad kommt aus der Konfiguration; ihr SHA-256 steht
  im Bericht.
- **W3** Nur die Klasse `table` zählt, und nur bei `konfidenz >= layout_schwelle` und
  Fläche `>= layout_min_flaeche`. Alle anderen Klassen und Kästen werden ignoriert.
- **W4** Der Ausschnitt ist der Kasten plus `layout_rand`, begrenzt auf die Seite,
  gerendert aus derselben Seite mit `layout_dpi_ocr`.
- **W5** Der Ausschnitt geht mit `ocr_paddle_prompt` an `paddleocr-vl`. Ist das Ergebnis
  eine Tabelle mit mindestens zwei Spalten und mindestens einer Datenzeile, entstehen
  Zeilen-Chunks; sonst Warnung und kein Tabellen-Chunk. Die Chunks tragen die Seite der
  Region.
- **W6** Der Tabellentext fällt aus dem Seitentext, damit er nicht doppelt im Index
  steht (D44). Fehlt die Zeilenkoordinate, bleibt der Text stehen und der Bericht trägt
  eine Warnung. Kein stiller Verlust und keine stille Dopplung.
- **W7** Der Bericht nennt je Datei Seiten mit Tabellen, Tabellen und Zeilen-Chunks.
  Warnungen: Modell fehlt, Erkennung fehlgeschlagen, Tabelle leer, Text nicht entfernt.
- **W8** Auf Seiten ohne Textebene bleibt die dots-Lesung der Text. Mit Layout entfällt
  der Prompt über die ganze Seite; gelesen werden nur erkannte Tabellen. Damit
  verschwindet der Fehlalarm auf dem Trennblatt.
- **W9** Mit `layout: false` ist alles wie heute (V9).
- **W10** Unbekannte Schlüssel bleiben ein Fehler (A3). Der Fingerabdruck enthält
  `layout`, `layout_modell`, `layout_schwelle`, `layout_min_flaeche`, `layout_dpi`,
  `layout_dpi_ocr` und `layout_rand`. Fehlt Modell oder Laufzeit, läuft der Ingest mit
  Warnung weiter, Layout ist aus.

#### Neue Konfiguration

| Schlüssel | Default | Bedeutung |
|---|---|---|
| `layout` | `false` | Erkennung ein/aus (D48) |
| `layout_modell` | `""` | Pfad zur ONNX-Datei |
| `layout_schwelle` | 0.25 | Mindestkonfidenz für eine Region |
| `layout_min_flaeche` | 0.01 | Mindestfläche als Anteil der Seite |
| `layout_dpi` | 200 | Auflösung für die Erkennung |
| `layout_dpi_ocr` | 300 | Auflösung für den Tabellenausschnitt |
| `layout_rand` | 0.01 | Rand um den Kasten als Seitenanteil |

#### Setup

| Punkt | Inhalt |
|---|---|
| Neue Abhängigkeit | `onnxruntime` (MIT, rund 27 MB). Kein Torch, kein `ultralytics` |
| Neues Artefakt | ONNX-Datei, 75 MB, Apache-2.0; Download über `scripts/hole_layout_modell.py`, SHA-256 im Bericht, **nicht** im Repository |
| Neue Dateien | `src/arbiter/layout.py`, `tests/test_layout.py`, `scripts/hole_layout_modell.py`, `scripts/messe_layout.py` |
| Geändert | `loaders.py` (Zeilenkästen, Tabellentext entfernen), `pipeline.py` (Haken, Bericht, Fingerabdruck), `chunk.py` (Chunks aus Ausschnitten), `config.py` |
| Aufwand | rund 2 Tage samt Probe, Tests und Messung |

#### Arbeitspakete

| Paket | Inhalt | Aufwand |
|---|---|---|
| 1 | Formatprüfung: liefert `pypdfium2` Zeilenkästen (`get_rect`) oder nur Zeichenkästen (`get_charbox`)? Eine Seite aus `ix.26.13.pdf`, Ausgabe nur als Zahlen | 0,5 h |
| 2 | `layout.py`: Letterbox, Dekodierung der sechs Werte je Kasten, Rückrechnung auf Seitenanteile, Schwelle, Mindestfläche, Klassen. Tests ohne Modell mit festen Ausgaben | 3 h |
| 3 | Loader und Pipeline: Haken je Seite, Ausschnitt, `paddleocr-vl`, `tabellen_chunks`, Seitenzuordnung, Textentfernung, Bericht | 4 h |
| 4 | Konfiguration, Fingerabdruck, Fail-soft | 1 h |
| 5 | Messung: Zensus, Recall vorher/nachher, Stichprobe von Hand | 2 h |

#### Messung

1. **Zensus**: alle Seiten des Korpus. Zahl der Seiten mit Tabellenregion, Zahl der
   Regionen je Seite. Stichprobe von 20 Seiten (zehn mit Region, zehn ohne) von Hand:
   Ist es eine Datentabelle, ja oder nein?
2. **Vorher/Nachher**: `scripts/messe_chunking.py --strategien auto`. Recall@10 muss
   mindestens 0,692 bleiben. Dazu Zahl der Tabellen-Chunks und Indexgröße.
3. **Zeilen-Qualität**: zehn erkannte Tabellen von Hand zeilenweise vergleichen
   (erkannte Zeilen gegen Zeilen auf der Seite).
4. **Zeit**: Dauer des Ingests vorher und nachher, je Seite und gesamt.
5. **Fehlalarme**: Anteil der Tabellenregionen ohne Datentabelle. Bekannt ist der
   Anzeigenkasten auf S. 145.

Der Bericht steht danach in `docs/EVIDENCE-layout.md`.

#### Grenzen dieser Stufe

- **Keine Lesereihenfolge aus Kästen.** Der Text kommt weiter aus der Textebene. Eine
  Layout-basierte Reihenfolge wäre eine eigene Stufe.
- **Keine gedrehten oder verzerrten Seiten.**
- **Keine eigenen Einheiten für Abbildungen, Formeln und Fußnoten.**
- **Die Erkennung ist ein Modell, keine Garantie.** Fehlalarme werden gezählt, nicht
  weggeredet. Der Anzeigenkasten auf S. 145 ist bereits einer.
- **Die Zeilen-Chunks erben die Grenzen von `tabellen_chunks`**: Kopfzeile, Obergrenze,
  Seitenwechsel, eine Zeile je Chunk.
- **Key-Value und Schema sind nicht Teil der Stufe.**

Alternative, falls die Erkennung auf unseren Layouts versagt: PP-DocLayout (Apache-2.0,
Paddle-Laufzeit) oder PP-StructureV3. Beide brauchen eine Paddle-Installation; sie sind
in dieser Stufe nicht vorgesehen.

**Status:** Entwurf 2026-11-09. Wartet auf Freigabe von D42–D48 und W1–W10.
Kein Code vor der Freigabe.


