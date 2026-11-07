# ArbiterRAG — SPEC

**Status:** freigegeben 2026-11-07 (rev. 2, Antworten D1–D9 eingearbeitet). Offen: D10–D12.
**Zweck:** RAG-System mit HyDE, Reranker und einer Prüfschicht über Logprobs.
**Arbeitsverzeichnis:** `/home/speedy/Dokumente/dev/ArbiterRAG`

Diese SPEC ist der Vertrag. Jede Änderung wird sichtbar ergänzt, nie still
überschrieben.

---

## 1. Was das System tut

Drei Stufen, jede mit einer eigenen Aufgabe:

| Stufe | Aufgabe | Wirkung |
|---|---|---|
| HyDE | erzeugt ein hypothetisches Antwortdokument und sucht damit zusätzlich | verändert die **Kandidatenmenge** (Recall) |
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
| D3 | HyDE | **b** — Schalter, Default an, Original- und HyDE-Suche fusioniert | Verhalten D2 |
| D4 | HyDE-Modell | **llama-swap** | Modell-ID ist Konfigurationswert; Vorgabe `gemma-4-12B` (kein chinesisches Modell) |
| D5 | Reranker | **a** — HTTP `/v1/rerank` | Eintrag `bge-reranker-v2-m3` ist defekt, siehe D11 |
| D6 | Prüfschicht-Umfang | **c** — zuerst Gate, Auswahl und Widerspruch/Mengenprüfung geplant; Widerspruch/Mengen verworfen, weil gemessen ohne Wirkung (F8) | Messung 2026-11-07 |
| D7 | Prüfschicht-Ausfall | **b** — keine Antwort | Verhalten F6, fail-closed |
| D8 | Antwort erzeugen | **a** — ja, abschaltbar | Verhalten G |
| D9 | Goldset | **b** — eigener Korpus, 30–40 originale deutsche Fragen | Stufe 5 |

### Offen

| ID | Frage | Optionen | Empfehlung |
|---|---|---|---|
| D10 | PDF **ohne Textebene** (Scan) | **b** — OCR über llama-swap, Modelle `dots-mocr` **und** `paddleocr-vl` | Verhalten B8; Kombination offen (D13) |
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
- **B8** Eine PDF-Seite ohne Textebene wird bei `ocr: true` mit `ocr_dpi`
  (Default 200) gerendert und an **beide** OCR-Modelle geschickt. `dots-mocr`
  liefert den Text, der indiziert wird: seine Layout-JSON wird in Lesereihenfolge
  zu Text reduziert, ohne Übersetzung, Bilder ohne Text. `paddleocr-vl` läuft mit
  dem Aufgabenschalter `OCR:` auf derselben Seite; seine Ausgabe steht als
  `ocr_paddle` im Datensatz, die Wortüberlappung beider Lesungen als
  `ocr_ueberlappung`. Die beiden Texte werden **nicht** gemischt (D13).
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
- **C5** HyDE-Listen gehen mit `hyde_gewicht` (Default 0.5) in die Fusion ein, die
  Listen der Originalfrage mit 1.0.
- **C6** `hybrid: false` lässt nur die dichte Suche laufen. Gemessen 2026-11-07:
  dichte Suche allein Recall 0,71, mit Volltextteil 0,65 — die wiederholte
  Kopfzeile der Foliensätze belegt sonst viele Plätze.

### D. HyDE

- **D1** Das HyDE-Dokument wird aus Frage und RAG-Beschreibung über
  `/v1/chat/completions` erzeugt und auf `hyde_max_chars` (Default 800) gekürzt.
- **D2** Ist HyDE eingeschaltet (Default an, D3=b), laufen zwei Suchen und
  werden fusioniert: eine mit der Originalfrage, eine mit dem HyDE-Dokument.
- **D3** Schlägt der HyDE-Aufruf fehl, läuft die Suche mit der Originalfrage
  weiter. Genau eine Warnung mit Fehlergrund. Kein Abbruch, kein stiller
  Ausfall.
- **D4** Das erzeugte Dokument wird im Ergebnis mitgeführt (`hyde_text`), damit
  sichtbar ist, was gesucht wurde.

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
  nDCG@10 und Antwortrichtigkeit, je Stufe getrennt: nur Fusion, mit HyDE, mit
  Reranker, mit Prüfschicht.
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
