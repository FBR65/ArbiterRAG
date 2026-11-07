# ArbiterRAG

Retrieval-Augmented Generation mit drei getrennten Stufen: HyDE erweitert die
Kandidatenmenge, ein Reranker ordnet sie, und eine Prüfschicht entscheidet, welche
Passage die Antwort belegen darf. Die Prüfschicht fragt das Chatmodell je Passage
nach Ja oder Nein und liest die Wahrscheinlichkeit aus den Logprobs.

Der Punkt: Ein Reranker kann nicht sagen „keine dieser Passagen belegt die
Antwort". ArbiterRAG trennt Ordnen von Urteilen.

## Stand

| Stufe | Inhalt | Zustand |
|---|---|---|
| 1 | Konfiguration, PDF-Index, Hybrid-Suche, RRF, CLI `index`/`search` | fertig, siehe `docs/EVIDENCE-stufe1.md` |
| 2 | HyDE | fertig, gemessen ohne Nutzen, abgeschaltet |
| 3 | Reranker über `/v1/rerank` | fertig |
| 4 | Prüfschicht: Ja/Nein je Passage über Logprobs | fertig, siehe `docs/EVIDENCE-verbesserungen.md` |
| 5 | Antwort, Goldset, Schwellen aus der Messung | fertig |

Der Vertrag steht in `SPEC.md`.

## Aufbau

```
src/arbiter/
  config.py     YAML, ${VAR}-Expansion, unbekannte Schlüssel sind Fehler
  loaders.py    PDF (pypdf) und Text, leere Seiten werden gezählt
  chunk.py      Abschnitte ≤ 600 Zeichen, an Absatzgrenzen, 100 Überlappung
  embedding.py  dichte Vektoren über /v1/embeddings (OpenAI-kompatibel)
  store.py      LanceDB, eine Tabelle je RAG, dicht + Volltextindex
  fusion.py     Reciprocal Rank Fusion, k=60
  pipeline.py   Indexieren und Suchen
  cli.py        dünner Client der Bibliothek
```

## Benutzung

```bash
uv sync
export ARBITER_CONFIG=examples/arbiter.yaml
export ARBITER_DB=/pfad/zur/datenbank
export ARBITER_BASE_URL=http://127.0.0.1:8080
export ARBITER_EMBED_MODEL=bge-m3-gguf
export ARBITER_CORPUS=/pfad/zum/korpus

uv run arbiter index probe
uv run arbiter search probe "Wie hoch ist die Miete?"
```

Ein neuer Korpus ist ein neuer Eintrag unter `rags:`, kein neuer Code.

## Entwicklung

```bash
uv run pytest -q                      # Unit-Tests, ohne Netz
uv run pytest -q -m integration       # braucht Endpunkt, noch keine Tests vorhanden
uv run ruff check . && uv run mypy
uv run python scripts/mutants.py      # Mutationslauf, setzt sauberen Arbeitsbaum voraus
```

## Lizenzen

Code: MIT. Modelle, die der Client anspricht: `bge-m3` (MIT),
`bge-reranker-v2-m3` (Apache-2.0).
