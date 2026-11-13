# ArbiterRAG

Retrieval-augmented generation over a local document corpus, with a separate decision
step between retrieval and answer generation.

A reranker can only reorder what the search found; it cannot say "none of these passages
answers the question". ArbiterRAG therefore separates ordering from judging. The
reranker orders candidates, a gate decides which candidates may be cited, and the answer
is written only from those passages. Every sentence of the answer is then checked against
the passage it cites before the answer leaves the system.

All models are called through one OpenAI-compatible endpoint (llama-swap on
`localhost`). No external API is contacted at query time.

## Contents

- [Status](#status)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Install](#install)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Command line](#command-line)
  - [Terminal UI](#terminal-ui)
  - [HTTP API](#http-api)
- [Indexing](#indexing)
- [Chunking](#chunking)
- [Tables](#tables)
- [Retrieval and reranking](#retrieval-and-reranking)
- [Gate](#gate)
- [Answer and sentence check](#answer-and-sentence-check)
- [Measured results](#measured-results)
- [Known limitations](#known-limitations)
- [Testing and quality gates](#testing-and-quality-gates)
- [Repository layout](#repository-layout)
- [Specification](#specification)
- [License](#license)

## Status

| Area | State |
|---|---|
| Indexing: PDF and text, deduplication, language filter, OCR | implemented |
| Chunking: `fenster`, `struktur`, `qes`, automatic decider | implemented |
| Table chunking from structured OCR (OTSL) | implemented |
| Layout detection for tables (DocLayout-YOLO over ONNX) | implemented |
| Search: dense retrieval, full-text, reciprocal rank fusion | implemented |
| Reranking via `/v1/rerank` | implemented |
| Gate: one yes/no decision per candidate, read from logprobs | implemented |
| Answer generation with per-sentence support check | implemented |
| HTTP service (FastAPI) | implemented |
| Terminal UI (Textual) | implemented |
| Goldset evaluation | implemented |
| Calibrated sentence check | not implemented |

The status is meant literally. "Not implemented" items are named with their reason in
[Known limitations](#known-limitations); [Measured results](#measured-results) contains
the numbers behind the table.

## Architecture

```
 question
    |
    +- dense search      /v1/embeddings        (bge-m3)
    +- full-text search  LanceDB FTS           (optional)
    |
    +- reciprocal rank fusion                  (k = 60)
    +- reranking         /v1/rerank            (bge-reranker-v2-m3)
    |
    +- gate              /v1/chat/completions  (grammar: "ja" | "nein", logprobs)
    |     one request per candidate; P("ja") decides admission
    |
    +- answer            /v1/chat/completions
    |     every sentence checked against the passage it cites
    |     unsupported sentences are dropped; if none survive, the source is quoted
    |
    +- answer with sources, or a refusal with a reason
```

Two properties define the design:

1. **The gate sees one passage per request.** A single prompt holding all passages made
   the model answer about the whole text instead of about one candidate; every
   probability came back between 0.98 and 0.99. One passage per request separates a real
   hit (0.941) from noise (0.001 to 0.028).
2. **The gate always decides.** Its answer is constrained by a GBNF grammar
   (`root ::= "ja" | "nein"`), so there is no undefined state: the model either says yes
   or no, and the probability of the first token is the decision.

The gate can refuse; the reranker cannot. That is the reason both steps exist side by
side.

## Requirements

- Linux, Python 3.12, [uv](https://docs.astral.sh/uv/)
- A [llama-swap](https://github.com/mostlygeek/llama-swap) instance at
  `http://127.0.0.1:8080` serving these models:
  - `bge-m3` — embeddings (`/v1/embeddings`)
  - `bge-reranker-v2-m3` — reranking (`/v1/rerank`, needs `--reranking`)
  - `qwen2.5-7b-instruct` — gate and answer (`/v1/chat/completions`, needs `logprobs`)
  - `paddleocr-vl` — structured reading of scanned pages; the table prompt returns OTSL
  - `dots-mocr` — layout reading of the same page, kept as the comparison
- For layout detection: `onnxruntime` (installed with the project) and the ONNX file
  with the Apache-2.0 weights of DocLayout-YOLO, about 75 MB. It is not part of the
  repository; fetch it once with `uv run python scripts/hole_layout_modell.py`.
- About 8 GB of GPU memory when all models are resident.

The model names are configuration, not code. Any OpenAI-compatible server that supports
the three endpoints can replace llama-swap; the gate needs `logprobs` and a grammar.

## Install

```bash
git clone git@github.com:FBR65/ArbiterRAG.git
cd ArbiterRAG
uv sync
cp examples/arbiter.yaml arbiter.yaml     # then edit paths and model names
cp examples/arbiter.env arbiter.env       # then edit paths and addresses
```

The CLI and the TUI read `arbiter.env` when a variable is missing; variables exported in
the shell always win. The service (`arbiter-api`) expects its variables from the process
manager and does not read the file itself.

## Configuration

`arbiter.yaml` holds one entry per corpus. `${VAR}` placeholders are expanded from the
environment; an unset or empty variable is an error, never a default. Unknown keys are
rejected, so a typo cannot silently disable a setting.

### Top level and backends

| Key | Required | Meaning |
|---|---|---|
| `db_path` | yes | LanceDB directory; one table per corpus |
| `backends.base_url` | yes | base URL of the model server |
| `backends.embed_model` | yes | embedding model |
| `backends.rerank_model` | yes | reranking model |
| `backends.chat_model` | yes | gate and answer model |
| `backends.ocr_dots_model` | with OCR | layout reading, kept as the comparison |
| `backends.ocr_paddle_model` | with OCR | structured reading; its table text is indexed |

### Corpus entry (`rags.<name>`)

| Key | Default | Meaning |
|---|---|---|
| `description` | — | free text, shown by CLI, TUI and API |
| `corpus_path` | — | root directory, read recursively |
| `extensions` | — | `[".pdf"]`, `[".txt", ".pdf"]`, ... |
| `top_k_dense` | 20 | dense candidates |
| `top_k_fts` | 20 | full-text candidates |
| `fuse_top_k` | 20 | candidates after fusion |
| `rerank_top_n` | 5 | candidates after reranking |
| `hybrid` | true | include the full-text half in the fusion |
| `fts_stem` | false | German stemming in the full-text index |
| `sprache` | `null` | keep only documents in this language (`de`) |
| `chunk_strategie` | `auto` | `auto`, `struktur`, `qes`, `fenster` |
| `chunk_chars` | 600 | characters per chunk |
| `chunk_overlap` | 100 | overlap between windows |
| `struktur_ebene` | 2 | heading level from which a section starts |
| `xref_max` | 3 | resolved cross-references per chunk |
| `xref_chars` | 200 | characters of the referenced section's body |
| `eltern_seite` | false | index small chunks, return the whole page as context |
| `qes_budget_faktor` | 3.0 | index budget as a multiple of the document length |
| `qe_fragen_max` | 60 | generated questions per file |
| `qe_fenster` | 6000 | characters per question-generation request |
| `qe_fenster_max` | 4 | question-generation requests per file |
| `qe_hits` | 5 | chunks a generated question may select |
| `qes_lambda` | 0.0001 | cost weight of the question-evidence selection |
| `qe_model` | `""` | empty = `backends.chat_model` |
| `ocr` | false | OCR on/off |
| `ocr_alle_seiten` | false | OCR every page, not only pages without a text layer |
| `ocr_dpi` | 200 | rendering resolution for `dots-mocr` |
| `ocr_dpi_paddle` | 300 | rendering resolution for `paddleocr-vl` |
| `ocr_paddle_prompt` | `Table Recognition:` | task prompt for the structured reading |
| `tabellen_kopf_zeilen` | 3 | rows in the table head chunk |
| `tabellen_max_zeilen` | 200 | row limit per table; above it a warning is reported |
| `layout` | false | detect table regions on every page and read them separately |
| `layout_modell` | `""` | path to the ONNX file; empty means layout is off |
| `layout_schwelle` | 0.25 | minimum confidence for a region |
| `layout_min_flaeche` | 0.01 | minimum area as a share of the page |
| `layout_dpi` | 200 | resolution for detection |
| `layout_dpi_ocr` | 300 | resolution for the table crop |
| `layout_rand` | 0.01 | padding around the table box, as a page share |
| `gate_kandidaten` | 5 | candidates the gate sees |
| `gate_schwelle` | 0.5 | P(ja) needed for admission |
| `gate_prompt` | `teil` | gate prompt variant: `teil`, `streng`, `beleg` |
| `antwort_passagen` | 3 | passages the answer may cite |
| `antwort_pruefung` | false | check every sentence against its source |
| `antwort_schwelle` | 0.5 | P(ja) needed for a sentence to be kept |
| `antwort_reparatur` | false | reserved; the repair prompt exists, the loop is not wired |

### Environment variables

| Variable | Meaning |
|---|---|
| `ARBITER_CONFIG` | path to `arbiter.yaml` |
| `ARBITER_DB` | LanceDB directory, referenced as `${ARBITER_DB}` |
| `ARBITER_BASE_URL` | model server, referenced as `${ARBITER_BASE_URL}` |
| `ARBITER_EMBED_MODEL` | embedding model, referenced as `${ARBITER_EMBED_MODEL}` |
| `ARBITER_CORPUS` | corpus root, referenced as `${ARBITER_CORPUS}` |
| `ARBITER_BIND` | bind address of `arbiter-api` (default `127.0.0.1`) |
| `ARBITER_PORT` | port of `arbiter-api` (default `8091`) |
| `ARBITER_ENV` | alternative path of `arbiter.env` |

## Usage

### Command line

```bash
uv run arbiter index downloads
uv run arbiter index downloads --strategie fenster
uv run arbiter search downloads "How is the processing run scheduled?"
uv run arbiter ask downloads "Why does LLM-based business intelligence often fail in operations?"
uv run arbiter ask downloads "..." --json
uv run arbiter eval downloads
uv run arbiter eval downloads --goldset goldset/downloads.yaml --limit 10
uv run arbiter-tui
uv run arbiter-api
```

| Command | Effect |
|---|---|
| `index <rag> [--strategie …]` | read the corpus, chunk, embed, upsert, purge |
| `search <rag> <frage>` | passages with source, page, RRF rank and reranker score |
| `ask <rag> <frage> [--json]` | answer with sources, or a refusal with a reason |
| `eval <rag> [--goldset] [--limit]` | recall@10, nDCG@10 and answer quality per stage |

Exit codes: `0` success, `1` failed run (store, embedder, or the model returned no
answer), `2` usage or configuration error. `--strategie` overrides the corpus setting for
this one run; it is not written to the configuration.

`index` prints one summary line and one line per file with the chosen strategy and the
reason, plus the number of tables and row chunks when the file had tables. Warnings
(skipped pages, failed OCR, unreadable files, parser warnings, duplicates) go to stderr.

### Terminal UI

```
Enter    ask
Ctrl+S   switch to search mode (passages with reranker score instead of an answer)
Ctrl+I   index in the background (asks first, because OCR can take hours)
Ctrl+L   clear the transcript
Ctrl+Q   quit
```

The status line shows corpus, chunk count, mode, reranker and gate settings.

### HTTP API

```bash
uv run arbiter-api        # 127.0.0.1:8091
```

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | `{"status": "ok", "version": "<version>"}`; no model call |
| GET | `/rags` | configured corpora with `name`, `description`, `abschnitte` |
| POST | `/search` | `{"rag": "...", "frage": "..."}` returns passages |
| POST | `/ask` | same body; returns an answer with sources, or a refusal with a reason |
| POST | `/index` | `{"rag": "..."}` returns `202 {"job_id": "..."}` |
| GET | `/index/{job_id}` | `status`: `laeuft`, `fertig` or `fehler`; report or message |

Indexing runs in a background thread. A second index run for the same corpus is refused
with `409` while the first one is still running. Errors map to `400` (question too long),
`409` (missing index) and `503` (embedder or gate unreachable). The JSON shapes of
`/search` and `/ask` are the same as `--json` on the command line; one module produces
both, so CLI and API cannot drift apart.

## Indexing

Each corpus maps to one LanceDB table plus one table `<name>_ingest` holding a
fingerprint per file. The fingerprint is a SHA-256 over the file content plus every
setting that changes the text: chunk size and overlap, strategy, heading level,
cross-reference limits, question-generation settings, OCR flags, both DPIs, the table
prompt, the table limits, language, and all model names.

If the fingerprint matches, the file is **not read** — no OCR, no embedding — and its
rows stay untouched. Measured on a five-document corpus: 71 s for the run that read the
files, 1.4 s for the next run that skipped all of them.

Other properties:

- Duplicate content is indexed once; the message names both paths.
- Files that disappeared from the corpus are removed from the index.
- With `sprache: de`, documents in another language are skipped and named in the report.
- Every chunk carries a header `[file name, page n]`, so the source is visible in the
  text itself. Structure chunks add the heading path: `[file, S. 8 > Chapter > Section]`.
- A file whose text is unchanged but whose strategy changed is rewritten; the strategy is
  part of the stored row, not only of the report.

## Chunking

Four strategies exist; `chunk_strategie` selects one, `auto` is the default.

| Strategy | What it does | When it wins |
|---|---|---|
| `struktur` | one chunk per section, with the heading path as a breadcrumb; long sections are packed at sentence ends; cross-references are resolved into the chunk | documents with headings and numbered clauses |
| `fenster` | windows of `chunk_chars`, snapped to paragraph boundaries, with overlap | flowing text, letters, protocols |
| `qes` | question-evidence supervision: generated questions pull the sentences they need into one chunk, under an index budget | only reachable with `--strategie qes`; measured worse than `fenster` on every file of the corpus, at 2.6 times the index |
| `auto` | binary structure detector: two or more headings lead to `struktur`, otherwise `fenster`; no score, no threshold | default |

Measured on the five-document German corpus, dense retrieval, complete-evidence recall@10
of the goldset:

| Strategy | Chunks | Index (characters) | Recall@10 |
|---|---|---|---|
| `fenster` (previous state) | 1511 | 943 919 | 0.538 |
| `struktur` | 1643 | 860 844 | **0.692** |
| `qes` | 6180 | 2 469 004 | 0.538 |
| `auto` (detector) | 1643 | 860 844 | **0.692** |

The detector chose `struktur` for four files and `fenster` for the one separator sheet
without headings: the value of the best measured strategy, with no configuration key and
a smaller index than before.

The limit that remains: the number of competing chunks a file contributes differs per
splitter, and the detector measures the shape of the document, not the retrieval result.
On this corpus both agree. Whether they agree on another corpus is a measurement, not a
property of the detector.

## Tables

Scanned pages are read twice. `paddleocr-vl` answers the table prompt with OTSL (Open
Table Structure Language: `<fcel>` cell, `<ecel>` empty cell, `<lcel>`/`<ucel>`/`<xcel>`
joined cell, `<nl>` row end). `dots-mocr` supplies the layout reading.

When the OTSL answer parses into a table, the Markdown of that table is what is indexed;
otherwise the dots reading stays. Both readings are kept, and parser warnings (unknown
token, no recognised header) appear in the report. A cell content that contains a line
break is flattened to one line, so a row cannot be split on its way into Markdown.

A table becomes one head chunk (header line plus up to `tabellen_kopf_zeilen` rows) and
one chunk per further row:

```
Kopf: Betrag, Datum, Posten | Zeile: 940,00, 01.04.2026, Miete
```

Every row carries its header. A wide row is split at cell borders with the header
repeated, and no row chunk exceeds `chunk_chars`. Before the prose chunkers run, each
table block is replaced by whitespace of the same length, so no window and no section can
cut into a table, and the row text stands only once in the index. Row chunks keep their
position in the document text; that position assigns them to the right page.

Measured with the running `paddleocr-vl` on a one-page tax notice: 5 columns, 12 rows, one
head chunk and 12 row chunks (three wide rows split at cell borders), longest chunk 571
characters, no warnings.

### Layout detection (optional)

With `layout: true`, every page of a PDF is rendered once at `layout_dpi` and passed to
DocLayout-YOLO, a YOLOv10-based detector (arXiv 2410.12628). Its code is AGPL-3.0 and is
not used here; only the Apache-2.0 weights are used, as an ONNX file through
`onnxruntime` (MIT), with the decoder in `layout.py`. The detector returns boxes for ten
classes; only `table` is used. Boxes below `layout_schwelle` or `layout_min_flaeche` are
dropped, and boxes that overlap a better one count as the same table.

Each remaining box is cropped with `layout_rand` padding at `layout_dpi_ocr`, read with
the table prompt, and turned into row chunks. The text inside the box is read from the
text layer with `get_text_bounded` and removed from the page text line by line, so the
table stands only once in the index. On pages without a text layer the dots reading
stays the page text and only detected tables are read; that removes the whole-page false
table which the unconditional prompt produced.

Measured on the corpus (292 pages, 200 dpi, CPU): 0.85 to 0.91 s per page, 30 table
regions in 27 pages (19 in the magazine, 12 in the slide deck, none in the three articles
and none on the separator sheet), all three pages of a hand-checked sample carried a real
table. With layout on the index grows from 1643 to 1767 chunks (+3.6 %); recall@10 stays
at 0.692. The ingest takes about 27 minutes instead of 7, because slide-sized crops at
300 dpi are expensive for the table reader.

## Retrieval and reranking

Fusion is reciprocal rank fusion with `k = 60`. The reranker scores `(question, passage)`
pairs and orders them; its output is a logit, **not** a probability, and is never used as
a threshold.

If the reranker fails, the fused order is used and the result carries
`rerank: "ausgefallen"`. The full-text half can be switched off (`hybrid: false`) because
a repeated page footer can match almost every question and steal ranks from real hits.

## Gate

One request per candidate passage, five candidates by default:

- The model answers with a single token; a grammar allows only `ja` or `nein`.
- The distribution over the first token comes from `top_logprobs`; the sum over all `ja`
  tokens is P(ja).
- A passage is admitted at `P(ja) >= gate_schwelle`.
- If no passage is admitted, no answer is generated. The reason is reported.
- If the model returns something other than `ja` or `nein`, the probability is undefined:
  the passage is not admitted and a warning is written.

Calibrated on `deutsche-telekom/Ger-RAG-eval` task 4 (`context_question_match`,
500 yes / 500 no, CC-BY-SA-4.0):

| Threshold | yes recognised | no rejected | accuracy |
|---|---|---|---|
| 0.30 | 490/500 | 497/500 | 98.7 % |
| 0.50 | 489/500 | 497/500 | 98.6 % |
| 0.90 | 479/500 | 497/500 | 97.6 % |

In 467 of 1000 cases the model did not answer with `ja` or `nein` at all but began a
sentence; those cases count as rejection. Without that rule the accuracy would look
better than it is.

## Answer and sentence check

The answer prompt allows a partial answer: if the passages answer the question only in
part, the model answers with what they support. It must not talk about the passages. For
a missing answer it returns the single token `KEINE_ANTWORT`, which becomes a refusal; an
earlier version spelled the refusal sentence out in the prompt and the model reproduced
it inside the answer.

With `antwort_pruefung: true`, every sentence is checked against the passage it names, by
the same logprob method as the gate. The check reads the **whole page**, not only the
600-character chunk that was retrieved. Sentences that no passage supports are dropped.
If nothing survives, the source is quoted verbatim; a quotation cannot be invented. A
refusal never carries sources; the checked pages appear in the warnings as
`geprüft, nicht ausreichend`.

A table row is a statement of its own: it carries the header and one row of values and
does not end with a period, so the sentence splitter would otherwise merge several rows
into one claim, and one wrong row would silence every right one. A citation on the first
row of a block applies to the rows below it.

## Measured results

All numbers below come from runs against the running models on a five-document German
corpus: 21 questions, 12 answerable, 9 not answerable.

| Setting | Answers | Correct source | Wrong source | Quotations |
|---|---|---|---|---|
| no gate, no check | 21 | 15 | 6 | 0 |
| gate, check at 0.5, refusal | 8 | 8 | 0 | 0 |
| gate, check at 0.5, drop sentences, quote as fallback | 21 | 16 | 5 | 10 |
| gate with grammar, check, partial answers allowed | 11 | 10 | 1 | 10 |

The gain of the decision layer is visible in the second row: the gate removes wrong
answers, at the price of refusing questions whose evidence it does not see. The fourth
row is the configuration in which the trade-off is acceptable.

An earlier measurement on the same corpus gave recall@10 0.71 and nDCG@10 0.65 (dense
only, 600-character windows); with the full-text half included, recall dropped to 0.65.
The strategy comparison with complete-evidence recall is in [Chunking](#chunking).

An acceptance test of eight questions against a 156-page magazine passed three. The
failures are documented in `docs/EVIDENCE-verbesserungen.md`, section J.

## Known limitations

- **The sentence check is not calibrated.** Its false-accept rate is unknown. It both
  drops correct statements and lets wrong ones through.
- **Table chunks need a scanned page or layout detection.** Without `layout: true` they
  come from the OCR path, which runs on pages without a text layer
  (`ocr_alle_seiten: false`, the default). With layout on, tables in digital PDFs are
  found and read as well, at the cost of the ingest time named above.
- **Without layout detection the table prompt is unconditional.** `paddleocr-vl` answers
  `Table Recognition:` on any page. Measured on a textless separator sheet without a
  table: it returned a table with a header and no rows, and the page was indexed as one
  head chunk. With `layout: true` only detected table crops are read, and the separator
  sheet yields no table.
- **A table across several pages is one unit.** All its row chunks carry the page on which
  the table starts. A citation of a row after a page break can be off by one page.
- **The structure chunker is line-based.** It reads headings, numbered clauses and
  paragraphs, not layout. A two-column PDF whose text order is wrong stays wrong.
- **OCR is expensive.** Roughly 75 s per page with both models at 200 and 300 dpi, so a
  156-page document takes about 3.3 hours on first ingest.
- **The goldset is small.** 31 questions originally, 21 after restricting the corpus to
  German; differences of one answer are within run-to-run variance.
- **`qes` is not maintained for tables.** The strategy does not mask table blocks and is
  not reachable through `auto`.
- **HyDE was removed.** It was measured with and without: one answer difference on
  bge-m3, none on nomic-embed-text. Section D of `SPEC.md` records the removal. Earlier
  gate variants (a per-passage scoring model, one choice over all candidates, a checker of
  the finished answer) were removed as well; all three removed correct answers and no
  wrong ones.

## Testing and quality gates

```bash
uv run pytest                          # 278 tests, no model needed
uv run pytest -m integration           # needs the real endpoints
uv run ruff check .                    # lint
uv run mypy                           # strict type check of src/
uv run pytest --cov=arbiter           # coverage
uv run python scripts/mutants.py      # 48 mutants, every one must die
```

The unit tests need no model: the HTTP layer is faked, the gate takes an injected logprob
map, and the terminal UI runs against a stub pipeline. The integration marker is excluded
by default.

Current state: 278 tests green, ruff and mypy without findings, 48/48 hand-written mutants
killed, line coverage 87 % overall (`tabellen.py` 96 %, `chunk.py` 99 %). The evidence for
the table and OCR work is in `docs/EVIDENCE-tabellen.md`, for the layout detection in
`docs/EVIDENCE-layout.md`; the chunking stage is documented in `docs/EVIDENCE-stufe7.md`.

The goldset used for the numbers above is not part of this repository: it consists of
questions about private documents and therefore stays local (`goldset/downloads.yaml`,
excluded by `.gitignore`).

## Repository layout

```
src/arbiter/
  config.py           YAML, ${VAR} expansion, unknown keys are errors
  runtime.py          reads arbiter.env for the CLI and the TUI
  loaders.py          PDF via pypdfium2 (pypdf fallback), text, duplicates, language filter
  ocr.py              page rendering, dots-mocr client, layout-to-text
  layout.py           DocLayout-YOLO over ONNX: boxes, table boxes, line removal
  tabellen.py         OTSL parser, Markdown tables, row chunks, table masking
  chunk.py            strategy switch, table blocks taken out first, window splitter
  chunk_structure.py  headings, sections, breadcrumbs, cross-references
  chunk_choice.py     binary structure detector (auto)
  chunk_qe.py         question-evidence generation
  chunk_qes.py        question-evidence chunking
  embedding.py        dense vectors via /v1/embeddings
  store.py            LanceDB: vectors, full-text index, file fingerprints
  fusion.py           reciprocal rank fusion
  rerank.py           /v1/rerank client
  decision.py         gate: prompt variants, logprobs to probability
  answer.py           answer prompt and source type
  check.py            claim splitting, support check, repair prompt
  pipeline.py         index, search, gate, answer, report
  output.py           JSON shapes for CLI --json and the API
  api.py              FastAPI service
  tui.py              terminal UI (Textual)
  cli.py              command line
  eval.py             goldset measurement per stage
scripts/              measurement and diagnosis scripts
docs/                 evidence per stage
tests/                unit and integration tests
SPEC.md               the contract behind the code
```

Measurement and diagnosis scripts worth knowing:

| Script | Purpose |
|---|---|
| `scripts/messe_chunking.py` | recall@10 per strategy on the real corpus, separate stores |
| `scripts/messe_pro_datei.py` | which strategy retrieves the evidence of one file |
| `scripts/calibrate_gate_ger_rag_eval.py` | gate thresholds on the public Ger-RAG-eval set |
| `scripts/measure_chain.py` | answer quality along the full chain |
| `scripts/pruefe_tabellenpfad.py` | OCR table path with the real models, no cell contents printed |
| `scripts/mutants.py` | mutation run: one real bug at a time, restore from git |

## Specification

`SPEC.md` is the contract behind the code. Requirement numbers (A, B, C, E, F, G, H, K, L,
M, N, O, P, R, V) are referenced in docstrings and commit messages. Decisions that were
made and later reversed carry their measurement with them, including the ones that
failed. The specification was approved before the code and amended visibly.

## License

MIT. See `License`.
