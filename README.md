# ArbiterRAG

Retrieval-augmented generation over a local document corpus, with a separate decision
step between retrieval and answer generation.

A reranker can only reorder what the search found; it cannot say "none of these
passages answers the question". ArbiterRAG therefore splits the work in two: the
reranker orders candidates, and a gate decides which candidates may be cited. The
answer is then written only from those, and every sentence of it is checked against
the passage it cites before the answer is released.

All models run behind a single llama-swap instance on `localhost`. No external API is
called at query time.

## Status

| Area | State |
|---|---|
| Indexing: PDF and text, deduplication, language filter, OCR | implemented |
| Search: dense retrieval, full-text, reciprocal rank fusion | implemented |
| Reranking via `/v1/rerank` | implemented |
| Gate: one yes/no decision per candidate, read from logprobs | implemented |
| Answer generation with per-sentence support check | implemented |
| HTTP service (FastAPI) | implemented |
| Terminal UI | implemented |
| Goldset evaluation | implemented |
| Article-aware chunking | not implemented |
| Calibrated sentence check | not implemented |

The status is meant literally. Section "Measured results" lists what was measured and
what failed; section "Known limitations" lists what is missing.

## Architecture

```
 question
    |
    +- dense search    /v1/embeddings              (bge-m3)
    +- full-text search (LanceDB FTS, optional)
    |
    +- reciprocal rank fusion                      (k = 60)
    +- reranking       /v1/rerank                  (bge-reranker-v2-m3)
    |
    +- gate            /v1/chat/completions        (qwen2.5-7b-instruct, logprobs, grammar)
    |     one request per candidate; P("ja") decides admission
    |
    +- answer          /v1/chat/completions
    |     every sentence checked against the passage it cites
    |     unsupported sentences are dropped; if none survive, the source is quoted
    |
    +- answer with sources, or a refusal with a reason
```

Two properties define the design:

1. **The gate sees one passage per request.** A single state holding all passages made
   the model answer about the whole text instead of about the marker; every probability
   came back between 0.98 and 0.99. One passage per request separates a real hit (0.941)
   from noise (0.001 to 0.028).
2. **The gate always decides.** Its answer is constrained by a GBNF grammar
   (`root ::= "ja" | "nein"`), so there is no undefined state: the model either says yes
   or no, and the probability of the first token is the decision.

## Requirements

- Linux, Python 3.12, [uv](https://docs.astral.sh/uv/)
- A [llama-swap](https://github.com/mostlygeek/llama-swap) instance at `http://127.0.0.1:8080`
  serving these models:
  - `bge-m3-gguf` — embeddings (`/v1/embeddings`)
  - `bge-reranker-v2-m3` — reranking (`/v1/rerank`, needs `--reranking`)
  - `qwen2.5-7b-instruct` — gate and answer (`/v1/chat/completions`, needs `logprobs`)
  - `dots-mocr` and `paddleocr-vl` — OCR, only with `ocr_alle_seiten: true`
- About 8 GB of GPU memory when all models are resident.

## Install

```bash
git clone git@github.com:FBR65/ArbiterRAG.git
cd ArbiterRAG
uv sync
cp examples/arbiter.yaml arbiter.yaml     # then edit paths and model names
```

## Configuration

`arbiter.yaml` holds one entry per corpus. `${VAR}` is replaced from the environment;
unknown keys are an error, so a typo never silently changes nothing.

| Key | Default | Meaning |
|---|---|---|
| `db_path` | — | LanceDB directory, one table per corpus |
| `backends.base_url` | — | llama-swap base URL |
| `backends.embed_model` | — | embeddings model |
| `backends.rerank_model` | — | reranking model |
| `backends.chat_model` | — | gate and answer model |
| `backends.ocr_dots_model` | — | OCR model that supplies the indexed text |
| `backends.ocr_paddle_model` | — | OCR second opinion |
| `rags.<name>.corpus_path` | — | root directory of the corpus |
| `rags.<name>.extensions` | — | `[".pdf"]`, `[".txt", ".pdf"]`, ... |
| `rags.<name>.top_k_dense` | 20 | dense candidates |
| `rags.<name>.top_k_fts` | 20 | full-text candidates |
| `rags.<name>.fuse_top_k` | 20 | candidates after fusion |
| `rags.<name>.rerank_top_n` | 5 | candidates after reranking |
| `rags.<name>.chunk_chars` | 600 | characters per chunk |
| `rags.<name>.chunk_overlap` | 100 | overlap between chunks |
| `rags.<name>.hybrid` | true | include full-text search in the fusion |
| `rags.<name>.fts_stem` | false | German stemming in the full-text index |
| `rags.<name>.sprache` | `null` | keep only documents in this language (`de`) |
| `rags.<name>.ocr` | false | OCR on/off |
| `rags.<name>.ocr_alle_seiten` | false | OCR every page, not only pages without a text layer |
| `rags.<name>.ocr_dpi` | 200 | rendering resolution for `dots-mocr` |
| `rags.<name>.ocr_dpi_paddle` | 300 | rendering resolution for `paddleocr-vl` |
| `rags.<name>.eltern_seite` | false | index small chunks, return the whole page as context |
| `rags.<name>.gate_kandidaten` | 5 | candidates the gate sees |
| `rags.<name>.gate_schwelle` | 0.5 | P(ja) needed for admission |
| `rags.<name>.gate_prompt` | `teil` | gate prompt variant: `teil`, `streng`, `beleg` |
| `rags.<name>.antwort_pruefung` | false | check every sentence against its source |
| `rags.<name>.antwort_schwelle` | 0.5 | P(ja) needed for a sentence to be kept |
| `rags.<name>.antwort_passagen` | 3 | passages the answer may cite |

## Usage

The CLI and the TUI read `arbiter.env` when variables are missing; values exported in
the shell always win.

```bash
uv run arbiter index downloads
uv run arbiter search downloads "How is the processing run scheduled?"
uv run arbiter ask downloads "Why does LLM-based business intelligence often fail in operations?"
uv run arbiter eval downloads
uv run arbiter-tui
uv run arbiter-api                 # 127.0.0.1:8091
```

`arbiter.env`:

```
ARBITER_CONFIG=/path/to/arbiter.yaml
ARBITER_DB=/home/user/.cache/arbiterrag/db
ARBITER_BASE_URL=http://127.0.0.1:8080
ARBITER_EMBED_MODEL=bge-m3-gguf
ARBITER_CORPUS=/home/user/Documents
```

### Terminal UI

```
Enter    ask
Ctrl+S   switch to search mode (passages with reranker score instead of an answer)
Ctrl+I   index in the background
Ctrl+L   clear the transcript
Ctrl+Q   quit
```

The status line shows corpus, chunk count, mode, reranker and gate settings.

### HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | service state and version |
| GET | `/rags` | configured corpora |
| POST | `/search` | passages for a question |
| POST | `/ask` | answer with sources, or a refusal with a reason |
| POST | `/index` | start indexing (`202` with `job_id`) |
| GET | `/index/{job_id}` | progress of an indexing run |

## Indexing

Each corpus maps to one LanceDB table plus one table `<name>_ingest` holding a
fingerprint per file: SHA-256 over the file content plus every setting that changes the
text (chunk size, overlap, OCR flags, both DPIs, language, model names).

If the fingerprint matches, the file is **not read** — no OCR, no embedding — and its
rows stay untouched. Measured on a five-document corpus: 71 s for the run that read the
files, 1.4 s for the next run that skipped all ten of them.

Other properties:

- Duplicate content is indexed once; the message names both paths.
- Files that disappeared from the corpus are removed from the index.
- With `sprache: de`, documents in another language are skipped and named in the report.
- Every chunk carries a header `[file name, page n]`, so the source is visible in the
  text itself.

## Retrieval and reranking

Fusion is reciprocal rank fusion with `k = 60`. The reranker scores `(question, passage)`
pairs and orders them; its output is a logit, **not** a probability, and is never used as
a threshold.

If the reranker fails, the fused order is used and the result carries
`rerank: "ausgefallen"`.

## Gate

One request per candidate passage, five candidates by default:

- The model answers with a single token; a grammar allows only `ja` or `nein`.
- The distribution over the first token comes from `top_logprobs`; the sum over all
  `ja` tokens is P(ja).
- A passage is admitted at `P(ja) >= gate_schwelle`.
- If no passage is admitted, no answer is generated. The reason is reported.

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
part, the model answers with what they support. It must not talk about the passages,
and for a missing answer it returns the single token `KEINE_ANTWORT`, which becomes a
refusal — an earlier version spelled the refusal sentence out in the prompt and the
model reproduced it inside the answer.

With `antwort_pruefung: true`, every sentence is checked against the passage it names,
by the same logprob method as the gate. The check reads the **whole page**, not only the
600-character chunk that was retrieved. Sentences that no passage supports are dropped.
If nothing survives, the source is quoted verbatim — a quotation cannot be invented.
A refusal never carries sources; the checked pages appear in the warnings as
"geprüft, nicht ausreichend".

## Measured results

All numbers below come from runs against the running models on a five-document German
corpus: 21 questions, 12 answerable, 9 not answerable.

| Setting | Answers | Correct source | Wrong source | Quotations |
|---|---|---|---|---|
| no gate, no check | 21 | 15 | 6 | 0 |
| gate, check at 0.5, refusal | 8 | 8 | 0 | 0 |
| gate, check at 0.5, drop sentences, quote as fallback | 21 | 16 | 5 | 10 |
| gate with grammar, check, partial answers allowed | 11 | 10 | 1 | 10 |

Retrieval on the same corpus: recall@10 0.71, nDCG@10 0.65 (dense only, 600-character
chunks). With the full-text half included, recall drops to 0.65 — the repeated footer of
the slide deck matches almost every question.

An acceptance test of eight questions against a 156-page magazine passed three. The
failures are documented in `docs/EVIDENCE-verbesserungen.md`, section J.

## Known limitations

- **The sentence check is not calibrated.** Its false-accept rate is unknown. It both
  drops correct statements and lets wrong ones through.
- **Chunking ignores document structure.** 600-character windows cross page and article
  boundaries. Article-aware chunking is the next step; a different window size was
  measured and changed nothing.
- **OCR is expensive.** Roughly 75 s per page with both models at 200 and 300 dpi, so a
  156-page document takes about 3.3 hours on first ingest.
- **The goldset is small.** 31 questions originally, 21 after restricting the corpus to
  German; differences of one answer are within run-to-run variance.
- **HyDE was removed.** It was measured with and without: one answer difference on
  bge-m3, none on nomic-embed-text. Section D of `SPEC.md` records the removal.

## Testing

```bash
uv run pytest                    # 144 tests
uv run ruff check .              # lint
uv run mypy                      # types
uv run python scripts/mutants.py # mutation run, 15 mutants
```

The tests need no model: the HTTP layer is faked, the gate takes an injected logprob
map, the terminal UI runs against a stub pipeline.

The goldset used for the numbers above is not part of this repository: it consists of
questions about private documents and therefore stays local (`goldset/downloads.yaml`,
excluded by `.gitignore`).

`SPEC.md` is the contract behind the code — requirement numbers (B, C, E, F, G, H) are
referenced in docstrings and commit messages. `docs/EVIDENCE-verbesserungen.md` records
every measurement, including the ones that failed.

## Repository layout

```
src/arbiter/
  config.py      YAML, ${VAR} expansion, unknown keys are errors
  loaders.py     PDF via pypdfium2 (pypdf fallback), text, duplicates, language filter
  chunk.py       chunking with paragraph boundaries and overlap
  embedding.py   dense vectors via /v1/embeddings
  store.py       LanceDB: vectors, full-text index, file fingerprints
  fusion.py      reciprocal rank fusion
  rerank.py      /v1/rerank client
  decision.py    gate: prompt variants, logprobs to probability
  answer.py      answer prompt
  check.py       sentence splitting, support check, repair prompt
  pipeline.py    index, search, gate, answer
  output.py      JSON shaping
  api.py         FastAPI service
  tui.py         terminal UI (Textual)
  cli.py         command line
  runtime.py     environment from arbiter.env
scripts/         measurement and diagnosis scripts
docs/            evidence per stage
tests/           unit and integration tests
```

## License

MIT. See `License`.
