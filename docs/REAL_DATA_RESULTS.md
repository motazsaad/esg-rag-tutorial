# Real-data lab: published ESG data, model matrix, RAG techniques

This document explains how the lab moves beyond the synthetic smoke test:
it runs the **same end-to-end pipeline** over a sample of the real ESG data
published by the presenters, with **2 LLM generators**, **2 embedding models**,
and **5 RAG techniques**.

## 1. Data (real, attributed)

Two published companion repositories are used; full provenance and licenses are
in [`data/PROVENANCE.md`](../data/PROVENANCE.md).

| Data | Source | Sample here | Used for |
|---|---|---|---|
| 6 real sustainability-report excerpts (Crédit Agricole, Volvo, Continental, Mercedes-Benz, ENGIE, Maersk) | [ESG-RAG-benchmark-ragas](https://github.com/datalab-unisalento/ESG-RAG-benchmark-ragas) — IEEE BigData 2026, MIT | `data/real_corpus/*.md` (~2.5–3 KB each, verbatim excerpts around the answer) | document corpus |
| 6 QA pairs with reference answers, tagged with an ESG pillar (Environmental / Social / Governance) | same repo, `outputs/testset/01_esg_testset.json` | `data/real_qa.json` | end-to-end answer quality, ESG-sliced analysis |
| 24 graded query/3-document records | [benchmark-embeddings-esg](https://github.com/datalab-unisalento/benchmark-embeddings-esg) — ITADATA 2026, Apache-2.0 | `data/real_retrieval_bench.json` | retrieval-only NDCG@10 / MRR / P@1 |

The excerpts keep the upstream wording and `*Page N*` markers; a small
`<!-- key: value -->` header adds company, year, language and full source URL
so the pipeline can do metadata-filtered retrieval and cite provenance.

## 2. Models

| Role | Models |
|---|---|
| Generators (2) | `qwen3:4b-instruct`, `gemma3:4b` |
| Embedders (2) | `embeddinggemma:latest` (768-d), `qwen3-embedding:0.6b` (1024-d) |
| Helper roles | rewrite / rerank / judge are parameterised separately (`models.rewrite`, `models.rerank`, `models.judge`) |

All model names live in `config.yaml` or can be overridden per command:

```bash
python3 -m esg_rag.cli --set models.generator=gemma3:4b query "..." --corpus real
python3 -m esg_rag.cli index --corpus real --embedder qwen3-embedding:0.6b
```

## 3. What is parameterised

Everything that affects a run is configurable, either in `config.yaml`, via
`ESG_RAG_<SECTION>_<KEY>` environment variables, or via the repeatable
`--set section.key=value` CLI flag (values are parsed as JSON when possible):

| Section | Controls |
|---|---|
| `ollama` | server URL, request timeout |
| `models` | generator, embedder, judge, rewrite, rerank, candidate LLM/embedder pools |
| `paths` | synthetic and real corpora, QA set, graded benchmark, index/result files |
| `chunking` | `strategy` (`section`/`fixed`/`semantic`), `max_words`, `overlap_words`, `min_words`, `semantic_threshold` |
| `retrieval` | `top_k`, `mode` (dense/bm25/hybrid), BM25 `k1`/`b`, RRF `k`, evidence length, candidate multiplier, LLM-inferred filters |
| `strategy` | active technique, number of query rewrites, rerank depth, self-RAG loops, decomposition depth |
| `generation` | temperature, `max_tokens`, seed, context size, thinking flag, insufficient-evidence string |
| `benchmark` | dataset (synthetic/real), limit, judge, answer metric |
| `system_metrics` | enable/disable telemetry, power source (`auto`/`rapl`/`nvidia`/`estimate`/`none`), fallback watts |
| `matrix` | model pools, technique list, dataset, questions per cell, judge |

Examples:

```bash
# semantic chunking, dense retrieval, 8 passages, no thinking
python3 -m esg_rag.cli --set chunking.strategy=semantic \
  --set chunking.semantic_threshold=0.2 \
  --set retrieval.mode=dense --set retrieval.top_k=8 \
  --set generation.think=false query "..." --corpus real

# LLM-inferred company/year/GRI predicates
python3 -m esg_rag.cli --set retrieval.infer_filters=true \
  query "What were Volvo's CO2 emissions in 2021?" --corpus real --strategy filtered
```

## 4. RAG techniques implemented

| Technique | Pipeline | Why it matters for regulated documents |
|---|---|---|
| `naive` | dense top-k → generate | baseline; shows the cost of no lexical matching |
| `hybrid` | dense + BM25 → reciprocal-rank fusion → generate | GRI codes, KPI names and units are literal tokens |
| `filtered` | hybrid + metadata predicates (company, year, GRI), optionally LLM-inferred | pre-filtering avoids cross-company/year contamination |
| `multi_query` | LLM rewrites question into N variants → retrieve each → fuse ranks | handles paraphrase and terminology mismatch |
| `rerank` | retrieve `top_k × candidate_multiplier` → LLM scores 0–5 → keep best | improves the candidate set before generation |
| `self_rag` | retrieve → grade evidence → rewrite + retry if unsupported → generate | reduces hallucination on absent KPIs |
| `decompose` | split multi-hop question → answer sub-questions → synthesize | multi-part regulatory questions |

Every technique returns the same result contract (answer, citations, evidence,
trace, latency, token counts) so comparisons are apples-to-apples. The
`trace` records each internal step; lineage stores it with the answer.

## 5. Reproduction commands

```bash
cd esg-rag
python3 -m pip install -r requirements.txt
ollama pull qwen3:4b-instruct gemma3:4b embeddinggemma:latest qwen3-embedding:0.6b

# one real-data run, hybrid
python3 -m esg_rag.cli --corpus real evaluate --dataset real --strategy hybrid --limit 6

# retrieval-only embedding comparison
python3 -m esg_rag.cli retrieval-bench --embedder embeddinggemma:latest
python3 -m esg_rag.cli retrieval-bench --embedder qwen3-embedding:0.6b

# the full 2 LLMs x 2 embedders x 5 techniques matrix (writes JSON/CSV/Markdown)
python3 -m esg_rag.cli matrix --llms qwen3:4b-instruct gemma3:4b \
  --embedders embeddinggemma:latest qwen3-embedding:0.6b \
  --strategies naive hybrid multi_query rerank self_rag \
  --dataset real --limit 2 --judge
```

Outputs (written **incrementally after every cell**, so partial results are
available while a run is still in progress):

| File | Format | Contents |
|---|---|---|
| `output/results/matrix.json` | JSON | full configuration, per-cell summaries, per-question rows, retrieval benchmarks |
| `output/results/matrix.csv` | CSV | one row per (LLM, embedder, technique) cell — easy to paste into a paper or spreadsheet |
| `output/results/matrix_report.md` | Markdown | human-readable tables |
| `output/lineage.jsonl` | JSONL | one append-only record per query with passages, trace, models and timings (local-only, gitignored) |

The summary artefacts under `output/results/` are versioned in this repository;
indexes and lineage logs stay local (see `.gitignore`).

### Live web dashboard

```bash
python3 dashboard.py --port 8080     # http://127.0.0.1:8080
```

The dashboard is a zero-dependency (stdlib server + vanilla HTML/JS) view over
the artefacts above. It auto-refreshes every 8 seconds and shows: run
progress, headline cards, a sortable/filterable cell table with expandable
per-question answers, embedding-quality bars (NDCG@10/MRR/P@1), a
live per-cell metrics (answer hit, retrieval hit, token F1, latency, tokens,
judge, decode speed, energy) directly from `lineage.jsonl` by matching each
question to its reference answer/source document, and shows an
`LIVE — from lineage` badge plus a progress bar. Once `matrix.json` appears,
the dashboard switches to the final artefacts automatically.

The results table has **five filters**: technique, LLM, **embedding model**,
**ESG pillar** (Environmental / Social / Governance / Other) and a free-text
**question search (`q`)**. ESG/query filters operate at the question level
(the displayed metrics remain cell-level, which the UI states explicitly);
older result files without an `esg_dimension` field are classified from the
question text by a keyword fallback.

The quality scatter plot has a labelled x-axis with numeric ticks and a metric
selector: **latency**, **decode speed (tok/s)**, **energy (J/query)** or
**average power (W)** — useful for quality–cost–latency trade-off discussions.

### Configure & Run page

The **⚙️ Configure & Run** tab is a small experiment launcher. It exposes the
main knobs of the pipeline — run type (single query / benchmark / matrix),
dataset, **checkbox groups for generator LLM(s), embedding model(s) and RAG
techniques** (check any combination), question, questions per cell, judge,
chunking strategy/size/overlap, retrieval mode/top-K/filter inference,
generation temperature/tokens, and power source/fallback watts. Checked
models/techniques are swept as a cartesian product in all three run modes;
the single-query and benchmark modes therefore return one result per
combination rather than silently ignoring the extra selections.
Submitting POSTs to `/api/run`, which **allowlists and clamps** every value
(models must exist in Ollama, strategies/chunking/retrieval/power must be from
fixed enums, numeric ranges are bounded, and at least one item per checkbox
group is required); no arbitrary paths or shell commands are accepted. Jobs run in a background thread, only one at a time, and the UI
polls `/api/run/status` for progress and the final result. Matrix jobs rewrite
the same `output/results/matrix.*` artefacts, so the Results tab picks them up
automatically.

API: `GET /api/config` (models, strategies, defaults), `POST /api/run`,
`GET /api/run/status?job=<id>`.

## 6b. System-level metrics

Besides answer quality, every query records (config section `system_metrics`):

| Metric | Source | Meaning |
|---|---|---|
| `tokens_per_second`, `ms_per_token` | Ollama `eval_duration` / `eval_count` | decode/inference speed of the generator |
| `process_rss_mb` | `/proc/self/statm` | resident memory of the lab process |
| `model_memory_mb`, `vram_mb` | Ollama `/api/ps` | memory of the currently loaded model(s) |
| `energy_j`, `avg_watts` | Intel RAPL / NVIDIA-SMI when readable, otherwise `estimated_watts × wall time` | energy per query and average power; `power_measured` marks whether it was a real sensor reading |
| `total_latency_s`, `prompt_tokens`, `completion_tokens`, `llm_calls` | pipeline timers / Ollama response | latency and token/call cost |

RAPL files are often root-only and this lab machine has no GPU, so energy here
is an **estimate** (clearly flagged with `power_measured: false` and a `*` in
the dashboard). On a machine with readable RAPL or NVIDIA-SMI the same code
reports measured joules automatically — set `system_metrics.power_source` to
`rapl`, `nvidia`, `estimate`, or `none` explicitly. These numbers are recorded
in `lineage.jsonl`, aggregated in `matrix.json`/`matrix.csv`, and shown in the
dashboard table and scatter plot.

### Telemetry addendum (measured on this laptop)

Because the 20-cell matrix above was launched before telemetry was added, a
short 4-cell rerun with the final code was used to populate the system columns
(`output/results/matrix_telemetry.{json,csv,md}`; same config, `qwen3:4b-instruct`
only, 2 questions/cell):

| Embedder | Technique | Latency (s) | Tok/s | RSS MB | Model MB | J/query (est.) | W (est.) | Answer hit |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| embeddinggemma:latest | hybrid | 71.679 | 7.09 | 37.6 | 3879.1 | 4659.21 | 65.0 | 1.0 |
| embeddinggemma:latest | self_rag | 62.666 | 7.93 | 37.6 | 3879.1 | 4073.32 | 65.0 | 1.0 |
| qwen3-embedding:0.6b | hybrid | 76.060 | 6.47 | 39.4 | 4424.0 | 4943.98 | 65.0 | 1.0 |
| qwen3-embedding:0.6b | self_rag | 66.271 | 7.37 | 39.4 | 4424.0 | 4307.62 | 65.0 | 1.0 |

Observations: the lab process itself is tiny (~38–39 MB RSS); the loaded Ollama
models dominate memory (~3.9–4.4 GB, all in RAM because `size_vram = 0` on this
machine). Decode speed is ~6–8 tokens/s on CPU-only hardware, and the energy
column is a 65 W × wall-time estimate (`power_measured = false`), so it mainly
tracks latency. These are exactly the system metrics the tutorial recommends
reporting next to answer quality.

## 6. Results

Run completed 2026-11-16 on a CPU-only Intel i7-1065G7 laptop with local
Ollama. Configuration: **2 questions per cell** (the Amundi assets-under-
management question and the Volvo 139 g/km question), `temperature=0`, `seed=42`,
`max_tokens=160`, judge enabled, `top_k=4`, hybrid BM25+dense retrieval,
section-aware chunking. Full data: `output/results/matrix.{json,csv}`;
generated tables: `output/results/matrix_report.md`.

### Generation matrix (20 cells)

| LLM | Embedder | Technique | Ans.hit | F1 | Latency (s) | Tokens | Calls | Judge |
|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3:4b-instruct | embeddinggemma:latest | naive | 1.0 | 0.639 | 63.652 | 92.0 | 2.0 | 1.0 |
| qwen3:4b-instruct | embeddinggemma:latest | hybrid | 1.0 | 0.639 | 68.408 | 90.0 | 2.0 | 1.0 |
| qwen3:4b-instruct | embeddinggemma:latest | multi_query | 1.0 | 0.639 | 85.631 | 211.0 | 3.0 | 1.0 |
| qwen3:4b-instruct | embeddinggemma:latest | rerank | 1.0 | 0.639 | 134.116 | 110.0 | 3.0 | 1.0 |
| qwen3:4b-instruct | embeddinggemma:latest | self_rag | 1.0 | 0.657 | 65.313 | 115.5 | 3.0 | 0.5 |
| gemma3:4b | embeddinggemma:latest | naive | 1.0 | 0.627 | 58.364 | 99.5 | 2.0 | 1.0 |
| gemma3:4b | embeddinggemma:latest | hybrid | 1.0 | 0.585 | 46.461 | 101.5 | 2.0 | 1.0 |
| gemma3:4b | embeddinggemma:latest | multi_query | 1.0 | 0.627 | 68.844 | 212.5 | 3.0 | 1.0 |
| gemma3:4b | embeddinggemma:latest | rerank | 0.5 | 0.475 | 116.767 | 109.0 | 3.0 | 1.0 |
| gemma3:4b | embeddinggemma:latest | self_rag | 1.0 | 0.637 | 61.77 | 116.5 | 3.0 | 0.5 |
| qwen3:4b-instruct | qwen3-embedding:0.6b | naive | 1.0 | 0.639 | 66.781 | 92.5 | 2.0 | 1.0 |
| qwen3:4b-instruct | qwen3-embedding:0.6b | hybrid | 1.0 | 0.639 | 68.441 | 90.0 | 2.0 | 1.0 |
| qwen3:4b-instruct | qwen3-embedding:0.6b | multi_query | 1.0 | 0.639 | 86.77 | 211.0 | 3.0 | 1.0 |
| qwen3:4b-instruct | qwen3-embedding:0.6b | rerank | 1.0 | 0.639 | 147.382 | 183.0 | 3.0 | 1.0 |
| qwen3:4b-instruct | qwen3-embedding:0.6b | self_rag | 1.0 | 0.657 | 67.266 | 115.5 | 3.0 | 0.5 |
| gemma3:4b | qwen3-embedding:0.6b | naive | 1.0 | 0.609 | 78.159 | 98.0 | 2.0 | 1.0 |
| gemma3:4b | qwen3-embedding:0.6b | hybrid | 1.0 | 0.627 | 56.842 | 97.0 | 2.0 | 1.0 |
| gemma3:4b | qwen3-embedding:0.6b | multi_query | 1.0 | 0.627 | 72.164 | 212.5 | 3.0 | 1.0 |
| gemma3:4b | qwen3-embedding:0.6b | rerank | 0.5 | 0.491 | 131.151 | 183.5 | 3.0 | 0.0 |
| gemma3:4b | qwen3-embedding:0.6b | self_rag | 1.0 | 0.637 | 56.367 | 119.5 | 3.0 | 0.5 |

`Calls` includes the judge call (2 = answer + judge; 3 = answer + one helper +
judge). `Judge` is the share of answers the judge considered fully supported.

### Retrieval-only benchmark (24 graded ESG queries)

| Embedder | NDCG@10 | MRR | P@1 |
|---|---:|---:|---:|
| embeddinggemma:latest | 0.9697 | 1.0 | 1.0 |
| qwen3-embedding:0.6b | 0.9697 | 1.0 | 1.0 |

### Observations

1. **Retrieval was not the bottleneck.** `retrieval_hit@k = 1.0` in all 20
   cells and both embedders score identically on the graded IR set; the
   2-document sample is too easy to separate them. Use the full 312-record
   benchmark for an embedding comparison.
2. **Generator matters more than technique here.** `qwen3:4b-instruct` reaches
   1.0 answer hit with all five techniques; `gemma3:4b` drops to 0.5 with
   `rerank` (one answer missed). LLM re-ranking can promote a fluent but
   non-answer-bearing passage — a useful cautionary result for the tutorial's
   re-ranking discussion.
3. **Advanced techniques are not free.** Averaged over the 4 cells per
   technique: `naive` 66.7 s, `hybrid` 60.0 s, `multi_query` 78.4 s,
   `self_rag` 62.7 s, `rerank` 132.4 s — roughly 2× the latency of the single-
   shot routes, with no quality gain on this small sample.
4. **Judge behaviour differs.** `self_rag` cells show `judge_supported_rate =
   0.5` for both generators, versus 1.0 for most other techniques: the judge
   penalises answers whose evidence was graded/retrieved in a second pass.
   This illustrates the judge-bias and nondeterminism caveat from Part 3 of
   the tutorial; do not treat these rates as ground truth.
5. **ESG slicing.** The two matrix questions cover Environmental (Volvo CO₂,
   20/20 answers hit) and Governance (Amundi AUM, 18/20). The Social pillar
   (Mercedes part-time ratio) is in the 6-question set but not in this 2-question
   matrix slice; run with `--limit 6` to include it, or use the ESG filter in
   the dashboard once those results exist.
6. **Scope.** These are 2 questions × 5 techniques, on a CPU-only laptop — a
   smoke-test-scale demonstration of the methodology, not a paper-quality
   comparison. The reproduced matrices in the companion repositories use the
   full 284-pair test set.

## 7. Interpreting the numbers

* `retrieval_hit@k` — at least one retrieved passage comes from the source
  report of the QA pair (no human-level gold passage labels are needed).
* `answer_hit_rate` — the reference answer's numeric values are present and
  token overlap with the reference clears the threshold; it is a cheap proxy,
  not a substitute for RAGAS-style judging.
* `judge_supported_rate` — share of answers the LLM judge considers fully
  supported by the cited passages (read alongside the tutorial's discussion
  of judge bias and nondeterminism).
* `mean_llm_calls` — cost of the technique: advanced strategies buy quality
  with extra model calls and latency.
* The retrieval-only benchmark grades the candidate documents of the
  embedding-benchmark paper, so it isolates embedding quality from generation.
