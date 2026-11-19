# ESG RAG — end-to-end, local-first pipeline

Reference implementation for the ESG RAG tutorial
*Retrieval-Augmented Analytics for Regulated Documents: Data Pipelines,
Benchmarking, and Structured Extraction*.

This is a **working** pipeline, not a slideware sketch. It ingests ESG report
text, chunks it by section, embeds it with a local Ollama embedding model,
indexes it with metadata, retrieves with **filtered hybrid search** (dense +
BM25), and produces either a **cited natural-language answer** or a **structured
KPI value** ready for a relational table. Every query writes an append-only
**lineage** record with the exact source chunks, configuration, models, latency
and token counts.

Everything runs on a laptop with no API keys and no cloud services.

```
parse → section-aware chunk → embedgemma embeddings → local vector index
      → filtered hybrid retrieval (dense + BM25, RRF fusion)
      → qwen3 grounded answer [S#]  /  KPI JSON extraction
      → lineage log + systems metrics
```

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | tested on 3.14 |
| [Ollama](https://ollama.com) | running locally on port 11434 |
| `qwen3:4b-instruct` | generator (and optional LLM judge) |
| `embeddinggemma:latest` | embedder, 768 dimensions |

```bash
ollama serve &                       # if not already running
ollama pull qwen3:4b-instruct
ollama pull embeddinggemma:latest

python3 -m pip install -r requirements.txt
```

Only `requests` and `PyYAML` are required. NumPy is optional (accelerates the
cosine product); if absent, pure Python is used. FAISS, Chroma, sentence-
transformers and HuggingFace downloads are **not** needed.

## 2. Smoke test (one command)

```bash
./run_smoke_test.sh
# equivalent: python3 -m esg_rag.cli smoke-test
```

It performs: health check → index build → one English RAG query → one Italian
RAG query → one structured KPI extraction → a 3-question benchmark, and prints
a PASS/FAIL table. Expected on the included synthetic corpus: `PASSED`,
`retrieval_hit@k = 1.0`, `answer_hit_rate = 1.0`, ~50–80 s on a CPU laptop.

```bash
python3 -m unittest discover -s tests    # same checks as unittest, optional
```

> **Real ESG data + model matrix:** see
> [`docs/REAL_DATA_RESULTS.md`](docs/REAL_DATA_RESULTS.md) and
> [`data/PROVENANCE.md`](data/PROVENANCE.md). A 26-slide walkthrough (ESG
> reporting background + implemented pipeline) is in
> [`slides/pipeline-lab.pdf`](slides/pipeline-lab.pdf).
> The lab includes a small sample
> of the published ESG corpora and runs the same pipeline with two generators
> (`qwen3:4b-instruct`, `gemma3:4b`), two embedders
> (`embeddinggemma:latest`, `qwen3-embedding:0.6b`) and seven RAG techniques.

## 3. CLI walkthrough

```bash
# 0. environment check
python3 -m esg_rag.cli doctor

# 1. parse + chunk + embed + index (one index per corpus AND embedder)
python3 -m esg_rag.cli index
python3 -m esg_rag.cli index --corpus real --embedder qwen3-embedding:0.6b
python3 -m esg_rag.cli index --force           # rebuild after editing corpus

# 2. grounded question answering (note the metadata filters)
python3 -m esg_rag.cli query \
  "What was the total number of employees at Nordwind in 2023?" \
  --company Nordwind --year 2023 --top-k 4 --mode hybrid --strategy hybrid --judge

# advanced techniques: multi-query, LLM rerank, self-RAG, decomposition, ...
python3 -m esg_rag.cli query "..." --strategy self_rag --corpus real
python3 -m esg_rag.cli --set retrieval.infer_filters=true query "..." --strategy filtered

# 3. multilingual question — embeddinggemma + qwen3 handle Italian
python3 -m esg_rag.cli query \
  "Quali sono le emissioni Scope 1 di Solaria nel 2023?" \
  --company Solaria --year 2023

# 4. structured extraction into a schema (JSON mode enforced by Ollama)
python3 -m esg_rag.cli extract "GRI 305_305-1_32" --company Nordwind

# 5. retrieval / answer benchmark and provenance
python3 -m esg_rag.cli evaluate --limit 3 --judge
python3 -m esg_rag.cli --corpus real evaluate --dataset real --limit 6

# embedding comparison and the full model/technique matrix
python3 -m esg_rag.cli retrieval-bench --embedder qwen3-embedding:0.6b
python3 -m esg_rag.cli matrix --dataset real --limit 2 --judge

python3 -m esg_rag.cli lineage --summary
python3 -m esg_rag.cli lineage --last 3
```

Any setting can be overridden with `--set section.key=value` (repeatable),
before the subcommand:

```bash
python3 -m esg_rag.cli --set chunking.strategy=semantic \
  --set retrieval.top_k=8 --set generation.max_tokens=160 query "..."
```

Add `--json` to any command for machine-readable output, e.g.
`python3 -m esg_rag.cli query "..." --json`.

### REST service (live demo)

```bash
python3 dashboard.py --port 8080       # results dashboard (auto-refresh, live progress)
python3 serve.py                       # REST demo: http://127.0.0.1:8000/docs
curl -s localhost:8000/health
curl -s -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"question":"What were the Scope 1 emissions at Nordwind in 2023?",
       "filters":{"company":"Nordwind"}}'
```

The dashboard (`dashboard.py`) additionally has a **Configure & Run** tab for
launching parameterised experiments from the browser (checkbox groups for
LLM/embedder/technique combinations, allowlisted parameters, single background
job, live status). See
[`docs/REAL_DATA_RESULTS.md`](docs/REAL_DATA_RESULTS.md).

`GET /health`, `POST /index`, `POST /query`, `POST /extract`, `GET /lineage`.
The HTTP route calls the same `ESGRAGPipeline` methods as the CLI.

## 4. What is in the box

```
esg-rag/
├── config.yaml                  # single reproducible configuration
├── requirements.txt
├── run_smoke_test.sh
├── run_real_benchmark.sh        # real-data benchmark + matrix in one command
├── dashboard.py                 # zero-dependency live results dashboard (:8080)
├── dashboard/index.html         # dashboard UI (cards, tables, charts, lineage)
├── serve.py                     # optional stdlib REST demo service
├── docs/REAL_DATA_RESULTS.md    # real-data provenance, techniques, matrix results
├── data/
│   ├── corpus/                  # 2 synthetic EN/IT reports (smoke test)
│   ├── real_corpus/             # 6 verbatim excerpts of published ESG reports
│   ├── real_qa.json             # 6 published QA pairs with reference answers
│   ├── real_retrieval_bench.json# 24 graded query/doc records (IR evaluation)
│   ├── PROVENANCE.md            # licenses, upstream paths, sampling method
│   ├── kpi_questions.yaml       # 5 GRI prompts from the SESG 59-prompt set
│   └── gold.jsonl               # 3 literal-answer benchmark items
├── esg_rag/
│   ├── config.py                # config, env + --set overrides
│   ├── ollama.py                # /api/embed, /api/chat, JSON mode, timings
│   ├── chunking.py              # section/fixed/semantic chunking, page/GRI metadata
│   ├── indexing.py              # per-embedder vector stores, cosine search, filters
│   ├── retrieval.py             # BM25, dense, hybrid (RRF) retrieval
│   ├── strategies.py            # naive / hybrid / filtered / multi_query / rerank / self_rag / decompose
│   ├── generation.py            # cited answers + "INSUFFICIENT EVIDENCE"
│   ├── extraction.py            # structured KPI extraction
│   ├── evaluate.py              # retrieval/answer metric helpers
│   ├── lineage.py               # append-only audit log
│   ├── pipeline.py              # orchestrator, matrix runner, smoke test
│   └── cli.py                   # command line UI
├── tests/test_smoke.py
└── output/                      # per-embedder indexes, lineage.jsonl, results/ (gitignored)
```

### Design decisions that map to the tutorial

| Tutorial theme | Where it appears here |
|---|---|
| Section-aware chunking, metadata, temporal/GRI provenance | `chunking.py`, `indexing.py` |
| Embedding model selection (multilingual) | `embeddinggemma:latest` in `config.yaml` |
| Filtered retrieval by company / year / GRI | `--company`, `--year`, `--gri`, `--filter k=v`, optional LLM inference |
| Hybrid BM25 + dense fusion, re-ranking, self-RAG | `retrieval.py` (RRF), `strategies.py` (7 techniques) |
| Multi-model / multi-configuration benchmarking | `matrix` command; 2 LLMs × 2 embedders × techniques; `output/results/` |
| Real published ESG corpus and QA set | `data/real_corpus/`, `data/real_qa.json`, `data/PROVENANCE.md` |
| Faithfulness, attribution, verifiability | `[S#]` citations + `INSUFFICIENT EVIDENCE` guard + optional LLM judge |
| Document → structured data route | `extraction.py` + `kpi_questions.yaml` (JSON mode) |
| Reproducible benchmark, no human labels | `gold.jsonl`, `evaluate.py` |
| Data lineage + systems metrics | `lineage.py`, `metrics` in every result |
| Scale-out discussion | swap `indexing.VectorStore` for FAISS/Milvus/DiskANN; interface unchanged |

## 5. Configuration and customization

`config.yaml` holds all knobs. Override any value with
`ESG_RAG_<SECTION>_<KEY>` environment variables, e.g.:

```bash
ESG_RAG_MODELS_GENERATOR=gemma3:4b python3 -m esg_rag.cli query "..."
ESG_RAG_RETRIEVAL_MODE=dense python3 -m esg_rag.cli evaluate
```

* **Real corpus:** add `.txt`/`.md` files to `data/corpus/` with the same
  `<!-- key: value -->` header and `<!-- page: N -->` markers as the examples,
  then `index --force`. Upstream PDF/HTML parsers only need to emit this text
  format.
* **Full SESG prompt set:** copy
  `materials/esg/sesg-materials/03_platform_documentation/prompts.yaml` over
  `data/kpi_questions.yaml` — the schema is identical (59 GRI questions).
* **Structured output:** `json_schema` on a KPI definition is passed directly
  to Ollama's constrained JSON mode; otherwise "json" in the question text
  enables plain JSON mode.
* **At scale** (the 3-hour tutorial's Part 2 extension): `VectorStore` is the
  only component that knows about storage. Replace it with an ANN index that
  implements `dense_search` / `all_records` and keep the rest unchanged.
* **Real data:** `python3 -m esg_rag.cli --corpus real evaluate --dataset real`;
  the matrix runner builds one index per embedder automatically.

## 6. Data lineage record

One JSON object per query is appended to `output/lineage.jsonl`:

```json
{
  "route": "rag_answer",
  "question": "...",
  "filters": {"company": "Nordwind", "year": "2023"},
  "config": {"generator": "...", "embedder": "...", "retrieval": {...},
             "corpus_fingerprint": "…"},
  "retrieval": {"mode": "hybrid", "passages": [{"chunk_id": "…", "score": …, "metadata": {...}}]},
  "answer": "... [S2]",
  "citations": ["2"],
  "metrics": {"total_latency_s": …, "embedding_latency_s": …,
              "generation_latency_s": …, "prompt_tokens": …,
              "completion_tokens": …}
}
```

This is the audit trail that makes answers attributable and reproducible.

## 7. Notes

* The bundled reports are **fictional** and exist only so the lab needs no
  network access beyond Ollama. They are not real ESG disclosures.
* Benchmark pass criteria in `smoke_test()` are structural (index built,
  citations present, value parsed, benchmark ran); the reported hit rates are
  informational because small local models vary run to run.
* Code is released under Apache-2.0, matching the tutorial proposal's
  commitment to open-source lab artifacts.
