# ESG RAG — end-to-end, local-first pipeline

Companion implementation for the ICDE 2027 tutorial proposal
**"Data Management for ESG and Regulated Document Analytics: Pipelines, Retrieval, Provenance, and Evaluation"**
by Motaz K. H. Saad and Antonella Longo (University of Salento).

This repository provides a **working, local-first** RAG/analytics pipeline and lab materials for regulated-document (ESG) settings. It demonstrates data-management-oriented components of the tutorial: corpus ingestion and versioned artifacts, section-aware chunking, metadata-aware indexing and filtered hybrid retrieval (dense + BM25), grounded answer generation with citations, optional structured KPI extraction, and append-only **lineage/provenance** records (sources, configuration, models, latency, token/cost-related metrics).

The tutorial framing also covers ESG data management topics such as reporting standards/interoperability (e.g. CSRD/ESRS, GRI, ISSB, digital reporting), data quality/comparability, and evidence-grounded claim verification relevant to greenwashing analysis; this codebase serves as a practical lab companion for pipeline, retrieval, provenance, and evaluation workflows.

Everything runs on a laptop with no API keys and no cloud services.

```
parse → section-aware chunk → embeddings → local vector index
      → filtered hybrid retrieval (dense + BM25, RRF fusion)
      → grounded answer [S#] / structured KPI extraction
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
> [`data/PROVENANCE.md`](data/PROVENANCE.md). A walkthrough of the implemented
> pipeline is in [`slides/pipeline-lab.pdf`](slides/pipeline-lab.pdf).
> The lab includes a small sample of the published ESG corpora and runs the same
> pipeline with multiple generators/embedders and RAG techniques.

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

## Slides

- `slides/tutorial-preview.tex` / `slides/tutorial-preview.pdf` — updated tutorial proposal preview (aligned with ICDE 2027 proposal)
- `slides/pipeline-lab.tex` / `slides/pipeline-lab.pdf` — implemented pipeline and lab walkthrough

## License

Open-source, Apache-2.0.
