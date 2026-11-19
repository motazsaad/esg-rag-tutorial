# Provenance of the real ESG data

The smoke-test corpus in `data/corpus/` is synthetic. The data in
`data/real_corpus/`, `data/real_qa.json` and `data/retrieval_bench.json` is a
small, attributed sample of the DataLab/University of Salento publications
listed in the root `README.md`.

## 1. Real ESG reports and QA pairs — ESG-RAG-benchmark-ragas

- Repository: https://github.com/datalab-unisalento/ESG-RAG-benchmark-ragas
- Paper: *Empirical Evaluation of Open-Source Large Language Models for
  Retrieval-Augmented Generation in ESG Domain*, paper BigD275, IEEE BigData
  2026 (accepted).
- License: **MIT** — Copyright (c) 2026 the authors of BigD275, University of
  Salento, University of Naples Federico II, and IFAB Foundation.
- Full corpus: 498 ESG reports converted to Markdown (`data/documents/`), 284
  synthetic QA pairs (`outputs/testset/01_esg_testset.json`), generated with
  `qwen3:4b-instruct` and reviewed by domain experts.

**What is sampled here**

| Local file | Upstream source |
|---|---|
| `real_corpus/375_58_2020.md` | `data/documents/375_58_2020.md` (Crédit Agricole, 2020) |
| `real_corpus/377_60_2021.md` | `data/documents/377_60_2021.md` (Volvo Cars, 2021) |
| `real_corpus/327_52_2022.md` | `data/documents/327_52_2022.md` (Continental AG, 2022) |
| `real_corpus/44_7_2023.md` | `data/documents/44_7_2023.md` (Mercedes-Benz Group, 2023) |
| `real_corpus/132_18_2020.md` | `data/documents/132_18_2020.md` (ENGIE, 2020) |
| `real_corpus/300_45_2020.md` | `data/documents/300_45_2020.md` (Maersk, 2020) |
| `real_qa.json` (6 questions) | `outputs/testset/01_esg_testset.json` (indices 3, 52, 82, 190, 207, 160) |

Each report file is a **verbatim excerpt** (~2.5–3 KB) taken around the
answer-bearing passage, prefixed with a metadata header (`doc_id`, `company`,
`report_year`, `language`, `source_repo`, `source_path`, `source_url`,
`license`). Page markers `*Page N*` from the upstream Markdown are preserved
and parsed by `chunking.py`. The QA pairs keep the original `user_input` and
`reference` fields unchanged.

## 2. Graded retrieval benchmark — benchmark-embeddings-esg

- Repository: https://github.com/datalab-unisalento/benchmark-embeddings-esg
- Paper: *Benchmarking Embedding Models for ESG Data*, Saad, Cretì, Gentile,
  Kazemi, Longo; ITADATA 2026, CEUR-WS.
- License: **Apache-2.0**.
- Full dataset: 312 English and 312 Italian records; each record has a query
  and three candidate documents graded 3 (highly relevant), 2 (moderately
  relevant), 0 (irrelevant).

**What is sampled here:** `real_retrieval_bench.json` contains the first 24
English records, unchanged, from
`data/English-cleaned_data_embeddings_benchmark.json`. It is used by
`retrieval-bench` to compute NDCG@10, MRR and Precision@1 per embedding model.

## 3. Related companion repository (referenced, not sampled)

- https://github.com/kiannakazemi/esg-rag-reference-free-evaluation —
  reference-free data-completeness evaluation for ESG KPI extraction (no human
  answers or LLM judge). Not bundled here; see the root README.

## Citation

If you use these samples, cite the two papers above. The full upstream
repositories contain the complete data and are the authoritative source;
this lab intentionally keeps only a small excerpt to stay laptop-friendly.
