# Lab data

Two kinds of data are bundled.

## 1. Synthetic smoke-test data (`corpus/`, `kpi_questions.yaml`, `gold.jsonl`)

Fictional companies (Nordwind Energy S.p.A., Solaria Industrie S.r.l.) used by
`./run_smoke_test.sh`. No network access beyond Ollama is needed.

| File | Purpose |
|---|---|
| `corpus/en_nordwind_2023.md` | English sustainability report (GRI 2, 302, 305, 401, 403) |
| `corpus/it_solaria_2023.md` | Italian sustainability report (GRI 2, 305, 404) — multilingual retrieval |
| `kpi_questions.yaml` | 5 KPI prompts adapted from the SESG `prompts.yaml` (59 questions) |
| `gold.jsonl` | 3 question / literal-answer / expected-section pairs |

## 2. Real published ESG data (`real_corpus/`, `real_qa.json`, `real_retrieval_bench.json`)

A small, attributed sample of the presenters' published benchmark data:

| File | Purpose | Upstream |
|---|---|---|
| `real_corpus/*.md` | 6 verbatim report excerpts (Crédit Agricole, Volvo, Continental, Mercedes-Benz, ENGIE, Maersk) | [ESG-RAG-benchmark-ragas](https://github.com/datalab-unisalento/ESG-RAG-benchmark-ragas) (IEEE BigData 2026, MIT) |
| `real_qa.json` | 6 QA pairs with reference answers from the published 284-pair test set | same repo |
| `real_retrieval_bench.json` | 24 graded query/3-document records | [benchmark-embeddings-esg](https://github.com/datalab-unisalento/benchmark-embeddings-esg) (ITADATA 2026, Apache-2.0) |

See [`PROVENANCE.md`](PROVENANCE.md) for exact upstream paths, licenses and
sampling method, and [`../docs/REAL_DATA_RESULTS.md`](../docs/REAL_DATA_RESULTS.md)
for the experiments run on this data.

To use more real data, replace the files in `real_corpus/` with additional
upstream reports (keeping the metadata header) and point `paths.real_qa` at a
larger QA set.
