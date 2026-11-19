# ESG RAG model / technique matrix

Dataset: **real** · questions per cell: **2** · LLM judge: **True**

## Generation results

| LLM | Embedder | Strategy | Ret.hit@k | Answer hit | Token F1 | Latency (s) | Tok/s | RSS MB | Model MB | J/query | W avg | Judge |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3:4b-instruct | embeddinggemma:latest | hybrid | 1.0 | 1.0 | 0.639 | 71.679 | 7.09 | 37.6 | 3879.1 | 4659.21 | 65.00 | 1.0 |
| qwen3:4b-instruct | embeddinggemma:latest | self_rag | 1.0 | 1.0 | 0.657 | 62.666 | 7.93 | 37.6 | 3879.1 | 4073.32 | 65.00 | 0.5 |
| qwen3:4b-instruct | qwen3-embedding:0.6b | hybrid | 1.0 | 1.0 | 0.639 | 76.06 | 6.47 | 39.4 | 4424.0 | 4943.98 | 65.00 | 1.0 |
| qwen3:4b-instruct | qwen3-embedding:0.6b | self_rag | 1.0 | 1.0 | 0.657 | 66.271 | 7.37 | 39.4 | 4424.0 | 4307.62 | 65.00 | 0.5 |

## Retrieval-only benchmark (graded ESG query/doc pairs)

| Embedder | NDCG@10 | MRR | P@1 |
|---|---:|---:|---:|
| embeddinggemma:latest | 0.9697 | 1.0 | 1.0 |
| qwen3-embedding:0.6b | 0.9697 | 1.0 | 1.0 |

## System metrics notes

Tok/s is Ollama decode speed (completion tokens / eval time). RSS MB is the
lab process; Model MB is the loaded Ollama model size from `/api/ps`.
Energy is measured via RAPL/NVIDIA-SMI where the sensor is readable;
otherwise it is an **estimate** (`estimated_watts × wall time`) and
`power_measured` is false. Do not compare estimated joules across machines.
