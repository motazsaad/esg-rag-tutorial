# ESG RAG model / technique matrix

Dataset: **real** · questions per cell: **2** · LLM judge: **True**

## Generation results

| LLM | Embedder | Strategy | Ret.hit@k | Answer hit | Token F1 | Numeric | Latency (s) | Tokens | LLM calls | Judge |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3:4b-instruct | embeddinggemma:latest | naive | 1.0 | 1.0 | 0.639 | 1.0 | 63.652 | 92.0 | 2.0 | 1.0 |
| qwen3:4b-instruct | embeddinggemma:latest | hybrid | 1.0 | 1.0 | 0.639 | 1.0 | 68.408 | 90.0 | 2.0 | 1.0 |
| qwen3:4b-instruct | embeddinggemma:latest | multi_query | 1.0 | 1.0 | 0.639 | 1.0 | 85.631 | 211.0 | 3.0 | 1.0 |
| qwen3:4b-instruct | embeddinggemma:latest | rerank | 1.0 | 1.0 | 0.639 | 1.0 | 134.116 | 110.0 | 3.0 | 1.0 |
| qwen3:4b-instruct | embeddinggemma:latest | self_rag | 1.0 | 1.0 | 0.657 | 1.0 | 65.313 | 115.5 | 3.0 | 0.5 |
| gemma3:4b | embeddinggemma:latest | naive | 1.0 | 1.0 | 0.627 | 1.0 | 58.364 | 99.5 | 2.0 | 1.0 |
| gemma3:4b | embeddinggemma:latest | hybrid | 1.0 | 1.0 | 0.585 | 1.0 | 46.461 | 101.5 | 2.0 | 1.0 |
| gemma3:4b | embeddinggemma:latest | multi_query | 1.0 | 1.0 | 0.627 | 1.0 | 68.844 | 212.5 | 3.0 | 1.0 |
| gemma3:4b | embeddinggemma:latest | rerank | 1.0 | 0.5 | 0.475 | 0.5 | 116.767 | 109.0 | 3.0 | 1.0 |
| gemma3:4b | embeddinggemma:latest | self_rag | 1.0 | 1.0 | 0.637 | 1.0 | 61.77 | 116.5 | 3.0 | 0.5 |
| qwen3:4b-instruct | qwen3-embedding:0.6b | naive | 1.0 | 1.0 | 0.639 | 1.0 | 66.781 | 92.5 | 2.0 | 1.0 |
| qwen3:4b-instruct | qwen3-embedding:0.6b | hybrid | 1.0 | 1.0 | 0.639 | 1.0 | 68.441 | 90.0 | 2.0 | 1.0 |
| qwen3:4b-instruct | qwen3-embedding:0.6b | multi_query | 1.0 | 1.0 | 0.639 | 1.0 | 86.77 | 211.0 | 3.0 | 1.0 |
| qwen3:4b-instruct | qwen3-embedding:0.6b | rerank | 1.0 | 1.0 | 0.639 | 1.0 | 147.382 | 183.0 | 3.0 | 1.0 |
| qwen3:4b-instruct | qwen3-embedding:0.6b | self_rag | 1.0 | 1.0 | 0.657 | 1.0 | 67.266 | 115.5 | 3.0 | 0.5 |
| gemma3:4b | qwen3-embedding:0.6b | naive | 1.0 | 1.0 | 0.609 | 1.0 | 78.159 | 98.0 | 2.0 | 1.0 |
| gemma3:4b | qwen3-embedding:0.6b | hybrid | 1.0 | 1.0 | 0.627 | 1.0 | 56.842 | 97.0 | 2.0 | 1.0 |
| gemma3:4b | qwen3-embedding:0.6b | multi_query | 1.0 | 1.0 | 0.627 | 1.0 | 72.164 | 212.5 | 3.0 | 1.0 |
| gemma3:4b | qwen3-embedding:0.6b | rerank | 1.0 | 0.5 | 0.491 | 0.5 | 131.151 | 183.5 | 3.0 | 0.0 |
| gemma3:4b | qwen3-embedding:0.6b | self_rag | 1.0 | 1.0 | 0.637 | 1.0 | 56.367 | 119.5 | 3.0 | 0.5 |

## Retrieval-only benchmark (graded ESG query/doc pairs)

| Embedder | NDCG@10 | MRR | P@1 |
|---|---:|---:|---:|
| embeddinggemma:latest | 0.9697 | 1.0 | 1.0 |
| qwen3-embedding:0.6b | 0.9697 | 1.0 | 1.0 |
