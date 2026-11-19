#!/usr/bin/env bash
# Real-data benchmark on the sampled published ESG corpus.
# Prerequisites: Ollama with qwen3:4b-instruct, gemma3:4b,
# embeddinggemma:latest and qwen3-embedding:0.6b.
set -euo pipefail
cd "$(dirname "$0")"

echo "== Hybrid RAG on the real ESG sample =="
python3 -u -m esg_rag.cli --corpus real evaluate --dataset real --strategy hybrid --limit 6 "$@"

echo "== Embedding-only retrieval benchmark =="
python3 -u -m esg_rag.cli retrieval-bench --embedder embeddinggemma:latest
python3 -u -m esg_rag.cli retrieval-bench --embedder qwen3-embedding:0.6b

echo "== Full matrix (2 LLMs x 2 embedders x 5 techniques) =="
python3 -u -m esg_rag.cli matrix --dataset real --limit 2 --judge
