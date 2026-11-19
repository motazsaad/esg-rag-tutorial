#!/usr/bin/env bash
# One-command smoke test. Prerequisites: Ollama running with the two models.
#   ollama serve &
#   ollama pull qwen3:4b-instruct
#   ollama pull embeddinggemma:latest
set -euo pipefail
cd "$(dirname "$0")"
python3 -m esg_rag.cli smoke-test "$@"
