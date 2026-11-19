"""ESG RAG — an end-to-end, local-first retrieval-augmented generation pipeline.

Reference implementation for the ESG RAG tutorial
"Retrieval-Augmented Analytics for Regulated Documents".

Pipeline stages
---------------
    parse -> chunk -> embed -> index -> retrieve -> generate / extract -> lineage

The code deliberately depends only on ``requests`` (plus optional ``numpy`` and
``PyYAML``) so that it runs on a laptop with no model downloads beyond the two
Ollama models used in the smoke test:

    qwen3:4b-instruct        (generator / judge)
    embeddinggemma:latest    (embedder)
"""

__version__ = "0.1.0"
