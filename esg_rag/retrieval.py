"""Filtered dense / BM25 / hybrid retrieval (reciprocal rank fusion).

Why BM25 at all? Regulated documents are full of rare literal tokens (GRI
codes, KPI names, units such as ``tCO2e``). Dense embeddings handle paraphrases;
BM25 handles exact tokens. Reciprocal rank fusion combines the two candidate
lists without needing to calibrate their very different score scales.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .indexing import VectorStore
from .ollama import CallStats, OllamaClient

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    """Okapi BM25 over the in-memory chunk store (pure Python)."""

    def __init__(self, records: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: list[list[str]] = []
        self.lengths: list[int] = []
        self.records: list[dict[str, Any]] = []
        df: Counter[str] = Counter()
        for rec in records:
            tokens = tokenize(rec["text"])
            self.docs.append(tokens)
            self.lengths.append(len(tokens))
            self.records.append(rec)
            df.update(set(tokens))
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 1.0
        n = len(records)
        self.idf = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(
        self, query: str, top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[tuple[dict[str, Any], float]]:
        q_tokens = tokenize(query)
        scored: list[tuple[dict[str, Any], float]] = []
        for rec, tokens, length in zip(self.records, self.docs, self.lengths):
            if not _passes_metadata(rec["metadata"], filters):
                continue
            tf = Counter(tokens)
            score = 0.0
            for term in q_tokens:
                if term not in tf:
                    continue
                denom = tf[term] + self.k1 * (1 - self.b + self.b * length / self.avgdl)
                score += self.idf.get(term, 0.0) * tf[term] * (self.k1 + 1) / denom
            if score > 0:
                scored.append((rec, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


# Metadata filtering for BM25: import here to avoid duplicating alias logic.
from .indexing import matches_filters as _passes_metadata  # noqa: E402


@dataclass
class RetrievalResult:
    passages: list[dict[str, Any]] = field(default_factory=list)
    mode: str = "hybrid"
    embed_stats: CallStats | None = None
    embed_latency_s: float = 0.0

    def ids(self) -> list[str]:
        return [p["chunk_id"] for p in self.passages]


class Retriever:
    def __init__(self, store: VectorStore, client: OllamaClient, embedder: str, retrieval_cfg: dict[str, Any]):
        self.store = store
        self.client = client
        self.embedder = embedder
        self.cfg = retrieval_cfg
        self.bm25 = BM25Index(
            store.records,
            k1=float(retrieval_cfg.get("bm25_k1", 1.5)),
            b=float(retrieval_cfg.get("bm25_b", 0.75)),
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        mode: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        top_k = int(top_k or self.cfg.get("top_k", 5))
        mode = (mode or self.cfg.get("mode", "hybrid")).lower()
        if mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError(f"Unknown retrieval mode: {mode}")

        result = RetrievalResult(mode=mode)
        dense_hits: list[tuple[dict[str, Any], float]] = []
        if mode in {"dense", "hybrid"}:
            vectors, stats = self.client.embed([query], self.embedder)
            result.embed_stats = stats
            result.embed_latency_s = stats.latency_s
            dense_hits = self.store.dense_search(vectors[0], top_k=top_k, filters=filters)

        if mode == "dense":
            result.passages = [_package(rec, score, "dense", i + 1) for i, (rec, score) in enumerate(dense_hits)]
            return result

        if mode == "bm25":
            hits = self.bm25.search(query, top_k=top_k, filters=filters)
            result.passages = [_package(rec, score, "bm25", i + 1) for i, (rec, score) in enumerate(hits)]
            return result

        # hybrid: reciprocal rank fusion of dense + lexical candidate lists.
        bm25_hits = self.bm25.search(query, top_k=top_k, filters=filters)
        rrf_k = float(self.cfg.get("rrf_k", 60))
        fused: dict[str, dict[str, Any]] = {}
        for rank, (rec, score) in enumerate(dense_hits, start=1):
            fused[rec["chunk_id"]] = _package(rec, 0.0, "hybrid", rank)
            fused[rec["chunk_id"]]["rrf_score"] = 1.0 / (rrf_k + rank)
            fused[rec["chunk_id"]]["dense_score"] = round(score, 6)
        for rank, (rec, score) in enumerate(bm25_hits, start=1):
            entry = fused.get(rec["chunk_id"])
            if entry is None:
                entry = _package(rec, 0.0, "hybrid", rank)
                entry["rrf_score"] = 0.0
                fused[rec["chunk_id"]] = entry
            entry["rrf_score"] += 1.0 / (rrf_k + rank)
            entry["bm25_rank"] = rank
            entry["bm25_score"] = round(score, 4)
        ranked = sorted(fused.values(), key=lambda p: p["rrf_score"], reverse=True)[:top_k]
        for i, passage in enumerate(ranked, start=1):
            passage["rank"] = i
        result.passages = ranked
        return result


def _package(rec: dict[str, Any], score: float, mode: str, rank: int) -> dict[str, Any]:
    return {
        "chunk_id": rec["chunk_id"],
        "text": rec["text"],
        "metadata": rec["metadata"],
        "score": round(score, 6),
        "rank": rank,
        "retrieval_mode": mode,
        "dense_rank": rank if mode in {"dense", "hybrid"} else None,
        "bm25_rank": rank if mode == "bm25" else None,
    }
