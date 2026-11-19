"""Local vector index with metadata filtering.

This is a deliberately small, dependency-free reference implementation of the
vector store: vectors are L2-normalised and persisted as JSON, and search is
exact cosine similarity computed in pure Python (with optional NumPy
acceleration when NumPy happens to be installed).

For a 3,700-report corpus this would be swapped for FAISS / Milvus / DiskANN;
the interface (``build``, ``dense_search``, metadata filters) stays the same.
That swap is one of the discussion points of the tutorial's Part 2 extension.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Iterable

try:  # optional acceleration
    import numpy as np
except ImportError:  # pragma: no cover - depends on environment
    np = None


def normalize(vec: Iterable[float]) -> list[float]:
    v = [float(x) for x in vec]
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def dot(a: list[float], b: list[float]) -> float:
    if np is not None:
        return float(np.dot(a, b))
    return sum(x * y for x, y in zip(a, b))


_ALIASES = {
    "year": "report_year",
    "lang": "language",
    "gri": "gri_reference",
    "section": "section",
    "company": "company",
    "country": "country",
    "doc_id": "doc_id",
}


def matches_filters(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Return True when ``metadata`` satisfies every metadata predicate.

    String comparisons are case-insensitive and support substring matching via
    a trailing ``*`` (e.g. ``section="GRI 305*"``). Numeric/other values are
    compared directly (with a string-cast fallback).
    """
    if not filters:
        return True
    for raw_key, expected in filters.items():
        if expected in (None, "", "NA", "all"):
            continue
        key = _ALIASES.get(raw_key, raw_key)
        actual = metadata.get(key, metadata.get(raw_key))
        if actual is None:
            return False
        exp = str(expected)
        act = str(actual)
        if exp.endswith("*"):
            if not act.lower().startswith(exp[:-1].lower()):
                return False
        elif act.lower() != exp.lower() and exp.lower() not in act.lower():
            # Substring matching lets users write --company Nordwind for the
            # metadata value "Nordwind Energy S.p.A." (case-insensitive).
            return False
    return True


class VectorStore:
    """A list of embedded chunks plus exact cosine search."""

    def __init__(self, records: list[dict[str, Any]] | None = None):
        self.records: list[dict[str, Any]] = records or []

    def __len__(self) -> int:
        return len(self.records)

    # ---------- construction ----------
    def build(
        self,
        chunks: list[dict[str, Any]],
        embed_fn: Callable[[list[str]], tuple[list[list[float]], Any]],
        batch_size: int = 16,
    ) -> dict[str, Any]:
        """Embed all chunks in batches. ``embed_fn`` returns (vectors, stats)."""
        t0 = time.perf_counter()
        records: list[dict[str, Any]] = []
        embed_latency = 0.0
        prompt_tokens = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors, stats = embed_fn([c["text"] for c in batch])
            if len(vectors) != len(batch):
                raise RuntimeError("Embedder returned a different number of vectors")
            embed_latency += getattr(stats, "latency_s", 0.0)
            prompt_tokens += getattr(stats, "prompt_tokens", 0)
            for chunk, vector in zip(batch, vectors):
                records.append({**chunk, "embedding": normalize(vector)})
        self.records = records
        return {
            "chunks": len(records),
            "build_time_s": round(time.perf_counter() - t0, 4),
            "embedding_time_s": round(embed_latency, 4),
            "embedding_tokens": prompt_tokens,
            "dimensions": len(records[0]["embedding"]) if records else 0,
        }

    # ---------- persistence ----------
    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "esg-rag-vector-store-v1",
            "count": len(self.records),
            "records": self.records,
        }
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Index not found: {source} (run the 'index' command first)")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("format") != "esg-rag-vector-store-v1":
            raise ValueError(f"Unsupported index format in {source}")
        return cls(payload["records"])

    @classmethod
    def from_file_if_exists(cls, path: str | Path) -> "VectorStore | None":
        try:
            return cls.load(path)
        except FileNotFoundError:
            return None

    def fingerprint(self) -> str:
        """Short hash of the index contents, stored with each lineage record."""
        digest = hashlib.sha256()
        for rec in self.records:
            digest.update(rec["chunk_id"].encode("utf-8"))
            digest.update(rec["text"].encode("utf-8"))
        return digest.hexdigest()[:16]

    # ---------- search ----------
    def dense_search(
        self,
        query_vector: Iterable[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        q = normalize(query_vector)
        scored: list[tuple[dict[str, Any], float]] = []
        for rec in self.records:
            if not matches_filters(rec["metadata"], filters):
                continue
            scored.append((rec, dot(q, rec["embedding"])))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def all_records(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return [r for r in self.records if matches_filters(r["metadata"], filters)]
