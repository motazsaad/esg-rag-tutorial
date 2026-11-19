"""Small reproducible benchmark harness (retrieval + answer quality + cost).

No human labels are needed for the smoke test: gold answers are literal
substrings taken from the synthetic corpus, and gold passages are GRI section
references. This mirrors the tutorial's "test-set generation without human
labels" theme; for a real corpus you would swap in the RAGAS / judge-based
evaluation from the companion repositories.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from .generation import Generator
from .retrieval import Retriever


def load_gold(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Gold set not found: {source}")
    return [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def retrieval_hit(passages: list[dict[str, Any]], item: dict[str, Any]) -> bool:
    """True when a retrieved passage matches the expected GRI section/company."""
    wanted = [str(s).lower() for s in item.get("expected_sections", [])]
    for p in passages:
        meta = p["metadata"]
        section = str(meta.get("section", "")).lower()
        gri = str(meta.get("gri_reference", "")).lower()
        if any(w in section or w in gri for w in wanted):
            if item.get("company") and str(item["company"]).lower() not in str(meta.get("company", "")).lower():
                continue
            return True
    return False


def answer_hit(answer: str, item: dict[str, Any]) -> bool:
    normalized = re.sub(r"\s+", " ", answer.replace("\u00a0", " ")).lower()
    return all(str(x).lower() in normalized for x in item.get("expected_substrings", []))


def evaluate(
    retriever: Retriever,
    generator: Generator,
    gold_items: list[dict[str, Any]],
    *,
    judge: bool = False,
    verbose: bool = True,
    max_evidence_chars: int = 900,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in gold_items:
        filters = {
            k: v
            for k, v in {
                "company": item.get("company"),
                "year": item.get("year"),
            }.items()
            if v
        }
        t0 = time.perf_counter()
        retrieval = retriever.retrieve(item["question"], filters=filters)
        gen = generator.answer(
            item["question"], retrieval, judge=judge, max_evidence_chars=max_evidence_chars
        )
        rows.append(
            {
                "question": item["question"],
                "retrieval_hit": retrieval_hit(retrieval.passages, item),
                "answer_hit": answer_hit(gen.answer, item),
                "answer": gen.answer,
                "citations": gen.citations,
                "judge": gen.judge,
                "retrieval_latency_s": round(retrieval.embed_latency_s, 4),
                "generation_latency_s": round(gen.stats.latency_s, 4),
                "latency_s": round(time.perf_counter() - t0, 4),
                "completion_tokens": gen.stats.completion_tokens,
                "passages": retrieval.ids(),
            }
        )
        if verbose:
            flag = "OK " if rows[-1]["answer_hit"] else "MISS"
            print(f"  [{flag}] {item['question'][:70]}")
            print(f"         answer: {gen.answer[:160]}")

    n = len(rows) or 1
    summary = {
        "questions": len(rows),
        "retrieval_hit@k": round(sum(r["retrieval_hit"] for r in rows) / n, 3),
        "answer_hit_rate": round(sum(r["answer_hit"] for r in rows) / n, 3),
        "mean_latency_s": round(sum(r["latency_s"] for r in rows) / n, 3),
        "mean_generation_latency_s": round(sum(r["generation_latency_s"] for r in rows) / n, 3),
        "mean_completion_tokens": round(sum(r["completion_tokens"] for r in rows) / n, 1),
        "judge_supported_rate": (
            round(
                sum(1 for r in rows if r["judge"] and r["judge"].get("supported")) / n,
                3,
            )
            if judge
            else None
        ),
    }
    return {"summary": summary, "rows": rows}
