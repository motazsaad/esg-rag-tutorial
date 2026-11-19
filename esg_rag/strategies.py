"""Pluggable RAG strategies: from naive retrieval to self-reflective RAG.

All strategies share the same output contract, so the pipeline, benchmark and
matrix runner treat them identically::

    {
      "answer": str,
      "citations": [str],
      "evidence": [{"id","chunk_id","metadata"}],
      "passages": [retrieval passage dicts],
      "trace": [{"step", "detail", "seconds"}],
      "llm_calls": [CallStats.asdict()],
      "retrieval_calls": int,
      "filters": {...},
      "total_latency_s": float,
    }

Implemented techniques
----------------------
naive       dense top-k, one retrieval, one generation (the baseline)
hybrid      dense + BM25 reciprocal-rank fusion (default)
filtered    hybrid with metadata predicates (optionally LLM-inferred)
multi_query LLM rewrites the question into several retrieval queries, fuses ranks
rerank      retrieve a large candidate pool, LLM-scored re-ranking, then answer
self_rag    retrieve → grade evidence → rewrite-and-retry if unsupported → answer
decompose   split a multi-hop question, answer sub-questions, synthesize
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .generation import GenerationResult, format_evidence
from .ollama import CallStats, OllamaClient, parse_json_loose
from .retrieval import RetrievalResult, Retriever

STRATEGIES = [
    "naive",
    "hybrid",
    "filtered",
    "multi_query",
    "rerank",
    "self_rag",
    "decompose",
]


@dataclass
class StrategyContext:
    """Everything a strategy needs, injected by the pipeline (no global state)."""

    retriever: Retriever
    generator: Any
    client: OllamaClient
    retrieval_cfg: dict[str, Any]
    strategy_cfg: dict[str, Any]
    generation_cfg: dict[str, Any]
    models: dict[str, str]
    judge: bool = False

    def options(self) -> dict[str, Any]:
        return {
            "temperature": float(self.generation_cfg.get("temperature", 0.0)),
            "max_tokens": int(self.generation_cfg.get("max_tokens", 512)),
            "seed": self.generation_cfg.get("seed"),
            "num_ctx": self.generation_cfg.get("num_ctx"),
        }

    def call(
        self,
        prompt: str,
        *,
        role: str = "rewrite",
        system: str | None = None,
        json_mode: bool = True,
        max_tokens: int = 220,
    ) -> tuple[str, dict[str, Any]]:
        text, stats = self.client.chat(
            prompt,
            self.models.get(role, self.models["generator"]),
            system=system,
            temperature=0.0,
            max_tokens=max_tokens,
            json_mode=json_mode,
            seed=self.generation_cfg.get("seed"),
            num_ctx=self.generation_cfg.get("num_ctx"),
        )
        return text, stats.asdict()

    def answer(self, question: str, passages: list[dict[str, Any]], mode: str) -> GenerationResult:
        return self.generator.answer(
            question,
            RetrievalResult(passages=passages, mode=mode),
            judge=self.judge,
            judge_model=self.models.get("judge"),
            max_evidence_chars=int(self.retrieval_cfg.get("max_evidence_chars", 900)),
            options=self.options(),
        )


FILTER_PROMPT = """Extract ESG-report metadata filters from the question. Return JSON only:
{"company": "<company name or NA>", "year": "<YYYY or NA>", "gri": "<GRI code like GRI 305 or NA>"}
Question: {question}
"""

REWRITE_PROMPT = """You improve retrieval queries for a corpus of sustainability reports.
Produce {n} distinct reformulations of the question below (synonyms, explicit GRI/KPI terms).
Return a JSON object: {{"queries": ["...", "..."]}} with exactly {n} strings.
Question: {question}
"""

RERANK_PROMPT = """Score each evidence passage for how well it answers the question.
Return JSON only: {{"scores": [{{"id": "S1", "score": 0}}]}} where score is 0..5.

Question: {question}

Passages:
{evidence}
"""

GRADE_PROMPT = """Decide whether the evidence passages contain enough information to answer the question.
Return JSON only:
{{"sufficient": true/false, "relevant_ids": ["S1", "S3"]}}
Include only passages that actually contain answer-bearing facts.

Question: {question}

Passages:
{evidence}
"""

DECOMPOSE_PROMPT = """Split this ESG question into at most {n} simpler, independently answerable sub-questions.
Return JSON only: {{"subquestions": ["...", "..."]}}.
Question: {question}
"""


def _short_evidence(passages: list[dict[str, Any]], chars: int = 450) -> str:
    text, _ = format_evidence(passages, max_chars=chars)
    return text


def _json_or_default(text: str, default: Any) -> Any:
    try:
        return parse_json_loose(text)
    except Exception:
        return default


def _rrf_fuse(lists: list[list[dict[str, Any]]], k: float = 60.0) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    for passages in lists:
        for rank, passage in enumerate(passages, start=1):
            key = passage["chunk_id"]
            if key not in fused:
                fused[key] = {**passage, "rrf_score": 0.0}
            fused[key]["rrf_score"] += 1.0 / (k + rank)
    ranked = sorted(fused.values(), key=lambda p: p["rrf_score"], reverse=True)
    for i, passage in enumerate(ranked, start=1):
        passage["rank"] = i
    return ranked


def _infer_filters(ctx: StrategyContext, question: str) -> dict[str, Any]:
    raw, call = ctx.call(FILTER_PROMPT.format(question=question), role="rewrite")
    parsed = _json_or_default(raw, {})
    result = {
        k: str(v)
        for k, v in parsed.items()
        if k in {"company", "year", "gri"} and v and str(v).lower() not in {"na", "none", ""}
    }
    return result, call


def _rewrite(ctx: StrategyContext, question: str, n: int) -> tuple[list[str], dict[str, Any]]:
    raw, call = ctx.call(REWRITE_PROMPT.format(n=n, question=question), role="rewrite")
    parsed = _json_or_default(raw, {})
    queries = [str(q) for q in parsed.get("queries", []) if str(q).strip()][:n]
    if not queries:
        queries = [question]
    return queries, call


def _rerank(ctx: StrategyContext, question: str, passages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = {f"S{i + 1}": p["chunk_id"] for i, p in enumerate(passages)}
    raw, call = ctx.call(
        RERANK_PROMPT.format(question=question, evidence=_short_evidence(passages)), role="rerank"
    )
    parsed = _json_or_default(raw, {"scores": []})
    scores: dict[str, float] = {}
    for item in parsed.get("scores", []):
        try:
            scores[str(item["id"])] = float(item["score"])
        except (KeyError, TypeError, ValueError):
            continue
    ranked = sorted(
        passages,
        key=lambda p: scores.get(next((k for k, v in labels.items() if v == p["chunk_id"]), ""), 0.0),
        reverse=True,
    )
    return ranked, call


def _grade(
    ctx: StrategyContext, question: str, passages: list[dict[str, Any]]
) -> tuple[bool, list[str], dict[str, Any]]:
    labels = {f"S{i + 1}": p["chunk_id"] for i, p in enumerate(passages)}
    raw, call = ctx.call(
        GRADE_PROMPT.format(question=question, evidence=_short_evidence(passages)), role="judge"
    )
    parsed = _json_or_default(raw, {})
    ids = [labels.get(str(i), str(i)) for i in parsed.get("relevant_ids", [])]
    return bool(parsed.get("sufficient", False)), ids, call


def _base_result() -> dict[str, Any]:
    return {
        "answer": "",
        "citations": [],
        "evidence": [],
        "passages": [],
        "trace": [],
        "llm_calls": [],
        "retrieval_calls": 0,
        "filters": {},
        "total_latency_s": 0.0,
    }


def _finish(result: dict[str, Any], gen: GenerationResult, t0: float, mode: str) -> dict[str, Any]:
    result["answer"] = gen.answer
    result["citations"] = gen.citations
    result["evidence"] = [
        {"id": e["id"], "chunk_id": e["chunk_id"], "metadata": e["metadata"]} for e in gen.evidence
    ]
    result["judge"] = gen.judge
    result["llm_calls"].append(gen.stats.asdict())
    if gen.judge_stats:
        result["llm_calls"].append(gen.judge_stats.asdict())
    result["mode"] = mode
    result["total_latency_s"] = round(time.perf_counter() - t0, 4)
    return result


def run_strategy(
    ctx: StrategyContext,
    name: str,
    question: str,
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Execute one RAG technique and return the shared result contract."""
    name = name.lower()
    if name not in STRATEGIES:
        raise ValueError(f"Unknown RAG strategy '{name}'. Choose from {', '.join(STRATEGIES)}")
    t0 = time.perf_counter()
    result = _base_result()
    result["strategy"] = name
    result["filters"] = dict(filters or {})
    top_k = int(top_k or ctx.retrieval_cfg.get("top_k", 4))
    mode = "dense" if name == "naive" else str(ctx.retrieval_cfg.get("mode", "hybrid"))
    effective_filters = dict(filters or {})

    if name == "filtered" and ctx.retrieval_cfg.get("infer_filters"):
        t = time.perf_counter()
        inferred, infer_call = _infer_filters(ctx, question)
        effective_filters.update(inferred)
        result["filters"] = dict(effective_filters)
        result["trace"].append({"step": "infer_filters", "detail": inferred, "seconds": round(time.perf_counter() - t, 3)})
        result["llm_calls"].append(infer_call)

    if name in {"naive", "hybrid", "filtered"}:
        t = time.perf_counter()
        retrieval = ctx.retriever.retrieve(question, top_k=top_k, mode=mode, filters=effective_filters)
        result["retrieval_calls"] += 1
        result["passages"] = retrieval.passages
        result["trace"].append(
            {"step": "retrieve", "detail": f"{mode}, {len(retrieval.passages)} passages", "seconds": round(time.perf_counter() - t, 3)}
        )
        gen = ctx.answer(question, retrieval.passages, mode)
        return _finish(result, gen, t0, mode)

    if name == "multi_query":
        n = int(ctx.strategy_cfg.get("multi_query_variants", 3))
        t = time.perf_counter()
        queries, call = _rewrite(ctx, question, n)
        result["llm_calls"].append(call)
        result["trace"].append({"step": "rewrite", "detail": queries, "seconds": round(time.perf_counter() - t, 3)})
        lists = []
        for q in [question, *queries]:
            t = time.perf_counter()
            r = ctx.retriever.retrieve(q, top_k=top_k, mode="hybrid", filters=effective_filters)
            result["retrieval_calls"] += 1
            lists.append(r.passages)
            result["trace"].append({"step": "retrieve", "detail": q[:80], "seconds": round(time.perf_counter() - t, 3)})
        fused = _rrf_fuse(lists, float(ctx.retrieval_cfg.get("rrf_k", 60)))[:top_k]
        result["passages"] = fused
        gen = ctx.answer(question, fused, "multi_query")
        return _finish(result, gen, t0, "multi_query")

    if name == "rerank":
        pool = top_k * int(ctx.retrieval_cfg.get("candidate_multiplier", 3))
        t = time.perf_counter()
        retrieval = ctx.retriever.retrieve(question, top_k=pool, mode="hybrid", filters=effective_filters)
        result["retrieval_calls"] += 1
        result["trace"].append({"step": "retrieve_pool", "detail": f"{len(retrieval.passages)} candidates", "seconds": round(time.perf_counter() - t, 3)})
        t = time.perf_counter()
        ranked, call = _rerank(ctx, question, retrieval.passages)
        result["llm_calls"].append(call)
        result["trace"].append({"step": "rerank", "detail": "LLM scored candidates 0..5", "seconds": round(time.perf_counter() - t, 3)})
        keep = int(ctx.strategy_cfg.get("rerank_top_n") or top_k)
        selected = ranked[:keep]
        result["passages"] = selected
        gen = ctx.answer(question, selected, "rerank")
        return _finish(result, gen, t0, "rerank")

    if name == "self_rag":
        loops = int(ctx.strategy_cfg.get("self_rag_max_loops", 1))
        query = question
        final_passages: list[dict[str, Any]] = []
        for loop in range(loops + 1):
            t = time.perf_counter()
            retrieval = ctx.retriever.retrieve(query, top_k=top_k, mode="hybrid", filters=effective_filters)
            result["retrieval_calls"] += 1
            result["trace"].append({"step": f"retrieve_{loop}", "detail": query[:80], "seconds": round(time.perf_counter() - t, 3)})
            t = time.perf_counter()
            sufficient, relevant_ids, call = _grade(ctx, question, retrieval.passages)
            result["llm_calls"].append(call)
            result["trace"].append({"step": f"grade_{loop}", "detail": {"sufficient": sufficient, "relevant": len(relevant_ids)}, "seconds": round(time.perf_counter() - t, 3)})
            final_passages = [p for p in retrieval.passages if p["chunk_id"] in relevant_ids] if relevant_ids else retrieval.passages
            if sufficient and final_passages:
                break
            if loop < loops:
                t = time.perf_counter()
                rewritten, call = _rewrite(ctx, question, 1)
                result["llm_calls"].append(call)
                query = rewritten[0]
                result["trace"].append({"step": f"rewrite_{loop}", "detail": query[:80], "seconds": round(time.perf_counter() - t, 3)})
        result["passages"] = final_passages
        gen = ctx.answer(question, final_passages, "self_rag")
        return _finish(result, gen, t0, "self_rag")

    if name == "decompose":
        n = int(ctx.strategy_cfg.get("decompose_max_subqs", 3))
        t = time.perf_counter()
        raw, call = ctx.call(DECOMPOSE_PROMPT.format(n=n, question=question), role="rewrite")
        result["llm_calls"].append(call)
        parsed = _json_or_default(raw, {})
        subs = [str(s) for s in parsed.get("subquestions", []) if str(s).strip()][:n] or [question]
        result["trace"].append({"step": "decompose", "detail": subs, "seconds": round(time.perf_counter() - t, 3)})
        subanswers: list[str] = []
        all_passages: list[dict[str, Any]] = []
        for sub in subs:
            t = time.perf_counter()
            r = ctx.retriever.retrieve(sub, top_k=top_k, mode="hybrid", filters=effective_filters)
            result["retrieval_calls"] += 1
            g = ctx.answer(sub, r.passages, "decompose")
            result["llm_calls"].append(g.stats.asdict())
            subanswers.append(f"Q: {sub}\nA: {g.answer}")
            all_passages.extend(r.passages)
            result["trace"].append({"step": "subanswer", "detail": sub[:80], "seconds": round(time.perf_counter() - t, 3)})
        # Deduplicate passages, then synthesize the final answer over the union.
        seen: set[str] = set()
        unique = [p for p in all_passages if not (p["chunk_id"] in seen or seen.add(p["chunk_id"]))]
        result["passages"] = unique[: top_k * 2]
        gen = ctx.answer(
            f"{question}\n\nEarlier sub-answers (use only if supported by evidence):\n" + "\n".join(subanswers),
            result["passages"],
            "decompose",
        )
        return _finish(result, gen, t0, "decompose")

    raise RuntimeError(f"Unhandled strategy: {name}")  # pragma: no cover
