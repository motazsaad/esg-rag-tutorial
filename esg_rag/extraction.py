"""Structured KPI extraction — the document-to-table route of the pipeline.

Questions mirror the SESG case study (59 GRI-referenced prompts in the
project's ``prompts.yaml``). Each question carries a target JSON schema in its
text; the LLM is constrained with Ollama's JSON mode and the parsed object is
attached to the retrieved passages, the model, and the timing data so the
result can be stored in a relational KPI table *with provenance*.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .generation import format_evidence
from .ollama import OllamaClient, parse_json_loose
from .retrieval import RetrievalResult, Retriever

EXTRACT_SYSTEM = """You are an expert ESG data extractor. You extract values ONLY from the provided sustainability-report passages.
Rules:
- Never infer, compute or guess a value that is not written in the passages.
- Use "NA" (string) when information is missing, unless the requested schema says otherwise.
- Preserve numbers, units, currency codes and years exactly as reported.
- Respect the output schema and formatting rules in the question exactly.
- Output the final answer only, with no preamble, comments or markdown.
""".strip()

EXTRACT_PROMPT = """### Target KPI
metric: {metric}
GRI reference: {reference}
category: {category}

### Question and required output format
{question}

### Evidence passages
{evidence}

### Extracted value
"""


def load_kpi_questions(path: str | Path) -> list[dict[str, Any]]:
    """Load KPI prompt definitions from YAML (or a JSON list, for convenience)."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"KPI question file not found: {source}")
    text = source.read_text(encoding="utf-8")
    if source.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyYAML is required to read the KPI question file") from exc
        data = yaml.safe_load(text) or {}
        return data["questions"] if isinstance(data, dict) else data
    import json

    data = json.loads(text)
    return data["questions"] if isinstance(data, dict) else data


def find_kpi(questions: list[dict[str, Any]], identifier: str) -> dict[str, Any]:
    """Find a KPI definition by q_id, kpi_id, id, or metric (case-insensitive)."""
    needle = identifier.strip().lower()
    for q in questions:
        for key in ("q_id", "kpi_id", "id", "metric"):
            if str(q.get(key, "")).strip().lower() == needle:
                return q
    raise KeyError(f"Unknown KPI: {identifier}")


def _expects_json(question: dict[str, Any]) -> bool:
    if "json_schema" in question:
        return True
    if "expect_json" in question:
        return bool(question["expect_json"])
    return "json" in str(question.get("question", "")).lower()


def extract_kpi(
    retriever: Retriever,
    client: OllamaClient,
    question: dict[str, Any],
    *,
    model: str,
    top_k: int | None = None,
    filters: dict[str, Any] | None = None,
    max_evidence_chars: int = 2200,
) -> dict[str, Any]:
    """Retrieve evidence, then extract one structured value for a KPI question."""
    retrieval: RetrievalResult = retriever.retrieve(query=question["question"], top_k=top_k, filters=filters)
    evidence_text, shown = format_evidence(retrieval.passages, max_chars=max_evidence_chars)
    prompt = EXTRACT_PROMPT.format(
        metric=question.get("metric", "NA"),
        reference=question.get("reference", "NA"),
        category=question.get("esg_category", "NA"),
        question=question["question"],
        evidence=evidence_text or "(no passages retrieved)",
    )
    as_json = _expects_json(question)
    raw, stats = client.chat(
        prompt,
        model,
        system=EXTRACT_SYSTEM,
        temperature=0.0,
        max_tokens=320,
        json_mode=as_json,
        json_schema=question.get("json_schema"),
    )
    value: Any = raw
    parse_error = None
    if as_json:
        try:
            value = parse_json_loose(raw)
        except Exception as exc:
            parse_error, value = str(exc), raw
    return {
        "kpi_id": question.get("kpi_id", question.get("q_id")),
        "q_id": question.get("q_id"),
        "metric": question.get("metric"),
        "reference": question.get("reference"),
        "esg_category": question.get("esg_category"),
        "target_table": question.get("db_table_name", "kpi_extraction_atomic"),
        "question": question["question"],
        "value": value,
        "raw_response": raw,
        "parse_error": parse_error,
        "filters": filters or {},
        "retrieval": {
            "mode": retrieval.mode,
            "passages": [
                {"chunk_id": p["chunk_id"], "score": p["score"], "metadata": p["metadata"]}
                for p in retrieval.passages
            ],
            "embed_call": retrieval.embed_stats.asdict() if retrieval.embed_stats else None,
        },
        "generation": stats.asdict(),
        "evidence": [{"id": e["id"], "chunk_id": e["chunk_id"], "metadata": e["metadata"]} for e in shown],
    }
