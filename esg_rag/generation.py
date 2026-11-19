"""Evidence-grounded answer generation and LLM-as-judge faithfulness check.

Trustworthiness design choices that matter for regulated documents:

* the generator sees *only* retrieved passages, never the raw corpus;
* every sentence must carry an ``[S#]`` citation pointing to a stored chunk;
* if the evidence does not contain the answer, the model must say so instead
  of hallucinating -- this is what makes downstream verification possible;
* an optional LLM judge produces a binary faithfulness decision plus a short
  rationale (the tutorial discusses judge bias / nondeterminism).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .ollama import CallStats, OllamaClient, parse_json_loose
from .retrieval import RetrievalResult

_CITE_RE = re.compile(r"\[S(\d+)\]")


SYSTEM_PROMPT = """You are a careful analyst answering questions about sustainability (ESG) reports.

Rules you MUST follow:
1. Use ONLY the numbered evidence passages provided. Do not use any outside knowledge and do not guess.
2. Cite the passage(s) that support each statement using the exact form [S1], [S2], etc.
3. Keep the answer concise and factual. Prefer the exact figures and units as written in the evidence.
4. If the evidence does not contain enough information to answer, reply with exactly: {insufficient}
5. Never invent a number, company name, year, unit, or GRI reference.
6. Answer in AT MOST TWO SHORT SENTENCES. Do not quote passages verbatim, do not compare or critique the evidence, and do not add a final "therefore" restatement.
""".strip()


EVIDENCE_HEADER = "### Evidence passages\n"

ANSWER_PROMPT = """### Question
{question}

{evidence}

### Task
Answer the question using only the evidence passages above.
Cite every factual claim with [S#]. If the answer is not in the evidence, say exactly "{insufficient}".

### Answer (at most two sentences)
"""


JUDGE_PROMPT = """You are a strict faithfulness judge for retrieval-augmented answers.

### Question
{question}

### Evidence passages
{evidence}

### Generated answer
{answer}

Decide whether EVERY factual claim in the answer is supported by the evidence.
Return a JSON object with exactly two fields:
  "supported": true or false
  "explanation": one short sentence (max 25 words)
Treat a missing citation, an invented number, or a claim not in the evidence as UNSUPPORTED.
"""


def format_evidence(
    passages: list[dict[str, Any]], max_chars: int = 1400
) -> tuple[str, list[dict[str, Any]]]:
    """Render retrieved passages as numbered [S#] blocks for the prompt."""
    blocks: list[str] = []
    shown: list[dict[str, Any]] = []
    for i, passage in enumerate(passages, start=1):
        text = passage["text"]
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + " …"
        meta = passage["metadata"]
        label = (
            f"[S{i}] chunk={passage['chunk_id']} "
            f"company={meta.get('company', 'NA')} "
            f"year={meta.get('report_year', 'NA')} "
            f"section={meta.get('section', 'NA')} "
            f"page={meta.get('page', 'NA')}"
        )
        blocks.append(f"{label}\n{text}")
        shown.append({"id": f"S{i}", "chunk_id": passage["chunk_id"], "text": text, "metadata": meta})
    return "\n\n".join(blocks), shown


@dataclass
class GenerationResult:
    answer: str
    citations: list[str]
    evidence: list[dict[str, Any]]
    stats: CallStats
    judge: dict[str, Any] | None = None
    judge_stats: CallStats | None = None

    def asdict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "answer": self.answer,
            "citations": self.citations,
            "evidence": [
                {"id": e["id"], "chunk_id": e["chunk_id"], "metadata": e["metadata"]}
                for e in self.evidence
            ],
            "generation": self.stats.asdict(),
        }
        if self.judge is not None:
            out["judge"] = self.judge
            out["judge_call"] = self.judge_stats.asdict() if self.judge_stats else None
        return out


class Generator:
    def __init__(self, client: OllamaClient, model: str, generation_cfg: dict[str, Any]):
        self.client = client
        self.model = model
        self.cfg = generation_cfg

    def answer(
        self,
        question: str,
        retrieval: RetrievalResult,
        *,
        judge: bool = False,
        judge_model: str | None = None,
        max_evidence_chars: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> GenerationResult:
        evidence_text, shown = format_evidence(
            retrieval.passages,
            int(max_evidence_chars or self.cfg.get("max_evidence_chars", 1400)),
        )
        insufficient = self.cfg.get("insufficient_evidence", "INSUFFICIENT EVIDENCE")
        prompt = ANSWER_PROMPT.format(
            question=question, evidence=evidence_text or "(no passages retrieved)", insufficient=insufficient
        )
        system = SYSTEM_PROMPT.format(insufficient=insufficient)
        answer, stats = self.client.chat(
            prompt,
            self.model,
            system=system,
            temperature=float(self.cfg.get("temperature", 0.0)),
            max_tokens=int(self.cfg.get("max_tokens", 512)),
            **(
                {
                    "seed": options.get("seed"),
                    "num_ctx": options.get("num_ctx"),
                    "think": bool(options.get("think", False)),
                }
                if options
                else {}
            ),
        )
        citations = sorted(set(_CITE_RE.findall(answer)))
        result = GenerationResult(answer=answer, citations=citations, evidence=shown, stats=stats)

        if judge:
            judge_prompt = JUDGE_PROMPT.format(
                question=question, evidence=evidence_text or "(none)", answer=answer
            )
            raw, jstats = self.client.chat(
                judge_prompt,
                judge_model or self.model,
                temperature=0.0,
                max_tokens=160,
                json_mode=True,
            )
            try:
                parsed = parse_json_loose(raw)
                verdict = {
                    "supported": bool(parsed.get("supported", False)),
                    "explanation": str(parsed.get("explanation", "")),
                }
            except Exception:
                verdict = {"supported": None, "explanation": f"Unparseable judge output: {raw[:200]}"}
            result.judge = verdict
            result.judge_stats = jstats
        return result
