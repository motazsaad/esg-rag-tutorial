"""Command-line interface for the ESG RAG lab.

Everything is parameterised: use ``--set section.key=value`` (repeatable) to
override ANY config value, and the dedicated flags for the common knobs.

Examples
--------
    python -m esg_rag.cli doctor
    python -m esg_rag.cli index --corpus real --embedder qwen3-embedding:0.6b
    python -m esg_rag.cli query "..." --strategy self_rag --set retrieval.top_k=8
    python -m esg_rag.cli evaluate --dataset real --strategy hybrid --limit 6
    python -m esg_rag.cli retrieval-bench --embedder gemma3:4b
    python -m esg_rag.cli matrix --limit 3 --judge
    python -m esg_rag.cli smoke-test
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .config import Config, load_config, parse_value
from .pipeline import ESGRAGPipeline
from .strategies import STRATEGIES


def _dump(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _pretty(payload)


def _pretty(payload: dict[str, Any]) -> None:
    cmd = payload.get("command")
    if cmd == "doctor":
        print("ESG RAG doctor")
        print(f"  Ollama reachable : {payload.get('ollama_reachable')}")
        for model, ok in payload.get("models_available", {}).items():
            print(f"    {'OK ' if ok else 'MISSING'} {model}")
        for name, files in payload.get("corpora", {}).items():
            print(f"  Corpus [{name:9s}]: {len(files)} files — {', '.join(files[:3]) or 'none'}")
        print(f"  Index present    : {payload.get('index_present')}")
        print(f"  => {'READY' if payload.get('ok') else 'NOT READY'}")
        if payload.get("error"):
            print(f"  error: {payload['error']}")

    elif cmd == "index":
        print(f"Index {payload['status']}: {payload.get('chunks', 0)} chunks")
        print(f"  corpus      : {payload.get('corpus')}")
        print(f"  embedder    : {payload.get('embedder')}")
        print(f"  path        : {payload.get('index_path')}")
        print(f"  build time  : {payload.get('build_time_s', 0)} s")
        print(f"  dimensions  : {payload.get('dimensions', 'cached')}")

    elif cmd == "query":
        print(f"\nQuestion: {payload['question']}")
        print(f"Strategy: {payload['strategy']}   Filters: {payload['filters']}")
        print(f"\nAnswer: {payload['answer']}")
        print(f"Citations: {', '.join(payload['citations']) or '(none)'}")
        print("\nEvidence used:")
        for ev in payload["evidence"]:
            meta = ev["metadata"]
            print(f"  [{ev['id']}] {ev['chunk_id']} | {meta.get('company')} | {meta.get('report_year')} | {meta.get('section', '')[:55]}")
        if payload.get("judge"):
            print(f"\nJudge: {payload['judge']}")
        m = payload["metrics"]
        print(
            f"\nMetrics: total={m['total_latency_s']}s  embed={m['embedding_latency_s']}s  "
            f"gen={m['generation_latency_s']}s  tokens(in/out)={m['prompt_tokens']}/{m['completion_tokens']}  "
            f"llm_calls={m['llm_calls']}"
        )
        if payload.get("trace"):
            print("Trace:")
            for step in payload["trace"]:
                print(f"  - {step['step']}: {str(step['detail'])[:90]} ({step['seconds']}s)")

    elif cmd == "extract":
        print(f"\nKPI {payload['kpi_id']} — {payload['metric']} ({payload['reference']})")
        print("Extracted value:")
        print(json.dumps(payload["value"], ensure_ascii=False, indent=2))
        if payload.get("parse_error"):
            print(f"  [warn] JSON parse failed: {payload['parse_error']}")
        print(f"Evidence chunks: {', '.join(e['chunk_id'] for e in payload['evidence'])}")
        m = payload["metrics"]
        print(f"Metrics: total={m['total_latency_s']}s  gen={m['generation_latency_s']}s")

    elif cmd in {"evaluate", "benchmark"}:
        s = payload["summary"]
        print("\nBenchmark summary")
        for key, value in s.items():
            print(f"  {key:30s}: {value}")

    elif cmd == "retrieval-bench":
        print("\nRetrieval benchmark (graded ESG query/doc pairs)")
        for key, value in payload.items():
            if key not in {"command", "source"}:
                print(f"  {key:20s}: {value}")

    elif cmd == "matrix":
        print("\nModel / technique matrix")
        for rr in payload.get("retrieval_benchmarks", []):
            print(f"  retrieval {rr['embedder']}: NDCG@10={rr['ndcg@10']}  MRR={rr['mrr']}  P@1={rr['precision@1']}")
        print()
        for r in payload.get("results", []):
            s = r["summary"]
            print(
                f"  {s['llm']:24s} | {s['embedder']:26s} | {s['strategy']:11s} | "
                f"hit={s['answer_hit_rate']:.2f}  f1={s['mean_token_f1']:.2f}  "
                f"lat={s['mean_latency_s']}s  calls={s['mean_llm_calls']}"
            )
        print(f"\nReports: {payload.get('paths', {})}")

    elif cmd == "lineage":
        if "summary" in payload:
            print(json.dumps(payload["summary"], indent=2))
        else:
            for row in payload.get("records", []):
                print(f"{row.get('ts')} | {row.get('route'):15s} | {row.get('strategy', '-'):11s} | {row.get('question', '')[:70]}")

    elif cmd == "smoke-test":
        print("\nSMOKE TEST", "PASSED" if payload["passed"] else "FAILED")
        for step in payload["steps"]:
            detail = step["detail"]
            if isinstance(detail, dict):
                detail = str(detail.get("summary") or detail.get("status") or f"{detail.get('chunks', '?')} chunks")[:110]
            print(f"  [{step['status']:4s}] {step['step']:14s} {step['seconds']:6.2f}s  {detail}")
        print("\nCriteria:")
        for key, ok in payload.get("criteria", {}).items():
            print(f"  {'OK  ' if ok else 'FAIL'} {key}")
        if payload.get("benchmark_summary"):
            print(f"\nBenchmark: {payload['benchmark_summary']}")
        if payload.get("hint"):
            print(f"\nHint: {payload['hint']}")

    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="esg-rag", description="End-to-end ESG RAG lab (Ollama, local)")
    parser.add_argument("--config", help="path to config.yaml")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--corpus", choices=["synthetic", "real", "all"], help="corpus/index to use for this run")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override any config value (repeatable)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check Ollama, models and data files")

    p_index = sub.add_parser("index", help="build an embedder-specific vector index")
    p_index.add_argument("--corpus", choices=["synthetic", "real", "all"], default="synthetic")
    p_index.add_argument("--embedder")
    p_index.add_argument("--force", action="store_true")

    p_query = sub.add_parser("query", help="answer a question with a selected RAG technique")
    p_query.add_argument("question")
    p_query.add_argument("--company")
    p_query.add_argument("--year")
    p_query.add_argument("--gri")
    p_query.add_argument("--section")
    p_query.add_argument("--filter", action="append", default=[], metavar="KEY=VALUE")
    p_query.add_argument("--mode", choices=["dense", "bm25", "hybrid"])
    p_query.add_argument("--top-k", type=int)
    p_query.add_argument("--judge", action="store_true")
    p_query.add_argument("--strategy", choices=STRATEGIES)

    p_extract = sub.add_parser("extract", help="extract a structured KPI value")
    p_extract.add_argument("kpi")
    p_extract.add_argument("--company")
    p_extract.add_argument("--year")
    p_extract.add_argument("--gri")
    p_extract.add_argument("--top-k", type=int)

    p_eval = sub.add_parser("evaluate", aliases=["benchmark"], help="run a benchmark dataset")
    p_eval.add_argument("--dataset", choices=["synthetic", "real"], default=None)
    p_eval.add_argument("--strategy", choices=STRATEGIES)
    p_eval.add_argument("--judge", action="store_true")
    p_eval.add_argument("--limit", type=int)

    p_rb = sub.add_parser("retrieval-bench", help="graded retrieval benchmark per embedder")
    p_rb.add_argument("--embedder")
    p_rb.add_argument("--limit", type=int)

    p_matrix = sub.add_parser("matrix", help="2 LLMs x 2 embedders x techniques")
    p_matrix.add_argument("--llms", nargs="+")
    p_matrix.add_argument("--embedders", nargs="+")
    p_matrix.add_argument("--strategies", nargs="+", choices=STRATEGIES)
    p_matrix.add_argument("--dataset", choices=["synthetic", "real"])
    p_matrix.add_argument("--limit", type=int)
    p_matrix.add_argument("--judge", action="store_true")
    p_matrix.add_argument("--no-judge", action="store_true")

    p_lineage = sub.add_parser("lineage", help="inspect the append-only lineage log")
    p_lineage.add_argument("--last", type=int, default=5)
    p_lineage.add_argument("--summary", action="store_true")

    sub.add_parser("smoke-test", help="doctor + index + RAG + extraction + benchmark")
    return parser


def _filters(args: argparse.Namespace) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    for key in ("company", "year", "gri", "section"):
        value = getattr(args, key, None)
        if value:
            filters[key] = value
    for pair in getattr(args, "filter", []) or []:
        if "=" not in pair:
            raise SystemExit(f"Invalid filter (expected KEY=VALUE): {pair}")
        key, value = pair.split("=", 1)
        filters[key.strip()] = value.strip()
    return filters


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = Config(load_config(args.config))
    for override in args.set:
        if "=" not in override:
            raise SystemExit(f"Invalid --set (expected SECTION.KEY=VALUE): {override}")
        key, value = override.split("=", 1)
        config.override(key.strip(), parse_value(value.strip()))
    pipeline = ESGRAGPipeline(config=config)
    if args.corpus:
        pipeline.use_corpus(args.corpus)

    try:
        if args.command == "doctor":
            payload = {"command": "doctor", **pipeline.doctor()}
        elif args.command == "index":
            if args.embedder:
                pipeline.set_models(embedder=args.embedder)
            payload = {"command": "index", **pipeline.index(force=args.force, corpus=args.corpus, embedder=args.embedder)}
        elif args.command == "query":
            result = pipeline.query(args.question, filters=_filters(args), top_k=args.top_k, mode=args.mode, judge=args.judge, strategy=args.strategy)
            payload = {"command": "query", **result}
        elif args.command == "extract":
            result = pipeline.extract(args.kpi, filters=_filters(args), top_k=args.top_k)
            payload = {"command": "extract", **result}
        elif args.command in {"evaluate", "benchmark"}:
            result = pipeline.benchmark(dataset=args.dataset, strategy=args.strategy, limit=args.limit, judge=args.judge or None)
            payload = {"command": "evaluate", "summary": result["summary"], "rows": result["rows"]}
        elif args.command == "retrieval-bench":
            payload = {"command": "retrieval-bench", **pipeline.retrieval_benchmark(embedder=args.embedder, limit=args.limit)}
        elif args.command == "matrix":
            judge = None if not (args.judge or args.no_judge) else args.judge
            result = pipeline.matrix_run(
                llms=args.llms, embedders=args.embedders, strategies=args.strategies,
                dataset=args.dataset, limit=args.limit, judge=judge,
            )
            payload = {"command": "matrix", **result}
        elif args.command == "lineage":
            payload = (
                {"command": "lineage", "summary": pipeline.lineage.summary()}
                if args.summary
                else {"command": "lineage", "records": pipeline.lineage.recent(args.last)}
            )
        elif args.command == "smoke-test":
            payload = {"command": "smoke-test", **pipeline.smoke_test()}
        else:  # pragma: no cover
            parser.error(f"Unknown command: {args.command}")
    except (RuntimeError, FileNotFoundError, ValueError, KeyError) as exc:
        payload = {"command": args.command, "status": "error", "error": str(exc)}

    _dump(payload, args.json)
    if payload.get("status") == "error":
        return 1
    if args.command == "doctor" and not payload.get("ok"):
        return 1
    if args.command == "smoke-test" and not payload.get("passed"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
