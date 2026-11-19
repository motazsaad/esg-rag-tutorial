"""Zero-dependency web dashboard for ESG RAG matrix results.

    python3 dashboard.py --port 8080
    # open http://127.0.0.1:8080

Serves the JSON/CSV artefacts produced by `matrix` and, while a run is still
in progress, derives live per-cell progress from the append-only lineage log.
No Flask/React/CDN needed — stdlib HTTP server + vanilla HTML/JS.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from esg_rag.config import Config, load_config, set_value
from esg_rag.ollama import OllamaClient
from esg_rag.pipeline import ESGRAGPipeline
from esg_rag.strategies import STRATEGIES

INDEX = Path(__file__).resolve().parent / "dashboard" / "index.html"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _lineage_rows(path: Path, last: int = 400) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines[-last:]:
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _qa_index(config: Config) -> dict[str, dict[str, Any]]:
    """Question -> reference answer / source document, for live scoring."""
    path = config.path("real_qa")
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    questions = payload.get("questions", payload) if isinstance(payload, dict) else payload
    return {q["question"]: q for q in questions if isinstance(q, dict) and "question" in q}


def _live_score(answer: str, reference: str) -> tuple[float, bool, bool]:
    """Cheap proxy metrics matching the matrix runner's answer scoring."""
    a = re.sub(r"\s+", " ", answer.replace("\u00a0", " ")).lower()
    r = re.sub(r"\s+", " ", reference.replace("\u00a0", " ")).lower()
    nums = [re.sub(r"[.,]+$", "", n) for n in re.findall(r"\d[\d.,]*", r)]
    numeric = all(n in a for n in nums) if nums else True
    stop = {"the", "a", "an", "of", "in", "was", "were", "is", "are", "to", "and", "for", "by", "from", "that", "with", "as", "at"}
    ta = [w for w in re.findall(r"\w+", a) if w not in stop]
    tr = [w for w in re.findall(r"\w+", r) if w not in stop]
    overlap = len(set(ta) & set(tr))
    f1 = 2 * overlap / (len(ta) + len(tr)) if ta and tr else 0.0
    return round(f1, 3), numeric, numeric and f1 >= 0.35


def _progress(rows: list[dict[str, Any]], qa: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Derive real, live per-cell metrics from the append-only lineage log.

    This works even for matrix runs that started before incremental result
    files were implemented: lineage is written per query, so partial progress
    (including quality metrics) is always available.
    """
    qa = qa or {}
    cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        cfg = row.get("config", {})
        if row.get("route") != "rag_answer" or cfg.get("corpus") != "real":
            continue
        key = (cfg.get("generator", "?"), cfg.get("embedder", "?"), row.get("strategy", "?"))
        cell = cells.setdefault(
            key,
            {
                "llm": key[0], "embedder": key[1], "strategy": key[2],
                "n": 0, "ans_hits": 0, "ret_hits": 0, "numeric": 0, "f1": 0.0,
                "latency": 0.0, "tokens": 0, "calls": 0, "judge_seen": 0, "judge_ok": 0,
                "tps": 0.0, "rss": 0.0, "model_mb": 0.0, "energy": 0.0, "watts": 0.0,
                "sys_n": 0, "power_measured": False, "rows": [],
            },
        )
        cell["n"] += 1
        question, answer = row.get("question", ""), row.get("answer", "")
        reference = qa.get(question, {}).get("reference", "")
        source = Path(str(qa.get(question, {}).get("document_source", ""))).name
        f1, numeric, ans_hit = _live_score(answer, reference)
        passages = row.get("retrieval", {}).get("passages", [])
        ret_hit = bool(source) and any(
            p.get("metadata", {}).get("source_file") == source for p in passages
        )
        metrics = row.get("metrics", {})
        cell["ans_hits"] += int(ans_hit)
        cell["ret_hits"] += int(ret_hit)
        cell["numeric"] += int(numeric)
        cell["f1"] += f1
        cell["latency"] += float(metrics.get("total_latency_s", 0))
        cell["tokens"] += int(metrics.get("completion_tokens", 0))
        cell["calls"] += int(metrics.get("llm_calls", 1))
        judge = row.get("judge")
        if judge and judge.get("supported") is not None:
            cell["judge_seen"] += 1
            cell["judge_ok"] += int(bool(judge["supported"]))
        if metrics.get("tokens_per_second") is not None:
            cell["tps"] += float(metrics["tokens_per_second"])
            cell["rss"] += float(metrics.get("process_rss_mb") or 0)
            cell["model_mb"] += float(metrics.get("model_memory_mb") or 0)
            cell["energy"] += float(metrics.get("energy_j") or 0)
            cell["watts"] += float(metrics.get("avg_watts") or 0)
            cell["sys_n"] += 1
            cell["power_measured"] = cell["power_measured"] or bool(metrics.get("power_measured"))
        cell["rows"].append(
            {"question": question, "reference": reference, "answer": answer,
             "esg_dimension": row.get("esg_dimension"),
             "answer_hit": ans_hit, "citations": row.get("citations", [])}
        )

    out = []
    for cell in cells.values():
        n = cell["n"]
        out.append(
            {
                "llm": cell["llm"], "embedder": cell["embedder"], "strategy": cell["strategy"],
                "summary": {
                    "questions": n,
                    "answer_hit_rate": round(cell["ans_hits"] / n, 3),
                    "retrieval_hit@k": round(cell["ret_hits"] / n, 3),
                    "numeric_match_rate": round(cell["numeric"] / n, 3),
                    "mean_token_f1": round(cell["f1"] / n, 3),
                    "mean_latency_s": round(cell["latency"] / n, 3),
                    "mean_completion_tokens": round(cell["tokens"] / n, 1),
                    "mean_llm_calls": round(cell["calls"] / n, 2),
                    "judge_supported_rate": (
                        round(cell["judge_ok"] / cell["judge_seen"], 3) if cell["judge_seen"] else None
                    ),
                    "mean_tokens_per_second": round(cell["tps"] / cell["sys_n"], 2) if cell["sys_n"] else None,
                    "mean_process_rss_mb": round(cell["rss"] / cell["sys_n"], 1) if cell["sys_n"] else None,
                    "model_memory_mb": round(cell["model_mb"] / cell["sys_n"], 1) if cell["sys_n"] else None,
                    "mean_energy_j": round(cell["energy"] / cell["sys_n"], 2) if cell["sys_n"] else None,
                    "mean_avg_watts": round(cell["watts"] / cell["sys_n"], 2) if cell["sys_n"] else None,
                    "power_measured": cell["power_measured"],
                },
                "rows": cell["rows"],
            }
        )
    return {"cells": out, "queries": sum(c["n"] for c in cells.values()), "live": True}


# ---------------------------------------------------------------- experiment runner
_JOBS: dict[str, dict[str, Any]] = {}
_JOB_LOCK = threading.Lock()


def _available_models(config: Config) -> list[str]:
    try:
        return OllamaClient(config.base_url).list_models()
    except Exception:
        return []


def _config_payload(config: Config) -> dict[str, Any]:
    return {
        "available_models": _available_models(config),
        "strategies": STRATEGIES,
        "datasets": ["synthetic", "real"],
        "defaults": {
            "chunking": config.data["chunking"],
            "retrieval": config.data["retrieval"],
            "generation": config.data["generation"],
            "benchmark": config.data["benchmark"],
            "matrix": config.data["matrix"],
            "system_metrics": config.data["system_metrics"],
        },
        "active": {
            "generator": config.generator,
            "embedder": config.embedder,
            "strategy": config.strategy_name,
        },
    }


def _validate_run(body: dict[str, Any], config: Config) -> dict[str, Any]:
    """Allowlist and clamp every user-supplied parameter (no arbitrary paths/commands)."""
    available = _available_models(config)
    mode = body.get("mode", "query")
    if mode not in {"query", "benchmark", "matrix"}:
        raise ValueError("mode must be query, benchmark or matrix")
    def pick_list(key: str, singular: str, default: str, pool: list[str]) -> list[str]:
        raw = body.get(key)
        if raw is None:
            raw = [body.get(singular) or default]
        if isinstance(raw, str):
            raw = [item.strip() for item in raw.split(",") if item.strip()]
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"Invalid {key} selection")
        values = [str(item) for item in raw if str(item) in pool]
        if not values:
            raise ValueError(f"Unsupported {key}: {raw}")
        return values
    llms = pick_list("llms", "llm", config.generator, available)
    embedders = pick_list("embedders", "embedder", config.embedder, available)
    strategies = pick_list("strategies", "strategy", config.strategy_name, STRATEGIES)
    llm, emb, strategy = llms[0], embedders[0], strategies[0]
    dataset = str(body.get("dataset") or "real")
    if dataset not in {"synthetic", "real"}:
        raise ValueError("Invalid dataset")
    limit = max(1, min(20, int(body.get("limit") or 1)))
    top_k = max(1, min(20, int(body.get("top_k") or config.data["retrieval"]["top_k"])))
    chunk_strategy = str(body.get("chunk_strategy") or config.data["chunking"]["strategy"])
    if chunk_strategy not in {"section", "fixed", "semantic"}:
        raise ValueError("Invalid chunking strategy")
    retrieval_mode = str(body.get("retrieval_mode") or config.data["retrieval"]["mode"])
    if retrieval_mode not in {"dense", "bm25", "hybrid"}:
        raise ValueError("Invalid retrieval mode")
    power_source = str(body.get("power_source") or config.data["system_metrics"]["power_source"])
    if power_source not in {"auto", "rapl", "nvidia", "estimate", "none"}:
        raise ValueError("Invalid power source")
    question = str(body.get("question") or "").strip()[:1000]
    if mode == "query" and not question:
        raise ValueError("A question is required for a single query")
    return {
        "mode": mode, "question": question, "llm": llm, "embedder": emb,
        "strategy": strategy,
        "llms": llms, "embedders": embedders, "strategies": strategies,
        "dataset": dataset, "limit": limit, "judge": bool(body.get("judge", False)),
        "corpus": "real" if dataset == "real" else "synthetic",
        "filters": body.get("filters") if isinstance(body.get("filters"), dict) else {},
        "overrides": {
            "models.generator": llm,
            "models.embedder": emb,
            "chunking.strategy": chunk_strategy,
            "chunking.max_words": max(40, min(400, int(body.get("max_words") or 140))),
            "chunking.overlap_words": max(0, min(100, int(body.get("overlap_words") or 25))),
            "retrieval.mode": retrieval_mode,
            "retrieval.top_k": top_k,
            "retrieval.infer_filters": bool(body.get("infer_filters", False)),
            "generation.max_tokens": max(32, min(1024, int(body.get("max_tokens") or 160))),
            "generation.temperature": max(0.0, min(1.0, float(body.get("temperature") or 0.0))),
            "system_metrics.power_source": power_source,
            "system_metrics.estimated_watts": max(1.0, min(500.0, float(body.get("estimated_watts") or 65.0))),
        },
    }


def _run_job(job_id: str, params: dict[str, Any]) -> None:
    def update(status: str, **extra: Any) -> None:
        with _JOB_LOCK:
            _JOBS[job_id].update(status=status, **extra)

    try:
        update("running", started_at=time.time())
        cfg = Config(load_config())
        for key, value in params["overrides"].items():
            set_value(cfg.data, key, value)
        pipeline = ESGRAGPipeline(config=cfg)
        pipeline.use_corpus(params["corpus"])

        mode = params["mode"]
        if mode in {"query", "benchmark"}:
            # Sweep the cartesian product of every checked model / technique.
            runs: list[dict[str, Any]] = []
            for embedder in params["embedders"]:
                pipeline.index(corpus=params["corpus"], embedder=embedder)
                for llm in params["llms"]:
                    pipeline.set_models(generator=llm, embedder=embedder)
                    for strategy in params["strategies"]:
                        if mode == "query":
                            q = pipeline.query(
                                params["question"], filters=params["filters"],
                                strategy=strategy, judge=params["judge"],
                            )
                            runs.append({
                                "llm": llm, "embedder": embedder, "strategy": strategy,
                                "answer": q["answer"], "citations": q["citations"],
                                "metrics": q["metrics"], "trace": q["trace"],
                            })
                        else:
                            bench = pipeline.benchmark(
                                dataset=params["dataset"], strategy=strategy,
                                limit=params["limit"], judge=params["judge"], verbose=False,
                            )
                            runs.append({
                                "llm": llm, "embedder": embedder, "strategy": strategy,
                                "summary": bench["summary"],
                            })
            update("done", result={"runs": runs, "combinations": len(runs)})
        else:
            matrix = pipeline.matrix_run(
                llms=params["llms"], embedders=params["embedders"],
                strategies=params["strategies"], dataset=params["dataset"],
                limit=params["limit"], judge=params["judge"],
            )
            update("done", result={
                "summary": [r["summary"] for r in matrix["results"]],
                "paths": matrix["paths"],
                "retrieval_benchmarks": matrix["retrieval_benchmarks"],
            })
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        update("error", error=f"{type(exc).__name__}: {exc}")


class Handler(BaseHTTPRequestHandler):
    config: Config

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[dashboard] {self.address_string()} - {fmt % args}")

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/", "/index.html"}:
            self._send(INDEX.read_bytes(), "text/html; charset=utf-8")
            return

        if path == "/api/config":
            self._json(_config_payload(self.config))
            return
        if path == "/api/run/status":
            job_id = parse_qs(parsed.query).get("job", [""])[0]
            with _JOB_LOCK:
                job = dict(_JOBS.get(job_id, {"status": "unknown"}))
            self._json({"job_id": job_id, **job})
            return

        matrix_path = self.config.path("matrix_output")
        lineage_path = self.config.path("lineage")
        rows = _lineage_rows(lineage_path)

        if path == "/api/matrix":
            self._json(_read_json(matrix_path, {}))
        elif path == "/api/progress":
            self._json(_progress(rows, _qa_index(self.config)))
        elif path == "/api/lineage":
            last = int(parse_qs(parsed.query).get("last", ["25"])[0])
            self._json({"records": rows[-last:]})
        elif path == "/api/status":
            matrix = _read_json(matrix_path, {})
            self._json(
                {
                    "matrix_exists": matrix_path.exists(),
                    "matrix_mtime": matrix_path.stat().st_mtime if matrix_path.exists() else None,
                    "completed_cells": matrix.get("completed_cells", len(matrix.get("results", []))),
                    "expected_cells": matrix.get("expected_cells") or (
                        len(self.config.data["matrix"]["llms"])
                        * len(self.config.data["matrix"]["embedders"])
                        * len(self.config.data["matrix"]["strategies"])
                    ),
                    "lineage_records": len(rows),
                    "csv_exists": self.config.path("matrix_csv").exists(),
                    "json_path": str(matrix_path),
                    "csv_path": str(self.config.path("matrix_csv")),
                }
            )
        else:
            self._json({"error": "not found"}, 404)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/run":
            self._json({"error": "not found"}, 404)
            return
        try:
            params = _validate_run(self._body(), self.config)
        except (ValueError, TypeError) as exc:
            self._json({"error": str(exc)}, 400)
            return
        with _JOB_LOCK:
            if any(j.get("status") == "running" for j in _JOBS.values()):
                self._json({"error": "An experiment is already running; wait for it to finish."}, 429)
                return
            job_id = f"job-{len(_JOBS) + 1}"
            _JOBS[job_id] = {
                "status": "queued", "params": params,
                "submitted_at": time.time(),
            }
        threading.Thread(target=_run_job, args=(job_id, params), daemon=True).start()
        self._json({"job_id": job_id, "status": "queued", "params": params})


def main() -> None:
    parser = argparse.ArgumentParser(description="ESG RAG results dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    Handler.config = Config()
    print(f"ESG RAG dashboard on http://{args.host}:{args.port}  (auto-refresh every 8 s)")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
