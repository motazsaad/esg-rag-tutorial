"""The end-to-end orchestrator.

    corpus (synthetic or real published data)
      -> chunk(strategy) -> embed(selected model) -> index
      -> RAG technique (naive / hybrid / filtered / multi_query / rerank /
                        self_rag / decompose)
      -> cited answer or structured KPI
      -> lineage + latency/token/quality metrics
      -> 2 LLMs x 2 embedders x N techniques matrix results

Every public method returns plain dictionaries so the CLI, REST service, smoke
test and matrix runner all exercise identical code.
"""

from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from typing import Any

from . import __version__
from .chunking import chunk_documents, load_documents
from .config import DEFAULTS, PROJECT_DIR, Config, load_config
from .evaluate import answer_hit as synthetic_answer_hit, retrieval_hit as synthetic_retrieval_hit, load_gold
from .extraction import extract_kpi, find_kpi, load_kpi_questions
from .generation import Generator
from .indexing import VectorStore, dot, normalize
from .lineage import LineageStore
from .ollama import OllamaClient
from .retrieval import Retriever
from .strategies import StrategyContext, run_strategy
from .system_metrics import PowerMeter, collect as collect_system_metrics


def _mean(rows: list[dict[str, Any]], key: str, n: int, digits: int) -> float | None:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return round(sum(values) / len(values), digits) if values else None


class ESGRAGPipeline:
    def __init__(self, config: Config | None = None, *, config_path: str | Path | None = None):
        self.config = config if config is not None else Config(load_config(config_path))
        self.client = OllamaClient(self.config.base_url, self.config.data["ollama"]["request_timeout"])
        self.lineage = LineageStore(self.config.path("lineage"))
        self.active_corpus = "synthetic"
        self._stores: dict[tuple[str, str], VectorStore] = {}
        self.store: VectorStore | None = None
        self._retriever: Retriever | None = None
        self._generator: Generator | None = None
        self._load_active_store()

    # ------------------------------------------------------------------ models
    def set_models(
        self,
        *,
        generator: str | None = None,
        embedder: str | None = None,
        judge: str | None = None,
    ) -> None:
        """Switch active models at runtime (used by the matrix runner)."""
        if generator:
            self.config.data["models"]["generator"] = generator
            self._generator = None
        if embedder and embedder != self.config.embedder:
            self.config.data["models"]["embedder"] = embedder
            self._retriever = None
            self._load_active_store()
        if judge:
            self.config.data["models"]["judge"] = judge
        if generator or embedder or judge:
            self._generator = None

    def use_corpus(self, corpus: str) -> None:
        """Select 'synthetic', 'real' or 'all' (both corpora merged)."""
        if corpus not in {"synthetic", "real", "all"}:
            raise ValueError(f"Unknown corpus: {corpus}")
        self.active_corpus = corpus
        self._load_active_store()

    def index_path(self, embedder: str | None = None, corpus: str | None = None) -> Path:
        emb = embedder or self.config.embedder
        corp = corpus or self.active_corpus
        if corp == "synthetic" and emb == DEFAULTS["models"]["embedder"]:
            return self.config.path("index")
        slug = self.config.model_slug(emb)
        return PROJECT_DIR / "output" / f"index_{corp}_{slug}.json"

    def _load_active_store(self) -> None:
        path = self.index_path()
        self.store = VectorStore.from_file_if_exists(path)
        if self.store is not None:
            self._stores[(self.active_corpus, self.config.embedder)] = self.store
        else:
            self.store = None

    # -------------------------------------------------------------- stage 1-3
    def doctor(self) -> dict[str, Any]:
        required = set(self.config.data["models"]["llms"] + self.config.data["models"]["embedders"])
        required.update([self.config.generator, self.config.embedder, self.config.judge])
        try:
            model_status = self.client.check_models(required)
            ollama_ok, error = True, None
        except Exception as exc:
            model_status = {m: False for m in required}
            ollama_ok, error = False, str(exc)
        corpora = {
            name: sorted(
                p.name
                for p in self.config.path("corpus_dir" if name == "synthetic" else "real_corpus_dir").glob("*")
                if p.suffix in {".txt", ".md"}
            )
            for name in ("synthetic", "real")
        }
        checks = {
            "ollama_reachable": ollama_ok,
            "models_available": model_status,
            "corpora": corpora,
            "index_present": self.store is not None,
            "kpi_questions_present": self.config.path("kpi_questions").exists(),
            "real_qa_present": self.config.path("real_qa").exists(),
            "gold_set_present": self.config.path("gold_set").exists(),
        }
        checks["ok"] = (
            ollama_ok
            and model_status.get(self.config.generator, False)
            and model_status.get(self.config.embedder, False)
            and bool(corpora["synthetic"] or corpora["real"])
        )
        if error:
            checks["error"] = error
        return checks

    def corpus_dirs(self, corpus: str | None = None) -> list[Path]:
        corp = corpus or self.active_corpus
        dirs = []
        if corp in {"synthetic", "all"}:
            dirs.append(self.config.path("corpus_dir"))
        if corp in {"real", "all"}:
            dirs.append(self.config.path("real_corpus_dir"))
        return dirs

    def index(self, *, force: bool = False, corpus: str | None = None, embedder: str | None = None) -> dict[str, Any]:
        corp = corpus or self.active_corpus
        emb = embedder or self.config.embedder
        key = (corp, emb)
        target = self.index_path(emb, corp)
        if target.exists() and not force:
            if key not in self._stores:
                self._stores[key] = VectorStore.load(target)
            self.store = self._stores[key]
            self.active_corpus, self.config.data["models"]["embedder"] = corp, emb
            self._retriever, self._generator = None, None
            return {"status": "cached", "chunks": len(self._stores[key]), "corpus": corp, "embedder": emb, "index_path": str(target)}
        docs: list[dict[str, Any]] = []
        for directory in self.corpus_dirs(corp):
            docs.extend(load_documents(directory))
        chunks = chunk_documents(docs, self.config.data["chunking"])
        if not chunks:
            raise ValueError("Chunking produced zero chunks — check the corpus files")
        store = VectorStore()
        build = store.build(
            chunks,
            embed_fn=lambda texts: self.client.embed(texts, emb),
        )
        store.save(target)
        self._stores[key] = store
        self.active_corpus, self.config.data["models"]["embedder"] = corp, emb
        self.store, self._retriever, self._generator = store, None, None
        return {
            "status": "built",
            **build,
            "corpus": corp,
            "embedder": emb,
            "documents": len(docs),
            "chunks": len(store),
            "index_path": str(target),
        }

    # --------------------------------------------------------------- accessors
    @property
    def retriever(self) -> Retriever:
        if self.store is None:
            raise RuntimeError("No index loaded — run the 'index' command first")
        if self._retriever is None:
            self._retriever = Retriever(self.store, self.client, self.config.embedder, self.config.data["retrieval"])
        return self._retriever

    @property
    def generator(self) -> Generator:
        if self._generator is None:
            self._generator = Generator(self.client, self.config.generator, self.config.data["generation"])
        return self._generator

    def _context(self, *, judge: bool = False) -> StrategyContext:
        return StrategyContext(
            retriever=self.retriever,
            generator=self.generator,
            client=self.client,
            retrieval_cfg=self.config.data["retrieval"],
            strategy_cfg=self.config.data["strategy"],
            generation_cfg=self.config.data["generation"],
            models={
                "generator": self.config.generator,
                "judge": self.config.judge,
                "rewrite": self.config.data["models"].get("rewrite", self.config.generator),
                "rerank": self.config.data["models"].get("rerank", self.config.generator),
            },
            judge=judge,
        )

    # ------------------------------------------------------------------ query
    def query(
        self,
        question: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        mode: str | None = None,
        judge: bool = False,
        strategy: str | None = None,
        esg_dimension: str | None = None,
    ) -> dict[str, Any]:
        if mode:  # explicit mode override for backward compatibility
            self.config.data["retrieval"]["mode"] = mode

        sys_cfg = self.config.data["system_metrics"]
        meter = None
        if sys_cfg.get("enabled", True) and sys_cfg.get("power_source", "auto") != "none":
            meter = PowerMeter(sys_cfg.get("power_source", "auto"), sys_cfg.get("estimated_watts", 65.0))
            meter.start()

        result = run_strategy(
            self._context(judge=judge),
            strategy or self.config.strategy_name,
            question,
            filters=filters,
            top_k=top_k,
        )
        power = meter.stop() if meter else {}
        system = collect_system_metrics(
            result["llm_calls"],
            power=power,
            client=self.client if sys_cfg.get("enabled", True) else None,
            models=[self.config.generator, self.config.embedder],
        )
        calls = result["llm_calls"]
        metrics = {
            "total_latency_s": result["total_latency_s"],
            "embedding_latency_s": result["retrieval_calls"] and next(
                (c["latency_s"] for c in calls if c.get("model") == self.config.embedder), 0.0
            ) or 0.0,
            "generation_latency_s": next((c["latency_s"] for c in calls if c.get("model") == self.config.generator), 0.0),
            "prompt_tokens": sum(int(c.get("prompt_tokens", 0)) for c in calls),
            "completion_tokens": sum(int(c.get("completion_tokens", 0)) for c in calls),
            "llm_calls": len(calls),
            "retrieval_calls": result["retrieval_calls"],
            "passages_retrieved": len(result["passages"]),
            # system-level metrics
            "tokens_per_second": system.get("tokens_per_second"),
            "ms_per_token": system.get("ms_per_token"),
            "process_rss_mb": system.get("process_rss_mb"),
            "model_memory_mb": system.get("ollama_memory", {}).get("total_model_mb"),
            "vram_mb": system.get("ollama_memory", {}).get("total_vram_mb"),
            "energy_j": power.get("joules"),
            "avg_watts": power.get("avg_watts"),
            "power_measured": power.get("measured"),
            "power_source": power.get("source"),
        }
        out = {
            "status": "ok",
            "question": question,
            "answer": result["answer"],
            "citations": result["citations"],
            "evidence": result["evidence"],
            "strategy": result["strategy"],
            "filters": result["filters"],
            "trace": result["trace"],
            "judge": result.get("judge"),
            "system": system,
            "metrics": metrics,
        }
        self.lineage.record(
            {
                "route": "rag_answer",
                "question": question,
                "esg_dimension": esg_dimension,
                "strategy": result["strategy"],
                "filters": result["filters"],
                "config": {
                    "version": __version__,
                    "generator": self.config.generator,
                    "embedder": self.config.embedder,
                    "corpus": self.active_corpus,
                    "chunking": self.config.data["chunking"],
                    "retrieval": self.config.data["retrieval"],
                    "corpus_fingerprint": self.store.fingerprint() if self.store else None,
                },
                "retrieval": {
                    "mode": result.get("mode"),
                    "passages": [
                        {"chunk_id": p["chunk_id"], "score": p.get("score"), "metadata": p["metadata"]}
                        for p in result["passages"]
                    ],
                },
                "answer": result["answer"],
                "citations": result["citations"],
                "judge": result.get("judge"),
                "trace": result["trace"],
                "system": system,
                "metrics": metrics,
            }
        )
        return out

    # ------------------------------------------------------------- extraction
    def extract(self, kpi_identifier: str, *, filters: dict[str, Any] | None = None, top_k: int | None = None) -> dict[str, Any]:
        questions = load_kpi_questions(self.config.path("kpi_questions"))
        kpi = find_kpi(questions, kpi_identifier)
        t0 = time.perf_counter()
        result = extract_kpi(
            self.retriever,
            self.client,
            kpi,
            model=self.config.generator,
            top_k=top_k,
            filters=filters,
            max_evidence_chars=int(self.config.data["retrieval"].get("max_evidence_chars", 900)),
        )
        result["metrics"] = {
            "total_latency_s": round(time.perf_counter() - t0, 4),
            "generation_latency_s": result["generation"]["latency_s"],
            "completion_tokens": result["generation"]["completion_tokens"],
        }
        self.lineage.record(
            {
                "route": "kpi_extraction",
                "question": kpi["question"],
                "filters": filters or {},
                "config": {
                    "version": __version__,
                    "generator": self.config.generator,
                    "embedder": self.config.embedder,
                    "corpus": self.active_corpus,
                },
                "kpi_id": result["kpi_id"],
                "target_table": result["target_table"],
                "retrieval": result["retrieval"],
                "value": result["value"],
                "evidence": result["evidence"],
                "metrics": result["metrics"],
            }
        )
        return {"status": "ok", **result}

    # --------------------------------------------------------------- benchmark
    def _benchmark_items(self, dataset: str, limit: int | None) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        if dataset == "synthetic":
            for item in load_gold(self.config.path("gold_set")):
                items.append(
                    {
                        "q_id": item.get("q_id", item["question"][:20]),
                        "question": item["question"],
                        "esg_dimension": item.get("esg_dimension"),
                        "reference": " ".join(item.get("expected_substrings", [])),
                        "source_doc": None,
                        "expected_sections": item.get("expected_sections", []),
                        "filters": {k: v for k, v in {"company": item.get("company"), "year": item.get("year")}.items() if v},
                    }
                )
        elif dataset == "real":
            payload = json.loads(self.config.path("real_qa").read_text(encoding="utf-8"))
            for item in payload.get("questions", payload):
                source = item.get("document_source", "")
                items.append(
                    {
                        "q_id": item.get("q_id"),
                        "question": item["question"],
                        "esg_dimension": item.get("esg_dimension"),
                        "reference": item["reference"],
                        "source_doc": Path(source).name,
                        "expected_sections": [],
                        "filters": {},
                    }
                )
        else:
            raise ValueError(f"Unknown benchmark dataset: {dataset}")
        return items[:limit] if limit else items

    @staticmethod
    def _answer_score(answer: str, reference: str, metric: str) -> tuple[bool, float, bool]:
        """Return (hit, token_f1, numeric_match) against a reference answer."""
        norm = lambda s: re.sub(r"\s+", " ", s.replace("\u00a0", " ")).lower()
        a, r = norm(answer), norm(reference)
        nums = [re.sub(r"[.,]+$", "", n) for n in re.findall(r"\d[\d.,]*", r)]
        numeric = all(n in a for n in nums) if nums else True
        stop = {"the", "a", "an", "of", "in", "was", "were", "is", "are", "to", "and", "for", "by", "from", "that", "with", "as", "at"}
        ta = [w for w in re.findall(r"\w+", a) if w not in stop]
        tr = [w for w in re.findall(r"\w+", r) if w not in stop]
        overlap = len(set(ta) & set(tr))
        f1 = 2 * overlap / (len(ta) + len(tr)) if ta and tr else 0.0
        if metric == "token_f1":
            return f1 >= 0.5, f1, numeric
        if metric == "judge":
            return numeric, f1, numeric  # judge handled separately by caller
        return numeric and f1 >= 0.35, f1, numeric

    def benchmark(
        self,
        *,
        dataset: str | None = None,
        strategy: str | None = None,
        limit: int | None = None,
        judge: bool | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        cfg = self.config.data["benchmark"]
        dataset = dataset or cfg.get("dataset", "synthetic")
        strategy = strategy or self.config.strategy_name
        limit = limit if limit is not None else cfg.get("limit")
        judge = cfg.get("judge", False) if judge is None else judge
        metric = cfg.get("answer_metric", "substring")
        wanted_corpus = "real" if dataset == "real" else "synthetic"
        if self.active_corpus != wanted_corpus and self.active_corpus != "all":
            self.use_corpus(wanted_corpus)
        items = self._benchmark_items(dataset, limit)
        rows: list[dict[str, Any]] = []
        for item in items:
            q = self.query(
                item["question"],
                filters=item["filters"],
                strategy=strategy,
                judge=judge,
                esg_dimension=item.get("esg_dimension"),
            )
            passages = [
                {
                    "chunk_id": e["chunk_id"],
                    "metadata": e["metadata"],
                }
                for e in q["evidence"]
            ]
            if item["source_doc"]:
                r_hit = any(item["source_doc"] == p["metadata"].get("source_file") for p in passages)
            else:
                r_hit = synthetic_retrieval_hit(passages, {"expected_sections": item["expected_sections"], **item["filters"]})
            hit, f1, numeric = self._answer_score(q["answer"], item["reference"], metric)
            row = {
                "q_id": item["q_id"],
                "question": item["question"],
                "esg_dimension": item.get("esg_dimension"),
                "reference": item["reference"],
                "answer": q["answer"],
                "retrieval_hit": r_hit,
                "answer_hit": hit,
                "token_f1": round(f1, 3),
                "numeric_match": numeric,
                "citations": q["citations"],
                "latency_s": q["metrics"]["total_latency_s"],
                "completion_tokens": q["metrics"]["completion_tokens"],
                "llm_calls": q["metrics"]["llm_calls"],
                "tokens_per_second": q["metrics"]["tokens_per_second"],
                "process_rss_mb": q["metrics"]["process_rss_mb"],
                "model_memory_mb": q["metrics"]["model_memory_mb"],
                "energy_j": q["metrics"]["energy_j"],
                "avg_watts": q["metrics"]["avg_watts"],
                "power_measured": q["metrics"]["power_measured"],
                "passages": [p["chunk_id"] for p in passages],
                "judge": q.get("judge"),
            }
            rows.append(row)
            if verbose:
                print(f"  [{'OK ' if hit else 'MISS'}] {item['question'][:75]}")
                print(f"         answer: {q['answer'][:150]}")
        n = len(rows) or 1
        summary = {
            "dataset": dataset,
            "strategy": strategy,
            "llm": self.config.generator,
            "embedder": self.config.embedder,
            "corpus": self.active_corpus,
            "questions": len(rows),
            "retrieval_hit@k": round(sum(r["retrieval_hit"] for r in rows) / n, 3),
            "answer_hit_rate": round(sum(r["answer_hit"] for r in rows) / n, 3),
            "mean_token_f1": round(sum(r["token_f1"] for r in rows) / n, 3),
            "numeric_match_rate": round(sum(r["numeric_match"] for r in rows) / n, 3),
            "mean_latency_s": round(sum(r["latency_s"] for r in rows) / n, 3),
            "mean_completion_tokens": round(sum(r["completion_tokens"] for r in rows) / n, 1),
            "mean_llm_calls": round(sum(r["llm_calls"] for r in rows) / n, 2),
            "mean_tokens_per_second": _mean(rows, "tokens_per_second", n, 2),
            "mean_process_rss_mb": _mean(rows, "process_rss_mb", n, 1),
            "model_memory_mb": _mean(rows, "model_memory_mb", n, 1),
            "mean_energy_j": _mean(rows, "energy_j", n, 2),
            "mean_avg_watts": _mean(rows, "avg_watts", n, 2),
            "power_measured": any(r["power_measured"] for r in rows),
        }
        if judge:
            supported = [r for r in rows if r["judge"] and r["judge"].get("supported") is not None]
            summary["judge_supported_rate"] = (
                round(sum(bool(r["judge"]["supported"]) for r in supported) / len(supported), 3) if supported else None
            )
        return {"summary": summary, "rows": rows}

    # ------------------------------------------------- retrieval-only benchmark
    def retrieval_benchmark(self, *, embedder: str | None = None, limit: int | None = None) -> dict[str, Any]:
        """NDCG@10 / MRR / P@1 on the graded ESG retrieval benchmark sample."""
        source = json.loads(self.config.path("real_retrieval_bench").read_text(encoding="utf-8"))
        records = source.get("records", source)[:limit] if limit else source.get("records", source)
        emb = embedder or self.config.embedder
        ndcg, mrr, p1 = [], [], []
        for rec in records:
            texts = [rec["query"], *(d["text"] for d in rec["documents"])]
            vectors, _ = self.client.embed(texts, emb)
            q = normalize(vectors[0])
            scores = [dot(q, normalize(v)) for v in vectors[1:]]
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            gains = [rec["documents"][i]["relevance"] for i in order]
            dcg = sum(g / __import__("math").log2(rank + 2) for rank, g in enumerate(gains))
            ideal = sorted((d["relevance"] for d in rec["documents"]), reverse=True)
            idcg = sum(g / __import__("math").log2(rank + 2) for rank, g in enumerate(ideal)) or 1.0
            ndcg.append(dcg / idcg)
            first_rel = next((i + 1 for i, g in enumerate(gains) if g > 1), None)
            mrr.append(1.0 / first_rel if first_rel else 0.0)
            p1.append(1.0 if gains[0] > 1 else 0.0)
        n = len(records) or 1
        return {
            "embedder": emb,
            "queries": len(records),
            "ndcg@10": round(sum(ndcg) / n, 4),
            "mrr": round(sum(mrr) / n, 4),
            "precision@1": round(sum(p1) / n, 4),
            "source": source.get("source", {}),
        }

    # -------------------------------------------------------------- model matrix
    def matrix_run(
        self,
        *,
        llms: list[str] | None = None,
        embedders: list[str] | None = None,
        strategies: list[str] | None = None,
        dataset: str | None = None,
        limit: int | None = None,
        judge: bool | None = None,
    ) -> dict[str, Any]:
        cfg = self.config.data["matrix"]
        llms = llms or cfg["llms"]
        embedders = embedders or cfg["embedders"]
        strategies = strategies or cfg["strategies"]
        dataset = dataset or cfg["dataset"]
        limit = limit if limit is not None else cfg["limit"]
        judge = cfg["judge"] if judge is None else judge

        available = set(self.client.list_models())
        if cfg.get("skip_missing_models", True):
            llms = [m for m in llms if m in available]
            embedders = [m for m in embedders if m in available]
        missing = [m for m in (llms + embedders) if m not in available]
        if not llms or not embedders:
            raise RuntimeError(f"No usable models for matrix; available: {sorted(available)}")

        corpus = "real" if dataset == "real" else "synthetic"
        results: list[dict[str, Any]] = []
        retrieval_results: list[dict[str, Any]] = []

        def persist() -> tuple[dict[str, Any], dict[str, str]]:
            """Write partial/final results so the dashboard can follow progress."""
            payload = {
                "config": {
                    "llms": llms, "embedders": embedders, "strategies": strategies,
                    "dataset": dataset, "limit": limit, "judge": judge,
                    "chunking": self.config.data["chunking"],
                    "retrieval": self.config.data["retrieval"],
                    "generation": self.config.data["generation"],
                },
                "missing_models": missing,
                "retrieval_benchmarks": retrieval_results,
                "results": results,
                "completed_cells": len(results),
                "expected_cells": len(llms) * len(embedders) * len(strategies),
            }
            return payload, self._write_matrix_outputs(payload)
        for emb in embedders:
            print(f"\n=== embedder {emb} ===")
            built = self.index(force=False, corpus=corpus, embedder=emb)
            print(f"  index: {built['status']} ({built['chunks']} chunks)")
            rr = self.retrieval_benchmark(embedder=emb, limit=20)
            retrieval_results.append(rr)
            persist()
            print(f"  retrieval bench: NDCG@10={rr['ndcg@10']}  MRR={rr['mrr']}")
            for llm in llms:
                self.set_models(generator=llm)
                for strategy in strategies:
                    print(f"  -> {llm} / {strategy}")
                    bench = self.benchmark(dataset=dataset, strategy=strategy, limit=limit, judge=judge, verbose=False)
                    results.append(
                        {
                            "llm": llm,
                            "embedder": emb,
                            "strategy": strategy,
                            "summary": bench["summary"],
                            "rows": bench["rows"],
                        }
                    )
                    persist()
        payload, paths = persist()
        return {**payload, "paths": paths}

    def _write_matrix_outputs(self, payload: dict[str, Any]) -> dict[str, str]:
        out_json = self.config.path("matrix_output")
        out_csv = self.config.path("matrix_csv")
        out_md = self.config.path("matrix_report")
        for path in (out_json, out_csv, out_md):
            path.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._write_matrix_csv(payload["results"], out_csv)
        out_md.write_text(self._matrix_markdown(payload), encoding="utf-8")
        return {"json": str(out_json), "csv": str(out_csv), "markdown": str(out_md)}

    @staticmethod
    def _write_matrix_csv(results: list[dict[str, Any]], path: Path) -> None:
        fields = [
            "llm", "embedder", "strategy", "questions", "retrieval_hit@k", "answer_hit_rate",
            "mean_token_f1", "numeric_match_rate", "mean_latency_s", "mean_completion_tokens",
            "mean_llm_calls", "judge_supported_rate",
            "mean_tokens_per_second", "mean_process_rss_mb", "model_memory_mb",
            "mean_energy_j", "mean_avg_watts", "power_measured",
        ]
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for r in results:
                writer.writerow({f: r["summary"].get(f, "") for f in fields})

    @staticmethod
    def _matrix_markdown(payload: dict[str, Any]) -> str:
        lines = [
            "# ESG RAG model / technique matrix",
            "",
            f"Dataset: **{payload['config']['dataset']}** · questions per cell: **{payload['config']['limit']}** · LLM judge: **{payload['config']['judge']}**",
            "",
            "## Generation results",
            "",
            "| LLM | Embedder | Strategy | Ret.hit@k | Answer hit | Token F1 | Latency (s) | Tok/s | RSS MB | Model MB | J/query | W avg | Judge |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in payload["results"]:
            s = r["summary"]
            cell = lambda v, d=2: "n/a" if v is None else f"{float(v):.{d}f}"
            lines.append(
                f"| {s['llm']} | {s['embedder']} | {s['strategy']} | {s['retrieval_hit@k']} | "
                f"{s['answer_hit_rate']} | {s['mean_token_f1']} | {s['mean_latency_s']} | "
                f"{cell(s.get('mean_tokens_per_second'))} | {cell(s.get('mean_process_rss_mb'),1)} | "
                f"{cell(s.get('model_memory_mb'),1)} | {cell(s.get('mean_energy_j'))} | "
                f"{cell(s.get('mean_avg_watts'))} | {s.get('judge_supported_rate', 'n/a')} |"
            )
        lines += ["", "## Retrieval-only benchmark (graded ESG query/doc pairs)", "", "| Embedder | NDCG@10 | MRR | P@1 |", "|---|---:|---:|---:|"]
        for rr in payload["retrieval_benchmarks"]:
            lines.append(f"| {rr['embedder']} | {rr['ndcg@10']} | {rr['mrr']} | {rr['precision@1']} |")
        lines += [
            "",
            "## System metrics notes",
            "",
            "Tok/s is Ollama decode speed (completion tokens / eval time). RSS MB is the",
            "lab process; Model MB is the loaded Ollama model size from `/api/ps`.",
            "Energy is measured via RAPL/NVIDIA-SMI where the sensor is readable;",
            "otherwise it is an **estimate** (`estimated_watts × wall time`) and",
            "`power_measured` is false. Do not compare estimated joules across machines.",
        ]
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------- smoke test
    def smoke_test(self) -> dict[str, Any]:
        """Doctor -> index -> RAG answers -> extraction -> benchmark."""
        steps: list[dict[str, Any]] = []

        def step(name: str, fn):
            t0 = time.perf_counter()
            try:
                payload = fn()
                steps.append({"step": name, "status": "PASS", "seconds": round(time.perf_counter() - t0, 3), "detail": payload})
                return payload
            except Exception as exc:
                steps.append({"step": name, "status": "FAIL", "seconds": round(time.perf_counter() - t0, 3), "detail": str(exc)})
                return None

        health = step("doctor", self.doctor)
        if not health or not health.get("ok"):
            return {"passed": False, "steps": steps, "hint": "Start Ollama and pull the configured models (see README)."}
        built = step("index", lambda: self.index(force=True, corpus="synthetic"))
        q1 = step("query:en", lambda: self.query("What was the total number of employees at Nordwind in 2023?", filters={"company": "Nordwind", "year": "2023"}, top_k=3))
        q2 = step("query:it", lambda: self.query("Quali sono le emissioni Scope 1 di Solaria nel 2023?", filters={"company": "Solaria", "year": "2023"}, top_k=3))
        extraction = step("extract:kpi", lambda: self.extract("GRI 305_305-1_32", filters={"company": "Nordwind"}, top_k=3))
        bench = step("evaluate", lambda: self.benchmark(dataset="synthetic", limit=3, judge=False, verbose=False))
        criteria = {
            "doctor_ok": bool(health.get("ok")),
            "indexed_chunks": bool(built and built.get("chunks", 0) >= 5),
            "answers_have_citations": all(r and r.get("citations") for r in (q1, q2)),
            "extraction_returned_value": bool(extraction and extraction.get("value") is not None),
            "benchmark_ran": bool(bench and bench["summary"]["questions"] == 3),
        }
        return {
            "passed": all(criteria.values()),
            "criteria": criteria,
            "steps": steps,
            "benchmark_summary": bench["summary"] if bench else None,
        }
