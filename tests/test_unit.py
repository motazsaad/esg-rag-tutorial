"""Fast unit tests (no Ollama calls) for parameterisation and utilities."""

from __future__ import annotations

import unittest

from esg_rag.chunking import chunk_documents, load_documents
from esg_rag.config import Config, parse_value, set_value
from esg_rag.pipeline import ESGRAGPipeline
from esg_rag.strategies import _rrf_fuse
from esg_rag.indexing import matches_filters


class UnitTest(unittest.TestCase):
    def test_config_overrides(self) -> None:
        cfg = Config()
        set_value(cfg.data, "chunking.strategy", "semantic")
        set_value(cfg.data, "retrieval.top_k", 9)
        cfg.override("generation.temperature", 0.5)
        self.assertEqual(cfg.data["chunking"]["strategy"], "semantic")
        self.assertEqual(cfg.data["retrieval"]["top_k"], 9)
        self.assertEqual(cfg.data["generation"]["temperature"], 0.5)
        self.assertEqual(parse_value("8"), 8)
        self.assertEqual(parse_value("true"), True)
        self.assertEqual(parse_value("qwen3:4b-instruct"), "qwen3:4b-instruct")

    def test_chunking_strategies(self) -> None:
        docs = load_documents("data/corpus")
        base = {"max_words": 140, "overlap_words": 25, "min_words": 30}
        section = chunk_documents(docs, {**base, "strategy": "section"})
        fixed = chunk_documents(docs, {**base, "strategy": "fixed"})
        semantic = chunk_documents(docs, {**base, "strategy": "semantic", "semantic_threshold": 0.15})
        self.assertGreaterEqual(len(section), 5)
        self.assertGreaterEqual(len(fixed), 1)
        self.assertGreaterEqual(len(semantic), 1)
        self.assertTrue(section[0]["metadata"]["company"])
        self.assertEqual(section[0]["metadata"]["language"], "en")

    def test_real_data_metadata(self) -> None:
        docs = load_documents("data/real_corpus")
        self.assertEqual(len(docs), 6)
        self.assertTrue(all(d["metadata"].get("source_url") for d in docs))
        self.assertTrue(any(d["metadata"].get("license", "").startswith("MIT") for d in docs))
        chunks = chunk_documents(docs, {"strategy": "section", "max_words": 140, "overlap_words": 25, "min_words": 30})
        self.assertTrue(any(c["metadata"].get("page") for c in chunks))

    def test_filters(self) -> None:
        meta = {"company": "Nordwind Energy S.p.A.", "report_year": "2023", "gri_reference": "GRI 305"}
        self.assertTrue(matches_filters(meta, {"company": "Nordwind", "year": "2023"}))
        self.assertTrue(matches_filters(meta, {"gri": "GRI 305"}))
        self.assertFalse(matches_filters(meta, {"year": "2022"}))

    def test_rrf_fusion(self) -> None:
        a = [{"chunk_id": "x"}, {"chunk_id": "y"}]
        b = [{"chunk_id": "y"}, {"chunk_id": "z"}]
        fused = _rrf_fuse([a, b])
        self.assertEqual(fused[0]["chunk_id"], "y")
        self.assertEqual(len(fused), 3)

    def test_dashboard_progress_from_lineage(self) -> None:
        import dashboard

        rows = [
            {
                "route": "rag_answer", "strategy": "hybrid", "question": "Q1", "answer": "A1",
                "config": {"generator": "qwen3:4b-instruct", "embedder": "embeddinggemma:latest", "corpus": "real"},
            },
            {
                "route": "rag_answer", "strategy": "hybrid", "question": "Q2", "answer": "A2",
                "config": {"generator": "qwen3:4b-instruct", "embedder": "embeddinggemma:latest", "corpus": "real"},
            },
            {"route": "kpi_extraction", "question": "ignored", "config": {"corpus": "real"}},
        ]
        qa = {
            "Q1": {"reference": "The value is 42.", "document_source": "real_corpus/doc.md"},
            "Q2": {"reference": "The value is 42.", "document_source": "real_corpus/doc.md"},
        }
        for row in rows[:2]:
            row["metrics"] = {"total_latency_s": 1.5, "completion_tokens": 10, "llm_calls": 1}
            row["retrieval"] = {"passages": [{"metadata": {"source_file": "doc.md"}}]}
            row["answer"] = "The value is 42."
        progress = dashboard._progress(rows, qa)
        self.assertEqual(progress["queries"], 2)
        self.assertEqual(progress["cells"][0]["summary"]["questions"], 2)
        self.assertEqual(progress["cells"][0]["summary"]["answer_hit_rate"], 1.0)
        self.assertEqual(progress["cells"][0]["summary"]["retrieval_hit@k"], 1.0)
        self.assertEqual(progress["cells"][0]["strategy"], "hybrid")

    def test_system_metrics_estimate(self) -> None:
        from esg_rag.system_metrics import PowerMeter, collect, process_rss_mb

        meter = PowerMeter("estimate", estimated_watts=100.0)
        meter.start()
        meter.stop()  # tiny duration
        reading = meter.stop()
        self.assertEqual(reading["source"], "estimate")
        self.assertFalse(reading["measured"])
        self.assertEqual(reading["avg_watts"], 100.0)
        self.assertGreaterEqual(reading["joules"], 0)
        self.assertGreaterEqual(process_rss_mb(), 0)
        stats = [{
            "model": "qwen3:4b-instruct", "completion_tokens": 100,
            "eval_duration_ns": 2_000_000_000, "tokens_per_second": 50.0,
        }]
        agg = collect(stats, power=reading)
        self.assertEqual(agg["tokens_per_second"], 50.0)
        self.assertEqual(agg["ms_per_token"], 20.0)

    def test_answer_score(self) -> None:
        score = ESGRAGPipeline._answer_score
        hit, f1, numeric = score("The value was €2.20 per share.", "The value was €2.20.", "substring")
        self.assertTrue(hit and numeric and f1 > 0.5)
        hit, _, numeric = score("No data available.", "The value was €2.20.", "substring")
        self.assertFalse(hit or numeric)


if __name__ == "__main__":
    unittest.main()
