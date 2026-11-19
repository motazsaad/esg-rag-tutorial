"""Integration smoke test: run with `python -m unittest discover -s tests`.

Requires a running Ollama with the two configured models (see README).
The test asserts the pipeline's structural guarantees, not model quality.
"""

from __future__ import annotations

import json
import unittest

from esg_rag.config import Config
from esg_rag.pipeline import ESGRAGPipeline


class PipelineSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = ESGRAGPipeline()

    def test_01_doctor_ready(self) -> None:
        health = self.pipeline.doctor()
        self.assertTrue(health["ok"], health)
        self.assertTrue(health["models_available"][Config().generator])

    def test_02_chunking_and_index(self) -> None:
        result = self.pipeline.index(force=True)
        self.assertEqual(result["status"], "built")
        self.assertGreaterEqual(result["chunks"], 5)
        self.assertEqual(result["dimensions"], 768)  # embeddinggemma:latest

    def test_03_metadata_filter(self) -> None:
        store = self.pipeline.store
        assert store is not None
        nordwind = store.all_records({"company": "Nordwind", "year": "2023"})
        solaria = store.all_records({"company": "Solaria"})
        self.assertEqual(len(nordwind), 5)
        self.assertEqual(len(solaria), 3)
        emissions = store.all_records({"company": "Nordwind", "gri": "GRI 305"})
        self.assertEqual(len(emissions), 1)

    def test_04_grounded_answer(self) -> None:
        r = self.pipeline.query(
            "What was the total number of employees at Nordwind in 2023?",
            filters={"company": "Nordwind"},
        )
        self.assertTrue(r["citations"], r)
        self.assertIn("12,480", r["answer"])
        self.assertGreater(r["metrics"]["completion_tokens"], 0)

    def test_05_structured_extraction(self) -> None:
        r = self.pipeline.extract("GRI 305_305-1_32", filters={"company": "Nordwind"}, top_k=3)
        self.assertIsNone(r["parse_error"])
        value = r["value"]
        self.assertEqual(str(value["reporting year"]), "2023")
        self.assertIn("84320", str(value["value"]))

    def test_06_full_smoke(self) -> None:
        result = self.pipeline.smoke_test()
        self.assertTrue(result["passed"], result)

    def test_07_real_data_present(self) -> None:
        corpus = sorted(self.pipeline.config.path("real_corpus_dir").glob("*.md"))
        qa = json.loads(self.pipeline.config.path("real_qa").read_text(encoding="utf-8"))
        bench = json.loads(self.pipeline.config.path("real_retrieval_bench").read_text(encoding="utf-8"))
        self.assertEqual(len(corpus), 6)
        self.assertEqual(len(qa["questions"]), 6)
        self.assertEqual(len(bench["records"]), 24)


if __name__ == "__main__":
    unittest.main()
