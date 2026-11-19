"""Configuration loading: YAML file + environment variable overrides.

Every knob used by the pipeline lives here so a run is reproducible from a
single config file. Environment variables use the ``ESG_RAG_`` prefix and
override the YAML values, e.g.::

    ESG_RAG_LLM_MODEL=qwen3:8b  python -m esg_rag.cli query "..."

The lineage record stores the effective configuration of every query.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # PyYAML is optional; a minimal built-in default is used.
    yaml = None


PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULTS: dict[str, Any] = {
    "ollama": {
        "base_url": "http://localhost:11434",
        "request_timeout": 300,
    },
    "models": {
        # Active models (overridable per command / matrix cell)
        "generator": "qwen3:4b-instruct",
        "embedder": "embeddinggemma:latest",
        "judge": "qwen3:4b-instruct",
        "rewrite": "qwen3:4b-instruct",
        "rerank": "qwen3:4b-instruct",
        # Candidate pools used by the matrix runner and `doctor`
        "llms": ["qwen3:4b-instruct", "gemma3:4b"],
        "embedders": ["embeddinggemma:latest", "qwen3-embedding:0.6b"],
    },
    "paths": {
        "corpus_dir": "data/corpus",
        "real_corpus_dir": "data/real_corpus",
        "kpi_questions": "data/kpi_questions.yaml",
        "gold_set": "data/gold.jsonl",
        "real_qa": "data/real_qa.json",
        "real_retrieval_bench": "data/real_retrieval_bench.json",
        "index": "output/index.json",
        "lineage": "output/lineage.jsonl",
        "results_dir": "output/results",
        "matrix_output": "output/results/matrix.json",
        "matrix_csv": "output/results/matrix.csv",
        "matrix_report": "output/results/matrix_report.md",
    },
    "chunking": {
        "strategy": "section",  # section | fixed | semantic
        "max_words": 140,
        "overlap_words": 25,
        "min_words": 30,
        "semantic_threshold": 0.15,
    },
    "retrieval": {
        "top_k": 4,
        "mode": "hybrid",  # dense | bm25 | hybrid
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "rrf_k": 60,
        "max_evidence_chars": 900,
        "candidate_multiplier": 3,  # pool size for rerank / multi-query
        "infer_filters": False,     # LLM-derived company/year/GRI filters
    },
    "strategy": {
        "name": "hybrid",  # naive | hybrid | filtered | multi_query | rerank | self_rag | decompose
        "multi_query_variants": 3,
        "rerank_top_n": None,
        "self_rag_max_loops": 1,
        "decompose_max_subqs": 3,
    },
    "generation": {
        "temperature": 0.0,
        "max_tokens": 256,
        "seed": 42,
        "num_ctx": 8192,
        "think": False,
        "insufficient_evidence": "INSUFFICIENT EVIDENCE",
    },
    "benchmark": {
        "dataset": "synthetic",  # synthetic | real
        "limit": None,
        "judge": False,
        "answer_metric": "substring",  # substring | token_f1 | judge
    },
    "system_metrics": {
        "enabled": True,
        "power_source": "auto",  # auto | rapl | nvidia | estimate | none
        "estimated_watts": 65.0,  # fallback assumption when no sensor is readable
    },
    "matrix": {
        "llms": ["qwen3:4b-instruct", "gemma3:4b"],
        "embedders": ["embeddinggemma:latest", "qwen3-embedding:0.6b"],
        "strategies": ["naive", "hybrid", "multi_query", "rerank", "self_rag"],
        "dataset": "real",
        "limit": 3,
        "judge": True,
        "skip_missing_models": True,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def parse_value(raw: str) -> Any:
    """Parse CLI override values as JSON when possible, else as plain text."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def set_value(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested config value from a dotted key, e.g. ``chunking.max_words``."""
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``config.yaml`` (if present), apply env overrides, return a dict."""
    cfg = copy.deepcopy(DEFAULTS)
    config_path = Path(path) if path else PROJECT_DIR / "config.yaml"
    if config_path.exists():
        if yaml is None:
            raise RuntimeError(
                "config.yaml found but PyYAML is not installed; "
                "install PyYAML or remove the file to use built-in defaults."
            )
        with config_path.open(encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh) or {}
        cfg = _deep_merge(cfg, user_cfg)

    # Environment overrides: ESG_RAG_SECTION_KEY -> cfg[section][key]
    for section, values in DEFAULTS.items():
        if not isinstance(values, dict):
            continue
        for key in values:
            env_key = f"ESG_RAG_{section.upper()}_{key.upper()}"
            if env_key in os.environ:
                raw = os.environ[env_key]
                # Preserve types for numeric / boolean values.
                default = values[key]
                if isinstance(default, bool):
                    cfg[section][key] = raw.lower() in ("1", "true", "yes", "on")
                elif isinstance(default, int):
                    cfg[section][key] = int(raw)
                elif isinstance(default, float):
                    cfg[section][key] = float(raw)
                else:
                    cfg[section][key] = raw
    return cfg


@dataclass
class Config:
    """Convenience wrapper around the configuration dictionary."""

    data: dict[str, Any] = field(default_factory=load_config)

    @property
    def base_url(self) -> str:
        return self.data["ollama"]["base_url"].rstrip("/")

    @property
    def generator(self) -> str:
        return self.data["models"]["generator"]

    @property
    def embedder(self) -> str:
        return self.data["models"]["embedder"]

    @property
    def judge(self) -> str:
        return self.data["models"]["judge"]

    def path(self, name: str) -> Path:
        p = Path(self.data["paths"][name])
        return p if p.is_absolute() else PROJECT_DIR / p

    def override(self, dotted_key: str, value: Any) -> "Config":
        """Apply one dotted override and return self (chainable)."""
        set_value(self.data, dotted_key, value)
        return self

    @property
    def strategy_name(self) -> str:
        return self.data["strategy"]["name"]

    @staticmethod
    def model_slug(name: str) -> str:
        """Filesystem-safe slug for model-specific index/result files."""
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

    def asdict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)
