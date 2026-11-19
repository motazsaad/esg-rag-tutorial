"""Queryable data-lineage log (one JSON object per query, append-only).

Each record ties together: the question, the route (RAG answer or KPI
extraction), the retrieval configuration and filters, the exact source chunks,
the prompts' models, the answer/extracted value, and system metrics
(latency and tokens). This is the audit trail required for regulated analytics.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class LineageStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def record(self, entry: dict[str, Any]) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **entry,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def recent(self, n: int = 5) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines[-n:] if line.strip()]

    def all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def summary(self) -> dict[str, Any]:
        rows = self.all()
        if not rows:
            return {"records": 0}
        modes: dict[str, int] = {}
        total_latency = 0.0
        total_tokens = 0
        for row in rows:
            modes[row.get("route", "unknown")] = modes.get(row.get("route", "unknown"), 0) + 1
            metrics = row.get("metrics", {})
            total_latency += float(metrics.get("total_latency_s", 0))
            total_tokens += int(metrics.get("completion_tokens", 0))
        return {
            "records": len(rows),
            "routes": modes,
            "mean_total_latency_s": round(total_latency / len(rows), 3),
            "total_completion_tokens": total_tokens,
            "path": str(self.path),
        }
