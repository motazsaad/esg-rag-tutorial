"""Optional zero-dependency REST service for the live demo.

    python serve.py                 # http://localhost:8000/docs
    curl -s localhost:8000/health
    curl -s -X POST localhost:8000/query \
      -H 'Content-Type: application/json' \
      -d '{"question":"What were the Scope 1 emissions at Nordwind in 2023?","filters":{"company":"Nordwind"}}'

It reuses the exact same pipeline object as the CLI, so the REST route and the
smoke test exercise identical code (important for reproducibility).
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from esg_rag.pipeline import ESGRAGPipeline


class Handler(BaseHTTPRequestHandler):
    pipeline: ESGRAGPipeline

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[http] {self.address_string()} - {fmt % args}")

    def _send(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._send({"status": "ok", "doctor": self.pipeline.doctor()})
        elif path == "/docs":
            self._send(
                {
                    "endpoints": {
                        "GET /health": "Ollama / model / corpus status",
                        "POST /index": "{} -> rebuild the vector index",
                        "POST /query": '{"question": str, "filters": {...}, "top_k": int, "mode": "hybrid|dense|bm25", "strategy": "hybrid|multi_query|rerank|self_rag|...", "corpus": "synthetic|real", "judge": bool}',
                        "POST /extract": '{"kpi": "GRI 305_305-1_32", "filters": {...}}',
                        "GET /lineage": "last 20 lineage records",
                    }
                }
            )
        elif path == "/lineage":
            self._send({"records": self.pipeline.lineage.recent(20)})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/index":
                result = self.pipeline.index(force=bool(body.get("force", True)))
            elif path == "/query":
                if body.get("corpus"):
                    self.pipeline.use_corpus(body["corpus"])
                result = self.pipeline.query(
                    body["question"],
                    filters=body.get("filters"),
                    top_k=body.get("top_k"),
                    mode=body.get("mode"),
                    judge=bool(body.get("judge", False)),
                    strategy=body.get("strategy"),
                )
            elif path == "/extract":
                result = self.pipeline.extract(
                    body["kpi"], filters=body.get("filters"), top_k=body.get("top_k")
                )
            else:
                self._send({"error": "not found"}, 404)
                return
            self._send(result)
        except (KeyError, RuntimeError, FileNotFoundError, ValueError) as exc:
            self._send({"error": str(exc)}, 400)


def main() -> None:
    parser = argparse.ArgumentParser(description="ESG RAG REST demo service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    Handler.pipeline = ESGRAGPipeline()
    print(f"ESG RAG service on http://{args.host}:{args.port}  (see /docs)")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
