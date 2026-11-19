"""Thin client for the local Ollama REST API.

Two endpoints are used:
  * ``/api/embed`` — embeddinggemma (or any Ollama embedding model)
  * ``/api/chat``  — qwen3 (or any Ollama chat model), with ``think=False``
                     because reasoning traces are not useful for grounded
                     answers and break strict JSON extraction.

No third-party Ollama SDK is required: only ``requests``.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

import requests


_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*```(?:json|text|sql)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` traces emitted by reasoning models."""
    return _THINK_RE.sub("", text or "").strip()


def parse_json_loose(text: str) -> Any:
    """Parse a JSON object from an LLM reply, tolerating markdown fences."""
    text = strip_thinking(text)
    fenced = _FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: first {...} block in the reply.
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


@dataclass
class CallStats:
    """Timing / token accounting used by the lineage and benchmark modules."""

    model: str = ""
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_duration_ns: int = 0
    eval_duration_ns: int = 0
    prompt_eval_duration_ns: int = 0
    load_duration_ns: int = 0

    @property
    def tokens_per_second(self) -> float | None:
        """Decode speed reported by Ollama (completion tokens / eval time)."""
        if self.eval_duration_ns and self.completion_tokens:
            return round(self.completion_tokens / (self.eval_duration_ns / 1e9), 2)
        return None

    @property
    def ms_per_token(self) -> float | None:
        tps = self.tokens_per_second
        return round(1000.0 / tps, 2) if tps else None

    def asdict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "latency_s": round(self.latency_s, 4),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_duration_ns": self.total_duration_ns,
            "eval_duration_ns": self.eval_duration_ns,
            "prompt_eval_duration_ns": self.prompt_eval_duration_ns,
            "load_duration_ns": self.load_duration_ns,
            "tokens_per_second": self.tokens_per_second,
            "ms_per_token": self.ms_per_token,
        }


class OllamaClient:
    def __init__(self, base_url: str, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    # ---------- health ----------
    def list_models(self) -> list[str]:
        r = self.session.get(f"{self.base_url}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def check_models(self, required: Iterable[str]) -> dict[str, bool]:
        available = set(self.list_models())
        return {name: name in available for name in required}

    def ps(self) -> list[dict[str, Any]]:
        """Currently loaded Ollama models (size in bytes, VRAM, processor)."""
        try:
            r = self.session.get(f"{self.base_url}/api/ps", timeout=10)
            r.raise_for_status()
            return r.json().get("models", [])
        except Exception:
            return []

    # ---------- embeddings ----------
    def embed(self, texts: list[str], model: str) -> tuple[list[list[float]], CallStats]:
        """Embed a batch of strings; returns vectors and call statistics."""
        t0 = time.perf_counter()
        r = self.session.post(
            f"{self.base_url}/api/embed",
            json={"model": model, "input": texts},
            timeout=self.timeout,
        )
        r.raise_for_status()
        payload = r.json()
        if "embeddings" not in payload:
            raise RuntimeError(f"Unexpected /api/embed response: {payload!r}")
        stats = CallStats(
            model=model,
            latency_s=time.perf_counter() - t0,
            prompt_tokens=payload.get("prompt_eval_count", 0),
            total_duration_ns=payload.get("total_duration", 0),
            load_duration_ns=payload.get("load_duration", 0),
            prompt_eval_duration_ns=payload.get("prompt_eval_duration", 0),
        )
        return payload["embeddings"], stats

    # ---------- generation ----------
    def chat(
        self,
        prompt: str,
        model: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
        think: bool = False,
        seed: int | None = None,
        num_ctx: int | None = None,
    ) -> tuple[str, CallStats]:
        """Single-turn chat completion. Optionally force structured JSON output."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        options: dict[str, Any] = {"temperature": temperature, "num_predict": max_tokens}
        if seed is not None:
            options["seed"] = seed
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": think,
            "options": options,
        }
        if json_mode or json_schema is not None:
            body["format"] = json_schema if json_schema is not None else "json"

        t0 = time.perf_counter()
        r = self.session.post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        payload = r.json()
        content = strip_thinking(payload.get("message", {}).get("content", ""))
        stats = CallStats(
            model=model,
            latency_s=time.perf_counter() - t0,
            prompt_tokens=payload.get("prompt_eval_count", 0),
            completion_tokens=payload.get("eval_count", 0),
            total_duration_ns=payload.get("total_duration", 0),
            eval_duration_ns=payload.get("eval_duration", 0),
            prompt_eval_duration_ns=payload.get("prompt_eval_duration", 0),
            load_duration_ns=payload.get("load_duration", 0),
        )
        return content, stats
