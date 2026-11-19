"""System-level metrics: memory, decode speed and energy.

Design goals
------------
* Use **real measurements** where the machine exposes them:
  - decode speed comes from Ollama's own ``eval_duration`` / ``eval_count``;
  - model memory comes from Ollama ``/api/ps``;
  - process memory comes from ``/proc/self/statm``;
  - CPU energy can come from Intel RAPL or NVIDIA-SMI when readable.
* Where neither RAPL nor NVIDIA-SMI is available, energy is **estimated** as
  ``assumed_watts × wall_seconds`` and explicitly labelled ``measured=False``.
  Never present an estimate as a measurement.

This module has no third-party dependencies and degrades gracefully on macOS /
Windows (where ``/proc`` and RAPL do not exist).
"""

from __future__ import annotations

import glob
import os
import subprocess
import time
from pathlib import Path
from typing import Any

_JOULES_IN_UJ = 1e-6


def process_rss_mb() -> float:
    """Resident set size of the current Python process in MB (0 if unavailable)."""
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return round(pages * os.sysconf("SC_PAGE_SIZE") / 1e6, 2)
    except (OSError, IndexError, ValueError, AttributeError):
        try:
            import resource

            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2)
        except Exception:
            return 0.0


class PowerMeter:
    """Measure (or, as a last resort, estimate) energy for one query."""

    def __init__(self, source: str = "auto", estimated_watts: float = 65.0):
        self.source = source
        self.estimated_watts = float(estimated_watts)
        self._rapl_files = self._find_rapl() if source in {"auto", "rapl"} else []
        if source == "auto":
            if self._rapl_files:
                self.source = "rapl"
            elif self._nvidia_available():
                self.source = "nvidia"
            else:
                self.source = "estimate"
        self._start_energy: float | None = None
        self._start_time: float | None = None

    @staticmethod
    def _find_rapl() -> list[Path]:
        found = []
        for pattern in (
            "/sys/class/powercap/intel-rapl:*/energy_uj",
            "/sys/class/powercap/intel-rapl-mmio:*/energy_uj",
        ):
            for name in glob.glob(pattern):
                path = Path(name)
                try:
                    path.read_text()
                    found.append(path)  # readable by this user
                except OSError:
                    continue
        return found

    @staticmethod
    def _nvidia_available() -> bool:
        try:
            return subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True,
                timeout=2,
            ).returncode == 0
        except (FileNotFoundError, subprocess.SubprocessError):
            return False

    def _energy_j(self) -> float | None:
        if self.source == "rapl" and self._rapl_files:
            try:
                return sum(int(p.read_text().strip()) for p in self._rapl_files) * _JOULES_IN_UJ
            except (OSError, ValueError):
                return None
        if self.source == "nvidia":
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2,
                ).stdout
                watts = sum(float(x) for x in out.split() if x.replace(".", "", 1).isdigit())
                return watts
            except (FileNotFoundError, ValueError, subprocess.SubprocessError):
                return None
        return None

    def start(self) -> None:
        self._start_time = time.perf_counter()
        self._start_energy = self._energy_j()

    def stop(self) -> dict[str, Any]:
        duration = max(0.0, time.perf_counter() - (self._start_time or time.perf_counter()))
        end_energy = self._energy_j()
        measured = self._start_energy is not None and end_energy is not None and end_energy >= self._start_energy
        if measured:
            joules = end_energy - self._start_energy  # type: ignore[operator]
            avg_watts = joules / duration if duration else 0.0
        else:
            joules = self.estimated_watts * duration
            avg_watts = self.estimated_watts
        return {
            "source": self.source,
            "measured": bool(measured),
            "joules": round(joules, 3),
            "avg_watts": round(avg_watts, 2),
            "duration_s": round(duration, 3),
            "note": (
                "measured via RAPL/NVIDIA-SMI"
                if measured
                else f"ESTIMATE: {self.estimated_watts} W assumed × wall time (no readable RAPL/GPU sensor)"
            ),
        }


def ollama_memory(client: Any, model_names: list[str]) -> dict[str, Any]:
    """Memory of the currently loaded Ollama models from /api/ps."""
    running = client.ps()
    per_model = {
        m.get("name", "?"): {
            "size_mb": round(m.get("size", 0) / 1e6, 1),
            "vram_mb": round(m.get("size_vram", 0) / 1e6, 1),
            "processor": m.get("processor", "?"),
        }
        for m in running
        if not model_names or m.get("name") in model_names
    }
    return {
        "total_model_mb": round(sum(m["size_mb"] for m in per_model.values()), 1),
        "total_vram_mb": round(sum(m["vram_mb"] for m in per_model.values()), 1),
        "per_model": per_model,
    }


def collect(
    call_stats: list[dict[str, Any]],
    *,
    power: dict[str, Any] | None = None,
    client: Any = None,
    models: list[str] | None = None,
) -> dict[str, Any]:
    """Aggregate system metrics for one pipeline query."""
    speeds = [c["tokens_per_second"] for c in call_stats if c.get("tokens_per_second")]
    decode_tokens = sum(int(c.get("completion_tokens", 0)) for c in call_stats)
    decode_ns = sum(int(c.get("eval_duration_ns", 0)) for c in call_stats)
    aggregate_tps = round(decode_tokens / (decode_ns / 1e9), 2) if decode_ns else None
    return {
        "tokens_per_second": aggregate_tps,
        "mean_call_tokens_per_second": round(sum(speeds) / len(speeds), 2) if speeds else None,
        "ms_per_token": round(1000.0 / aggregate_tps, 2) if aggregate_tps else None,
        "process_rss_mb": process_rss_mb(),
        "ollama_memory": ollama_memory(client, models or []) if client is not None else {},
        "power": power or {},
    }
