"""Document parsing and section-aware chunking.

Input files are plain text / Markdown with a small YAML-like HTML comment
header (see ``data/corpus/`` for examples). The same functions work for any
text extracted from PDF/HTML by an upstream parser — the ESG RAG pipeline only
needs text plus metadata, which is exactly what the index stores.

Chunking is *section aware*: headings delimit chunks, so a chunk never mixes
e.g. GRI 302 (Energy) with GRI 305 (Emissions). Long sections are split into
overlapping word windows. This mirrors the tutorial's discussion of fixed vs.
section-aware vs. semantic chunking.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

_HEADER_RE = re.compile(r"^\s*([a-z_]+)\s*:\s*(.*?)\s*$")
_PAGE_RE = re.compile(
    r"<!--\s*page\s*:\s*(\d+)\s*-->|^\s*\*+\s*Page\s+(\d+)\s*\*+\s*$",
    re.IGNORECASE,
)
_GRI_RE = re.compile(r"GRI\s+\d{1,3}(?:[-_ ]\d{1,3})*", re.IGNORECASE)


def _parse_header(lines: list[str]) -> tuple[dict[str, str], list[str]]:
    """Parse the leading metadata comment, if present.

    Supports both a block comment::

        <!--
        doc_id: NRW-2023
        company: Nordwind Energy S.p.A.
        -->

    and one ``<!-- key: value -->`` comment per line.
    """
    meta: dict[str, str] = {}
    body_start = 0
    if lines and lines[0].strip() == "<!--":
        for i, line in enumerate(lines[1:], start=1):
            stripped = line.strip()
            if "-->" in stripped:
                meta_part = stripped.split("-->", 1)[0]
                m = _HEADER_RE.match(meta_part.strip())
                if m:
                    meta[m.group(1)] = m.group(2)
                body_start = i + 1
                break
            m = _HEADER_RE.match(stripped)
            if m:
                meta[m.group(1)] = m.group(2)
                body_start = i + 1
            elif stripped:
                break
        return meta, lines[body_start:]

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            if meta:
                body_start = i + 1
                break
            continue
        if not (stripped.startswith("<!--") and stripped.endswith("-->")):
            break
        inner = stripped[4:-3].strip()
        m = _HEADER_RE.match(inner)
        if not m:
            break
        meta[m.group(1)] = m.group(2)
        body_start = i + 1
    return meta, lines[body_start:]


def load_documents(corpus_dir: str | Path) -> list[dict[str, Any]]:
    """Load all ``.txt`` / ``.md`` documents from ``corpus_dir``."""
    directory = Path(corpus_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {directory}")
    docs = []
    for path in sorted([*directory.glob("*.txt"), *directory.glob("*.md")]):
        raw = path.read_text(encoding="utf-8")
        lines = raw.splitlines()
        header, body = _parse_header(lines)
        # Keep every header field (source repo, license, provenance, ...) in the
        # metadata so it survives into the index and the lineage record.
        metadata = {
            "company": path.stem,
            "company_short": path.stem,
            "country": "NA",
            "report_year": "NA",
            "language": "en",
            "doc_type": "Sustainability Report",
            "source_url": "",
        }
        metadata.update(header)
        docs.append(
            {
                "doc_id": header.get("doc_id", path.stem),
                "source_file": path.name,
                "metadata": metadata,
                "text": "\n".join(body),
            }
        )
    if not docs:
        raise ValueError(f"No .txt/.md documents found in {directory}")
    return docs


def _split_sections(text: str) -> Iterator[tuple[str, str, int | None]]:
    """Yield (heading, section_text, page) triples."""
    current_heading = "Overview"
    buffer: list[str] = []
    page: int | None = None

    def flush() -> tuple[str, str, int | None]:
        return current_heading, "\n".join(buffer).strip(), page

    for line in text.splitlines():
        pm = _PAGE_RE.search(line)
        if pm:
            if buffer and "\n".join(buffer).strip():
                yield flush()
                buffer = []
            page = int(pm.group(1) or pm.group(2))
            continue
        hm = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", line)
        if not hm:
            # Also treat ALL-CAPS short lines as headings (common in reports).
            stripped = line.strip()
            if (
                stripped
                and len(stripped) < 90
                and stripped.upper() == stripped
                and any(c.isalpha() for c in stripped)
                and not stripped.startswith("<!--")
            ):
                hm = None
                current_heading = stripped.title()
                if buffer and "\n".join(buffer).strip():
                    yield flush()
                buffer = []
                continue
        if hm:
            if buffer and "\n".join(buffer).strip():
                yield flush()
            current_heading = hm.group(2).strip()
            buffer = []
        else:
            buffer.append(line)
    if buffer and "\n".join(buffer).strip():
        yield flush()


def _word_windows(text: str, max_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    out: list[str] = []
    step = max(1, max_words - overlap_words)
    for start in range(0, len(words), step):
        window = words[start : start + max_words]
        if len(window) < max_words // 3 and out:
            break  # avoid tiny tail chunks
        out.append(" ".join(window))
        if start + max_words >= len(words):
            break
    return out


def _semantic_groups(text: str, threshold: float) -> list[str]:
    """Group adjacent paragraphs whose vocabulary overlaps (lightweight semantics)."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    groups: list[str] = []
    current: list[str] = []
    for para in paragraphs:
        if not current:
            current = [para]
            continue
        a = set(re.findall(r"\w+", current[-1].lower()))
        b = set(re.findall(r"\w+", para.lower()))
        union = a | b
        jaccard = len(a & b) / len(union) if union else 0.0
        if jaccard >= threshold:
            current.append(para)
        else:
            groups.append(" ".join(current))
            current = [para]
    if current:
        groups.append(" ".join(current))
    return groups


def chunk_documents(
    documents: list[dict[str, Any]], chunk_cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """Turn documents into embedded-ready chunks with rich metadata.

    ``chunk_cfg['strategy']`` selects:
      * ``section``  — headings delimit chunks (recommended for ESG reports)
      * ``fixed``    — plain overlapping word windows over the whole document
      * ``semantic`` — adjacent paragraphs with vocabulary overlap are grouped
    """
    max_words = int(chunk_cfg.get("max_words", 140))
    overlap_words = int(chunk_cfg.get("overlap_words", 25))
    min_words = int(chunk_cfg.get("min_words", 30))
    strategy = chunk_cfg.get("strategy", "section").lower()
    threshold = float(chunk_cfg.get("semantic_threshold", 0.15))
    chunks: list[dict[str, Any]] = []

    for doc in documents:
        counter = 0
        units: list[tuple[str, str, int | None]] = []
        if strategy == "section":
            units = list(_split_sections(doc["text"]))
        elif strategy == "fixed":
            units = [("Document", doc["text"], None)]
        elif strategy == "semantic":
            for heading, section_text, page in _split_sections(doc["text"]):
                for group in _semantic_groups(section_text, threshold):
                    units.append((heading, group, page))
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        for heading, section_text, page in units:
            if len(section_text.split()) < 3:
                continue
            gri = _GRI_RE.search(heading) or _GRI_RE.search(section_text[:400])
            for part in _word_windows(section_text, max_words, overlap_words):
                if len(part.split()) < min_words:
                    continue
                counter += 1
                chunk_id = f"{doc['doc_id']}::c{counter:03d}"
                text = f"[{doc['metadata']['company']} — {doc['metadata']['report_year']}] {heading}\n{part}"
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": text,
                        "metadata": {
                            **doc["metadata"],
                            "doc_id": doc["doc_id"],
                            "source_file": doc["source_file"],
                            "section": heading,
                            "gri_reference": gri.group(0) if gri else "NA",
                            "page": page,
                            "chunk_index": counter,
                        },
                    }
                )
    return chunks
