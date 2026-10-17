from __future__ import annotations

import math
import os
import re
import threading
from collections import Counter

# ------------------------------------------------------------------
# BM25 keyword search over the indexed chunks
# ------------------------------------------------------------------
# Code questions are full of exact identifiers ("what calls fail_task?"),
# which keyword search matches precisely while embeddings only match
# approximately. Tokens are code-aware: `fail_task` → fail_task, fail, task
# and `TaskOrchestrator` → taskorchestrator, task, orchestrator, so both
# the exact name and its words match.
#
# The index is built in memory from the chunks already stored in the
# vector store (no extra storage) and rebuilt after the index changes.
# ------------------------------------------------------------------

_WORD     = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_SUBWORDS = re.compile(r"_+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "be", "of", "to", "in", "on", "for", "and", "or",
    "how", "what", "when", "where", "why", "which", "who", "does", "do", "it", "its", "this",
    "that", "with", "by", "from", "as", "at", "if", "into", "used", "use",
}


def tokenize(text: str) -> list[str]:
    tokens = []
    for word in _WORD.findall(text):
        lower = word.lower()
        if lower in _STOPWORDS:
            continue
        tokens.append(lower)
        parts = [p.lower() for p in _SUBWORDS.split(word) if p]
        if len(parts) > 1:
            tokens.extend(p for p in parts if p not in _STOPWORDS)
    return tokens


def chunk_text(chunk: dict) -> str:
    """What BM25 indexes for a chunk: file name + symbol name + code."""
    return f"{os.path.basename(chunk['source'])} {chunk['name']}\n{chunk['content']}"


class BM25Index:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.lengths = [len(d) for d in documents]
        self.avgdl   = (sum(self.lengths) / len(documents)) if documents else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, doc in enumerate(documents):
            for term, tf in Counter(doc).items():
                self.postings.setdefault(term, []).append((i, tf))
        n = len(documents)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def top(self, query: str, n: int) -> list[tuple[int, float]]:
        """Indexes and scores of the n best-matching documents (best first)."""
        scores: dict[int, float] = {}
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc, tf in self.postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[doc] / (self.avgdl or 1))
                scores[doc] = scores.get(doc, 0.0) + idf * tf * (self.k1 + 1) / (tf + norm)
        return sorted(scores.items(), key=lambda item: -item[1])[:n]


# ── cache: rebuilt when the vector index changes ────────────────

_lock = threading.Lock()
_generation = 0
_cache: tuple[int, BM25Index, list[dict]] | None = None


def invalidate() -> None:
    """Called after any index update, so the next search rebuilds the keyword index."""
    global _generation
    with _lock:
        _generation += 1


def search(query: str, n: int, load_chunks) -> list[dict]:
    """Top-n chunks by BM25. load_chunks() returns every indexed chunk (called only on rebuild)."""
    global _cache
    with _lock:
        if _cache is None or _cache[0] != _generation:
            chunks = load_chunks()
            _cache = (_generation, BM25Index([tokenize(chunk_text(c)) for c in chunks]), chunks)
        _, index, chunks = _cache
    return [chunks[i] | {"bm25_score": score} for i, score in index.top(query, n)]


def reciprocal_rank_fusion(rankings: list[list[dict]], k: int = 60) -> list[dict]:
    """
    Merge ranked result lists: each chunk scores Σ 1/(k + rank) over the lists it
    appears in. Rank-based, so dense distances and BM25 scores never need to be
    put on the same scale.
    """
    scores: dict[tuple, float] = {}
    chunks: dict[tuple, dict] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, 1):
            key = (chunk["source"], chunk["name"], chunk["start_line"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            chunks.setdefault(key, chunk)
    ordered = sorted(scores, key=lambda key: -scores[key])
    return [chunks[key] | {"rrf_score": scores[key]} for key in ordered]
