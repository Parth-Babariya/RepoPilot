from __future__ import annotations

import os
from functools import lru_cache

from repopilot.config import config
from repopilot.observability.logger import get_logger


logger = get_logger(__name__)

# ------------------------------------------------------------------
# Two-stage retrieval: vector search → cross-encoder rerank
# ------------------------------------------------------------------
# Vector search compares the question and each chunk as two separate
# embeddings — fast, but coarse. A cross-encoder reads the question and
# one chunk TOGETHER and scores how well that chunk answers it — much
# more accurate, but too slow to run over the whole index.
#
# So: fetch rerank.candidates chunks (e.g. 20) with vector search, score
# each with the cross-encoder, keep the best k (e.g. 5).
#
# Runs locally via fastembed (ONNX, already a dependency) — no API cost.
# The model is downloaded once on first use. If it can't be loaded, we
# log a warning and fall back to plain vector-search order.
# ------------------------------------------------------------------

_DOC_CHARS = 2000   # cross-encoders read ~512 tokens; longer text is wasted work


def rerank_config() -> dict:
    cfg = config.get("rerank", {}) or {}
    return {
        "enabled":    bool(cfg.get("enabled", False)),
        "model":      cfg.get("model", "jinaai/jina-reranker-v1-tiny-en"),
        "candidates": int(cfg.get("candidates", 20)),
    }


@lru_cache(maxsize=1)
def _get_encoder(model_name: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    logger.info(f"Loading reranker model {model_name} (downloaded on first use)")
    return TextCrossEncoder(model_name=model_name)


def _document(chunk: dict) -> str:
    # Path and symbol name give the model useful signal ("task_store.py :: fail_task")
    try:
        path = os.path.relpath(chunk["source"])
    except ValueError:
        path = chunk["source"]
    return f"{path} :: {chunk['name']}\n{chunk['content'][:_DOC_CHARS]}"


def rerank(query: str, chunks: list[dict], k: int, encoder=None) -> list[dict]:
    """
    Return the k chunks the cross-encoder scores as most relevant to query,
    each with a "rerank_score" added. encoder can be injected for tests.
    """
    if len(chunks) <= 1:
        return chunks[:k]
    try:
        encoder = encoder or _get_encoder(rerank_config()["model"])
        scores  = list(encoder.rerank(query, [_document(c) for c in chunks]))
    except Exception as e:
        logger.warning(f"Reranking failed, using vector-search order: {e}")
        return chunks[:k]

    ranked = sorted(zip(scores, range(len(chunks))), key=lambda pair: -pair[0])
    return [chunks[i] | {"rerank_score": float(score)} for score, i in ranked[:k]]


def with_reranking(retrieve):
    """Wrap a retrieve(query, k) function so it over-fetches and reranks."""
    settings = rerank_config()

    def retrieve_and_rerank(query: str, k: int = 5) -> list[dict]:
        candidates = retrieve(query, k=max(k, settings["candidates"]))
        reranked   = rerank(query, candidates, k)
        logger.info(f"Reranked {len(candidates)} candidates → top {len(reranked)}")
        return reranked

    return retrieve_and_rerank
