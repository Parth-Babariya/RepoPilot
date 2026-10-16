import chromadb

from repopilot.config import config
from repopilot.context.indexers.semantic_chroma import chroma_collection_name
from repopilot.context.retrievers import bm25
from repopilot.context.retrievers.semantic_chroma import retrieve as dense_retrieve
from repopilot.observability.logger import get_logger


logger = get_logger(__name__)

# Each list is fetched this deep before fusion, so a chunk ranked low by one
# method but high by the other still makes it into the merged list.
_MIN_CANDIDATES = 20


def _load_chunks() -> list[dict]:
    client = chromadb.PersistentClient(path=config["chromadb"]["persist_dir"])
    collection = client.get_or_create_collection(name=chroma_collection_name())
    stored = collection.get(include=["documents", "metadatas"])
    logger.info(f"Building BM25 index over {len(stored['ids'])} chunks")
    return [
        {
            "content":    doc,
            "source":     meta["source"],
            "name":       meta["name"],
            "type":       meta["type"],
            "start_line": meta["start_line"],
            "end_line":   meta["end_line"],
            "distance":   None,
        }
        for doc, meta in zip(stored["documents"], stored["metadatas"])
    ]


def retrieve(query: str, k: int = 5) -> list[dict]:
    """
    Hybrid search on Chroma: vector search + BM25 keyword search, merged with
    reciprocal rank fusion. Same return format as the other retrievers.
    """
    n = max(k, _MIN_CANDIDATES)
    dense  = dense_retrieve(query, k=n)
    sparse = bm25.search(query, n, _load_chunks)
    fused  = bm25.reciprocal_rank_fusion([dense, sparse])[:k]
    logger.info(f"Hybrid search: {len(dense)} dense + {len(sparse)} keyword → top {len(fused)}")
    return fused
