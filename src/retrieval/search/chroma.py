import chromadb

from repopilot.config import config
from repopilot.retrieval.indexing.chroma import chroma_collection_name
from repopilot.retrieval.search import bm25
from repopilot.services.llm_service import get_embedder
from repopilot.observability.logger import get_logger


logger = get_logger(__name__)

# In hybrid mode each list is fetched this deep before fusion, so a chunk
# ranked low by one method but high by the other still makes the merged list.
_MIN_CANDIDATES = 20


def _collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=config["chromadb"]["persist_dir"])
    return client.get_or_create_collection(name=chroma_collection_name())


def _dense(query: str, k: int) -> list[dict]:
    """Embed the query and find the k most similar chunks."""
    # Same embedder as the indexer — the model must match or results are meaningless
    query_embedding = get_embedder().embed_query(query)
    results = _collection().query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "content":    doc,
            "source":     meta["source"],
            "name":       meta["name"],
            "type":       meta["type"],
            "start_line": meta["start_line"],
            "end_line":   meta["end_line"],
            "distance":   dist,   # lower = more similar
        }
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ]


def _all_chunks() -> list[dict]:
    """Every stored chunk — the BM25 keyword index is built from these."""
    stored = _collection().get(include=["documents", "metadatas"])
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
    Search the Chroma index. rag.mode picks the search: "semantic" uses vector
    search; "hybrid" also runs BM25 keyword search and merges both rankings
    with reciprocal rank fusion.
    """
    if config["rag"]["mode"] != "hybrid":
        chunks = _dense(query, k)
        logger.info(f"Vector search: {len(chunks)} chunks for: {query}")
        return chunks

    n = max(k, _MIN_CANDIDATES)
    dense  = _dense(query, n)
    sparse = bm25.search(query, n, _all_chunks)
    fused  = bm25.reciprocal_rank_fusion([dense, sparse])[:k]
    logger.info(f"Hybrid search: {len(dense)} vector + {len(sparse)} keyword → top {len(fused)} for: {query}")
    return fused
