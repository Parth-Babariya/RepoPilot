from langchain_qdrant import RetrievalMode

from repopilot.config import config
from repopilot.context.indexers.qdrant import vector_store
from repopilot.observability.logger import get_logger


logger = get_logger(__name__)


def retrieve(query: str, k: int = 5) -> list[dict]:
    """
    Search the Qdrant index. rag.mode picks the search: "semantic" uses the
    dense vectors only, "hybrid" fuses dense + BM25 keyword vectors (Qdrant
    does the fusion). The index always holds both, so either works without
    re-indexing. Same return format as the other retrievers.
    """
    mode = RetrievalMode.HYBRID if config["rag"]["mode"] == "hybrid" else RetrievalMode.DENSE
    logger.info(f"Retrieving top {k} chunks from Qdrant ({mode.value}) for: {query}")
    results = vector_store(mode).similarity_search_with_score(query, k=k)

    chunks = []
    for doc, score in results:
        meta = doc.metadata
        chunks.append({
            "content":    doc.page_content,
            "source":     meta["source"],
            "name":       meta["name"],
            "type":       meta["type"],
            "start_line": meta["start_line"],
            "end_line":   meta["end_line"],
            "distance":   score,
        })
    logger.info(f"Retrieved {len(chunks)} chunks")
    return chunks
