from repopilot.config import config
from repopilot.retrieval.search.reranker import rerank_config, with_reranking
from repopilot.observability.logger import get_logger


logger = get_logger(__name__)




def get_retriever():
   """
   Return retrieve(query, k) for the configured vector store. When
   rerank.enabled is true, it over-fetches and reranks with a cross-encoder
   (see reranker.py) — used by both /ask and /eval.
   """
   # The store decides *where* to search; each store module reads rag.mode
   # itself to decide *how* (vector only, or vector + BM25 keyword search).
   if config["vector_store"]["provider"] == "qdrant":
       from .qdrant import retrieve
   else:
       from .chroma import retrieve


   if rerank_config()["enabled"]:
       return with_reranking(retrieve)
   return retrieve
