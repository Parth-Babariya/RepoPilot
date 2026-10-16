from repopilot.config import config
from repopilot.context.retrievers.reranker import rerank_config, with_reranking
from repopilot.observability.logger import get_logger


logger = get_logger(__name__)




def get_retriever():
   """
   Return retrieve(query, k) for the configured vector store. When
   rerank.enabled is true, it over-fetches and reranks with a cross-encoder
   (see reranker.py) — used by both /ask and /eval.
   """
   mode = config["rag"]["mode"]
   provider = config["vector_store"]["provider"]


   if mode == "hybrid" and provider == "qdrant":
       from .hybrid_qdrant import retrieve
   elif provider == "qdrant":
       from .semantic_qdrant import retrieve
   elif mode == "hybrid":
       from .hybrid_chroma import retrieve      # vector + BM25 keyword search, fused
   else:
       from .semantic_chroma import retrieve


   if rerank_config()["enabled"]:
       return with_reranking(retrieve)
   return retrieve
