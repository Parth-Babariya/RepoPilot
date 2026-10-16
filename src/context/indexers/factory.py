from repopilot.config import config
from repopilot.observability.logger import get_logger
from repopilot.observability.usage import track


logger = get_logger(__name__)


# The symbol / call-graph index (context/symbol_index.py) is kept in sync
# alongside the vector index, so every caller of get_indexer() and
# get_file_indexer() — startup, /reindex, after /plan, the watcher —
# updates both. A symbol-index failure is logged but never breaks
# vector indexing.

def _index_changed() -> None:
   # The in-memory BM25 keyword index (hybrid search) is rebuilt on next use
   from repopilot.context.retrievers import bm25
   bm25.invalidate()


def _sync_symbols(action, *args) -> None:
   from repopilot.context.symbol_index import symbols_enabled
   if not symbols_enabled():
       return
   try:
       action(*args)
   except Exception as e:
       logger.error(f"Symbol index update failed: {e}")


def get_indexer():
   mode = config["rag"]["mode"]
   provider = config["vector_store"]["provider"]


   if mode == "hybrid" and provider == "qdrant":
       from .hybrid_qdrant import index_codebase
   elif provider == "qdrant":
       from .semantic_qdrant import index_codebase
   else:
       from .semantic_chroma import index_codebase


   def index(repo_path: str):
       from repopilot.context.symbol_index import update_symbol_index
       with track("index"):
           result = index_codebase(repo_path)
       _index_changed()
       _sync_symbols(update_symbol_index, repo_path)
       return result


   return index


def get_file_indexer():
   """
   Return (index_single_file, remove_file_from_index) for the configured
   vector store — used by the watcher for real-time per-file updates.
   """
   mode = config["rag"]["mode"]
   provider = config["vector_store"]["provider"]


   if mode == "hybrid" and provider == "qdrant":
       from .hybrid_qdrant import index_single_file, remove_file_from_index
   elif provider == "qdrant":
       from .semantic_qdrant import index_single_file, remove_file_from_index
   else:
       from .semantic_chroma import index_single_file, remove_file_from_index


   def index_file(filepath: str) -> None:
       from repopilot.context.symbol_index import get_symbol_store
       with track("index"):
           index_single_file(filepath)
       _index_changed()
       _sync_symbols(lambda: get_symbol_store().index_file(filepath))


   def remove_file(filepath: str) -> None:
       from repopilot.context.symbol_index import get_symbol_store
       remove_file_from_index(filepath)
       _index_changed()
       _sync_symbols(lambda: get_symbol_store().remove_file(filepath))


   return index_file, remove_file


def get_index_inspector():
   """Return the right show_index function based on rag mode and vector_store in config."""
   mode = config["rag"]["mode"]
   provider = config["vector_store"]["provider"]


   if mode == "hybrid" and provider == "qdrant":
       from .hybrid_qdrant import show_index
   elif provider == "qdrant":
       from .semantic_qdrant import show_index
   else:
       from .semantic_chroma import show_index


   return show_index
