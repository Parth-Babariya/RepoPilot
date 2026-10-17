from repopilot.config import config
from repopilot.observability.logger import get_logger
from repopilot.observability.usage import track


logger = get_logger(__name__)


# The symbol / call-graph index (context/symbol_index.py) is kept in sync
# alongside the vector index, so every caller of get_indexer() and
# get_file_indexer() — startup, /reindex, after /plan, the watcher —
# updates both. A symbol-index failure is logged but never breaks
# vector indexing.
#
# Only the store (Chroma or Qdrant) changes how code is indexed; rag.mode
# only changes how it is searched, so it isn't consulted here.

def _store_module():
   if config["vector_store"]["provider"] == "qdrant":
       from . import qdrant
       return qdrant
   from . import chroma
   return chroma


def _index_changed() -> None:
   # The in-memory BM25 keyword index (hybrid search on Chroma) is rebuilt on next use
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
   store = _store_module()

   def index(repo_path: str):
       from repopilot.context.symbol_index import update_symbol_index
       with track("index"):
           result = store.index_codebase(repo_path)
       _index_changed()
       _sync_symbols(update_symbol_index, repo_path)
       return result

   return index


def get_file_indexer():
   """
   Return (index_file, remove_file) for the configured vector store —
   used by the watcher for real-time per-file updates.
   """
   store = _store_module()

   def index_file(filepath: str) -> None:
       from repopilot.context.symbol_index import get_symbol_store
       with track("index"):
           store.index_single_file(filepath)
       _index_changed()
       _sync_symbols(lambda: get_symbol_store().index_file(filepath))

   def remove_file(filepath: str) -> None:
       from repopilot.context.symbol_index import get_symbol_store
       store.remove_file_from_index(filepath)
       _index_changed()
       _sync_symbols(lambda: get_symbol_store().remove_file(filepath))

   return index_file, remove_file


def get_index_inspector():
   """Return the show_index function (/show_index) for the configured vector store."""
   return _store_module().show_index
