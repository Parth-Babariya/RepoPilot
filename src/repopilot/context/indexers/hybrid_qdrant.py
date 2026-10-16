from functools import lru_cache

from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse


from repopilot.config import config
from repopilot.context.indexers.qdrant_common import qdrant_collection_name
from repopilot.context.indexers import qdrant_common
from repopilot.observability.logger import get_logger


logger = get_logger(__name__)


RETRIEVAL_MODE_MAP = {
   "dense": RetrievalMode.DENSE,
   "sparse": RetrievalMode.SPARSE,
   "hybrid": RetrievalMode.HYBRID,
}



def _get_retrieval_mode() -> RetrievalMode:
   mode = config["vector_store"].get("retrieval_mode", "hybrid")
   return RETRIEVAL_MODE_MAP.get(mode, RetrievalMode.HYBRID)


@lru_cache(maxsize=1)
def _get_sparse_embedding() -> FastEmbedSparse:
   # Loading the BM25 model is slow — reuse it across watcher events.
   return FastEmbedSparse(model_name="Qdrant/bm25")


def _store_kwargs() -> dict:
   return {
       "sparse_embedding": _get_sparse_embedding(),
       "retrieval_mode": _get_retrieval_mode(),
   }


def index_codebase(repo_path: str) -> QdrantVectorStore | None:
   """
   Incrementally index all source files with dense + sparse vectors into Qdrant.
   Supports dense, sparse, and hybrid retrieval modes via config.
   Only new or changed files are embedded — see qdrant_common.py.
   """
   logger.info(f"Starting hybrid indexing of {repo_path}")
   return qdrant_common.index_codebase(repo_path, store_kwargs=_store_kwargs())


def index_single_file(filepath: str) -> None:
   qdrant_common.index_single_file(filepath, store_kwargs=_store_kwargs())


def remove_file_from_index(filepath: str) -> None:
   qdrant_common.remove_file_from_index(filepath)


def show_index(vector_store: QdrantVectorStore | None) -> None:
   from rich.console import Console
   console = Console()
   if vector_store is None:
       console.print("[yellow]The index is empty.[/yellow]")
       return
   client = vector_store.client
   collection_name = qdrant_collection_name()
   results = client.scroll(collection_name=collection_name, with_payload=True, with_vectors=False, limit=1000)
   points = results[0]
   console.print(f"\n[bold]Hybrid Index — {len(points)} chunks[/bold]\n")


   for i, point in enumerate(points):
       content = point.payload.get("page_content", "")
       meta = point.payload.get("metadata", {})
       console.print(f"[bold cyan]── Chunk {i+1} ──────────────────────────[/bold cyan]")
       console.print(f"  File     : {meta.get('source', 'unknown')}")
       console.print(f"  Name     : {meta.get('name', 'unknown')} ({meta.get('type', 'unknown')})")
       console.print(f"  Lines    : {meta.get('start_line')} - {meta.get('end_line')}")
       console.print(f"  Code     :\n[dim]{content[:300]}[/dim]\n")
