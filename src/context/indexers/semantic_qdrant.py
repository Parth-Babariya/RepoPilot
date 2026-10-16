from langchain_qdrant import QdrantVectorStore


from repopilot.context.indexers.qdrant_common import qdrant_collection_name
from repopilot.context.indexers import qdrant_common
from repopilot.observability.logger import get_logger


logger = get_logger(__name__)


def index_codebase(repo_path: str) -> QdrantVectorStore | None:
   """
   Incrementally index all source files in repo_path into Qdrant (dense vectors only).
   Only new or changed files are embedded — see qdrant_common.py.
   """
   logger.info(f"Starting semantic indexing of {repo_path}")
   return qdrant_common.index_codebase(repo_path, store_kwargs={})


def index_single_file(filepath: str) -> None:
   qdrant_common.index_single_file(filepath, store_kwargs={})


def remove_file_from_index(filepath: str) -> None:
   qdrant_common.remove_file_from_index(filepath)


def show_index(vector_store: QdrantVectorStore | None) -> None:
   """Display all documents stored in the Qdrant collection."""
   from rich.console import Console
   console = Console()
   if vector_store is None:
       console.print("[yellow]The index is empty.[/yellow]")
       return
   client = vector_store.client
   collection_name = qdrant_collection_name()
   results = client.scroll(collection_name=collection_name, with_payload=True, with_vectors=True, limit=1000)
   points = results[0]
   console.print(f"\n[bold]Semantic Index — {len(points)} chunks[/bold]\n")


   for i, point in enumerate(points):
       # LangChain stores Document.metadata under the "metadata" payload key
       content = point.payload.get("page_content", "")
       meta = point.payload.get("metadata", {})
       emb = point.vector
       console.print(f"[bold cyan]── Chunk {i+1} ──────────────────────────[/bold cyan]")
       console.print(f"  File     : {meta.get('source', 'unknown')}")
       console.print(f"  Name     : {meta.get('name', 'unknown')} ({meta.get('type', 'unknown')})")
       console.print(f"  Lines    : {meta.get('start_line')} - {meta.get('end_line')}")
       console.print(f"  Embedding: [{', '.join(f'{v:.4f}' for v in emb[:5])}...] ({len(emb)} dims)")
       console.print(f"  Code     :\n[dim]{content[:300]}[/dim]\n")
