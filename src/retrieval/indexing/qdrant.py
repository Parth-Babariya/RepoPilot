import hashlib
import os
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

from repopilot.config import config
from repopilot.retrieval.indexing.code_parser import parse_file, get_source_files
from repopilot.services.llm_service import embedding_slug, get_embedder
from repopilot.observability.logger import get_logger


logger = get_logger(__name__)

# ------------------------------------------------------------------
# Qdrant index — for a shared server or Qdrant Cloud (Chroma is the
# local default). One collection per repo and embedding model.
# ------------------------------------------------------------------
# Every point stores BOTH a dense vector (embeddings) and a sparse BM25
# vector, so the same index serves rag.mode semantic (dense search) and
# hybrid (dense + keyword) — switching mode never needs a re-index.
#
# Incremental, like chroma.py: every point stores its file's
# mtime; each run re-embeds only new or changed files and deletes points
# for files that are gone. LangChain stores Document.metadata under the
# "metadata" payload key, so filters target "metadata.source".
# ------------------------------------------------------------------

_BATCH_SIZE = 50
_SPARSE_VECTOR = "langchain-sparse"   # langchain_qdrant's default sparse vector name


def connection() -> dict:
    """QDRANT_URL / QDRANT_API_KEY from .env (unset URL → a local server on localhost:6333)."""
    return {"url": os.getenv("QDRANT_URL"), "api_key": os.getenv("QDRANT_API_KEY")}


def collection_name() -> str:
    """
    One collection per repo and embedding model:
    "<qdrant.collection_name>_<hash of the repo path>_<embedding model>".
    Qdrant is a shared server (unlike Chroma, which stores inside each repo's
    .repopilot/), so without this, indexing a second repo would treat the
    first repo's files as deleted and remove them.
    """
    repo_hash = hashlib.sha256(str(Path.cwd().resolve()).encode()).hexdigest()[:8]
    return f"{config['qdrant']['collection_name']}_{repo_hash}_{embedding_slug()}"


@lru_cache(maxsize=1)
def sparse_embedding() -> FastEmbedSparse:
    # Loading the BM25 model is slow — reuse it across watcher events.
    return FastEmbedSparse(model_name="Qdrant/bm25")


def vector_store(retrieval_mode: RetrievalMode = RetrievalMode.HYBRID) -> QdrantVectorStore:
    """Open the existing collection for searching or adding points."""
    return QdrantVectorStore.from_existing_collection(
        embedding=get_embedder(),
        sparse_embedding=sparse_embedding(),
        retrieval_mode=retrieval_mode,
        collection_name=collection_name(),
        **connection(),
    )


def _ensure_layout(client: QdrantClient, name: str) -> bool:
    """
    Return whether the collection exists with both vector kinds. Collections built
    by older versions had dense vectors only — the index is derived from the code,
    so such a collection is deleted and rebuilt rather than migrated.
    """
    if not client.collection_exists(name):
        return False
    sparse = client.get_collection(name).config.params.sparse_vectors or {}
    if _SPARSE_VECTOR in sparse:
        return True
    logger.warning(f"Qdrant collection {name} has no keyword vectors (older layout) — rebuilding it")
    client.delete_collection(name)
    return False


def _source_filter(filepath: str) -> models.Filter:
    return models.Filter(must=[
        models.FieldCondition(key="metadata.source", match=models.MatchValue(value=filepath))
    ])


def _get_indexed_mtimes(client: QdrantClient, name: str) -> dict[str, float]:
    """Scroll every point's metadata and return {filepath: mtime}."""
    mtimes: dict[str, float] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=name, with_payload=["metadata"], with_vectors=False, limit=256, offset=offset,
        )
        for point in points:
            meta   = (point.payload or {}).get("metadata", {})
            source = meta.get("source", "")
            mtime  = meta.get("mtime", 0.0)
            if source and mtime >= mtimes.get(source, 0.0):
                mtimes[source] = mtime
        if offset is None:
            return mtimes


def _delete_file_points(client: QdrantClient, name: str, filepath: str) -> None:
    client.delete(collection_name=name, points_selector=models.FilterSelector(filter=_source_filter(filepath)))
    logger.debug(f"Deleted points for {filepath}")


def _file_documents(filepath: str, mtime: float) -> list[Document]:
    try:
        chunks = parse_file(filepath)
    except (SyntaxError, ValueError) as e:
        logger.error(f"Skipping {filepath}: {e}")
        return []
    return [
        Document(
            page_content=chunk.content,
            metadata={
                "source":     chunk.source,
                "name":       chunk.name,
                "type":       chunk.type,
                "start_line": chunk.start_line,
                "end_line":   chunk.end_line,
                "mtime":      mtime,
            },
        )
        for chunk in chunks
    ]


def _add_documents(client: QdrantClient, docs: list[Document]) -> QdrantVectorStore | None:
    """Add docs, creating the collection (dense + sparse vectors) if it doesn't exist yet."""
    if not client.collection_exists(collection_name()):
        if not docs:
            return None
        return QdrantVectorStore.from_documents(
            docs, get_embedder(),
            sparse_embedding=sparse_embedding(),
            retrieval_mode=RetrievalMode.HYBRID,
            collection_name=collection_name(),
            batch_size=_BATCH_SIZE,
            **connection(),
        )
    store = vector_store()
    if docs:
        store.add_documents(docs, batch_size=_BATCH_SIZE)
    return store


def index_codebase(repo_path: str) -> QdrantVectorStore | None:
    """
    Incrementally index repo_path into Qdrant:
      new / changed files → (old points deleted) re-embedded
      deleted files       → points removed
      unchanged files     → skipped
    Returns None only when the repo has no indexable files and no collection exists.
    """
    name    = collection_name()
    client  = QdrantClient(**connection())
    exists  = _ensure_layout(client, name)

    current = {f: os.path.getmtime(f) for f in get_source_files(repo_path)}
    indexed = _get_indexed_mtimes(client, name) if exists else {}

    deleted  = set(indexed) - set(current)
    to_embed = [f for f, mtime in current.items() if indexed.get(f) != mtime]

    for filepath in deleted | (set(to_embed) & set(indexed)):
        _delete_file_points(client, name, filepath)

    docs  = [doc for f in to_embed for doc in _file_documents(f, current[f])]
    store = _add_documents(client, docs)

    changed = len(set(to_embed) & set(indexed))
    logger.info(
        f"Qdrant index updated — added: {len(to_embed) - changed}, changed: {changed}, "
        f"deleted: {len(deleted)}, skipped: {len(current) - len(to_embed)} | new chunks: {len(docs)}"
    )
    if store is None:
        logger.warning(f"Nothing to index in {repo_path}")
    return store


def index_single_file(filepath: str) -> None:
    """Re-index one file (called by the watcher on create/modify)."""
    if not os.path.exists(filepath):
        return
    client = QdrantClient(**connection())
    if _ensure_layout(client, collection_name()):
        _delete_file_points(client, collection_name(), filepath)
    docs = _file_documents(filepath, os.path.getmtime(filepath))
    _add_documents(client, docs)
    logger.info(f"Watchdog indexed {filepath} → {len(docs)} chunks")


def remove_file_from_index(filepath: str) -> None:
    """Remove all points for a deleted file (called by the watcher)."""
    client = QdrantClient(**connection())
    if client.collection_exists(collection_name()):
        _delete_file_points(client, collection_name(), filepath)
    logger.info(f"Watchdog removed {filepath} from index")


def show_index(store: QdrantVectorStore | None) -> None:
    """Display all chunks stored in the Qdrant collection (/show_index)."""
    from rich.console import Console
    console = Console()
    if store is None:
        console.print("[yellow]The index is empty.[/yellow]")
        return
    points, _ = store.client.scroll(collection_name=collection_name(), with_payload=True,
                                    with_vectors=False, limit=1000)
    console.print(f"\n[bold]Qdrant index — {len(points)} chunks[/bold]\n")
    for i, point in enumerate(points, 1):
        content = point.payload.get("page_content", "")
        meta = point.payload.get("metadata", {})
        console.print(f"[bold cyan]── Chunk {i} ──────────────────────────[/bold cyan]")
        console.print(f"  File     : {meta.get('source', 'unknown')}")
        console.print(f"  Name     : {meta.get('name', 'unknown')} ({meta.get('type', 'unknown')})")
        console.print(f"  Lines    : {meta.get('start_line')} - {meta.get('end_line')}")
        console.print(f"  Code     :\n[dim]{content[:300]}[/dim]\n")
