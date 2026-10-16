import hashlib
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from repopilot.config import config
from repopilot.context.indexers.code_parser import parse_file, get_source_files
from repopilot.llm.factory import embedding_slug, get_embedder
from repopilot.observability.logger import get_logger


logger = get_logger(__name__)

# ------------------------------------------------------------------
# Incremental indexing shared by semantic_qdrant.py and hybrid_qdrant.py
# ------------------------------------------------------------------
# Same idea as semantic_chroma.py: every point stores the file's mtime
# in its metadata. On each index_codebase() call we compare stored vs
# on-disk mtimes and only re-embed files that are new or changed, and
# delete points for files that no longer exist.
#
# LangChain stores Document.metadata under the "metadata" payload key,
# so filters target "metadata.source".
#
# store_kwargs carries what differs between the two modes — e.g. the
# sparse embedding and retrieval mode for hybrid search.
# ------------------------------------------------------------------

_BATCH_SIZE = 50


def _connection() -> dict:
    return {"url": os.getenv("QDRANT_URL"), "api_key": os.getenv("QDRANT_API_KEY")}


def qdrant_collection_name() -> str:
    """
    One collection per repo and embedding model:
    "<qdrant.collection_name>_<hash of the repo path>_<embedding model>".
    Qdrant is a shared server (unlike Chroma, which stores inside each repo's
    .repopilot/), so without this, indexing a second repo would treat the
    first repo's files as deleted and remove them.
    """
    repo_hash = hashlib.sha256(str(Path.cwd().resolve()).encode()).hexdigest()[:8]
    return f"{config['qdrant']['collection_name']}_{repo_hash}_{embedding_slug()}"


_collection_name = qdrant_collection_name


def _source_filter(filepath: str) -> models.Filter:
    return models.Filter(must=[
        models.FieldCondition(key="metadata.source", match=models.MatchValue(value=filepath))
    ])


def _get_indexed_mtimes(client: QdrantClient, collection_name: str) -> dict[str, float]:
    """Scroll every point's metadata and return {filepath: mtime}. Points indexed
    before mtimes were stored count as 0.0, so those files get re-embedded once."""
    mtimes: dict[str, float] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            with_payload=["metadata"],
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        for point in points:
            meta   = (point.payload or {}).get("metadata", {})
            source = meta.get("source", "")
            mtime  = meta.get("mtime", 0.0)
            if source and mtime >= mtimes.get(source, 0.0):
                mtimes[source] = mtime
        if offset is None:
            return mtimes


def _delete_file_points(client: QdrantClient, collection_name: str, filepath: str) -> None:
    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(filter=_source_filter(filepath)),
    )
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


def _add_documents(client: QdrantClient, docs: list[Document], store_kwargs: dict) -> QdrantVectorStore | None:
    """Add docs, creating the collection first if it doesn't exist yet."""
    collection_name = _collection_name()
    embedder        = get_embedder()

    if not client.collection_exists(collection_name):
        if not docs:
            return None
        return QdrantVectorStore.from_documents(
            docs, embedder,
            collection_name=collection_name,
            batch_size=_BATCH_SIZE,
            **_connection(),
            **store_kwargs,
        )

    store = QdrantVectorStore.from_existing_collection(
        embedding=embedder,
        collection_name=collection_name,
        **_connection(),
        **store_kwargs,
    )
    if docs:
        store.add_documents(docs, batch_size=_BATCH_SIZE)
    return store


def index_codebase(repo_path: str, store_kwargs: dict) -> QdrantVectorStore | None:
    """
    Incrementally index repo_path into Qdrant:
      new / changed files → (old points deleted) re-embedded
      deleted files       → points removed
      unchanged files     → skipped
    Returns None only when the repo has no indexable files and no collection exists.
    """
    collection_name = _collection_name()
    client          = QdrantClient(**_connection())
    exists          = client.collection_exists(collection_name)

    current = {f: os.path.getmtime(f) for f in get_source_files(repo_path)}
    indexed = _get_indexed_mtimes(client, collection_name) if exists else {}

    deleted  = set(indexed) - set(current)
    to_embed = [f for f, mtime in current.items() if indexed.get(f) != mtime]

    for filepath in deleted | (set(to_embed) & set(indexed)):
        _delete_file_points(client, collection_name, filepath)

    docs = [doc for f in to_embed for doc in _file_documents(f, current[f])]
    store = _add_documents(client, docs, store_kwargs)

    changed = len(set(to_embed) & set(indexed))
    logger.info(
        f"Qdrant index updated — added: {len(to_embed) - changed}, changed: {changed}, "
        f"deleted: {len(deleted)}, skipped: {len(current) - len(to_embed)} | new chunks: {len(docs)}"
    )
    if store is None:
        logger.warning(f"Nothing to index in {repo_path}")
    return store


def index_single_file(filepath: str, store_kwargs: dict) -> None:
    """Re-index one file (called by the watcher on create/modify)."""
    if not os.path.exists(filepath):
        return
    client = QdrantClient(**_connection())
    if client.collection_exists(_collection_name()):
        _delete_file_points(client, _collection_name(), filepath)
    docs = _file_documents(filepath, os.path.getmtime(filepath))
    _add_documents(client, docs, store_kwargs)
    logger.info(f"Watchdog indexed {filepath} → {len(docs)} chunks")


def remove_file_from_index(filepath: str) -> None:
    """Remove all points for a deleted file (called by the watcher)."""
    client = QdrantClient(**_connection())
    if client.collection_exists(_collection_name()):
        _delete_file_points(client, _collection_name(), filepath)
    logger.info(f"Watchdog removed {filepath} from index")
