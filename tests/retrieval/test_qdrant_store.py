"""
Integration test for the Qdrant store — runs only when a Qdrant server is reachable
at localhost:6333 (or QDRANT_URL); skipped otherwise, e.g. in CI.
Start one with: docker run -p 6333:6333 qdrant/qdrant
"""

import os
import socket
from urllib.parse import urlparse

import pytest
from repopilot.config import config


def _server_up() -> bool:
    url = urlparse(os.getenv("QDRANT_URL") or "http://localhost:6333")
    try:
        with socket.create_connection((url.hostname, url.port or 6333), timeout=1):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _server_up(), reason="no Qdrant server running")


@pytest.fixture
def qdrant_project(tmp_path, monkeypatch):
    """A small repo in a temp folder, indexed into its own Qdrant collection."""
    from qdrant_client import QdrantClient
    from repopilot.retrieval.indexing import qdrant

    (tmp_path / "billing.py").write_text(
        "def calculate_invoice_total(items):\n    return sum(i.price * i.quantity for i in items)\n"
    )
    (tmp_path / "users.py").write_text(
        "def register_user(email, password):\n    return {'email': email, 'hash': hash(password)}\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(config["vector_store"], "provider", "qdrant")
    monkeypatch.setitem(config["qdrant"], "collection_name", "repopilot_test")

    client = QdrantClient(**qdrant.connection())
    yield tmp_path, qdrant, client
    if client.collection_exists(qdrant.collection_name()):
        client.delete_collection(qdrant.collection_name())


def search(query: str, mode: str, monkeypatch) -> list[str]:
    from repopilot.retrieval.search.qdrant import retrieve

    monkeypatch.setitem(config["rag"], "mode", mode)
    return [os.path.basename(c["source"]) for c in retrieve(query, k=2)]


def test_one_index_serves_both_search_modes(qdrant_project, monkeypatch):
    root, qdrant, _ = qdrant_project
    qdrant.index_codebase(str(root))

    for mode in ("semantic", "hybrid"):  # no re-index between modes
        assert search("invoice total price", mode, monkeypatch)[0] == "billing.py"
        assert search("register_user", mode, monkeypatch)[0] == "users.py"


def test_watcher_updates_and_removals(qdrant_project, monkeypatch):
    root, qdrant, _ = qdrant_project
    qdrant.index_codebase(str(root))

    (root / "shipping.py").write_text("def shipping_cost(weight_kg):\n    return 5 + 2 * weight_kg\n")
    qdrant.index_single_file(str(root / "shipping.py"))
    assert search("shipping_cost weight", "hybrid", monkeypatch)[0] == "shipping.py"

    qdrant.remove_file_from_index(str(root / "shipping.py"))
    assert "shipping.py" not in search("shipping_cost weight", "hybrid", monkeypatch)


def test_old_dense_only_collection_is_rebuilt(qdrant_project, monkeypatch):
    from qdrant_client import models

    root, qdrant, client = qdrant_project
    # layout written by older versions: dense vectors only, no keyword vectors
    client.create_collection(
        qdrant.collection_name(), vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )

    qdrant.index_codebase(str(root))

    sparse = client.get_collection(qdrant.collection_name()).config.params.sparse_vectors
    assert sparse and "langchain-sparse" in sparse
    assert search("invoice total", "hybrid", monkeypatch)[0] == "billing.py"
