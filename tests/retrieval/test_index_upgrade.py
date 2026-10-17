from repopilot.retrieval.indexing import chroma
from repopilot.retrieval.indexing.code_parser import CHUNKER_VERSION


class FakeCollection:
    def __init__(self, metadatas):
        self.metadatas = metadatas

    def get(self, include):
        return {"metadatas": self.metadatas}


def test_files_chunked_by_an_older_chunker_are_reindexed():
    collection = FakeCollection([
        {"source": "new.py", "mtime": 5.0, "chunker": CHUNKER_VERSION},
        {"source": "new.py", "mtime": 5.0, "chunker": CHUNKER_VERSION},
        {"source": "old.py", "mtime": 7.0},                     # indexed before chunker versions
        {"source": "mixed.py", "mtime": 9.0, "chunker": CHUNKER_VERSION},
        {"source": "mixed.py", "mtime": 9.0, "chunker": CHUNKER_VERSION - 1},
    ])
    # 0.0 never equals a real mtime, so index_codebase deletes and re-chunks those files
    assert chroma._get_indexed_mtimes(collection) == {"new.py": 5.0, "old.py": 0.0, "mixed.py": 0.0}
