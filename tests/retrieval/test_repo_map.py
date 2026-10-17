import pytest

from repopilot.retrieval import repo_map
from repopilot.retrieval.repo_map import build_repo_map
from repopilot.retrieval.symbol_index import SymbolStore


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "__init__.py").write_text("")
    (tmp_path / "app" / "store.py").write_text(
        "class TaskStore:\n"
        "    def claim(self):\n        pass\n"
        "    def _connect(self):\n        pass\n"
        "\n"
        "def open_store():\n    pass\n"
        "\n"
        "def _helper():\n    pass\n"
    )
    (tmp_path / "tests").mkdir()
    for name in ("test_store.py", "test_cli.py"):
        (tmp_path / "tests" / name).write_text("def test_x():\n    pass\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'app'\n")

    store = SymbolStore(str(tmp_path / ".repopilot" / "symbols.db"))
    for f in ("app/store.py", "tests/test_store.py", "tests/test_cli.py"):
        store.index_file(str(tmp_path / f))
    monkeypatch.setattr(repo_map, "get_symbol_store", lambda: store)
    return tmp_path


def test_outline_shows_public_symbols_and_groups_tests(repo):
    text = build_repo_map(repo)
    assert "project files: pyproject.toml" in text
    assert "app/store.py" in text and "TaskStore(claim), open_store" in text
    assert "_connect" not in text and "_helper" not in text        # private helpers left out
    assert "app/__init__.py" not in text                            # empty file says nothing
    assert "tests/  (2 test files)" in text and "test_cli.py" not in text


def test_detail_is_dropped_until_it_fits(repo):
    full = build_repo_map(repo)
    no_methods = build_repo_map(repo, max_chars=len(full) - 1)        # methods go first
    assert "TaskStore, open_store" in no_methods and "(claim)" not in no_methods
    files_only = build_repo_map(repo, max_chars=len(no_methods) - 1)  # then all symbols
    assert "app/store.py" in files_only and "TaskStore" not in files_only
    assert len(build_repo_map(repo, max_chars=60)) <= 60              # then folders, cut to fit


def test_empty_folder(tmp_path):
    assert "no source files" in build_repo_map(tmp_path)
