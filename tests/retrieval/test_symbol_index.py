import os
import textwrap

import pytest

from repopilot.retrieval import symbol_index
from repopilot.retrieval.symbol_index import SymbolStore, extract, update_symbol_index

PYTHON_SOURCE = textwrap.dedent("""
    import json

    class Store:
        def fail_task(self, task_id):
            self._log(task_id)

        def _log(self, msg):
            print(msg)

    class Orchestrator:
        def execute(self, task):
            self.store.fail_task(task["id"])
            json.dumps(task)

    def run():
        Orchestrator().execute({})

    run()
""")


def test_extract_python_symbols_and_calls():
    symbols, calls = extract(PYTHON_SOURCE, "python")

    assert {s.qualname for s in symbols} == {
        "Store", "Store.fail_task", "Store._log", "Orchestrator", "Orchestrator.execute", "run",
    }
    kinds = {s.qualname: s.kind for s in symbols}
    assert kinds["Store"] == "class" and kinds["run"] == "function"

    pairs = {(c.caller, c.callee) for c in calls}
    assert ("Orchestrator.execute", "fail_task") in pairs   # self.store.fail_task → fail_task
    assert ("Store.fail_task", "_log") in pairs
    assert ("run", "Orchestrator") in pairs
    assert ("<module>", "run") in pairs


def test_extract_javascript_arrow_functions_and_methods():
    source = "const handler = () => { doThing(1); };\nclass A { run() { this.helper.go(); } }"
    symbols, calls = extract(source, "javascript")
    assert {s.qualname for s in symbols} == {"handler", "A", "A.run"}
    assert {(c.caller, c.callee) for c in calls} == {("handler", "doThing"), ("A.run", "go")}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "app.py").write_text(PYTHON_SOURCE)
    (tmp_path / "other.py").write_text("def fail_task():\n    pass\n")
    (tmp_path / "notes.md").write_text("# not code\n")
    store = SymbolStore(str(tmp_path / ".repopilot" / "symbols.db"))
    monkeypatch.setattr(symbol_index, "_store", store)
    return tmp_path, store


def test_index_and_query(repo):
    path, store = repo
    update_symbol_index(str(path))

    assert store.stats()["files"] == 2   # .md files have no symbols

    definitions = store.find_definitions("fail_task")
    assert {d["qualname"] for d in definitions} == {"Store.fail_task", "fail_task"}   # ambiguous name
    assert [d["qualname"] for d in store.find_definitions("Store.fail_task")] == ["Store.fail_task"]

    assert [c["caller"] for c in store.find_callers("Store.fail_task")] == ["Orchestrator.execute"]

    callees = {c["callee"]: c["in_repo"] for c in store.find_callees("execute")}
    assert callees == {"fail_task": True, "dumps": False}


def test_incremental_update_handles_changes_and_deletes(repo):
    path, store = repo
    update_symbol_index(str(path))

    (path / "other.py").unlink()
    app = path / "app.py"
    app.write_text("def renamed():\n    pass\n")
    os.utime(app, (os.path.getmtime(app) + 10,) * 2)   # make sure the mtime changes
    update_symbol_index(str(path))

    assert store.find_definitions("fail_task") == []
    assert [d["qualname"] for d in store.find_definitions("renamed")] == ["renamed"]
