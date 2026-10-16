import asyncio
import sys
import types

import pytest

from repopilot.tasks.task_store import SQLiteTaskStore
from tests.test_task_store import make_plan, statuses


@pytest.fixture
def orchestrator_module(monkeypatch):
    """
    Import tasks.orchestrator with its LLM-backed dependencies stubbed out,
    so the execution loop can be tested without API keys or network access.
    """
    stubs = {
        "repopilot.tasks.executor":           {"run_subtask_agent": None},
        "repopilot.tasks.planner":            {"create_plan": None},
        "repopilot.tasks.approval":           {"present_plan_for_approval": None},
        "repopilot.context.indexers.factory": {"get_indexer": None},
    }
    for name, attrs in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "repopilot.tasks.orchestrator", raising=False)

    import repopilot.tasks.orchestrator as orchestrator
    return orchestrator


def run(orchestrator, store, pid, failing: set[str]):
    async def fake_agent(task, dep_outputs=None):
        if task["id"] in failing:
            raise RuntimeError("agent error")
        return f"built {task['id']}"

    orchestrator.run_subtask_agent = fake_agent
    asyncio.run(orchestrator.TaskOrchestrator(store).run(pid))


def test_successful_run_marks_project_completed(orchestrator_module, tmp_path):
    store = SQLiteTaskStore(str(tmp_path / "tasks.db"))
    pid = store.create_project("goal", make_plan([("a", []), ("b", ["a"])]))

    run(orchestrator_module, store, pid, failing=set())

    assert statuses(store, pid) == {"a": "completed", "b": "completed"}
    assert store.get_project(pid)["status"] == "completed"
    assert store.get_latest_approved_project() is None


def test_failed_dependency_ends_run_and_blocks_dependents(orchestrator_module, tmp_path):
    store = SQLiteTaskStore(str(tmp_path / "tasks.db"))
    pid = store.create_project("goal", make_plan([("a", []), ("b", ["a"]), ("c", [])]))

    run(orchestrator_module, store, pid, failing={"a"})

    assert statuses(store, pid) == {"a": "failed", "b": "blocked", "c": "completed"}
    assert store.get_project(pid)["status"] == "failed"
    assert store.get_latest_approved_project() is None
