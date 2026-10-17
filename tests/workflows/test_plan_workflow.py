import asyncio
import sys
import types

import pytest

from repopilot.agents.planner_agent import PlannedTask, RevisedPlan
from repopilot.agents.task_spec import TaskType
from repopilot.workflows.task_store import SQLiteTaskStore
from tests.workflows.test_task_store import make_plan, statuses


@pytest.fixture
def orchestrator_module(monkeypatch):
    """
    Import tasks.orchestrator with its LLM-backed dependencies stubbed out,
    so the execution loop can be tested without API keys or network access.
    (The planner is imported for real; tests replace its replan() call.)
    """
    stubs = {
        "repopilot.workflows.task_execution":           {"execute_task": None},
        "repopilot.workflows.approval":           {"present_plan_for_approval": None},
        "repopilot.retrieval.indexing.factory": {"get_indexer": None},
    }
    for name, attrs in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "repopilot.workflows.plan_workflow", raising=False)

    import repopilot.workflows.plan_workflow as orchestrator
    return orchestrator


def run(orchestrator, store, pid, failing: set[str], max_replans: int = 0):
    async def fake_agent(task, dep_outputs=None):
        if task["id"] in failing:
            raise RuntimeError("agent error")
        return f"built {task['id']}"

    orchestrator.execute_task = fake_agent
    asyncio.run(orchestrator.TaskOrchestrator(store, max_replans=max_replans).run(pid))


def planned(task_id: str, deps: list[str]) -> PlannedTask:
    return PlannedTask(id=task_id, title=task_id, description="", task_type=TaskType.IMPLEMENT,
                       depends_on=deps, estimated_minutes=5, output_files=[], acceptance_criteria=[])


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


def test_approved_replan_replaces_failed_branch_and_finishes(orchestrator_module, tmp_path):
    store = SQLiteTaskStore(str(tmp_path / "tasks.db"))
    pid = store.create_project("goal", make_plan([("a", []), ("b", ["a"]), ("c", [])]))
    seen = {}

    def fake_replan(goal, completed, failed, affected, unaffected, existing_ids):
        seen.update(failed=failed["id"], affected=[t["id"] for t in affected])
        return RevisedPlan(reason="split a", tasks=[planned("task_101", []), planned("task_102", ["task_101"])])

    orchestrator_module.replan = fake_replan
    orchestrator_module.present_plan_for_approval = lambda plan, **kw: plan   # user approves

    run(orchestrator_module, store, pid, failing={"a"}, max_replans=1)

    assert seen == {"failed": "a", "affected": ["b"]}          # only the failed branch is replaced
    assert statuses(store, pid) == {"a": "replaced", "b": "replaced", "c": "completed",
                                    "task_101": "completed", "task_102": "completed"}
    assert store.get_project(pid)["status"] == "completed"


def test_rejected_replan_falls_back_to_blocking(orchestrator_module, tmp_path):
    store = SQLiteTaskStore(str(tmp_path / "tasks.db"))
    pid = store.create_project("goal", make_plan([("a", []), ("b", ["a"])]))

    orchestrator_module.replan = lambda *a, **kw: RevisedPlan(reason="r", tasks=[planned("task_101", [])])
    orchestrator_module.present_plan_for_approval = lambda plan, **kw: None   # user rejects

    run(orchestrator_module, store, pid, failing={"a"}, max_replans=1)

    assert statuses(store, pid) == {"a": "failed", "b": "blocked"}
    assert store.get_project(pid)["status"] == "failed"


def test_invalid_replan_is_not_applied(orchestrator_module, tmp_path):
    store = SQLiteTaskStore(str(tmp_path / "tasks.db"))
    pid = store.create_project("goal", make_plan([("a", []), ("b", ["a"])]))

    # depends on "b", which is being replaced — invalid
    orchestrator_module.replan = lambda *a, **kw: RevisedPlan(reason="r", tasks=[planned("task_101", ["b"])])
    orchestrator_module.present_plan_for_approval = lambda plan, **kw: pytest.fail("invalid plan was shown")

    run(orchestrator_module, store, pid, failing={"a"}, max_replans=1)

    assert statuses(store, pid) == {"a": "failed", "b": "blocked"}


def test_no_replan_when_budget_is_zero(orchestrator_module, tmp_path):
    store = SQLiteTaskStore(str(tmp_path / "tasks.db"))
    pid = store.create_project("goal", make_plan([("a", [])]))
    orchestrator_module.replan = lambda *a, **kw: pytest.fail("replan called with max_replans=0")

    run(orchestrator_module, store, pid, failing={"a"}, max_replans=0)

    assert statuses(store, pid) == {"a": "failed"}
