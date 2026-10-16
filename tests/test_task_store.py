import json
from types import SimpleNamespace

import pytest

from repopilot.tasks.task_store import SQLiteTaskStore


def make_plan(tasks: list[tuple[str, list[str]]], name: str = "demo"):
    """Build a minimal stand-in for planner.ExecutionPlan: [(task_id, depends_on), ...]."""
    plan_tasks = [
        SimpleNamespace(
            id=task_id,
            title=f"Task {task_id}",
            description="",
            task_type=SimpleNamespace(value="implement"),
            depends_on=deps,
            output_files=[],
            acceptance_criteria=[],
        )
        for task_id, deps in tasks
    ]
    return SimpleNamespace(
        project_name=name,
        tasks=plan_tasks,
        model_dump_json=lambda: json.dumps({"project_name": name}),
    )


@pytest.fixture
def store(tmp_path):
    return SQLiteTaskStore(str(tmp_path / "tasks.db"))


def statuses(store, project_id):
    return {t["id"]: t["status"] for t in store.get_all_tasks(project_id)}


def fail_permanently(store, task_id):
    for _ in range(3):  # max_retries = 3
        store.claim_task(task_id)
        store.fail_task(task_id, "boom")


def test_ready_tasks_respect_dependencies(store):
    pid = store.create_project("goal", make_plan([("a", []), ("b", ["a"])]))
    assert [t["id"] for t in store.get_ready_tasks(pid)] == ["a"]

    store.claim_task("a")
    store.complete_task("a", "done")
    assert [t["id"] for t in store.get_ready_tasks(pid)] == ["b"]


def test_failed_task_is_retried_then_failed(store):
    pid = store.create_project("goal", make_plan([("a", [])]))
    store.claim_task("a")
    store.fail_task("a", "boom")
    assert statuses(store, pid)["a"] == "pending"

    fail_permanently(store, "a")
    assert statuses(store, pid)["a"] == "failed"


def test_failure_blocks_whole_dependency_chain(store):
    # a ← b ← c, and d is independent
    pid = store.create_project("goal", make_plan([("a", []), ("b", ["a"]), ("c", ["b"]), ("d", [])]))
    fail_permanently(store, "a")

    assert store.block_unreachable_tasks(pid) == 2
    assert statuses(store, pid) == {"a": "failed", "b": "blocked", "c": "blocked", "d": "pending"}
    assert [t["id"] for t in store.get_ready_tasks(pid)] == ["d"]


def test_finished_project_is_not_resumed(store):
    pid = store.create_project("goal", make_plan([("a", [])]))
    assert store.get_latest_approved_project() == pid

    store.finish_project(pid, "completed")
    assert store.get_latest_approved_project() is None
    # /task_status can still show it
    assert store.get_latest_project() == pid
    assert store.get_project(pid)["status"] == "completed"


def test_latest_project_is_newest(store):
    first  = store.create_project("one", make_plan([("a1", [])]))
    second = store.create_project("two", make_plan([("b1", [])]))
    assert store.get_latest_project() == second
    store.finish_project(second, "abandoned")
    assert store.get_latest_approved_project() == first
