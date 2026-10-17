import json
import sqlite3
from types import SimpleNamespace

import pytest

from repopilot.tasks.task_store import SQLiteTaskStore


def make_task(task_id: str, deps: list[str], verify: str = ""):
    """Minimal stand-in for planner.PlannedTask."""
    return SimpleNamespace(
        id=task_id,
        title=f"Task {task_id}",
        description="",
        task_type=SimpleNamespace(value="implement"),
        depends_on=deps,
        output_files=[],
        acceptance_criteria=[],
        verify_command=verify,
    )


def make_plan(tasks: list[tuple[str, list[str]]], name: str = "demo"):
    """Build a minimal stand-in for planner.ExecutionPlan: [(task_id, depends_on), ...]."""
    return SimpleNamespace(
        project_name=name,
        tasks=[make_task(task_id, deps) for task_id, deps in tasks],
        model_dump_json=lambda: json.dumps({"project_name": name}),
    )


@pytest.fixture
def store(tmp_path):
    return SQLiteTaskStore(str(tmp_path / "tasks.db"))


def statuses(store, project_id):
    return {t["id"]: t["status"] for t in store.get_all_tasks(project_id)}


def fail_permanently(store, project_id, task_id):
    for _ in range(3):  # max_retries = 3
        store.claim_task(project_id, task_id)
        store.fail_task(project_id, task_id, "boom")


def test_ready_tasks_respect_dependencies(store):
    pid = store.create_project("goal", make_plan([("a", []), ("b", ["a"])]))
    assert [t["id"] for t in store.get_ready_tasks(pid)] == ["a"]

    store.claim_task(pid, "a")
    store.complete_task(pid, "a", "done")
    assert [t["id"] for t in store.get_ready_tasks(pid)] == ["b"]


def test_missing_dependency_never_counts_as_done(store):
    pid = store.create_project("goal", make_plan([("a", ["typo"])]))
    assert store.get_ready_tasks(pid) == []


def test_failed_task_is_retried_then_failed(store):
    pid = store.create_project("goal", make_plan([("a", [])]))
    store.claim_task(pid, "a")
    store.fail_task(pid, "a", "boom")
    task = store.get_all_tasks(pid)[0]
    assert (task["status"], task["retry_count"], task["error"]) == ("pending", 1, "boom")

    fail_permanently(store, pid, "a")
    assert statuses(store, pid)["a"] == "failed"


def test_failure_blocks_whole_dependency_chain(store):
    # a ← b ← c, and d is independent
    pid = store.create_project("goal", make_plan([("a", []), ("b", ["a"]), ("c", ["b"]), ("d", [])]))
    fail_permanently(store, pid, "a")

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


def test_two_projects_can_reuse_task_ids(store):
    # every plan numbers its tasks task_001…; that used to fail with UNIQUE constraint
    first  = store.create_project("one", make_plan([("task_001", [])]))
    second = store.create_project("two", make_plan([("task_001", [])]))

    store.claim_task(second, "task_001")
    store.complete_task(second, "task_001", "done")
    assert statuses(store, first)  == {"task_001": "pending"}
    assert statuses(store, second) == {"task_001": "completed"}


def test_verify_command_is_stored(store):
    plan = make_plan([])
    plan.tasks = [make_task("a", [], verify="  python -m pytest -q  ")]
    pid = store.create_project("goal", plan)
    assert store.get_all_tasks(pid)[0]["verify_command"] == "python -m pytest -q"


def test_apply_replan_replaces_tasks_and_appends_new_ones(store):
    pid = store.create_project("goal", make_plan([("a", []), ("b", ["a"]), ("c", [])]))
    fail_permanently(store, pid, "a")

    store.apply_replan(pid, ["a", "b"], [make_task("task_101", []), make_task("task_102", ["task_101"])], "split a")

    assert statuses(store, pid) == {"a": "replaced", "b": "replaced", "c": "pending",
                                    "task_101": "pending", "task_102": "pending"}
    assert [t["id"] for t in store.get_all_tasks(pid)][-2:] == ["task_101", "task_102"]   # appended in order
    assert store.block_unreachable_tasks(pid) == 0   # replaced tasks don't block anything


def test_old_database_is_migrated(tmp_path):
    # schema from before per-project task ids (id alone was the primary key)
    db = tmp_path / "tasks.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE projects (id TEXT PRIMARY KEY, name TEXT NOT NULL, goal TEXT NOT NULL,
                               plan_json TEXT, status TEXT, created_at REAL, approved_at REAL);
        CREATE TABLE tasks (id TEXT PRIMARY KEY, project_id TEXT, title TEXT NOT NULL,
                            description TEXT NOT NULL, task_type TEXT NOT NULL, status TEXT DEFAULT 'pending',
                            depends_on TEXT DEFAULT '[]', output_files TEXT DEFAULT '[]',
                            acceptance_criteria TEXT DEFAULT '[]', result TEXT, error TEXT,
                            retry_count INTEGER DEFAULT 0, max_retries INTEGER DEFAULT 3,
                            execution_order INTEGER DEFAULT 0, created_at REAL, started_at REAL, completed_at REAL);
        CREATE INDEX idx_tasks_project ON tasks(project_id, status);
        INSERT INTO projects VALUES ('p1', 'old', 'goal', '{}', 'completed', 1, NULL);
        INSERT INTO tasks (id, project_id, title, description, task_type, status)
               VALUES ('task_001', 'p1', 'Old task', '', 'implement', 'completed');
    """)
    conn.close()

    store = SQLiteTaskStore(str(db))
    assert statuses(store, "p1") == {"task_001": "completed"}            # old rows kept
    new = store.create_project("new goal", make_plan([("task_001", [])]))  # same id now allowed
    assert statuses(store, new) == {"task_001": "pending"}


def test_every_task_status_has_a_display_style():
    from repopilot.tasks.status import _STATUS_STYLE
    from repopilot.tasks.task_store import TaskStatus
    assert set(_STATUS_STYLE) == set(TaskStatus)


def test_project_records_when_its_plan_was_approved(store):
    pid = store.create_project("goal", make_plan([("a", [])]))
    with store._conn() as conn:
        row = conn.execute("SELECT approved_at, plan_json FROM projects WHERE id=?", (pid,)).fetchone()
    assert row["approved_at"] is not None and row["plan_json"]
