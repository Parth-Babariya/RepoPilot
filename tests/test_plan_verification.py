import json
import sys

import pytest

from repopilot.tasks.executor import _build_system_prompt, verify_task
from repopilot.tasks.planner import PlannedTask, RevisedPlan, next_task_id, validate_revision
from repopilot.tasks.task_store import TaskType
from repopilot.tools import safety


@pytest.fixture(autouse=True)
def in_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    safety.set_confirm_handler(None)
    return tmp_path


def task(**overrides) -> dict:
    base = {"id": "task_001", "title": "T", "description": "Do it", "task_type": "implement",
            "output_files": "[]", "acceptance_criteria": "[]", "verify_command": None,
            "retry_count": 0, "max_retries": 3, "error": None}
    return base | overrides


# ── #1 objective checks ─────────────────────────────────────────

def test_missing_output_file_fails(in_project):
    result = verify_task(task(output_files=json.dumps(["app.py"])))
    assert not result.passed and "missing output files: app.py" in result.report


def test_syntax_error_fails_and_writes_no_pycache(in_project):
    (in_project / "app.py").write_text("def broken(:\n")
    result = verify_task(task(output_files=json.dumps(["app.py"])))
    assert not result.passed and "app.py does not parse" in result.report
    assert not (in_project / "__pycache__").exists()


def test_invalid_json_fails(in_project):
    (in_project / "data.json").write_text("{not json")
    assert not verify_task(task(output_files=json.dumps(["data.json"]))).passed


def test_verify_command_decides(in_project):
    (in_project / "app.py").write_text("print('ok')\n")
    files = json.dumps(["app.py"])
    passing = verify_task(task(output_files=files, verify_command=f'"{sys.executable}" -c "print(1)"'))
    failing = verify_task(task(output_files=files,
                               verify_command=f'"{sys.executable}" -c "import sys; print(\'tests failed\'); sys.exit(1)"'))

    assert passing.passed and "✓ all 1 output files exist" in passing.report
    assert not failing.passed and "exited 1" in failing.report and "tests failed" in failing.report


def test_no_checks_passes(in_project):
    assert verify_task(task()).passed


# ── #4 reflection ───────────────────────────────────────────────

def test_retry_prompt_includes_previous_failure():
    first = _build_system_prompt(task(), [])
    retry = _build_system_prompt(task(retry_count=1, error="Automatic checks failed:\n✗ `pytest` exited 1"), [])

    assert "PREVIOUS ATTEMPT FAILED" not in first
    assert "PREVIOUS ATTEMPT FAILED (this is attempt 2 of 3)" in retry
    assert "`pytest` exited 1" in retry


# ── #3 re-plan validation ───────────────────────────────────────

def planned(task_id, deps):
    return PlannedTask(id=task_id, title=task_id, description="", task_type=TaskType.IMPLEMENT,
                       depends_on=deps, estimated_minutes=5, output_files=[], acceptance_criteria=[])


def test_next_task_id_starts_a_new_hundred():
    assert next_task_id(["task_001", "task_007"]) == "task_101"
    assert next_task_id(["task_001", "task_103"]) == "task_201"


def test_validate_revision():
    existing = {"task_001", "task_002", "task_003"}
    dependable = {"task_001"}                       # completed; 002 failed, 003 is being replaced

    good = RevisedPlan(reason="r", tasks=[planned("task_101", ["task_001"]), planned("task_102", ["task_101"])])
    assert validate_revision(good, dependable, existing) == []

    reused = RevisedPlan(reason="r", tasks=[planned("task_002", [])])
    assert "reuses existing task IDs: task_002" in validate_revision(reused, dependable, existing)

    replaced_dep = RevisedPlan(reason="r", tasks=[planned("task_101", ["task_003"])])
    assert any("task_003" in p for p in validate_revision(replaced_dep, dependable, existing))

    cycle = RevisedPlan(reason="r", tasks=[planned("task_101", ["task_102"]), planned("task_102", ["task_101"])])
    assert any("cycle" in p for p in validate_revision(cycle, dependable, existing))
