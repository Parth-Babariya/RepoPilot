from repopilot.agents.planner_agent import PlannedTask, RevisedPlan, next_task_id, validate_revision
from repopilot.agents.task_spec import TaskType


def planned(task_id, deps):
    return PlannedTask(
        id=task_id,
        title=task_id,
        description="",
        task_type=TaskType.IMPLEMENT,
        depends_on=deps,
        estimated_minutes=5,
        output_files=[],
        acceptance_criteria=[],
    )


def test_next_task_id_starts_a_new_hundred():
    assert next_task_id(["task_001", "task_007"]) == "task_101"
    assert next_task_id(["task_001", "task_103"]) == "task_201"


def test_validate_revision():
    existing = {"task_001", "task_002", "task_003"}
    dependable = {"task_001"}  # completed; 002 failed, 003 is being replaced

    good = RevisedPlan(reason="r", tasks=[planned("task_101", ["task_001"]), planned("task_102", ["task_101"])])
    assert validate_revision(good, dependable, existing) == []

    reused = RevisedPlan(reason="r", tasks=[planned("task_002", [])])
    assert "reuses existing task IDs: task_002" in validate_revision(reused, dependable, existing)

    replaced_dep = RevisedPlan(reason="r", tasks=[planned("task_101", ["task_003"])])
    assert any("task_003" in p for p in validate_revision(replaced_dep, dependable, existing))

    cycle = RevisedPlan(reason="r", tasks=[planned("task_101", ["task_102"]), planned("task_102", ["task_101"])])
    assert any("cycle" in p for p in validate_revision(cycle, dependable, existing))
