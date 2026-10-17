from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field
from langchain.agents import create_agent

from repopilot.services.llm_service import chat_model, fallback_middleware
from repopilot.agents.task_spec import TaskType
from repopilot.observability.logger import get_logger

logger = get_logger(__name__)


class PlannedTask(BaseModel):
    id: str                          # stable snake_case e.g. task_001
    title: str
    description: str
    task_type: TaskType
    depends_on: list[str]            # list of task IDs that must complete first
    estimated_minutes: int
    output_files: list[str]          # files this task will create/modify
    acceptance_criteria: list[str]   # what "done" looks like for the judge
    verify_command: str = Field(
        default="",
        description="One shell command, run from the project root, that exits 0 only when "
                    "this task is done (e.g. its tests). Empty if nothing can be run yet.",
    )


class ExecutionPlan(BaseModel):
    project_name: str
    goal_summary: str
    tech_stack: list[str]
    total_estimated_hours: float
    tasks: list[PlannedTask]
    risks: list[str]
    assumptions: list[str]


class RevisedPlan(BaseModel):
    reason: str                      # what changed and why, in one or two sentences
    tasks: list[PlannedTask]         # the new tasks that finish the goal from here


_TASK_RULES = """\
- depends_on must reference valid task IDs (no cycles)
- output_files must list every file the task will write to disk
- acceptance_criteria must be concrete and verifiable (3-5 items per task)
- task_type must be one of: design, implement, test, review, integrate, configure
- verify_command: for test and integrate tasks — and any task whose result can be
  checked by running something — give ONE shell command, run from the project root,
  that exits 0 only when the task is done (e.g. `python -m pytest tests/test_x.py -q`).
  No pipes, `&&` or redirection. Leave it empty when there is nothing to run yet
  (e.g. design tasks). The user approves these commands before anything runs.
"""

_SYSTEM_PROMPT = f"""\
You are a senior software architect. Given a software goal, produce a detailed
ExecutionPlan broken into concrete tasks for an AI agent to implement.

Rules:
- 5 to 20 tasks
- Task IDs must be stable snake_case: task_001, task_002, ...
- Ordering must form a valid DAG: architecture → schema → config → core → tests → integration
{_TASK_RULES}"""

_REPLAN_SYSTEM_PROMPT = f"""\
You are a senior software architect revising a plan after one of its tasks
failed repeatedly. Write the new tasks that replace the failed task and the
tasks that depended on it, so the goal can still be finished. You may split the
failed task into smaller steps, change the approach, or drop work that isn't
essential. Never redo completed tasks or duplicate unaffected ones.

Rules:
- New task IDs must continue the numbering you are given and must not reuse any existing ID
- depends_on may only reference completed task IDs or your new task IDs
{_TASK_RULES}"""


def create_plan(goal: str, extra_context: str = "") -> ExecutionPlan:
    """Call the LLM planner and return a structured ExecutionPlan."""
    llm = chat_model(temperature=0)

    planner_agent = create_agent(
        llm,
        tools=[],
        system_prompt=_SYSTEM_PROMPT,
        response_format=ExecutionPlan,
        middleware=fallback_middleware(temperature=0),
    )

    user_message = f"Goal: {goal}"
    if extra_context:
        user_message += f"\n\nAdditional context / change requests:\n{extra_context}"

    result = planner_agent.invoke({"messages": [{"role": "user", "content": user_message}]})
    plan: ExecutionPlan = result["structured_response"]
    logger.info(f"Plan created: {plan.project_name} with {len(plan.tasks)} tasks")
    return plan


# ------------------------------------------------------------------
# Re-planning after a failure
# ------------------------------------------------------------------

def next_task_id(existing_ids: list[str]) -> str:
    """First unused id after the highest task_NNN, rounded up to the next hundred (task_101, …)."""
    numbers = [int(m.group(1)) for i in existing_ids if (m := re.fullmatch(r"task_(\d+)", i))]
    start = (max(numbers, default=0) // 100 + 1) * 100 + 1
    return f"task_{start:03d}"


def _task_line(task: dict, extra: str = "") -> str:
    deps = ", ".join(json.loads(task.get("depends_on") or "[]")) or "none"
    return f"- {task['id']} [{task['task_type']}] {task['title']} (depends on: {deps}){extra}"


def build_replan_message(goal: str, completed: list[dict], failed: dict, affected: list[dict],
                         unaffected: list[dict], first_new_id: str) -> str:
    def lines(tasks: list[dict], detail) -> str:
        return "\n".join(_task_line(t, detail(t)) for t in tasks) or "(none)"

    return f"""GOAL: {goal}

COMPLETED TASKS (done — do not redo; new tasks may depend on these IDs):
{lines(completed, lambda t: f"{chr(10)}    result: {(t.get('result') or '')[:300]}")}

FAILED TASK (failed {failed.get('retry_count', 0)} times — will be replaced):
{_task_line(failed)}
    {failed['description']}
    last error: {(failed.get('error') or '(none recorded)')[:2000]}

TASKS THAT DEPENDED ON IT (not started — will be replaced by your new tasks):
{lines(affected, lambda t: f"{chr(10)}    {t['description']}")}

OTHER TASKS STILL TO RUN (unaffected — they stay; new tasks may depend on them):
{lines(unaffected, lambda t: "")}

Write the tasks that replace the failed task and the tasks that depended on it.
Number your new tasks starting at {first_new_id}."""


def validate_revision(revision: RevisedPlan, dependable_ids: set[str], existing_ids: set[str]) -> list[str]:
    """
    Return the problems that make a revised plan unusable (empty list if it's valid).
    dependable_ids: existing tasks new ones may depend on (completed or still to run).
    existing_ids:   every task ID already in the project — new IDs must not reuse them.
    """
    problems = []
    new_ids = [t.id for t in revision.tasks]
    if not new_ids:
        problems.append("the revision has no tasks")
    if len(set(new_ids)) != len(new_ids):
        problems.append("duplicate task IDs")
    if reused := sorted(set(new_ids) & existing_ids):
        problems.append(f"reuses existing task IDs: {', '.join(reused)}")
    allowed = dependable_ids | set(new_ids)
    for task in revision.tasks:
        if unknown := [d for d in task.depends_on if d not in allowed]:
            problems.append(f"{task.id} depends on unknown or replaced tasks: {', '.join(unknown)}")

    # cycle check (Kahn's algorithm) over the new tasks only
    pending = {t.id: {d for d in t.depends_on if d in set(new_ids)} for t in revision.tasks}
    while ready := [i for i, deps in pending.items() if not deps]:
        for i in ready:
            pending.pop(i)
        for deps in pending.values():
            deps.difference_update(ready)
    if pending:
        problems.append(f"dependency cycle between: {', '.join(sorted(pending))}")
    return problems


def replan(goal: str, completed: list[dict], failed: dict, affected: list[dict],
           unaffected: list[dict], existing_ids: list[str]) -> RevisedPlan:
    """Ask the planner for new tasks that replace the failed task and the tasks that depended on it."""
    llm = chat_model(temperature=0)
    agent = create_agent(llm, tools=[], system_prompt=_REPLAN_SYSTEM_PROMPT, response_format=RevisedPlan,
                         middleware=fallback_middleware(temperature=0))
    message = build_replan_message(goal, completed, failed, affected, unaffected, next_task_id(existing_ids))
    result = agent.invoke({"messages": [{"role": "user", "content": message}]})
    revision: RevisedPlan = result["structured_response"]
    logger.info(f"Re-plan proposed {len(revision.tasks)} tasks: {revision.reason}")
    return revision
