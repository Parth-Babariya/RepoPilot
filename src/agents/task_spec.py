from __future__ import annotations

import json
from enum import Enum

# ------------------------------------------------------------------
# What a /plan task is — shared by the agents that plan it (planner),
# build it (task agent) and judge it (critic), and by the task store
# that persists it. The task's lifecycle (TaskStatus) belongs to the
# workflow layer: see workflows/task_store.py.
# ------------------------------------------------------------------


class TaskType(str, Enum):
    """Decides which tools the task agent gets (agents/task_agent.py)."""

    DESIGN = "design"
    IMPLEMENT = "implement"
    TEST = "test"
    REVIEW = "review"
    INTEGRATE = "integrate"
    CONFIGURE = "configure"


def json_list(value) -> list:
    """Decode a JSON list field of a stored task (a list passes through; bad JSON → [])."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return []
    return value or []
