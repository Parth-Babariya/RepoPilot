from __future__ import annotations

import json

from langchain.agents import create_agent
from pydantic import BaseModel

from repopilot.guardrails.tool_safety import check_path, resolve_path
from repopilot.services.llm_service import fallback_middleware, get_judge_llm, response_format
from repopilot.agents.task_spec import json_list

# ------------------------------------------------------------------
# Critic agent — LLM-as-judge for /plan tasks.
# Runs only after the objective checks passed (workflows/verification.py),
# and grades the files the task actually produced against its acceptance
# criteria, using the check results as evidence. The task agent's own
# summary is shown last and labelled as a claim, not evidence.
# ------------------------------------------------------------------

_FILE_CHARS = 6000   # total file content shown to the critic


class CriticVerdict(BaseModel):
    passed: bool
    score: int    # 0-10
    reason: str   # one sentence


_SYSTEM_PROMPT = """\
You are a code review judge. Given a task description, its acceptance criteria,
the files the agent produced, the results of automatic checks, and the agent's
own summary, decide whether the task was completed satisfactorily.
Base your verdict on the files and check results, not on the agent's claims.

Score 0-10:
  8-10 → passed (all acceptance criteria met)
  5-7  → borderline (minor gaps, still usable)
  0-4  → failed (criteria not met, re-execution needed)

Set passed=true if score >= 6. One short sentence for reason."""


def file_excerpts(task: dict) -> str:
    """The task's output files (inside the project), trimmed to fit the critic's prompt."""
    files = [f for f in json_list(task.get("output_files")) if check_path(f) is None and resolve_path(f).is_file()]
    if not files:
        return "(no output files)"
    per_file = max(500, _FILE_CHARS // len(files))
    parts = []
    for f in files:
        try:
            text = resolve_path(f).read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            text = f"(could not read: {e})"
        cut = "\n… (truncated)" if len(text) > per_file else ""
        parts.append(f"=== {f} ===\n{text[:per_file]}{cut}")
    return "\n\n".join(parts)[:_FILE_CHARS + 500]


async def judge_task(task: dict, agent_output: str, check_report: str) -> CriticVerdict:
    """Grade one task with the cheaper judge_model; returns passed / score / reason."""
    llm = get_judge_llm("plan-judge", temperature=0, max_tokens=1000)
    critic = create_agent(llm, tools=[], system_prompt=_SYSTEM_PROMPT,
                          response_format=response_format(llm, CriticVerdict),
                          middleware=fallback_middleware(feature="plan-judge", temperature=0, max_tokens=1000))

    user_message = f"""TASK: {task['title']}
DESCRIPTION: {task['description']}
ACCEPTANCE CRITERIA: {json.dumps(json_list(task.get("acceptance_criteria")), indent=2)}

AUTOMATIC CHECKS (run by the system, not the agent):
{check_report}

FILES PRODUCED:
{file_excerpts(task)}

AGENT'S OWN SUMMARY:
{agent_output[:2000]}"""

    result = await critic.ainvoke({"messages": [{"role": "user", "content": user_message}]})
    return result["structured_response"]
