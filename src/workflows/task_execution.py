from __future__ import annotations

from repopilot.agents.critic_agent import judge_task
from repopilot.agents.task_agent import run_task_agent
from repopilot.guardrails.budgets import ensure_budget
from repopilot.observability.logger import get_logger
from repopilot.workflows.verification import verify_task

logger = get_logger(__name__)

_ERROR_CHARS = 2000   # how much of a failure report is stored for the retry


async def execute_task(task: dict, dep_outputs: list[dict] | None = None) -> str:
    """
    Run one /plan task end to end — called by the orchestrator loop:

        task agent  →  objective checks  →  critic agent

    Either check failing raises ValueError; the orchestrator records the
    message as the task's error, so the retry's prompt explains what to fix.
    Raises BudgetExceeded if the /plan token budget ran out (the agent may have
    been stopped mid-task, so its work is not judged). Returns the agent's summary.
    """
    output = await run_task_agent(task, dep_outputs)
    ensure_budget()

    verification = verify_task(task)
    logger.info(f"Verification for {task['id']}: passed={verification.passed}\n{verification.report}")
    if not verification.passed:
        raise ValueError(f"Automatic checks failed:\n{verification.report[:_ERROR_CHARS]}")

    verdict = await judge_task(task, output, verification.report)
    logger.info(f"Critic verdict for {task['id']}: score={verdict.score}, passed={verdict.passed}, "
                f"reason={verdict.reason}")
    if not verdict.passed:
        raise ValueError(f"Judge rejected output (score={verdict.score}/10): {verdict.reason}")

    return output
