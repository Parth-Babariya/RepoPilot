from __future__ import annotations

import contextvars
from contextlib import contextmanager

from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware, ToolCallLimitMiddleware, hook_config
from langchain_core.messages import AIMessage
from repopilot.config import config
from repopilot.observability.usage import metered, run_tokens

# ------------------------------------------------------------------
# Budgets: hard limits so a looping agent can't run up cost
# ------------------------------------------------------------------
# Two kinds, both set in config.yaml budgets:
#
#   Step limits  — per agent run: at most N tool calls and M model calls.
#                  Past the tool limit the agent must answer with what it
#                  has; past the model limit the run ends with a message.
#   Token budget — per command (one /ask, one /plan run), across every LLM
#                  call it makes: agents, critics, the summariser. Tokens are
#                  counted by observability (UsageCallback); this module only
#                  decides when the count is too high.
#
# When the token budget runs out, an agent's loop stops before its next
# model call. /plan stops starting new tasks and keeps its progress, so
# running /plan again resumes it.
# ------------------------------------------------------------------

_DEFAULTS = {
    "ask_tokens": 250_000,  # largest /ask measured so far used ~75k
    "plan_tokens": 1_000_000,  # whole /plan runs measured so far used ~40k-105k
    "qa_tool_calls": 15,
    "task_tool_calls": 40,
    "model_calls": 30,
}

_limit: contextvars.ContextVar[int | None] = contextvars.ContextVar("repopilot_token_limit", default=None)


class BudgetExceeded(Exception):
    """The current command used up its token budget."""


def budget(name: str) -> int:
    value = (config.get("budgets", {}) or {}).get(name, _DEFAULTS[name])
    return int(value)


@contextmanager
def token_budget(name: str):
    """Limit the LLM tokens used inside this block to budgets.<name> (0 = unlimited)."""
    limit = budget(name)
    token = _limit.set(limit or None)
    try:
        with metered():
            yield
    finally:
        _limit.reset(token)


def budget_exhausted() -> str | None:
    """A message if the current command's token budget is used up, else None."""
    limit, used = _limit.get(), run_tokens()
    if limit is None or used is None or used < limit:
        return None
    return f"Stopped: the token budget for this command is used up ({used:,} of {limit:,} tokens)."


def ensure_budget() -> None:
    """Raise BudgetExceeded if the token budget is used up (checked between steps)."""
    if message := budget_exhausted():
        raise BudgetExceeded(message)


class TokenBudgetMiddleware(AgentMiddleware):
    """Ends an agent's loop before its next model call once the command's token budget is used up."""

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        if message := budget_exhausted():
            return {"jump_to": "end", "messages": [AIMessage(content=message)]}
        return None

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state, runtime):
        return self.before_model(state, runtime)


def agent_limits(tool_calls: str) -> list:
    """
    Middleware for a tool-using agent: the token budget plus its step limits.
    tool_calls names the budget for this agent ("qa_tool_calls" or "task_tool_calls").
    """
    return [
        TokenBudgetMiddleware(),
        # "continue": past the limit tool calls are refused and the model is told
        # so — it then answers with what it already found instead of crashing.
        ToolCallLimitMiddleware(run_limit=budget(tool_calls), exit_behavior="continue"),
        ModelCallLimitMiddleware(run_limit=budget("model_calls"), exit_behavior="end"),
    ]
