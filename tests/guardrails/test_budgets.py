import pytest
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware

from repopilot.guardrails import budgets
from repopilot.guardrails.budgets import (
    BudgetExceeded, TokenBudgetMiddleware, agent_limits, budget_exhausted, ensure_budget, token_budget,
)
from repopilot.observability.usage import UsageCallback, run_tokens
from tests.observability.test_usage import fake_llm_end, store  # noqa: F401  (store is a fixture)


@pytest.fixture
def limits(monkeypatch):
    def set_limits(**values):
        monkeypatch.setitem(budgets.config, "budgets", values)
    return set_limits


def test_token_budget_counts_every_llm_call_inside_it(store, limits):
    limits(ask_tokens=1000)
    assert run_tokens() is None                       # nothing is counted outside a budget
    with token_budget("ask_tokens"):
        fake_llm_end(UsageCallback("gpt-4o-mini"), 300, 100)
        assert run_tokens() == 400 and budget_exhausted() is None
        fake_llm_end(UsageCallback("gpt-4o-mini"), 500, 100)
        assert "1,000 of 1,000" in budget_exhausted()
        with pytest.raises(BudgetExceeded):
            ensure_budget()
    assert budget_exhausted() is None                 # the limit ends with the block


def test_zero_means_unlimited(store, limits):
    limits(plan_tokens=0)
    with token_budget("plan_tokens"):
        fake_llm_end(UsageCallback("gpt-4o-mini"), 10_000_000, 0)
        assert budget_exhausted() is None


def test_middleware_ends_the_agent_loop_once_the_budget_is_used(store, limits):
    limits(ask_tokens=100)
    middleware = TokenBudgetMiddleware()
    with token_budget("ask_tokens"):
        assert middleware.before_model({"messages": []}, None) is None
        fake_llm_end(UsageCallback("gpt-4o-mini"), 100, 0)
        update = middleware.before_model({"messages": []}, None)
    assert update["jump_to"] == "end"
    assert "token budget" in update["messages"][0].content


def test_agent_limits_use_configured_step_limits(limits):
    limits(qa_tool_calls=7, model_calls=9)
    token, tools, models = agent_limits("qa_tool_calls")
    assert isinstance(token, TokenBudgetMiddleware)
    assert isinstance(tools, ToolCallLimitMiddleware) and tools.run_limit == 7
    assert isinstance(models, ModelCallLimitMiddleware) and models.run_limit == 9
