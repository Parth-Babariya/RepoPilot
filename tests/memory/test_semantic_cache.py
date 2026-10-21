import asyncio

import pytest
from repopilot.memory import semantic_cache
from repopilot.memory.semantic_cache import SameQuestion, SemanticCache, same_question


class FakeJudge:
    """Stands in for the judge model: answers 'same' from a fixed rule and counts calls."""

    def __init__(self, same: bool = False, error: Exception | None = None):
        self.same, self.error, self.prompts = same, error, []

    def with_structured_output(self, schema, **kwargs):
        assert schema is SameQuestion
        return self

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return SameQuestion(same=self.same)


@pytest.fixture
def judge(monkeypatch):
    def use(**kwargs):
        fake = FakeJudge(**kwargs)
        monkeypatch.setattr(semantic_cache, "get_judge_llm", lambda feature, **kw: fake)
        return fake

    return use


def test_the_same_text_needs_no_model_call(judge):
    fake = judge(same=False)
    assert asyncio.run(same_question("What calls fail_task?", "what calls  fail_task"))
    assert fake.prompts == []


def test_a_near_match_is_confirmed_by_the_judge(judge):
    fake = judge(same=False)
    assert not asyncio.run(same_question("What calls fail_task?", "What does fail_task call?"))
    assert "What does fail_task call?" in fake.prompts[0]
    judge(same=True)
    assert asyncio.run(same_question("How are failed tasks retried?", "How are failed plan tasks retried?"))


def test_a_failed_check_counts_as_a_miss(judge):
    judge(error=RuntimeError("quota"))
    assert not asyncio.run(same_question("What calls X?", "What does X call?"))


def cache_with(match: dict | None, confirm: bool = True) -> SemanticCache:
    """A SemanticCache without Redis: _top_match returns `match`."""
    cache = SemanticCache.__new__(SemanticCache)
    cache.threshold, cache.confirm_matches = 0.85, confirm

    async def top_match(query, domain, model):
        return match

    cache._top_match = top_match
    return cache


def test_get_rejects_a_close_match_that_asks_something_else(judge):
    judge(same=False)
    stored = {"response": "Called by _execute.", "query_text": "What calls fail_task?", "vector_distance": "0.016"}
    assert asyncio.run(cache_with(stored).get("What does fail_task call?", "d", "m")) is None
    assert asyncio.run(cache_with(stored).get("What calls fail_task?", "d", "m")) == "Called by _execute."
    assert asyncio.run(cache_with(stored, confirm=False).get("What does fail_task call?", "d", "m")) is not None


def test_get_ignores_matches_below_the_threshold(judge):
    fake = judge(same=True)
    far = {"response": "r", "query_text": "Something else", "vector_distance": "0.4"}
    assert asyncio.run(cache_with(far).get("q", "d", "m")) is None
    assert fake.prompts == []  # no model call for an obvious miss


class FakeRedis:
    def __init__(self):
        self.values = {}

    async def get(self, key):
        value = self.values.get(key)
        return value.encode() if value is not None else None

    async def set(self, key, value):
        self.values[key] = value


def test_entries_are_dropped_when_the_code_changed_between_runs():
    cache = cache_with(None)
    cache.client, dropped = FakeRedis(), []

    async def invalidate_domain(domain):
        dropped.append(domain)

    cache.invalidate_domain = invalidate_domain

    def run(fingerprint):
        return asyncio.run(cache.invalidate_if_changed("repo", fingerprint))

    assert run("v1") is True  # first start: entries may predate the fingerprint
    assert run("v1") is False  # same code: cache kept
    assert run("v2") is True  # code changed while RepoPilot was closed
    assert dropped == ["repo", "repo"]
