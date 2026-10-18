import asyncio
import uuid

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from repopilot.observability import usage
from repopilot.observability.usage import TrackedEmbeddings, UsageCallback, UsageStore, track


@pytest.fixture
def store(tmp_path, monkeypatch):
    store = UsageStore(str(tmp_path / "usage.db"))
    monkeypatch.setattr(usage, "_store", store)
    return store


def fake_llm_end(callback: UsageCallback, input_tokens: int, output_tokens: int) -> None:
    run_id = uuid.uuid4()
    message = AIMessage(
        content="hi",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )
    callback.on_chat_model_start({}, [[]], run_id=run_id)
    callback.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=run_id)


def test_cost_uses_configured_prices():
    # config.yaml: gpt-4o-mini = $0.15 in / $0.60 out per 1M tokens
    assert usage.estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)
    assert usage.estimate_cost("some-unknown-model", 100, 100) is None


def test_llm_calls_are_tagged_with_feature_and_session(store):
    with track("ask", "session-1"):
        fake_llm_end(UsageCallback("gpt-4o-mini"), 1000, 500)
    with track("plan", "session-1"):
        fake_llm_end(UsageCallback("gpt-4o-mini"), 2000, 0)
    with track("ask", "session-2"):
        fake_llm_end(UsageCallback("gpt-4o-mini"), 1000, 0)

    rows = {r["feature"]: r for r in store.summary_by_feature("session-1")}
    assert set(rows) == {"ask", "plan"}
    assert rows["ask"]["tokens"] == 1500
    assert rows["ask"]["cost"] == pytest.approx((1000 * 0.15 + 500 * 0.60) / 1e6)
    assert sum(r["calls"] for r in store.summary_by_feature()) == 3  # all time


def test_fixed_feature_overrides_context(store):
    with track("ask", "s"):
        fake_llm_end(UsageCallback("gpt-4o-mini", feature="summarize"), 10, 10)
    assert [r["feature"] for r in store.summary_by_feature("s")] == ["summarize"]


def test_unpriced_model_is_reported(store):
    with track("ask", "s"):
        fake_llm_end(UsageCallback("mystery-model"), 10, 10)
    assert store.summary_by_feature("s")[0]["unpriced"] == 1
    assert store.unpriced_models("s") == ["mystery-model"]


def test_cache_savings_estimate(store):
    with track("ask", "s"):
        fake_llm_end(UsageCallback("gpt-4o-mini"), 1_000_000, 0)  # $0.15 for one uncached question
        store.record_question(cache_hit=False, latency_ms=1000)
        store.record_question(cache_hit=True, latency_ms=50)
        store.record_question(cache_hit=True, latency_ms=50)
    q = store.question_summary("s")
    assert (q["questions"], q["hits"]) == (3, 2)
    assert q["estimated_saved"] == pytest.approx(0.30)


class _FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.0] for _ in texts]

    def embed_query(self, text):
        return [0.0]


def test_embeddings_are_tracked(store):
    embedder = TrackedEmbeddings(_FakeEmbeddings(), "text-embedding-3-small")
    with track("index", "s"):
        embedder.embed_documents(["a" * 400, "b" * 400])
        asyncio.run(embedder.aembed_query("c" * 40))
    rows = store.summary_by_feature("s")
    assert rows[0]["kind"] == "embedding"
    assert rows[0]["calls"] == 2
    assert rows[0]["tokens"] == 210  # ~4 chars per token


def test_eval_cost_excludes_judge(store):
    with track("eval", "eval:run:q1"):
        fake_llm_end(UsageCallback("gpt-4o-mini"), 1_000_000, 0)
        fake_llm_end(UsageCallback("gpt-4o-mini", feature="eval-judge"), 1_000_000, 0)
    assert store.total_cost("eval:run:q1", exclude_feature="eval-judge") == pytest.approx(0.15)
    assert store.total_cost("eval:run:q1") == pytest.approx(0.30)
