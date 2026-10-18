from __future__ import annotations

import contextvars
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.embeddings import Embeddings
from repopilot.config import config
from repopilot.observability.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# How usage tracking works
# ------------------------------------------------------------------
# Every chat model is created with a UsageCallback and every embedder is
# wrapped in TrackedEmbeddings. Each call is written as one row to a
# SQLite table, tagged with:
#   feature — which part of RepoPilot made the call ("ask", "plan", "eval"…)
#   session — the conversation (or eval run) it belongs to
#
# Feature and session come from context variables set with track(...)
# around each command, so the code that makes the call doesn't need to
# know about tracking. Threads that never call track() (e.g. the file
# watcher) record as feature "other" unless they set their own.
#
# Cost = tokens × price from config.yaml usage.prices (USD per 1M tokens).
# A model with no price recorded gets cost NULL and shows as "?" in /stats.
# Recording must never break the app, so every write swallows errors.
# ------------------------------------------------------------------

current_feature: contextvars.ContextVar[str] = contextvars.ContextVar("repopilot_feature", default="other")
current_session: contextvars.ContextVar[str] = contextvars.ContextVar("repopilot_session", default="")


@contextmanager
def track(feature: str, session: str | None = None):
    """Tag every model/embedding call made inside this block with feature (and session)."""
    feature_token = current_feature.set(feature)
    session_token = current_session.set(session) if session is not None else None
    try:
        yield
    finally:
        if session_token is not None:
            current_session.reset(session_token)
        current_feature.reset(feature_token)


# Tokens used by the current command (one /ask, one /plan run), counted by
# UsageCallback for every LLM call made inside metered(). Token budgets
# (guardrails/budgets.py) read it; it is a one-item list so the count is shared
# with the copies of the context that LangGraph runs tools and callbacks in.
_run_tokens: contextvars.ContextVar[list[int] | None] = contextvars.ContextVar("repopilot_run_tokens", default=None)


@contextmanager
def metered():
    """Count the tokens of every LLM call made inside this block (see run_tokens())."""
    token = _run_tokens.set([0])
    try:
        yield
    finally:
        _run_tokens.reset(token)


def run_tokens() -> int | None:
    """Tokens used so far inside the current metered() block, or None outside one."""
    counter = _run_tokens.get()
    return counter[0] if counter is not None else None


def _price(model: str) -> dict | None:
    return (config.get("usage", {}).get("prices") or {}).get(model)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    price = _price(model)
    if price is None:
        return None
    return (input_tokens * price.get("input", 0) + output_tokens * price.get("output", 0)) / 1_000_000


class UsageStore:
    """SQLite log of model calls and /ask questions. One connection per write (thread-safe)."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS calls (
                    ts            REAL NOT NULL,
                    session       TEXT,
                    feature       TEXT,
                    kind          TEXT NOT NULL,      -- 'llm' | 'embedding'
                    model         TEXT,
                    input_tokens  INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    cost          REAL,               -- NULL when the model has no price in config
                    latency_ms    REAL
                );
                CREATE TABLE IF NOT EXISTS questions (
                    ts          REAL NOT NULL,
                    session     TEXT,
                    cache_hit   INTEGER NOT NULL,
                    latency_ms  REAL
                );
                CREATE INDEX IF NOT EXISTS idx_calls_session ON calls(session);
            """)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def record_call(
        self,
        kind: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float | None,
        feature: str | None = None,
    ) -> None:
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO calls VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        time.time(),
                        current_session.get(),
                        feature or current_feature.get(),
                        kind,
                        model,
                        input_tokens,
                        output_tokens,
                        estimate_cost(model, input_tokens, output_tokens),
                        latency_ms,
                    ),
                )
        except Exception as e:
            logger.warning(f"Could not record usage: {e}")

    def record_question(self, cache_hit: bool, latency_ms: float) -> None:
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO questions VALUES (?,?,?,?)",
                    (time.time(), current_session.get(), int(cache_hit), latency_ms),
                )
        except Exception as e:
            logger.warning(f"Could not record question: {e}")

    # ── Queries (session=None means all time) ───────────────────────

    def _where(self, session: str | None) -> tuple[str, tuple]:
        return ("WHERE session = ?", (session,)) if session is not None else ("", ())

    def summary_by_feature(self, session: str | None = None) -> list[dict]:
        where, args = self._where(session)
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT kind, feature,
                       COUNT(*)                        AS calls,
                       SUM(input_tokens + output_tokens) AS tokens,
                       SUM(cost)                       AS cost,
                       SUM(cost IS NULL)               AS unpriced,
                       AVG(latency_ms)                 AS avg_latency_ms
                FROM calls {where}
                GROUP BY kind, feature
                ORDER BY kind DESC, cost DESC
            """,
                args,
            ).fetchall()
        return [dict(r) for r in rows]

    def unpriced_models(self, session: str | None = None) -> list[str]:
        where, args = self._where(session)
        clause = f"{where} AND cost IS NULL" if where else "WHERE cost IS NULL"
        with self._conn() as conn:
            rows = conn.execute(f"SELECT DISTINCT model FROM calls {clause}", args).fetchall()
        return [r["model"] for r in rows]

    def question_summary(self, session: str | None = None) -> dict:
        where, args = self._where(session)
        with self._conn() as conn:
            q = conn.execute(
                f"""
                SELECT COUNT(*) AS questions, COALESCE(SUM(cache_hit), 0) AS hits,
                       AVG(latency_ms) AS avg_latency_ms
                FROM questions {where}
            """,
                args,
            ).fetchone()
            ask_cost = conn.execute(
                f"SELECT SUM(cost) FROM calls {where + ' AND' if where else 'WHERE'} feature = 'ask'", args
            ).fetchone()[0]
        result = dict(q)
        misses = result["questions"] - result["hits"]
        # A cache hit saves roughly what an uncached /ask costs on average.
        avg_ask = (ask_cost / misses) if ask_cost and misses else None
        result["estimated_saved"] = avg_ask * result["hits"] if avg_ask is not None else None
        return result

    def total_cost(self, session: str, exclude_feature: str | None = None) -> float:
        query = "SELECT COALESCE(SUM(cost), 0) FROM calls WHERE session = ?"
        args: tuple = (session,)
        if exclude_feature:
            query += " AND feature != ?"
            args += (exclude_feature,)
        with self._conn() as conn:
            return conn.execute(query, args).fetchone()[0]


_store: UsageStore | None = None


def get_usage_store() -> UsageStore:
    global _store
    if _store is None:
        _store = UsageStore(config.get("usage", {}).get("db_path", ".repopilot/usage.db"))
    return _store


# ------------------------------------------------------------------
# LangChain hooks
# ------------------------------------------------------------------


class UsageCallback(BaseCallbackHandler):
    """
    Records token usage and latency for every chat model call.

    model is the configured model name (used for pricing — API responses often
    return dated names like "gpt-4o-mini-2024-07-18"). feature, if given,
    overrides the context's feature, e.g. "summarize" for the summariser.
    """

    # Run in the caller's context so current_feature/current_session are visible.
    run_inline = True

    def __init__(self, model: str, feature: str | None = None) -> None:
        self.model = model
        self.feature = feature
        self._started: dict = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        self._started[run_id] = time.perf_counter()

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs) -> None:
        self._started[run_id] = time.perf_counter()

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        started = self._started.pop(run_id, None)
        latency_ms = (time.perf_counter() - started) * 1000 if started else None

        input_tokens = output_tokens = 0
        for generations in response.generations:
            for gen in generations:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
                input_tokens += usage.get("input_tokens", 0)
                output_tokens += usage.get("output_tokens", 0)
        if not (input_tokens or output_tokens):
            # Older-style providers report usage in llm_output instead
            token_usage = (response.llm_output or {}).get("token_usage") or {}
            input_tokens = token_usage.get("prompt_tokens", 0)
            output_tokens = token_usage.get("completion_tokens", 0)

        if (counter := _run_tokens.get()) is not None:
            counter[0] += input_tokens + output_tokens
        get_usage_store().record_call("llm", self.model, input_tokens, output_tokens, latency_ms, self.feature)

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        self._started.pop(run_id, None)


def usage_callbacks(model: str, feature: str | None = None) -> list[BaseCallbackHandler]:
    return [UsageCallback(model, feature)]


def _estimate_tokens(texts: list[str]) -> int:
    # Embedding APIs don't return token counts through LangChain; ~4 chars/token
    # is the usual estimate for English and code.
    return sum(len(t) for t in texts) // 4


class TrackedEmbeddings(Embeddings):
    """Wraps any LangChain embedder and records (estimated) tokens per call."""

    def __init__(self, inner: Embeddings, model: str) -> None:
        self.inner = inner
        self.model = model

    def _record(self, texts: list[str], started: float) -> None:
        get_usage_store().record_call(
            "embedding", self.model, _estimate_tokens(texts), 0, (time.perf_counter() - started) * 1000
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        started = time.perf_counter()
        result = self.inner.embed_documents(texts)
        self._record(texts, started)
        return result

    def embed_query(self, text: str) -> list[float]:
        started = time.perf_counter()
        result = self.inner.embed_query(text)
        self._record([text], started)
        return result

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        started = time.perf_counter()
        result = await self.inner.aembed_documents(texts)
        self._record(texts, started)
        return result

    async def aembed_query(self, text: str) -> list[float]:
        started = time.perf_counter()
        result = await self.inner.aembed_query(text)
        self._record([text], started)
        return result
