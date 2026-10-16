import time
from dataclasses import dataclass, field

from repopilot.agent.sources import Source, collect_sources
from repopilot.config import config
from repopilot.observability.logger import get_logger
from repopilot.observability.usage import get_usage_store


logger = get_logger(__name__)


@dataclass
class QueryResult:
    answer: str
    sources: list[Source] = field(default_factory=list)
    cached: bool = False
    error: bool = False


def _content(message) -> str:
    # Claude returns content as a list of blocks; OpenAI returns a plain string.
    content = getattr(message, "content", "")
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content or ""


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


async def _has_history(agent, agent_config: dict) -> bool:
    """True if this session (thread) already has messages in the checkpointer."""
    try:
        state = await agent.aget_state(agent_config)
        return bool(state.values.get("messages"))
    except Exception as e:
        # If we can't tell, assume a follow-up — a cache miss is safe, a wrong hit isn't.
        logger.warning(f"Could not read session history, skipping cache: {e}")
        return True


async def handle_query(
    agent,
    question: str,
    thread_id: str,
    semantic_cache=None,
    cache_domain: str | None = None,
) -> QueryResult:
    """Entry point for all user queries - invokes the agent.

    Returns the answer plus the code locations the agent's search tools
    returned while answering (shown as "Sources" under the answer).

    When semantic_cache/cache_domain are provided, a cache hit returns the
    stored answer directly and skips the agent (and every tool call it would
    have made, including search_codebase) entirely. A miss falls through to
    the normal agent call and stores the fresh answer for next time.

    The cache is keyed on the question text alone, so it is only used for the
    first question in a session. A follow-up like "explain that more" depends
    on the earlier conversation and must never get (or store) a cached answer.
    """
    logger.info(f"Handling query for session {thread_id}: {question}")
    started = time.perf_counter()
    model = config["llm"]["model"]
    agent_config = {"configurable": {"thread_id": thread_id}}

    use_cache = semantic_cache is not None and cache_domain is not None
    if use_cache and await _has_history(agent, agent_config):
        logger.info("Session has prior messages - skipping semantic cache")
        use_cache = False

    if use_cache:
        try:
            cached_response = await semantic_cache.get(question, domain=cache_domain, model=model)
        except Exception as e:
            logger.warning(f"Semantic cache lookup failed, falling back to agent: {e}")
            cached_response = None
        if cached_response is not None:
            logger.info("Semantic cache HIT - skipping agent/tool calls")
            get_usage_store().record_question(cache_hit=True, latency_ms=_elapsed_ms(started))
            return QueryResult(cached_response, cached=True)

    try:
        with collect_sources() as sources:
            response = await agent.ainvoke(
                {"messages": [{"role": "user", "content": question}]}, agent_config
            )
        answer = _content(response["messages"][-1])
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return QueryResult(f"**Error:** {e}", error=True)
    get_usage_store().record_question(cache_hit=False, latency_ms=_elapsed_ms(started))

    if use_cache:
        try:
            ttl = config.get("semantic_cache", {}).get("ttl", 86400)
            await semantic_cache.put(question, answer, domain=cache_domain, model=model, ttl=ttl)
        except Exception as e:
            logger.warning(f"Semantic cache store failed: {e}")

    return QueryResult(answer, sources=sources)