import re
import time
from dataclasses import dataclass, field

from repopilot.agents.qa_agent import SEARCH_RESULTS_MARKER
from repopilot.config import config
from repopilot.guardrails.budgets import budget_exhausted, token_budget
from repopilot.observability.logger import get_logger
from repopilot.observability.usage import get_usage_store
from repopilot.tools.code_search import search_codebase, search_memory
from repopilot.tools.sources import Source, collect_sources
from repopilot.workflows.answer_verification import AnswerCheck, max_revisions, verify_answer

logger = get_logger(__name__)


@dataclass
class QueryResult:
    answer: str
    sources: list[Source] = field(default_factory=list)
    cached: bool = False
    error: bool = False
    check: AnswerCheck | None = None  # result of answer verification (None = not checked)
    revised: bool = False  # the answer was rewritten after failing a check


_REVISION_PROMPT = """An automatic check of your last answer found problems:
{problems}
Look at the code again with your tools if you need to, then write your complete corrected
answer (the whole answer, not only the changes). State only what the code supports; if
something can't be confirmed from the code, say so."""


_INITIAL_SEARCH = (
    """{question}

"""
    + SEARCH_RESULTS_MARKER
    + """; the results are below. Answer from
them if they are enough, and search only for what is missing.]

{results}"""
)

# "explain that more", "why does it do this?" — about the previous answer, not new code
_BACK_REFERENCE = re.compile(r"\b(that|this|it|those|these|above|previous|earlier)\b", re.IGNORECASE)


def with_initial_search(question: str, has_history: bool = False) -> tuple[str, list[str]]:
    """
    The message to send the agent: the question plus the results of searching
    for it, so the agent can often answer without spending a model call just to
    decide to search (in /eval it found the right files 94% of the time; most
    answers' first step was that search). Returns (message, [search results])
    — the results also count as evidence for the answer checks.

    Skipped for follow-ups about the previous answer, where a fresh search would
    only add unrelated code, and when ask.initial_search is off.
    """
    if not (config.get("ask", {}) or {}).get("initial_search", True):
        return question, []
    if has_history and _BACK_REFERENCE.search(question) and len(question.split()) <= 12:
        return question, []
    results = search_codebase.invoke({"query": question})
    if results.startswith("No relevant code"):
        return question, []
    return _INITIAL_SEARCH.format(question=question, results=results), [results]


def _content(message) -> str:
    # Claude returns content as a list of blocks; OpenAI returns a plain string.
    content = getattr(message, "content", "")
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content or ""


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _tool_outputs(messages) -> list[str]:
    return [_content(m) for m in messages if getattr(m, "type", "") == "tool"]


async def _ask(agent, text: str, agent_config: dict, found: list[str] | None = None) -> tuple[str, list[str]]:
    """
    Send one message to the agent. Returns its answer and the code behind it:
    `found` (the initial search) plus this turn's tool outputs. A follow-up that
    retrieved nothing ("explain that more") is backed by the code retrieved
    earlier in the session instead.
    """
    response = await agent.ainvoke({"messages": [{"role": "user", "content": text}]}, agent_config)
    messages = response["messages"]
    start = next(
        (
            i
            for i in range(len(messages) - 1, -1, -1)
            if getattr(messages[i], "type", "") == "human" and _content(messages[i]) == text
        ),
        -1,
    )
    evidence = (found or []) + _tool_outputs(messages[start + 1 :])
    return _content(messages[-1]), evidence or _tool_outputs(messages)[-10:]


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

    Before it is returned, the answer is verified (workflows/answer_verification.py):
    its file/line references must exist and the answer critic must find its
    claims backed by the retrieved code; a failure gets one revision.

    The cache is keyed on the question text alone, so it is only used for the
    first question in a session. A follow-up like "explain that more" depends
    on the earlier conversation and must never get (or store) a cached answer.
    """
    logger.info(f"Handling query for session {thread_id}: {question}")
    started = time.perf_counter()
    model = config["llm"]["model"]
    agent_config = {"configurable": {"thread_id": thread_id}}

    has_history = await _has_history(agent, agent_config)
    use_cache = semantic_cache is not None and cache_domain is not None
    if use_cache and has_history:
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

    # Answer → verify (citations, then the answer critic) → at most
    # ask.max_revisions rewrites. Every LLM call here counts against budgets.ask_tokens.
    revised = False
    try:
        with token_budget("ask_tokens"), collect_sources() as sources, search_memory():
            message, found = with_initial_search(question, has_history)
            answer, evidence = await _ask(agent, message, agent_config, found)
            check = None if budget_exhausted() else await verify_answer(question, answer, sources, evidence)
            for _ in range(max_revisions()):
                if check is None or check.passed or budget_exhausted():
                    break
                logger.info(f"Answer failed verification, asking for a revision: {check.feedback}")
                problems = "\n".join(f"- {f}" for f in check.feedback)
                answer, more = await _ask(agent, _REVISION_PROMPT.format(problems=problems), agent_config)
                evidence += more
                revised = True
                check = await verify_answer(question, answer, sources, evidence)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return QueryResult(f"**Error:** {e}", error=True)
    get_usage_store().record_question(cache_hit=False, latency_ms=_elapsed_ms(started))

    # Never cache an answer that failed its checks.
    if use_cache and (check is None or check.passed):
        try:
            ttl = config.get("semantic_cache", {}).get("ttl", 86400)
            await semantic_cache.put(question, answer, domain=cache_domain, model=model, ttl=ttl)
        except Exception as e:
            logger.warning(f"Semantic cache store failed: {e}")

    return QueryResult(answer, sources=sources, check=check, revised=revised)
