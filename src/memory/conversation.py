from pathlib import Path

from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import HumanMessage
from repopilot.config import config
from repopilot.observability.logger import get_logger
from repopilot.services.llm_service import get_llm

logger = get_logger(__name__)


def get_checkpointer_db_path() -> str:
    """Where conversations are saved. main.py opens the checkpointer with it, as an
    async context manager, so the connection closes cleanly when RepoPilot exits."""
    db_path = config["memory"]["db_path"]
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using SQLite checkpointer at {db_path}")
    return db_path


class TurnSafeSummarization(SummarizationMiddleware):
    """
    Summarises only earlier turns, never the question being answered.

    SummarizationMiddleware keeps the last N messages, but one answer can
    have more than N: 10 tool calls are 20 messages, so the user's question
    was summarised away mid-answer and the agent replied as if the
    conversation had been reset (seen in /eval). The cut point is moved back
    to the latest user message, so the current turn always stays intact.
    """

    def _determine_cutoff_index(self, messages):
        cutoff = super()._determine_cutoff_index(messages)
        last_user = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)
        return min(cutoff, last_user)


def get_summarization_middleware() -> SummarizationMiddleware:
    return TurnSafeSummarization(
        model=get_llm(feature="summarize"),
        trigger=("tokens", config["memory"]["summarize_at_tokens"]),
        keep=("messages", config["memory"]["keep_last_messages"]),
    )
