from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain.agents.middleware import SummarizationMiddleware
from pathlib import Path


from repopilot.config import config
from repopilot.llm.factory import get_llm
from repopilot.observability.logger import get_logger
logger = get_logger(__name__)


def get_checkpointer_db_path() -> str:
 db_path = config["memory"]["db_path"]
 Path(db_path).parent.mkdir(parents=True, exist_ok=True)
 logger.info(f"Using SQLite checkpointer at {db_path}")
 return db_path


def get_checkpointer() -> AsyncSqliteSaver:
 db_path = get_checkpointer_db_path()
 return AsyncSqliteSaver.from_conn_string(db_path)


def get_summarization_middleware() -> SummarizationMiddleware:
 return SummarizationMiddleware(
     model=get_llm(feature="summarize"),
     trigger=("tokens", config["memory"]["summarize_at_tokens"]),
     keep=("messages", config["memory"]["keep_last_messages"]),
 )
