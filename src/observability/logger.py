import logging
from pathlib import Path

# Console shows WARNING and above only — the REPL output stays readable.
# Full DEBUG logs go to a file once main.py calls enable_file_logging().
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# Harmless warnings repeated once per MCP tool when Gemini converts tool schemas
logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    # Our own loggers record everything; handlers decide where it goes
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger


def enable_file_logging(path: str = ".repopilot/logs/repopilot.log") -> Path:
    """
    Send RepoPilot's own DEBUG/INFO logs to a file and keep them off the console.
    Called once by main.py at startup (not at import, so tests don't create files).
    """
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    repopilot_logger = logging.getLogger("repopilot")
    repopilot_logger.setLevel(logging.DEBUG)
    repopilot_logger.addHandler(file_handler)

    # The console (root) handler should only show warnings and errors
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.WARNING)
    return log_path
