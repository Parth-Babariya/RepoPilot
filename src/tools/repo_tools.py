from langchain.tools import tool

from repopilot.guardrails.tool_safety import check_path, resolve_path
from repopilot.observability.logger import get_logger
from repopilot.retrieval.repo_map import build_repo_map
from repopilot.tools.sources import add_source

logger = get_logger(__name__)

_MAX_LINES = 200


@tool
def repo_map() -> str:
    """
    An outline of the whole repository: every source file with its classes,
    methods and functions, and where the tests are. Use it for questions about
    overall structure ("how is this project organised?", "where does X live?")
    before searching for details.
    """
    logger.info("Tool called: repo_map")
    return build_repo_map()


@tool
def read_lines(file_path: str, start_line: int, end_line: int) -> str:
    """
    Read lines start_line to end_line (inclusive, 1-based) of a file, with line
    numbers. Use it to open just the part of a file a search result pointed to —
    cheaper than reading the whole file. At most 200 lines per call.
    """
    if error := check_path(file_path):
        return error
    path = resolve_path(file_path)
    if not path.is_file():
        return f"Error: file not found: {file_path}"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        return f"Error: {e}"

    start = max(1, int(start_line))
    end = min(len(lines), int(end_line), start + _MAX_LINES - 1)
    if start > len(lines):
        return f"Error: {file_path} has only {len(lines)} lines"
    add_source(str(path), start, end)
    width = len(str(end))
    body = "\n".join(f"{n:>{width}}  {lines[n - 1]}" for n in range(start, end + 1))
    more = f"\n… (file continues to line {len(lines)})" if end < len(lines) else ""
    return f"{file_path} lines {start}-{end}:\n{body}{more}"
