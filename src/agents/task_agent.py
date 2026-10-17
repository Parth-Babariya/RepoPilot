from __future__ import annotations

from langchain.agents import create_agent

from repopilot.observability.logger import get_logger
from repopilot.guardrails.budgets import agent_limits
from repopilot.services.llm_service import chat_model, fallback_middleware
from repopilot.guardrails.tool_safety import workspace_root
from repopilot.retrieval.indexing.code_parser import get_source_files
from repopilot.tools.code_search import search_codebase
from repopilot.tools.filesystem_tools import append_file, list_directory, read_file, write_file
from repopilot.tools.repo_tools import read_lines
from repopilot.tools.symbol_tools import find_symbol
from repopilot.tools.terminal_tools import run_command
from repopilot.agents.task_spec import json_list

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Task agent — builds one /plan task (code, tests, config…).
# A fresh agent per task: it sees only its task, the outputs of the tasks
# it depends on, and — on a retry — why the previous attempt failed.
# Checking the result is not its job (see workflows/task_execution.py).
# ------------------------------------------------------------------

# Read-only ways to find existing code. Since /plan works in a worktree of a real
# repo, a task like "add retry logging to the orchestrator" must first find that
# code; with only list_directory and read_file it could only browse folders.
# Results show repo-relative paths, which open the worktree's copy of the file.
FIND = [search_codebase, find_symbol, read_lines]

# Each task type gets a minimal, focused toolset — least privilege per task.
TOOLS_BY_TYPE: dict[str, list] = {
    "design":    [read_file, write_file, list_directory, *FIND],
    "implement": [read_file, write_file, append_file, list_directory, *FIND],
    "test":      [read_file, write_file, append_file, list_directory, run_command, *FIND],
    "review":    [read_file, write_file, *FIND],
    "integrate": [read_file, write_file, append_file, list_directory, run_command, *FIND],
    "configure": [read_file, write_file, list_directory, *FIND],
}

_DEFAULT_TOOLS = [read_file, write_file, list_directory, *FIND]
_ERROR_CHARS = 2000   # how much of the previous failure a retry sees


def build_system_prompt(task: dict, dep_outputs: list[dict]) -> str:
    """
    Build a task-specific system prompt.

    dep_outputs — results from completed dependency tasks — are injected as
    "PRIOR TASK OUTPUTS" so the agent has memory of what was already built,
    without needing a shared checkpointer across tasks.
    """
    criteria_lines = "\n".join(f"  - {c}" for c in json_list(task.get("acceptance_criteria")))
    output_files = "\n".join(f"  - {f}" for f in json_list(task.get("output_files")))

    retry_context = ""
    if task.get("retry_count", 0) > 0 and task.get("error"):
        # Reflection: a retry with the same prompt tends to fail the same way,
        # so tell the agent exactly why the previous attempt was rejected.
        retry_context = f"""

PREVIOUS ATTEMPT FAILED (this is attempt {task['retry_count'] + 1} of {task.get('max_retries', 3)}):
{task['error'][:_ERROR_CHARS]}
Fix these problems. Files from the previous attempt may already exist — read them
and correct them rather than starting over."""

    prior_context = ""
    if dep_outputs:
        parts = [
            f"[{dep['id']}] {dep['title']}\n{dep['result'] or '(no output recorded)'}"
            for dep in dep_outputs
        ]
        prior_context = "\n\nPRIOR TASK OUTPUTS (from your dependencies):\n" + "\n\n".join(parts)

    return f"""You are an expert software engineer executing a single well-defined task.
Be thorough and complete. Always write all files to disk before finishing.

TASK ID:   {task['id']}
TASK TYPE: {task['task_type']}
TITLE:     {task['title']}

DESCRIPTION:
{task['description']}

FILES TO PRODUCE:
{output_files or '  (none specified)'}

ACCEPTANCE CRITERIA (your output must satisfy ALL of these):
{criteria_lines or '  (none specified)'}{prior_context}{retry_context}

When done, summarise what you implemented in 3-5 bullet points.
Do NOT leave any implementation incomplete."""


def _content(msg) -> str:
    # Claude returns content as a list of blocks; OpenAI and Gemini return a plain string.
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content or ""


def opening_message(task: dict) -> str:
    """The first user message: retry, new project, or change to an existing codebase."""
    if task.get("retry_count", 0) > 0:
        return (
            f"{task['description']}\n\n"
            "Your previous attempt failed — the reason is in your instructions. Read the files\n"
            "you wrote, fix the problems, and write the corrected files with write_file."
        )
    if not get_source_files(str(workspace_root())):
        # A new project: create files directly rather than explore an empty directory.
        return (
            f"{task['description']}\n\n"
            "The project directory is empty — there is no existing code to read.\n"
            "You must CREATE all output files from scratch using write_file.\n"
            "Do not spend time listing directories. Go directly to writing the output files."
        )
    return (
        f"{task['description']}\n\n"
        "This is an existing codebase. First find the code your task touches: search_codebase\n"
        "for behaviour, find_symbol for an exact function or class, read_lines to open the part\n"
        "a result points to. Search shows the code as it was when the plan started — files that\n"
        "earlier tasks wrote are in PRIOR TASK OUTPUTS; open those with read_file. Then write\n"
        "your files with write_file, keeping the existing style and changing only what the task needs."
    )


async def run_task_agent(task: dict, dep_outputs: list[dict] | None = None) -> str:
    """Build a fresh agent for one task, run it, and return its summary of what it did."""
    llm   = chat_model(temperature=0, max_tokens=3000)
    tools = TOOLS_BY_TYPE.get(task.get("task_type", ""), _DEFAULT_TOOLS)
    agent = create_agent(
        llm, tools=tools, system_prompt=build_system_prompt(task, dep_outputs or []),
        middleware=[*agent_limits("task_tool_calls"), *fallback_middleware(temperature=0, max_tokens=3000)],
    )
    logger.info(f"Building agent for task {task['id']} (type={task['task_type']}, tools={[t.name for t in tools]})")

    final_state = None
    async for step in agent.astream({"messages": [{"role": "user", "content": opening_message(task)}]},
                                    stream_mode="values"):
        last_msg   = step["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", None)
        if tool_calls:
            logger.info(f"Task {task['id']} → tool calls: {[tc['name'] for tc in tool_calls]}")
        else:
            logger.info(f"Task {task['id']} → {type(last_msg).__name__}: {_content(last_msg)[:200]}")
        final_state = step

    # The final AIMessage can be empty for reasoning models (reasoning tokens are internal).
    # Walk backwards to find the last message that actually has visible content.
    output = next(
        (_content(msg) for msg in reversed(final_state["messages"])
         if type(msg).__name__ == "AIMessage" and _content(msg).strip()),
        "",
    )
    if not output.strip():
        raise ValueError("Agent returned empty output — it did not write any files or produce a summary")

    logger.info(f"Task {task['id']} agent returned {len(output)} chars")
    return output
