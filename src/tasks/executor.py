from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from langchain.agents import create_agent


from repopilot.tools.filesystem_tools import (
    append_file,
    file_exists,
    list_directory,
    read_file,
    write_file,
)
from repopilot.tools.terminal_tools import run_command
from repopilot.tools.safety import check_path, run_guarded
from repopilot.config import config
from repopilot.observability.logger import get_logger
from repopilot.llm.factory import chat_model, get_judge_llm

logger = get_logger(__name__)


# Each task type gets a minimal, focused toolset — least privilege per task.
_TOOLS_BY_TYPE: dict[str, list] = {
    "design":    [read_file, write_file, list_directory],
    "implement": [read_file, write_file, append_file, list_directory],
    "test":      [read_file, write_file, append_file, list_directory, run_command],
    "review":    [read_file, write_file],
    "integrate": [read_file, write_file, append_file, list_directory, run_command],
    "configure": [read_file, write_file, list_directory, file_exists],
}

_DEFAULT_TOOLS = [read_file, write_file, list_directory]


def _parse_json_field(val) -> list:
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return []
    return val or []


def _build_system_prompt(task: dict, dep_outputs: list[dict]) -> str:
    """
    Build a task-specific system prompt.

    dep_outputs — results from completed dependency tasks — are injected as
    "PRIOR TASK OUTPUTS" so the agent has memory of what was already built,
    without needing a shared checkpointer across tasks.
    """
    criteria_lines = "\n".join(
        f"  - {c}" for c in _parse_json_field(task.get("acceptance_criteria"))
    )
    output_files = "\n".join(
        f"  - {f}" for f in _parse_json_field(task.get("output_files"))
    )

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


# ------------------------------------------------------------------
# Verification — objective checks the agent can't talk its way past
# ------------------------------------------------------------------
# Run after the agent finishes and BEFORE the LLM judge:
#   1. every file in output_files exists
#   2. produced .py / .json / .js files parse
#   3. the task's verify_command (approved with the plan) exits 0
# A failure skips the judge and goes straight to retry, with the report
# fed back to the agent. On success the report is evidence for the judge.
# ------------------------------------------------------------------

_ERROR_CHARS        = 2000   # how much of a failure report a retry sees
_OUTPUT_TAIL_CHARS  = 1500   # tail of verify_command output kept in the report
_JUDGE_FILE_CHARS   = 6000   # total file content shown to the judge


@dataclass
class Verification:
    passed: bool
    report: str


def _syntax_error(path: Path) -> str | None:
    suffix = path.suffix.lower()
    try:
        if suffix == ".py":
            compile(path.read_text(encoding="utf-8"), str(path), "exec")   # no .pyc written
        elif suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif suffix in (".js", ".mjs", ".cjs") and shutil.which("node"):
            result = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return result.stderr.strip()[-500:]
    except SyntaxError as e:
        return f"line {e.lineno}: {e.msg}"
    except (ValueError, OSError, subprocess.TimeoutExpired) as e:
        return str(e)[-500:]
    return None


def verify_task(task: dict) -> Verification:
    lines, passed = [], True

    files = [f for f in _parse_json_field(task.get("output_files")) if check_path(f) is None]
    missing = [f for f in files if not Path(f).is_file()]
    if missing:
        passed = False
        lines.append(f"✗ missing output files: {', '.join(missing)}")
    elif files:
        lines.append(f"✓ all {len(files)} output files exist")

    for f in files:
        if f not in missing and (error := _syntax_error(Path(f))):
            passed = False
            lines.append(f"✗ {f} does not parse: {error}")

    command = (task.get("verify_command") or "").strip()
    if command:
        timeout = int(config.get("tasks", {}).get("verify_timeout_seconds", 120))
        code, output = run_guarded(command, timeout=timeout, pre_approved=True)
        ok = code == 0
        passed = passed and ok
        tail = output[-_OUTPUT_TAIL_CHARS:]
        lines.append(f"{'✓' if ok else '✗'} `{command}` exited {code if code is not None else 'without running'}\n{tail}")

    return Verification(passed, "\n".join(lines) or "(no checks configured for this task)")


def _file_excerpts(task: dict) -> str:
    files = [f for f in _parse_json_field(task.get("output_files"))
             if check_path(f) is None and Path(f).is_file()]
    if not files:
        return "(no output files)"
    per_file = max(500, _JUDGE_FILE_CHARS // len(files))
    parts = []
    for f in files:
        try:
            text = Path(f).read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            text = f"(could not read: {e})"
        cut = "\n… (truncated)" if len(text) > per_file else ""
        parts.append(f"=== {f} ===\n{text[:per_file]}{cut}")
    return "\n\n".join(parts)[:_JUDGE_FILE_CHARS + 500]


# ------------------------------------------------------------------
# LLM-as-judge
# ------------------------------------------------------------------

class _JudgeVerdict(BaseModel):
    passed: bool
    score: int    # 0-10
    reason: str   # one sentence


_JUDGE_SYSTEM_PROMPT = """\
You are a code review judge. Given a task description, its acceptance criteria,
the files the agent produced, the results of automatic checks, and the agent's
own summary, decide whether the task was completed satisfactorily.
Base your verdict on the files and check results, not on the agent's claims.

Score 0-10:
  8-10 → passed (all acceptance criteria met)
  5-7  → borderline (minor gaps, still usable)
  0-4  → failed (criteria not met, re-execution needed)

Set passed=true if score >= 6. One short sentence for reason."""


async def _judge_task(task: dict, agent_output: str, verification: Verification) -> _JudgeVerdict:
    """
    Lightweight LLM-as-judge. Uses judge_model (cheaper) from config.
    Returns a structured verdict with passed/score/reason.
    """
    llm = get_judge_llm("plan-judge", temperature=0, max_tokens=1000)
    judge_agent = create_agent(
        llm,
        tools=[],
        system_prompt=_JUDGE_SYSTEM_PROMPT,
        response_format=_JudgeVerdict,
    )

    criteria = _parse_json_field(task.get("acceptance_criteria"))
    user_message = f"""TASK: {task['title']}
DESCRIPTION: {task['description']}
ACCEPTANCE CRITERIA: {json.dumps(criteria, indent=2)}

AUTOMATIC CHECKS (run by the system, not the agent):
{verification.report}

FILES PRODUCED:
{_file_excerpts(task)}

AGENT'S OWN SUMMARY:
{agent_output[:2000]}"""

    result = await judge_agent.ainvoke({"messages": [{"role": "user", "content": user_message}]})
    return result["structured_response"]


# ------------------------------------------------------------------
# Main entry point — called by orchestrator._execute()
# ------------------------------------------------------------------

async def run_subtask_agent(task: dict, dep_outputs: list[dict] | None = None) -> str:
    """
    Build a fresh agent for a single task and invoke it.

    After the agent returns, objective checks run first (files exist and
    parse, verify_command passes), then an LLM judge grades the files
    against acceptance_criteria. Either failing raises ValueError, so the
    orchestrator's retry logic kicks in and the reason reaches the retry.
    """
    llm = chat_model(temperature=0, max_tokens=3000)

    tools         = _TOOLS_BY_TYPE.get(task.get("task_type", ""), _DEFAULT_TOOLS)
    system_prompt = _build_system_prompt(task, dep_outputs or [])

    logger.info(f"Building agent for task {task['id']} (type={task['task_type']}, tools={[t.name for t in tools]})")

    agent = create_agent(llm, tools=tools, system_prompt=system_prompt)

    if task.get("retry_count", 0) > 0:
        user_message = (
            f"{task['description']}\n\n"
            "Your previous attempt failed — the reason is in your instructions. Read the files\n"
            "you wrote, fix the problems, and write the corrected files with write_file."
        )
    else:
        # The project directory may be empty on first run — tell the agent to create
        # files directly rather than spending turns exploring an empty directory.
        user_message = (
            f"{task['description']}\n\n"
            "The project directory may be empty — there is no existing code to read.\n"
            "You must CREATE all output files from scratch using write_file.\n"
            "Do not spend time listing directories. Go directly to writing the output files."
        )

    final_state = None
    async for step in agent.astream(
        {"messages": [{"role": "user", "content": user_message}]},
        stream_mode="values",
    ):
        last_msg   = step["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", None)
        if tool_calls:
            logger.info(f"Task {task['id']} → tool calls: {[tc['name'] for tc in tool_calls]}")
        else:
            logger.info(f"Task {task['id']} → {type(last_msg).__name__}: {str(getattr(last_msg, 'content', ''))[:200]}")
        final_state = step

    def _get_content(msg) -> str:
        # Claude returns content as a list of blocks; OpenAI returns a plain string.
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
        return content or ""

    # The final AIMessage can be empty for reasoning models (reasoning tokens are internal).
    # Walk backwards to find the last message that actually has visible content.
    output = next(
        (_get_content(msg) for msg in reversed(final_state["messages"])
         if type(msg).__name__ == "AIMessage" and _get_content(msg).strip()),
        ""
    )

    if not output.strip():
        raise ValueError("Agent returned empty output — it did not write any files or produce a summary")

    logger.info(f"Task {task['id']} agent returned {len(output)} chars")

    # ── Objective checks, then LLM-as-judge ──────────────────────────
    verification = verify_task(task)
    logger.info(f"Verification for {task['id']}: passed={verification.passed}\n{verification.report}")
    if not verification.passed:
        raise ValueError(f"Automatic checks failed:\n{verification.report[:_ERROR_CHARS]}")

    verdict = await _judge_task(task, output, verification)
    logger.info(
        f"Judge verdict for {task['id']}: score={verdict.score}, "
        f"passed={verdict.passed}, reason={verdict.reason}"
    )
    if not verdict.passed:
        raise ValueError(f"Judge rejected output (score={verdict.score}/10): {verdict.reason}")

    return output