from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from repopilot.config import config
from repopilot.guardrails.tool_safety import check_path, resolve_path, run_guarded
from repopilot.agents.task_spec import json_list

# ------------------------------------------------------------------
# Verification — objective checks the agent can't talk its way past
# ------------------------------------------------------------------
# Run after the task agent finishes and BEFORE the critic agent:
#   1. every file in output_files exists
#   2. produced .py / .json / .js files parse
#   3. the task's verify_command (approved with the plan) exits 0
# A failure skips the critic and goes straight to retry, with the report
# fed back to the agent. On success the report is evidence for the critic.
# ------------------------------------------------------------------

_OUTPUT_TAIL_CHARS = 1500   # tail of verify_command output kept in the report


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

    files = [f for f in json_list(task.get("output_files")) if check_path(f) is None]
    missing = [f for f in files if not resolve_path(f).is_file()]
    if missing:
        passed = False
        lines.append(f"✗ missing output files: {', '.join(missing)}")
    elif files:
        lines.append(f"✓ all {len(files)} output files exist")

    for f in files:
        if f not in missing and (error := _syntax_error(resolve_path(f))):
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
