from __future__ import annotations

import contextvars
import os
import re
import shlex
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from repopilot.config import config
from repopilot.observability.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Tool safety: what agents may touch and run
# ------------------------------------------------------------------
# Agents read untrusted text (repo files, GitHub issues), so a prompt
# injection like "run curl … | sh" or "read .env" must not just work.
#
# Files:    every path must resolve inside the project folder (symlinks
#           included); secret files (.env, .env.local, …) can't be read or
#           written; .git/ and .repopilot/ can't be written.
# Commands: an allowlist of common read-only / build / test commands runs
#           directly; anything else is shown to you for approval (refused
#           when nobody can answer, e.g. during /eval); a few destructive
#           patterns are always refused. Chaining, pipes, redirection and
#           $(…) always need approval, because they can hide a second command.
#
# Limits: allowlisted commands still run code the agent wrote (e.g. its
# tests) with your permissions. For full isolation run RepoPilot in Docker.
# ------------------------------------------------------------------

_ALWAYS_DENIED = ("rm -rf /", "rm -rf ~", "mkfs", "dd if=", ":(){", "format c:", "del /s /q c:\\")
_SHELL_OPERATORS = re.compile(r"[;&|`<>\n]|\$\(")
_SECRET_FILE = re.compile(r"^\.env(\..+)?$", re.IGNORECASE)
_SECRET_FILE_OK = {".env.example", ".env.sample", ".env.template"}
_NO_WRITE_DIRS = {".git", ".repopilot"}

_READ_ONLY_PROGRAMS = {
    "ls", "dir", "cat", "type", "head", "tail", "wc", "pwd", "echo",
    "grep", "findstr", "where", "which", "tree",
}
_GIT_READ_ONLY = {"status", "diff", "log", "show", "branch", "rev-parse", "ls-files", "blame"}
_PIP_READ_ONLY = {"list", "show", "freeze", "--version", "-V"}
_LINTERS = {"flake8", "pylint", "mypy", "pyflakes", "pycodestyle"}          # only ever read code
_FORMATTER_CHECKS = {"black": "--check", "isort": "--check-only", "ruff": "check"}   # read-only only with this
_PYTHON_MODULES = {"pytest", "unittest", "py_compile", "compileall", "json.tool", *_LINTERS}


def _settings() -> dict:
    cfg = config.get("safety", {}) or {}
    return {
        "workspace_only":         bool(cfg.get("workspace_only", True)),
        "confirm_other_commands": bool(cfg.get("confirm_other_commands", True)),
    }


# /plan points this at its git worktree while it runs (workflows/worktree.py),
# so every tool and check works in the worktree without changing the process's
# current directory — RepoPilot's own data (.repopilot/) stays where it is.
_workspace: contextvars.ContextVar[Path | None] = contextvars.ContextVar("repopilot_workspace", default=None)


def workspace_root() -> Path:
    """The folder agents work in (the project, or /plan's worktree) — they can't leave it."""
    return _workspace.get() or Path.cwd().resolve()


@contextmanager
def use_workspace(path: str | Path):
    """Make agents' relative paths and commands resolve inside `path` for this block."""
    token = _workspace.set(Path(path).resolve())
    try:
        yield
    finally:
        _workspace.reset(token)


def resolve_path(path: str | Path) -> Path:
    """An agent-supplied path, made absolute against the workspace (not the process's cwd)."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else workspace_root() / p


# ------------------------------------------------------------------
# Files
# ------------------------------------------------------------------

def _is_secret(path: Path) -> bool:
    name = path.name.lower()
    return bool(_SECRET_FILE.match(name)) and name not in _SECRET_FILE_OK


def is_secret_path(value: str) -> bool:
    """True if a string names a secret file such as .env (used to guard MCP tool arguments)."""
    return _is_secret(Path(value.strip()))


# MCP tools that only read (GitHub: get_/list_/search_…, filesystem: read_…/directory_tree).
_MCP_READ_PREFIXES = ("get_", "list_", "search_", "read_")
_MCP_READ_NAMES = {"directory_tree"}


def is_read_only_mcp_tool(name: str) -> bool:
    return name.startswith(_MCP_READ_PREFIXES) or name in _MCP_READ_NAMES


def mcp_read_only() -> bool:
    return bool((config.get("safety", {}) or {}).get("mcp_read_only", True))


def check_path(path: str, write: bool = False) -> str | None:
    """Return an error message if an agent may not use `path`, else None."""
    resolved = Path(os.path.realpath(resolve_path(path)))
    if _is_secret(resolved):
        return f"Error: access to secret files is not allowed: {path}"

    if not _settings()["workspace_only"]:
        return None
    root = workspace_root()
    if resolved != root and not resolved.is_relative_to(root):
        return f"Error: {path} is outside the project folder ({root})"
    parts = resolved.relative_to(root).parts
    if write and parts and parts[0].lower() in _NO_WRITE_DIRS:
        return f"Error: writing inside {parts[0]}/ is not allowed"
    return None


# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------

def _split(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=os.name != "nt")
    except ValueError:
        return command.split()


def _program(token: str) -> str:
    name = os.path.basename(token.strip("\"'")).lower()
    return name.removesuffix(".exe").removesuffix(".cmd")


def _workspace_script(token: str, suffixes: tuple[str, ...]) -> bool:
    return token.lower().endswith(suffixes) and check_path(token) is None


def classify_command(command: str) -> tuple[str, str]:
    """
    Decide how to treat a command: ("allow" | "ask" | "deny", reason).
    Pure function (no side effects), so the policy is unit-tested directly.
    """
    lowered = command.lower()
    if any(pattern in lowered for pattern in _ALWAYS_DENIED):
        return "deny", "destructive command"
    if _SHELL_OPERATORS.search(command):
        return "ask", "uses chaining, pipes, redirection or substitution"

    tokens = _split(command)
    if not tokens:
        return "deny", "empty command"
    program, args = _program(tokens[0]), tokens[1:]
    first = args[0].lower() if args else ""

    if program in _READ_ONLY_PROGRAMS:
        return "allow", "read-only command"
    if program == "find":
        if any(a in ("-delete", "-exec", "-execdir", "-ok") for a in args):
            return "ask", "find with -delete/-exec"
        return "allow", "read-only command"
    if program in ("python", "python3", "py"):
        if first in ("--version", "-v"):
            return "allow", "version check"
        if first == "-m" and len(args) > 1 and args[1] in _PYTHON_MODULES:
            return "allow", f"runs python -m {args[1]}"
        if first and _workspace_script(args[0], (".py",)):
            return "allow", "runs a script in the project"
        return "ask", "python with arbitrary code or module"
    if program == "pytest":
        return "allow", "runs tests"
    if program in _LINTERS:
        return "allow", "linter (read-only)"
    if program in _FORMATTER_CHECKS:
        if _FORMATTER_CHECKS[program] in args and "--fix" not in args:
            return "allow", "check-only run"
        return "ask", "may rewrite files"
    if program == "node":
        if first in ("--version", "--check", "-c") or (first and _workspace_script(args[0], (".js", ".mjs", ".cjs"))):
            return "allow", "runs a script in the project"
        return "ask", "node with arbitrary code"
    if program == "npm":
        if first in ("test", "t", "--version", "-v", "ls", "list") or (first == "run" and len(args) > 1):
            return "allow", "runs a project script"
        return "ask", "npm command that installs or changes packages"
    if program == "git":
        if first in _GIT_READ_ONLY:
            return "allow", "read-only git command"
        return "ask", "git command that changes the repository"
    if program in ("pip", "pip3"):
        if first in _PIP_READ_ONLY:
            return "allow", "read-only pip command"
        return "ask", "installs or removes packages"
    return "ask", "not on the allowlist"


# Set by the REPL while a person is at the keyboard. None (e.g. /eval) → refuse.
_confirm_handler: Callable[[str], bool] | None = None


def set_confirm_handler(handler: Callable[[str], bool] | None) -> None:
    global _confirm_handler
    _confirm_handler = handler


def _approved(command: str, reason: str) -> bool:
    if not _settings()["confirm_other_commands"] or _confirm_handler is None:
        return False
    try:
        return bool(_confirm_handler(f"Agent wants to run: {command}\n  ({reason})  Allow?"))
    except (EOFError, KeyboardInterrupt):
        return False


def _format_result(result: subprocess.CompletedProcess) -> str:
    parts = []
    if result.stdout:
        parts.append(result.stdout.rstrip())
    if result.stderr:
        parts.append(f"STDERR: {result.stderr.rstrip()}")
    if result.returncode != 0:
        parts.append(f"Exit code: {result.returncode}")
    return "\n".join(parts) if parts else "(no output)"


def run_guarded(command: str, cwd: str | None = None, timeout: int = 30,
                pre_approved: bool = False) -> tuple[int | None, str]:
    """
    Run a shell command under the policy. Returns (exit code, output);
    exit code is None when the command was refused or timed out.
    pre_approved: a person already approved this exact command (e.g. a
    plan's verify_command) — skips the prompt, but never the hard denials.
    """
    if not command or not command.strip():
        return None, "Error: command cannot be empty"
    if cwd is not None and (error := check_path(cwd)):
        return None, error
    cwd = str(resolve_path(cwd)) if cwd is not None else str(workspace_root())

    decision, reason = classify_command(command)
    if decision == "deny":
        logger.warning(f"Refused command ({reason}): {command}")
        return None, f"Error: command refused ({reason})"
    if decision == "ask" and not pre_approved and not _approved(command, reason):
        logger.info(f"Command not approved ({reason}): {command}")
        return None, f"Error: command not run — it needs approval ({reason}) and was not approved"

    logger.info(f"Running command ({'approved' if decision == 'ask' else reason}): {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True,
                                cwd=cwd, timeout=timeout)
        return result.returncode, _format_result(result)
    except subprocess.TimeoutExpired:
        return None, f"Error: command timed out after {timeout} seconds"
    except Exception as e:
        return None, f"Error: {e}"
