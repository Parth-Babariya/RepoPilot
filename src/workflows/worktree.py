from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from repopilot.config import config
from repopilot.observability.logger import get_logger
from rich.console import Console
from rich.markup import escape
from rich.syntax import Syntax

logger = get_logger(__name__)
console = Console()

# ------------------------------------------------------------------
# Worktree isolation for /plan
# ------------------------------------------------------------------
# Task agents write files and run commands. Instead of doing that in your
# checkout, /plan creates a git worktree — a second checkout of the repo on
# its own branch, repopilot/plan-<id> — under .repopilot/worktrees/, and every
# tool and check works there (guardrails.tool_safety.use_workspace).
#
# When the plan finishes you see what changed and choose:
#   apply   — the changes land in your checkout as staged, uncommitted
#             changes (git merge --squash); you review and commit them
#   keep    — the branch stays, to diff or merge yourself later
#   discard — the worktree and its branch are deleted
# A plan that is interrupted or paused keeps its worktree; resuming continues there.
#
# Needs a git repo with at least one commit. Otherwise, or with
# tasks.isolation: none, tasks write directly into the project folder.
# The worktree starts from your last commit, so uncommitted changes and
# untracked files (e.g. node_modules/) are not in it.
# ------------------------------------------------------------------

_AUTHOR = ["-c", "user.name=RepoPilot", "-c", "user.email=repopilot@localhost"]

# Build/test leftovers that are never part of the work (test runs create them).
_NOT_WORK = [
    f":(exclude,glob)**/{p}"
    for p in ("__pycache__/**", "*.pyc", ".pytest_cache/**", ".mypy_cache/**", "node_modules/**")
]


@dataclass
class Worktree:
    path: Path  # top level of the worktree
    workspace: Path  # where agents work: the worktree's copy of the folder RepoPilot runs in
    branch: str
    repo: Path  # top level of your checkout


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace"
    )


def isolation_enabled() -> bool:
    return (config.get("tasks", {}) or {}).get("isolation", "worktree") == "worktree"


def _repo_root(cwd: Path) -> Path | None:
    """Top level of the git repo containing cwd, or None if there's no repo or no commit yet."""
    top = _git("rev-parse", "--show-toplevel", cwd=cwd)
    if top.returncode != 0 or _git("rev-parse", "--verify", "--quiet", "HEAD", cwd=cwd).returncode != 0:
        return None
    return Path(top.stdout.strip()).resolve()


def open_worktree(project_id: str, cwd: Path | None = None) -> Worktree | None:
    """
    The worktree for a /plan project — created on first use, reused when the
    project resumes. None means tasks work in place (isolation off, or no git).
    """
    if not isolation_enabled():
        return None
    cwd = (cwd or Path.cwd()).resolve()
    repo = _repo_root(cwd)
    if repo is None:
        console.print(
            "[yellow]Not a git repository with at least one commit — "
            "tasks will write directly into this folder.[/yellow]"
        )
        return None

    short = project_id[:8]
    branch = f"repopilot/plan-{short}"
    path = cwd / ".repopilot" / "worktrees" / short
    try:
        workspace = path / cwd.relative_to(repo)
    except ValueError:
        workspace = path

    if not (path / ".git").exists():
        _git("worktree", "prune", cwd=repo)  # forget worktrees whose folder was deleted
        has_branch = _git("rev-parse", "--verify", "--quiet", f"refs/heads/{branch}", cwd=repo).returncode == 0
        args = (
            ["worktree", "add", str(path), branch]
            if has_branch
            else ["worktree", "add", "-b", branch, str(path), "HEAD"]
        )
        result = _git(*args, cwd=repo)
        if result.returncode != 0:
            logger.warning(f"git worktree add failed: {result.stderr}")
            console.print(
                f"[yellow]Couldn't create a git worktree ({escape(result.stderr.strip()[:200])}) — "
                "tasks will write directly into this folder.[/yellow]"
            )
            return None
        if _git("status", "--porcelain", "--untracked-files=no", cwd=repo).stdout.strip():
            console.print(
                "[yellow]Note: you have uncommitted changes. The plan works from your last "
                "commit, without them.[/yellow]"
            )

    workspace.mkdir(parents=True, exist_ok=True)
    return Worktree(path, workspace, branch, repo)


def commit_work(wt: Worktree, message: str) -> str:
    """Commit everything the plan produced on its branch; returns `git diff --stat` against your HEAD."""
    _git("add", "-A", "--", ".", *_NOT_WORK, cwd=wt.path)
    if _git("diff", "--cached", "--quiet", cwd=wt.path).returncode != 0:
        result = _git(*_AUTHOR, "commit", "-q", "--no-verify", "-m", message, cwd=wt.path)
        if result.returncode != 0:
            raise RuntimeError(f"git commit in the worktree failed: {result.stderr.strip()[:300]}")
    return _git("diff", "--stat", f"HEAD...{wt.branch}", cwd=wt.repo).stdout.strip()


def remove_worktree(wt: Worktree, delete_branch: bool) -> None:
    _git("worktree", "remove", "--force", str(wt.path), cwd=wt.repo)
    if delete_branch:
        _git("branch", "-D", wt.branch, cwd=wt.repo)


def apply_work(wt: Worktree) -> tuple[bool, str]:
    """Bring the branch's changes into your checkout as staged changes. (ok, git's message)"""
    result = _git("merge", "--squash", wt.branch, cwd=wt.repo)
    return result.returncode == 0, (result.stdout + result.stderr).strip()


def review_changes(wt: Worktree, goal: str) -> bool:
    """
    Show what the plan changed and let the user apply, keep or discard it.
    Returns True if the changes were applied to the checkout.
    """
    try:
        stat = commit_work(wt, f"RepoPilot /plan: {goal[:72]}")
    except RuntimeError as e:
        console.print(f"[red]{escape(str(e))}[/red]\n[dim]The worktree is kept at {wt.path}[/dim]")
        return False
    if not stat:
        console.print("[dim]The plan didn't change any files.[/dim]")
        remove_worktree(wt, delete_branch=True)
        return False

    console.print(f"\n[bold]Changes made by the plan[/bold] [dim](branch {wt.branch})[/dim]")
    console.print(escape(stat), highlight=False)
    while True:
        choice = (
            input("\n[A]pply to your checkout · [V]iew the full diff · [K]eep the branch for later · [D]iscard  > ")
            .strip()
            .lower()[:1]
        )
        if choice == "v":
            diff = _git("diff", f"HEAD...{wt.branch}", cwd=wt.repo).stdout
            console.print(Syntax(diff, "diff", theme="monokai", word_wrap=True))
        elif choice == "a":
            remove_worktree(wt, delete_branch=False)
            ok, message = apply_work(wt)
            if ok:
                _git("branch", "-D", wt.branch, cwd=wt.repo)
                console.print(
                    "[green]✓ Applied as staged, uncommitted changes — review them with "
                    "`git diff --cached`, then commit.[/green]"
                )
                return True
            console.print(f"[red]Couldn't apply the changes:[/red] {escape(message[:400])}")
            console.print(
                f"[dim]They are kept on branch {wt.branch} — commit or stash your own changes, "
                f"then run: git merge --squash {wt.branch}[/dim]"
            )
            return False
        elif choice == "k":
            remove_worktree(wt, delete_branch=False)
            console.print(
                f"[dim]Kept on branch {wt.branch}. See it with `git diff HEAD...{wt.branch}`; "
                f"apply it with `git merge --squash {wt.branch}`.[/dim]"
            )
            return False
        elif choice == "d":
            remove_worktree(wt, delete_branch=True)
            console.print("[dim]Discarded the plan's changes.[/dim]")
            return False
