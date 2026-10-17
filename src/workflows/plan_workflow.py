from __future__ import annotations

import asyncio
import json

from rich.console import Console
from rich.markup import escape

from pathlib import Path

from repopilot.config import config
from repopilot.workflows.task_store import SQLiteTaskStore
from repopilot.workflows.task_execution import execute_task
from repopilot.agents.planner_agent import ExecutionPlan, create_plan, replan, validate_revision
from repopilot.workflows.approval import present_plan_for_approval
from repopilot.workflows.recovery import RecoveryManager
from repopilot.workflows.worktree import open_worktree, review_changes
from repopilot.guardrails.budgets import BudgetExceeded, budget_exhausted, token_budget
from repopilot.guardrails.tool_safety import use_workspace
from repopilot.retrieval.indexing.factory import get_indexer
from repopilot.observability.logger import get_logger

logger = get_logger(__name__)
console = Console()


class TaskOrchestrator:
    """
    Main execution loop: finds tasks whose dependencies are met,
    claims them atomically, and dispatches them to subtask agents.

    When a task fails all its retries, it asks the planner to replace that
    task and the tasks depending on it (with human approval), up to
    tasks.max_replans times per run; otherwise its dependents are blocked.

    Serial by default (max_concurrent=1) for cost control and determinism.
    """

    def __init__(self, store: SQLiteTaskStore, max_concurrent: int = 1,
                 max_replans: int | None = None) -> None:
        self.store = store
        self.max_concurrent = max_concurrent
        if max_replans is None:
            max_replans = int(config.get("tasks", {}).get("max_replans", 1))
        self.replans_left = max_replans
        self._handled_failures: set[str] = set()   # failed task IDs already considered for re-planning

    async def run(self, project_id: str) -> bool:
        """
        Loop until all tasks reach a terminal state. Returns True when the
        project finished, False when it paused because the token budget ran out
        (its progress is kept; the next /plan resumes it).

        Each iteration:
          1. Check progress — exit if nothing pending or in-progress
          2. Get ready tasks (all deps completed)
          3. Dispatch up to max_concurrent tasks
          4. Sleep 5s and repeat if waiting on in-progress tasks
        """
        console.print(f"\n[bold blue]Starting execution for project {project_id}[/bold blue]\n")

        while True:
            if message := budget_exhausted():
                console.print(f"\n[yellow]⏸ {message}[/yellow]")
                console.print("[yellow]  Progress is saved — run /plan with the same goal to continue "
                              "(raise budgets.plan_tokens in config.yaml to allow more).[/yellow]")
                return False

            # A task that used up its retries: try to re-plan around it first.
            for failed in self._new_failures(project_id):
                self._try_replan(project_id, failed)

            # A failed task means its dependents can never run — mark them
            # blocked so the loop can terminate instead of waiting forever.
            self.store.block_unreachable_tasks(project_id)

            progress    = self.store.get_progress(project_id)
            pending     = progress.get("pending", 0)
            in_progress = progress.get("in_progress", 0)
            completed   = progress.get("completed", 0)
            failed      = progress.get("failed", 0)
            blocked     = progress.get("blocked", 0)
            replaced    = progress.get("replaced", 0)
            total       = sum(progress.values()) - replaced

            console.print(
                f"[dim]Progress: {completed}/{total} completed"
                f" · {in_progress} in-progress"
                f" · {pending} pending"
                f" · {failed} failed"
                f" · {blocked} blocked"
                + (f" · {replaced} replaced" if replaced else "") + "[/dim]"
            )
            # all tasks are done - nothing in pending and nothing is in progress
            if pending == 0 and in_progress == 0:
                _print_final_summary(progress)
                # Mark the project finished so the next /plan starts fresh
                # instead of resuming this one.
                status = "completed" if failed == 0 and blocked == 0 else "failed"
                self.store.finish_project(project_id, status)
                return True

            ready = self.store.get_ready_tasks(project_id)
            if not ready:
                if in_progress > 0:
                    console.print("[dim]⏳ Waiting for in-progress tasks...[/dim]")
                else:
                    # Only reachable if a task depends on an ID that doesn't exist
                    # (failed dependencies are already handled by block_unreachable_tasks).
                    console.print("[yellow]⚠ No tasks are ready — some depend on tasks that don't exist.[/yellow]")
                    console.print("[yellow]  Use /task_status to inspect.[/yellow]")
                    self.store.finish_project(project_id, "failed")
                    return True
                await asyncio.sleep(5)
                continue

            batch = ready[:self.max_concurrent]
            await asyncio.gather(*[self._execute(task) for task in batch])

    async def _execute(self, task: dict) -> None:
        """Claim and execute a single task, handling retries via fail_task."""
        # escape(): "[task_001]" would otherwise be read as a Rich style tag and vanish
        label = escape(f"[{task['id']}] {task['title']}")
        console.print(f"\n[bold]▶ Starting:[/bold] {label}")

        project_id = task["project_id"]
        if not self.store.claim_task(project_id, task["id"]):
            console.print(f"[dim]⚠ Task {task['id']} already claimed — skipping[/dim]")
            return

        try:
            # Fetch what dependency tasks actually produced and inject into the agent.
            dep_ids     = json.loads(task.get("depends_on") or "[]")
            dep_outputs = self.store.get_dep_results(project_id, dep_ids)

            result = await execute_task(task, dep_outputs=dep_outputs)
            self.store.complete_task(project_id, task["id"], result)
            console.print(f"[green]✅ Completed:[/green] {label}")

        except BudgetExceeded:
            # Interrupted, not judged: back to pending without using a retry.
            self.store.release_task(project_id, task["id"])
            console.print(f"[yellow]⏸ Paused:[/yellow]    {label}")

        except Exception as e:
            error_msg = str(e)
            self.store.fail_task(project_id, task["id"], error_msg)
            console.print(f"[red]❌ Failed:[/red]    {label}: {escape(error_msg[:120])}")
            logger.error(f"Task {task['id']} failed: {error_msg}")


    # ------------------------------------------------------------------
    # Re-planning
    # ------------------------------------------------------------------

    def _new_failures(self, project_id: str) -> list[dict]:
        failures = [t for t in self.store.get_all_tasks(project_id)
                    if t["status"] == "failed" and t["id"] not in self._handled_failures]
        self._handled_failures.update(t["id"] for t in failures)
        return failures

    def _try_replan(self, project_id: str, failed: dict) -> bool:
        """
        Replace a permanently failed task and everything downstream of it with
        new tasks from the planner, if the user approves. Returns True if applied.
        """
        if self.replans_left <= 0:
            return False

        tasks      = self.store.get_all_tasks(project_id)
        downstream = _downstream_ids(tasks, failed["id"])
        affected   = [t for t in tasks if t["id"] in downstream and t["status"] in ("pending", "blocked")]
        unaffected = [t for t in tasks if t["id"] not in downstream and t["status"] in ("pending", "in_progress")]
        completed  = [t for t in tasks if t["status"] == "completed"]
        project    = self.store.get_project(project_id)

        console.print(
            f"\n[yellow]⟳ {failed['id']} failed after {failed['retry_count']} attempts — "
            f"asking the planner for a revised plan ({len(affected)} dependent task(s) affected)...[/yellow]"
        )
        try:
            revision = replan(project["goal"], completed, failed, affected, unaffected,
                              existing_ids=[t["id"] for t in tasks])
        except Exception as e:
            logger.error(f"Re-planning failed: {e}")
            console.print(f"[red]Re-planning failed: {e}[/red]")
            return False

        problems = validate_revision(
            revision,
            dependable_ids={t["id"] for t in completed + unaffected},
            existing_ids={t["id"] for t in tasks},
        )
        if problems:
            console.print("[red]The revised plan is invalid, so it was not used:[/red]")
            for problem in problems:
                console.print(f"  • {problem}")
            return False

        proposal = ExecutionPlan(
            project_name=project["name"],
            goal_summary=revision.reason,
            tech_stack=[],
            total_estimated_hours=round(sum(t.estimated_minutes for t in revision.tasks) / 60, 1),
            tasks=revision.tasks,
            risks=[],
            assumptions=[],
        )
        approved = present_plan_for_approval(proposal, title="Revised plan",
                                             reject_label="[R]eject (stop and keep the failure)")
        if approved is None:
            console.print("[dim]Revision rejected — tasks depending on the failure will be blocked.[/dim]")
            return False

        self.replans_left -= 1
        self.store.apply_replan(project_id, [failed["id"], *(t["id"] for t in affected)],
                                approved.tasks, revision.reason)
        console.print(f"[green]✓ Revised plan applied: {len(approved.tasks)} new task(s).[/green]")
        return True


def _downstream_ids(tasks: list[dict], root_id: str) -> set[str]:
    """IDs of every task that depends on root_id, directly or transitively."""
    dependents: dict[str, list[str]] = {}
    for t in tasks:
        for dep in json.loads(t.get("depends_on") or "[]"):
            dependents.setdefault(dep, []).append(t["id"])
    found, stack = set(), [root_id]
    while stack:
        for child in dependents.get(stack.pop(), []):
            if child not in found:
                found.add(child)
                stack.append(child)
    return found


async def handle_plan_command(goal: str) -> None:
    """
    Full /plan flow — entry point called by main.py.

      1. Check DB for an unfinished project. If its goal matches, resume it;
         if the goal is different, ask whether to resume it or start the new one
      2. Otherwise: plan → human approval loop → persist
      3. Execute in an isolated git worktree, then review the diff:
         apply it to your checkout, keep the branch, or discard it

    Every LLM call in the run counts against budgets.plan_tokens.
    """
    with token_budget("plan_tokens"):
        await _run_plan(goal)


async def _run_plan(goal: str) -> None:
    db_path = config.get("tasks", {}).get("db_path", ".repopilot/tasks.db")
    store   = SQLiteTaskStore(db_path)
    recover = RecoveryManager(store)

    project_id = store.get_latest_approved_project()
    if project_id:
        project = store.get_project(project_id)
        if project and project["goal"].strip().lower() != goal.strip().lower():
            console.print(
                f"\n[yellow]An unfinished project exists:[/yellow] {project['name']}\n"
                f"[dim]  Goal: {project['goal']}[/dim]"
            )
            choice = input("Resume it (r) or abandon it and start your new goal (n)? [r/n] ").strip().lower()
            if choice.startswith("n"):
                store.finish_project(project_id, "abandoned")
                project_id = None

    if project_id:
        console.print(f"\n[yellow]↩ Resuming existing project {project_id}...[/yellow]")
        recovered = recover.recover(project_id)
        if recovered:
            console.print(f"[dim]Recovered {recovered} crashed task(s)[/dim]")
    else:
        console.print("\n[dim]Planning with LLM (this may take a moment)...[/dim]")
        extra_context = ""
        approved_plan = None

        while approved_plan is None:
            raw_plan      = create_plan(goal, extra_context)
            approved_plan = present_plan_for_approval(raw_plan)
            if approved_plan is None:
                extra_context = input("What should change in the re-plan?\n> ").strip()
                console.print("\n[dim]Re-planning with your feedback...[/dim]")

        project_id = store.create_project(goal, approved_plan)
        console.print(f"\n[dim]Project {project_id} saved.[/dim]")

    worktree = open_worktree(project_id)
    if worktree is not None:
        console.print(f"[dim]Working in a separate git worktree on branch {worktree.branch} — "
                      "your checkout is untouched until you apply the changes.[/dim]")

    orchestrator = TaskOrchestrator(store, max_concurrent=1)
    with use_workspace(worktree.workspace if worktree else Path.cwd()):
        finished = await orchestrator.run(project_id)

    if not finished:
        if worktree is not None:
            console.print(f"[dim]The work so far is in {worktree.path}[/dim]")
        return
    if worktree is not None and not review_changes(worktree, store.get_project(project_id)["goal"]):
        return

    # Re-index the project directory so /ask follow-up questions can find
    # the files that were just generated by the plan tasks.
    console.print("\n[dim]Re-indexing generated files so /ask can query them...[/dim]")
    try:
        get_indexer()(str(Path.cwd()))
        console.print("[green]✓ Index updated — you can now use /ask to ask about the generated code.[/green]")
    except Exception as e:
        logger.warning(f"Re-index after /plan failed: {e}")
        console.print(f"[yellow]⚠ Re-index failed: {e}[/yellow]")


def _print_final_summary(progress: dict[str, int]) -> None:
    completed = progress.get("completed", 0)
    failed    = progress.get("failed", 0)
    blocked   = progress.get("blocked", 0)
    replaced  = progress.get("replaced", 0)

    if failed == 0 and blocked == 0:
        console.print(f"\n[bold green]🎉 All {completed} tasks completed successfully![/bold green]")
        if replaced:
            console.print(f"[dim]  ({replaced} task(s) were replaced by a revised plan along the way)[/dim]")
    else:
        console.print(f"\n[bold yellow]⚠ Execution finished with issues:[/bold yellow]")
        console.print(f"  ✅ Completed: {completed}")
        if failed:
            console.print(f"  ❌ Failed:    {failed}  (run /task_status to review)")
        if blocked:
            console.print(f"  🚫 Blocked:   {blocked}  (dependencies failed)")
        if replaced:
            console.print(f"  ⟳ Replaced:  {replaced}  (by a revised plan)")