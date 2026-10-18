from __future__ import annotations

from repopilot.config import config
from repopilot.workflows.task_store import SQLiteTaskStore, TaskStatus
from rich.console import Console
from rich.markup import escape
from rich.table import Table

console = Console()

# Keyed by TaskStatus; a test checks every status has a style
_STATUS_STYLE = {
    TaskStatus.COMPLETED: "green",
    TaskStatus.FAILED: "red",
    TaskStatus.IN_PROGRESS: "yellow",
    TaskStatus.PENDING: "dim",
    TaskStatus.BLOCKED: "red",
    TaskStatus.REPLACED: "dim",
}


def show_task_status() -> None:
    """Print a Rich table of all tasks for the most recent project (finished or not)."""
    db_path = config.get("tasks", {}).get("db_path", ".repopilot/tasks.db")
    store = SQLiteTaskStore(db_path)
    project_id = store.get_latest_project()

    if not project_id:
        console.print("[yellow]No project found. Run /plan <goal> first.[/yellow]")
        return

    project = store.get_project(project_id)

    tasks = store.get_all_tasks(project_id)
    progress = store.get_progress(project_id)
    replaced = progress.get("replaced", 0)
    total = sum(progress.values()) - replaced  # replaced tasks are history, not work left
    done = progress.get("completed", 0)

    console.print(f"\n[bold]Project:[/bold] {escape(project['name'])} [dim]({project_id} · {project['status']})[/dim]")
    console.print(
        f"[dim]Progress: {done}/{total} completed"
        + (f" · {replaced} replaced by a revised plan" if replaced else "")
        + "[/dim]\n"
    )

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", width=4)
    table.add_column("ID", width=12)
    table.add_column("Type", width=10)
    table.add_column("Title", width=35)
    table.add_column("Status", width=12)
    table.add_column("Retries", width=8)
    table.add_column("Error", width=40)

    for i, task in enumerate(tasks, 1):
        style = _STATUS_STYLE.get(task["status"], "default")
        error = (task.get("error") or "")[:60]
        table.add_row(
            str(i),
            task["id"],
            task["task_type"],
            escape(task["title"]),
            f"[{style}]{task['status']}[/{style}]",
            f"{task['retry_count']}/{task['max_retries']}",
            escape(error),  # escape(): "[task_001]" etc. would otherwise be read as Rich markup
        )

    console.print(table)
