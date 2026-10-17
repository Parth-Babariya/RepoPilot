from __future__ import annotations

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from repopilot.agents.planner_agent import ExecutionPlan, PlannedTask

console = Console()


def present_plan_for_approval(plan: ExecutionPlan, title: str = "Plan",
                              reject_label: str = "[R]eject and re-plan") -> ExecutionPlan | None:
    """
    Render the plan as a Rich table and prompt the user to:
      [A] Approve — returns the plan as-is
      [M] Modify  — edit a task's description or verify command and re-render
      [R] Reject  — returns None so the caller re-plans with feedback

    Approving also approves each task's verify command: those run without a
    further prompt, so they are listed explicitly and can be edited or removed.

    Loop continues until the user approves or rejects.
    """
    while True:
        _render_plan(plan, title)
        choice = input(f"\n[A]pprove / [M]odify task / {reject_label}: ").strip().upper()

        if choice == "A":
            return plan

        elif choice == "M":
            task_id = input("Enter task ID to modify: ").strip()
            task = next((t for t in plan.tasks if t.id == task_id), None)
            if not task:
                console.print(f"[red]Task '{task_id}' not found.[/red]")
                continue
            _modify_task(task)

        elif choice == "R":
            return None

        else:
            console.print("[yellow]Please enter A, M, or R.[/yellow]")


def _modify_task(task: PlannedTask) -> None:
    console.print(f"\nCurrent description:\n{escape(task.description)}\n")
    new_desc = input("New description (Enter to keep): ").strip()
    if new_desc:
        task.description = new_desc

    console.print(f"Current verify command: {escape(task.verify_command) or '(none)'}")
    new_cmd = input("New verify command (Enter to keep, '-' to remove): ").strip()
    if new_cmd == "-":
        task.verify_command = ""
    elif new_cmd:
        task.verify_command = new_cmd
    console.print("[green]Task updated.[/green]")


def _render_plan(plan: ExecutionPlan, title: str) -> None:
    console.print(f"\n[bold blue]{escape(title)}: {escape(plan.project_name)}[/bold blue]")
    console.print(f"[dim]{escape(plan.goal_summary)}[/dim]")
    if plan.tech_stack:
        console.print(f"[dim]Stack: {escape(', '.join(plan.tech_stack))} | Est: {plan.total_estimated_hours}h[/dim]")
    console.print()

    table = Table(show_header=True, header_style="bold")
    table.add_column("ID",          style="dim",    width=12)
    table.add_column("Type",        width=10)
    table.add_column("Title",       width=32)
    table.add_column("Depends on",  width=20)
    table.add_column("Output files",width=30)

    for task in plan.tasks:
        table.add_row(
            task.id,
            task.task_type.value,
            escape(task.title),
            ", ".join(task.depends_on) or "—",
            escape("\n".join(task.output_files)) or "—",
        )

    console.print(table)

    checks = [t for t in plan.tasks if t.verify_command.strip()]
    if checks:
        console.print("\n[bold]Verify commands[/bold] [dim](run after each task — approving the plan allows them)[/dim]")
        for task in checks:
            console.print(f"  {task.id}: [cyan]{escape(task.verify_command)}[/cyan]")

    # Assumptions are the planner's guesses about what you want — the part of the
    # plan most worth checking before approving.
    if plan.assumptions:
        console.print("\n[cyan]Assumptions (check these):[/cyan]")
        for a in plan.assumptions:
            console.print(f"  • {escape(a)}")

    if plan.risks:
        console.print("\n[yellow]Risks:[/yellow]")
        for r in plan.risks:
            console.print(f"  • {escape(r)}")
