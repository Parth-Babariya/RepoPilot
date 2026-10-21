from __future__ import annotations

from repopilot.observability.usage import get_usage_store
from rich.console import Console
from rich.table import Table

console = Console()

_FEATURE_LABELS = {
    "ask": "/ask",
    "plan": "/plan",
    "plan-judge": "/plan judge",
    "ask-verify": "/ask answer checks",
    "cache-check": "Cache match checks",
    "summarize": "Summarising",
    "eval": "/eval answers",
    "eval-judge": "/eval judge",
    "index": "Indexing",
    "other": "Other",
}


def _money(cost: float | None, unpriced: int = 0) -> str:
    if cost is None:
        return "?"
    text = f"${cost:.4f}" if cost < 0.01 else f"${cost:.2f}"
    return text + ("*" if unpriced else "")


def _seconds(ms: float | None) -> str:
    return f"{ms / 1000:.1f}s" if ms is not None else "–"


def _render(title: str, session: str | None) -> None:
    store = get_usage_store()
    rows = store.summary_by_feature(session)
    q = store.question_summary(session)

    console.print(f"\n[bold]{title}[/bold]")
    if not rows and not q["questions"]:
        console.print("[dim]  No usage recorded yet.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Source")
    table.add_column("Calls", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Avg time", justify="right")

    total_cost = 0.0
    any_unpriced = False
    for row in rows:
        label = _FEATURE_LABELS.get(row["feature"], row["feature"])
        if row["kind"] == "embedding":
            label = f"Embeddings ({label})"
            tokens = f"~{row['tokens']:,}"  # estimated; ASCII so legacy Windows consoles can print it
        else:
            tokens = f"{row['tokens']:,}"
        total_cost += row["cost"] or 0
        any_unpriced |= bool(row["unpriced"])
        table.add_row(
            label,
            str(row["calls"]),
            tokens,
            _money(row["cost"], row["unpriced"]),
            _seconds(row["avg_latency_ms"]),
        )
    table.add_row("[bold]Total[/bold]", "", "", f"[bold]{_money(total_cost)}[/bold]", "")
    console.print(table)

    if q["questions"]:
        hit_rate = 100 * q["hits"] / q["questions"]
        saved = f", about {_money(q['estimated_saved'])} saved" if q["estimated_saved"] else ""
        console.print(
            f"  Questions: {q['questions']} · avg answer time {_seconds(q['avg_latency_ms'])}"
            f" · cache hits {q['hits']}/{q['questions']} ({hit_rate:.0f}%){saved}"
        )
    if any_unpriced:
        models = ", ".join(store.unpriced_models(session))
        console.print(
            f"  [yellow]* Some calls have no price — add these models to usage.prices in config.yaml: {models}[/yellow]"
        )


def show_stats(session_id: str) -> None:
    """Print token usage and estimated cost for the current session and all time."""
    _render(f"This session ({session_id[:8]}...)", session_id)
    _render("All time", None)
    console.print(
        "[dim]Costs are estimates from usage.prices in config.yaml; embedding tokens are approximate.[/dim]\n"
    )
