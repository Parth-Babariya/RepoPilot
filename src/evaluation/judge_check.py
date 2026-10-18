from __future__ import annotations

import json
from pathlib import Path

from repopilot.config import config
from repopilot.evaluation.metrics import load_questions
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.prompt import Prompt
from rich.table import Table

console = Console()

# ------------------------------------------------------------------
# /judge_check — how far can the LLM judge be trusted?
# ------------------------------------------------------------------
# The judge's faithfulness/correctness scores are only meaningful if they
# match what a person would say. /judge_check shows the answers from an
# /eval run one at a time — with the judge's scores hidden, so they can't
# bias you — and asks two yes/no questions. Labels are saved, so you can
# stop any time and continue later. It then reports, per metric, how often
# the judge agrees with you and Cohen's kappa (agreement beyond chance:
# 1 = perfect, 0 = no better than chance).
# ------------------------------------------------------------------

JUDGE_PASS = 0.5  # a judge score at or above this counts as "yes"


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """Agreement between two raters beyond what chance would give. None if undefined."""
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(a == b for a, b in pairs) / n
    p_a = sum(a for a, _ in pairs) / n
    p_b = sum(b for _, b in pairs) / n
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    if expected == 1:
        return None  # both raters gave the same label every time — kappa is undefined
    return (observed - expected) / (1 - expected)


def agreement(results: list[dict], labels: dict, metric: str) -> dict:
    """Compare judge scores with human labels for one metric ("faithfulness" or "correctness")."""
    pairs, disagreements = [], []
    for r in results:
        label = labels.get(r["id"], {}).get(metric)
        if label is None or r.get(metric) is None:
            continue
        judge_says = r[metric] >= JUDGE_PASS
        pairs.append((judge_says, label))
        if judge_says != label:
            disagreements.append({"id": r["id"], "judge": r[metric], "human": label})
    n = len(pairs)
    return {
        "n": n,
        "agreement": (sum(a == b for a, b in pairs) / n) if n else None,
        "kappa": cohen_kappa(pairs),
        "disagreements": disagreements,
    }


# ── files ───────────────────────────────────────────────────────


def _results_dir() -> Path:
    return Path(config.get("eval", {}).get("results_dir", ".repopilot/evals"))


def _latest_run() -> Path | None:
    runs = sorted(_results_dir().glob("run_*.json"))
    return runs[-1] if runs else None


def _labels_path(run_path: Path) -> Path:
    return run_path.with_name(run_path.stem.replace("run_", "labels_") + ".json")


def _expected_facts(run: dict) -> dict[str, list[str]]:
    try:
        return {q.id: q.expected_facts for q in load_questions(run["questions_file"])}
    except Exception:
        return {}


# ── interactive labelling ───────────────────────────────────────


def _ask_yes_no(question: str) -> bool | None:
    """True / False, None to skip, raises StopIteration to quit."""
    answer = Prompt.ask(f"{question} [y/n/s=skip/q=quit]", choices=["y", "n", "s", "q"], show_choices=False).lower()
    if answer == "q":
        raise StopIteration
    return {"y": True, "n": False}.get(answer)


def _label(run: dict, labels: dict, save) -> None:
    facts = _expected_facts(run)
    todo = [r for r in run["results"] if "error" not in r and r["id"] not in labels]
    console.print(f"[dim]{len(todo)} answers to label (judge scores are hidden). q quits; progress is saved.[/dim]")
    for i, r in enumerate(todo, 1):
        console.rule(f"[bold]{i}/{len(todo)} · {r['id']}[/bold]")
        console.print(f"[bold]Question:[/bold] {escape(r['question'])}")
        if facts.get(r["id"]):
            console.print(f"[dim]A correct answer should mention: {escape(', '.join(facts[r['id']]))}[/dim]")
        console.print(Markdown(r.get("answer") or "(empty answer)"))
        try:
            correct = _ask_yes_no("Is the answer correct and complete?")
            invented = _ask_yes_no("Does it state anything false or invented about the code?")
        except StopIteration:
            break
        labels[r["id"]] = {"correctness": correct, "faithfulness": None if invented is None else not invented}
        save()


def _report(run: dict, labels: dict) -> None:
    table = Table(show_header=True, header_style="bold", title="LLM judge vs your labels")
    for col in ("Metric", "Labelled", "Agreement", "Cohen's kappa"):
        table.add_column(col, justify="left" if col == "Metric" else "right")
    stats = {m: agreement(run["results"], labels, m) for m in ("faithfulness", "correctness")}
    for metric, s in stats.items():
        table.add_row(
            metric.capitalize(),
            str(s["n"]),
            "–" if s["agreement"] is None else f"{s['agreement']:.0%}",
            "–" if s["kappa"] is None else f"{s['kappa']:.2f}",
        )
    console.print(table)
    console.print(
        f"[dim]Judge score ≥ {JUDGE_PASS} counts as yes. "
        "Kappa: >0.6 substantial, 0.4–0.6 moderate, <0.4 weak agreement.[/dim]"
    )
    for metric, s in stats.items():
        for d in s["disagreements"]:
            console.print(
                f"  [yellow]≠[/yellow] {d['id']} · {metric}: judge {d['judge']:.2f}, "
                f"you said {'yes' if d['human'] else 'no'}"
            )


def run_judge_check(run_file: str | None = None) -> None:
    run_path = Path(run_file) if run_file else _latest_run()
    if run_path is None or not run_path.exists():
        console.print("[yellow]No /eval run found — run /eval first, then /judge_check.[/yellow]")
        return

    run = json.loads(run_path.read_text(encoding="utf-8"))
    labels_path = _labels_path(run_path)
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}

    def save() -> None:
        labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")

    console.print(f"\n[bold]Judge check[/bold] — eval run {run['run_id']} ({run_path})")
    _label(run, labels, save)
    _report(run, labels)
    console.print(f"[dim]Labels saved to {labels_path}[/dim]\n")
