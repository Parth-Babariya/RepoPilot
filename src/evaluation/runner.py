from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path


from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from repopilot.config import config
from repopilot.context.retrievers.factory import get_retriever
from repopilot.context.retrievers.reranker import rerank_config
from repopilot.context.symbol_index import symbols_enabled
from repopilot.evaluation.metrics import (
    EvalQuestion, context_recall, fact_recall, load_questions, mean, reciprocal_rank, retrieval_rank,
)
from repopilot.observability.logger import get_logger
from repopilot.llm.factory import get_judge_llm
from repopilot.observability.usage import get_usage_store, track

logger  = get_logger(__name__)
console = Console()

# ------------------------------------------------------------------
# How /eval works
# ------------------------------------------------------------------
# For each question in the questions file:
#   1. Raw retrieval — call the configured retriever directly and record
#      the rank of the first chunk from an expected file (hit@k, MRR).
#   2. Full answer — run the real /ask agent on a fresh thread (no history,
#      no semantic cache) and capture its answer and every tool output.
#   3. Score — free deterministic metrics (context recall, fact recall)
#      plus an LLM judge (judge_model) for faithfulness and correctness.
#   4. Cost/latency — every call is tagged with session "eval:<run>:<id>"
#      via usage tracking, so per-question cost comes from the usage DB.
#
# Each run is saved as JSON with a config snapshot, and the summary is
# compared against the previous run so the effect of a change is visible.
# ------------------------------------------------------------------

_JUDGE_CONTEXT_CHARS = 12_000


class _JudgeVerdict(BaseModel):
    faithfulness: float = Field(description="0-1: how fully the answer's claims are supported by the retrieved code")
    correctness:  float = Field(description="0-1: how well the answer correctly and completely answers the question")
    reason: str         = Field(description="One short sentence")


_JUDGE_PROMPT = """You grade answers from a code assistant.

QUESTION:
{question}

FACTS A CORRECT ANSWER SHOULD MENTION (may be empty):
{facts}

CODE THE ASSISTANT RETRIEVED (its only evidence):
{context}

ASSISTANT'S ANSWER:
{answer}

Score two things from 0 to 1:
- faithfulness: are the answer's claims supported by the retrieved code? Penalise
  invented functions, files or behaviour. An honest "I couldn't find it" is faithful.
- correctness: does it correctly and completely answer the question, covering the facts listed?"""


def _content(message) -> str:
    # Claude returns content as a list of blocks; OpenAI returns a plain string.
    content = getattr(message, "content", "")
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content or ""


async def _judge(question: EvalQuestion, answer: str, tool_outputs: list[str]) -> _JudgeVerdict:
    llm = get_judge_llm("eval-judge", temperature=0).with_structured_output(_JudgeVerdict)

    prompt = _JUDGE_PROMPT.format(
        question=question.question,
        facts="\n".join(f"- {f}" for f in question.expected_facts) or "(none)",
        context="\n---\n".join(tool_outputs)[:_JUDGE_CONTEXT_CHARS] or "(nothing retrieved)",
        answer=answer,
    )
    verdict = await llm.ainvoke(prompt)
    verdict.faithfulness = min(max(verdict.faithfulness, 0.0), 1.0)
    verdict.correctness  = min(max(verdict.correctness, 0.0), 1.0)
    return verdict


async def _evaluate_one(agent, question: EvalQuestion, run_id: str, top_k: int) -> dict:
    session = f"eval:{run_id}:{question.id}"
    result: dict = {"id": question.id, "question": question.question}

    with track("eval", session):
        # 1. Raw retrieval quality
        chunks = get_retriever()(question.question, k=top_k)
        rank   = retrieval_rank([c["source"] for c in chunks], question.expected_files)
        if question.expected_files:
            result["retrieval_rank"] = rank
            result["hit"] = rank is not None
            result["reciprocal_rank"] = reciprocal_rank(rank)

        # 2. Full agent answer on a fresh thread (no history, no cache)
        started  = time.perf_counter()
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": question.question}]},
            {"configurable": {"thread_id": session}},
        )
        result["latency_s"] = round(time.perf_counter() - started, 2)

        messages     = response["messages"]
        answer       = _content(messages[-1])
        tool_outputs = [_content(m) for m in messages if getattr(m, "type", "") == "tool"]
        result["answer"]     = answer
        result["tools_used"] = [tc["name"] for m in messages for tc in (getattr(m, "tool_calls", None) or [])]

        # 3. Scores
        result["context_recall"] = context_recall(tool_outputs, question.expected_files)
        result["fact_recall"]    = fact_recall(answer, question.expected_facts)
        verdict = await _judge(question, answer, tool_outputs)
        result["faithfulness"] = verdict.faithfulness
        result["correctness"]  = verdict.correctness
        result["judge_reason"] = verdict.reason

    # 4. Cost of answering (judge cost reported separately in the summary)
    store = get_usage_store()
    result["cost"]       = store.total_cost(session, exclude_feature="eval-judge")
    result["judge_cost"] = store.total_cost(session) - result["cost"]
    return result


def _summarise(results: list[dict]) -> dict:
    ok = [r for r in results if "error" not in r]
    return {
        "questions":       len(results),
        "errors":          len(results) - len(ok),
        "hit_rate":        mean([float(r["hit"]) for r in ok if "hit" in r]),
        "mrr":             mean([r["reciprocal_rank"] for r in ok if "reciprocal_rank" in r]),
        "context_recall":  mean([r.get("context_recall") for r in ok]),
        "fact_recall":     mean([r.get("fact_recall") for r in ok]),
        "faithfulness":    mean([r.get("faithfulness") for r in ok]),
        "correctness":     mean([r.get("correctness") for r in ok]),
        "avg_latency_s":   mean([r.get("latency_s") for r in ok]),
        "avg_cost":        mean([r.get("cost") for r in ok]),
        "judge_cost":      sum(r.get("judge_cost", 0) for r in ok),
    }


def _config_snapshot(top_k: int) -> dict:
    return {
        "llm":            config["llm"]["model"],
        "judge":          config["llm"].get("judge_model"),
        "embeddings":     config["embeddings"]["model"],
        "vector_store":   config["vector_store"]["provider"],
        "rag_mode":       config["rag"]["mode"],
        "retrieval_mode": config["vector_store"].get("retrieval_mode"),
        "symbol_tools":   symbols_enabled(),
        "rerank":         rerank_config()["model"] if rerank_config()["enabled"] else False,
        "top_k":          top_k,
    }


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------

_SUMMARY_ROWS = [
    # key, label, format, higher_is_better
    ("hit_rate",       "Retrieval hit rate",  "pct",   True),
    ("mrr",            "Retrieval MRR",       "num",   True),
    ("context_recall", "Context recall",      "pct",   True),
    ("fact_recall",    "Fact recall",         "pct",   True),
    ("faithfulness",   "Faithfulness",        "num",   True),
    ("correctness",    "Correctness",         "num",   True),
    ("avg_latency_s",  "Avg answer time",     "sec",   False),
    ("avg_cost",       "Avg cost / question", "money", False),
]


def _fmt(value, kind: str) -> str:
    if value is None:
        return "–"
    return {"pct": f"{value:.0%}", "num": f"{value:.2f}", "sec": f"{value:.1f}s", "money": f"${value:.4f}"}[kind]


def _delta(new, old, kind: str, higher_is_better: bool) -> str:
    if new is None or old is None or new == old:
        return ""
    diff  = new - old
    good  = (diff > 0) == higher_is_better
    color = "green" if good else "red"
    sign  = "+" if diff > 0 else "−"
    text  = {"pct": f"{abs(diff):.0%}", "num": f"{abs(diff):.2f}", "sec": f"{abs(diff):.1f}s",
             "money": f"${abs(diff):.4f}"}[kind]
    return f"[{color}]{sign}{text}[/{color}]"


def _previous_run(results_dir: Path, current: Path) -> dict | None:
    runs = sorted(p for p in results_dir.glob("run_*.json") if p != current)
    if not runs:
        return None
    try:
        return json.loads(runs[-1].read_text(encoding="utf-8"))
    except Exception:
        return None


def _print_report(run: dict, previous: dict | None) -> None:
    table = Table(show_header=True, header_style="bold", title="Per question")
    for col in ("ID", "Search rank", "Context", "Facts", "Faithful", "Correct", "Time", "Cost"):
        table.add_column(col, justify="left" if col == "ID" else "right")
    for r in run["results"]:
        if "error" in r:
            table.add_row(r["id"], f"[red]error: {r['error'][:50]}[/red]", "", "", "", "", "", "")
            continue
        rank = r.get("retrieval_rank")
        table.add_row(
            r["id"],
            "–" if "hit" not in r else (f"#{rank}" if rank else "[red]miss[/red]"),
            _fmt(r.get("context_recall"), "pct"), _fmt(r.get("fact_recall"), "pct"),
            _fmt(r.get("faithfulness"), "num"), _fmt(r.get("correctness"), "num"),
            _fmt(r.get("latency_s"), "sec"), _fmt(r.get("cost"), "money"),
        )
    console.print(table)

    summary = run["summary"]
    old     = previous["summary"] if previous else {}
    totals  = Table(show_header=True, header_style="bold", title="Summary")
    totals.add_column("Metric")
    totals.add_column("This run", justify="right")
    totals.add_column(f"vs {previous['run_id']}" if previous else "vs previous", justify="right")
    for key, label, kind, higher_is_better in _SUMMARY_ROWS:
        totals.add_row(label, _fmt(summary.get(key), kind),
                       _delta(summary.get(key), old.get(key), kind, higher_is_better))
    console.print(totals)

    cfg = run["config"]
    console.print(
        f"[dim]{summary['questions']} questions · {summary['errors']} errors · judge cost ${summary['judge_cost']:.4f}"
        f" · {cfg['vector_store']}/{cfg['rag_mode']} · symbol tools {'on' if cfg['symbol_tools'] else 'off'}"
        f" · rerank {cfg.get('rerank') or 'off'}[/dim]"
    )
    if previous and previous.get("config") != cfg:
        changed = [k for k in cfg if previous["config"].get(k) != cfg[k]]
        console.print(f"[dim]Config changed since previous run: {', '.join(changed)}[/dim]")


# ------------------------------------------------------------------
# Entry point — called by main.py for /eval [questions_file]
# ------------------------------------------------------------------

async def run_eval(agent, questions_file: str | None = None) -> dict | None:
    eval_cfg    = config.get("eval", {})
    path        = Path(questions_file or eval_cfg.get("questions_file", "evals/questions.yaml"))
    results_dir = Path(eval_cfg.get("results_dir", ".repopilot/evals"))
    top_k       = int(eval_cfg.get("top_k", 5))

    if not path.exists():
        console.print(f"[yellow]Questions file not found: {path}[/yellow]")
        console.print("[dim]Create one (see evals/questions.yaml in the RepoPilot repo for the format) "
                      "or pass a path: /eval path/to/questions.yaml[/dim]")
        return None

    questions = load_questions(path)
    run_id    = datetime.now().strftime("%Y%m%d_%H%M%S")
    console.print(f"\n[bold]Eval run {run_id}[/bold] — {len(questions)} questions from {path}")
    console.print("[dim]Each question runs the full /ask agent plus a judge call; this costs real API usage.[/dim]\n")

    delay   = float(eval_cfg.get("delay_seconds", 0))
    results = []
    for i, question in enumerate(questions, 1):
        if i > 1 and delay:
            await asyncio.sleep(delay)   # stay under free-tier requests-per-minute limits
        console.print(f"[dim][{i}/{len(questions)}] {question.question}[/dim]")
        try:
            results.append(await _evaluate_one(agent, question, run_id, top_k))
        except Exception as e:
            logger.error(f"Eval question {question.id} failed: {e}")
            results.append({"id": question.id, "question": question.question, "error": str(e)})

    run = {
        "run_id":         run_id,
        "questions_file": str(path),
        "config":         _config_snapshot(top_k),
        "summary":        _summarise(results),
        "results":        results,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"run_{run_id}.json"
    out_path.write_text(json.dumps(run, indent=2), encoding="utf-8")

    _print_report(run, _previous_run(results_dir, out_path))
    console.print(f"[dim]Full results (answers, judge reasons): {out_path}[/dim]\n")
    return run
