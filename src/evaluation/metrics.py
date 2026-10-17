from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# ------------------------------------------------------------------
# Deterministic eval metrics — no LLM involved, so they are free,
# repeatable, and unit-tested (tests/test_eval_metrics.py).
# ------------------------------------------------------------------


@dataclass
class EvalQuestion:
    id: str
    question: str
    expected_files: list[str] = field(default_factory=list)   # repo-relative paths
    expected_facts: list[str] = field(default_factory=list)   # names/phrases a correct answer mentions


def load_questions(path: str | Path) -> list[EvalQuestion]:
    """
    Load a questions file:
        - id: retry_logic                  # optional, defaults to q1, q2, ...
          question: "Where are failed tasks retried?"
          expected_files: ["workflows/task_store.py"]
          expected_facts: ["fail_task", "max_retries"]
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    questions = []
    for i, item in enumerate(raw, 1):
        if not item.get("question"):
            raise ValueError(f"Question #{i} in {path} has no 'question' field")
        questions.append(EvalQuestion(
            id=str(item.get("id") or f"q{i}"),
            question=item["question"],
            expected_files=list(item.get("expected_files") or []),
            expected_facts=list(item.get("expected_facts") or []),
        ))
    return questions


def _normalise(path: str) -> str:
    return path.replace("\\", "/").lower().strip("/")


def file_matches(source: str, expected: str) -> bool:
    """True if retrieved absolute path `source` is the repo-relative file `expected`."""
    source, expected = _normalise(source), _normalise(expected)
    return source == expected or source.endswith("/" + expected)


def retrieval_rank(sources: list[str], expected_files: list[str]) -> int | None:
    """1-based rank of the first retrieved chunk from an expected file, or None if none matched."""
    for rank, source in enumerate(sources, 1):
        if any(file_matches(source, expected) for expected in expected_files):
            return rank
    return None


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def context_recall(tool_outputs: list[str], expected_files: list[str]) -> float | None:
    """
    Fraction of expected files that showed up anywhere in what the agent's tools
    returned while answering. Unlike retrieval_rank (raw vector search only), this
    credits every tool — including find_symbol / find_callers.
    """
    if not expected_files:
        return None
    blob = _normalise("\n".join(tool_outputs))
    found = sum(1 for expected in expected_files if _normalise(expected) in blob)
    return found / len(expected_files)


def fact_recall(answer: str, expected_facts: list[str]) -> float | None:
    """Fraction of expected facts mentioned in the answer (case-insensitive substring)."""
    if not expected_facts:
        return None
    answer = answer.lower()
    return sum(1 for fact in expected_facts if fact.lower() in answer) / len(expected_facts)


def mean(values: list[float | None]) -> float | None:
    """Average of the non-None values (a metric is None when a question doesn't define it)."""
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None
