from __future__ import annotations

import contextvars
import os
from contextlib import contextmanager
from dataclasses import dataclass

# ------------------------------------------------------------------
# Sources shown under each /ask answer
# ------------------------------------------------------------------
# handle_query() opens collect_sources() around the agent call. While it
# is open, every search tool reports the code it returned via
# add_source(). LangGraph runs tools with a copy of the caller's context,
# and the copy points at the same list, so the tools' additions are
# visible to handle_query afterwards.
#
# Outside collect_sources() (e.g. /eval or /plan) add_source() is a no-op.
# ------------------------------------------------------------------

_collected: contextvars.ContextVar[list | None] = contextvars.ContextVar("repopilot_sources", default=None)


@dataclass(frozen=True)
class Source:
    path: str
    start_line: int
    end_line: int


@contextmanager
def collect_sources():
    sources: list[Source] = []
    token = _collected.set(sources)
    try:
        yield sources
    finally:
        _collected.reset(token)


def add_source(path: str, start_line: int, end_line: int | None = None) -> None:
    sources = _collected.get()
    if sources is not None:
        sources.append(Source(path, int(start_line), int(end_line or start_line)))


def _rel(path: str) -> str:
    try:
        return os.path.relpath(path).replace("\\", "/")
    except ValueError:  # different drive on Windows
        return path


def format_sources(sources: list[Source], limit: int = 8) -> list[str]:
    """
    Group by file (in the order first seen) and merge overlapping line ranges:
      ["tasks/orchestrator.py:97-115", "tasks/task_store.py:247-262, 270"]
    """
    by_file: dict[str, list[tuple[int, int]]] = {}
    for s in sources:
        by_file.setdefault(s.path, []).append((s.start_line, s.end_line))

    lines = []
    for path, ranges in list(by_file.items())[:limit]:
        merged: list[list[int]] = []
        for start, end in sorted(ranges):
            if merged and start <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        spans = ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in merged)
        lines.append(f"{_rel(path)}:{spans}")
    if len(by_file) > limit:
        lines.append(f"... and {len(by_file) - limit} more files")
    return lines
