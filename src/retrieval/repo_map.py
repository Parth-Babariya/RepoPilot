from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

from repopilot.retrieval.indexing.code_parser import get_source_files
from repopilot.retrieval.symbol_index import get_symbol_store, symbols_enabled

# ------------------------------------------------------------------
# Repo map — a compact outline of the repository, built without any LLM
# ------------------------------------------------------------------
# One line per file with its classes (and their methods) and functions,
# from the symbol index that already exists:
#
#   src/workflows/task_store.py   SQLiteTaskStore(claim_task, complete_task, fail_task, …)
#
# The planner gets it with every goal, so plans name files that exist and follow
# the repo's layout instead of assuming them; the Q&A agent can ask for it to
# answer "how is this project structured?" without a dozen searches.
#
# It must fit in a prompt, so detail is dropped in steps until it fits:
# methods, then all symbols, then files are grouped into folders.
# ------------------------------------------------------------------

_MAX_CHARS = 6000
_MAX_METHODS = 6
_MAX_SYMBOLS_PER_FILE = 8
_MANIFESTS = ("pyproject.toml", "requirements.txt", "setup.py", "package.json", "go.mod", "Cargo.toml",
              "pom.xml", "build.gradle", "Makefile", "Dockerfile", "pytest.ini", "conftest.py", "tox.ini")


def _is_test(path: str) -> bool:
    p = Path(path)
    return (any(part in ("tests", "test", "__tests__", "spec") for part in p.parts[:-1])
            or p.name.startswith("test_") or p.stem.endswith(("_test", ".test", ".spec")) or p.name == "conftest.py")


def _symbols_by_file(root: Path) -> dict[str, list[dict]]:
    if not symbols_enabled():
        return {}
    with get_symbol_store()._conn() as conn:
        rows = conn.execute("SELECT file, name, qualname, kind, start_line FROM symbols ORDER BY file, start_line")
        by_file: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            try:
                rel = Path(os.path.relpath(row["file"], root)).as_posix()
            except ValueError:      # another drive on Windows: not in this repo
                continue
            if not rel.startswith(".."):
                by_file[rel].append(dict(row))
    return by_file


def _outline(symbols: list[dict], with_methods: bool) -> str:
    # The public parts: _private helpers are most of the names and least of the meaning
    symbols = [s for s in symbols if not s["name"].startswith("_")]
    methods: dict[str, list[str]] = defaultdict(list)
    for s in symbols:
        if "." in s["qualname"]:
            methods[s["qualname"].split(".")[0]].append(s["name"])
    parts = []
    for s in symbols:
        if "." in s["qualname"]:
            continue                                    # methods are listed under their class
        if s["kind"] == "class" and with_methods and methods[s["name"]]:
            names = [m for m in methods[s["name"]] if not m.startswith("__")][:_MAX_METHODS]
            more = ", …" if len(methods[s["name"]]) > len(names) else ""
            parts.append(f"{s['name']}({', '.join(names)}{more})")
        else:
            parts.append(s["name"])
    shown = parts[:_MAX_SYMBOLS_PER_FILE]
    return ", ".join(shown) + (f", … (+{len(parts) - len(shown)})" if len(parts) > len(shown) else "")


def build_repo_map(root: str | Path | None = None, max_chars: int = _MAX_CHARS) -> str:
    """Outline of the repository at root (default: the current folder), at most max_chars long."""
    root = Path(root or Path.cwd()).resolve()
    files = sorted(Path(os.path.relpath(f, root)).as_posix() for f in get_source_files(str(root)))
    if not files:
        return "(the folder has no source files yet)"

    manifests = [f for f in files if Path(f).name in _MANIFESTS]
    header = [f"{len(files)} source files" + (f" · project files: {', '.join(manifests[:8])}" if manifests else "")]
    symbols = _symbols_by_file(root)

    # Tests are summarised per folder: a plan needs to know where tests live and how
    # many there are, not every test's name. Empty __init__.py files say nothing.
    tests: dict[str, int] = defaultdict(int)
    listed = []
    for f in files:
        if _is_test(f):
            tests[Path(f).parent.as_posix()] += 1
        elif not (Path(f).name == "__init__.py" and not symbols.get(f)):
            listed.append(f)
    test_lines = [f"{folder}/  ({count} test file{'s' if count != 1 else ''})"
                  for folder, count in sorted(tests.items())]

    def render(with_symbols: bool, with_methods: bool) -> str:
        width = min(max((len(f) for f in listed), default=0), 48)
        lines = [f"{f:<{width}}  {_outline(symbols.get(f, []), with_methods) if with_symbols else ''}".rstrip()
                 for f in listed]
        return "\n".join(header + lines + test_lines)

    for with_symbols, with_methods in ((True, True), (True, False), (False, False)):
        text = render(with_symbols, with_methods)
        if len(text) <= max_chars:
            return text

    # Still too long: one line per folder with its file count
    folders: dict[str, int] = defaultdict(int)
    for f in files:
        folders[str(Path(f).parent.as_posix())] += 1
    lines = [f"{folder}/  ({count} files)" for folder, count in sorted(folders.items())]
    return "\n".join(header + lines)[:max_chars]
