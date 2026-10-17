from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from tree_sitter_languages import get_parser

from repopilot.config import config
from repopilot.retrieval.indexing.code_parser import EXTENSION_TO_LANGUAGE, get_source_files
from repopilot.observability.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# How the symbol / call-graph index works
# ------------------------------------------------------------------
# Semantic search finds code by meaning, which is unreliable for exact
# questions like "what calls fail_task?". This index answers those.
#
# For every source file, tree-sitter gives us the syntax tree. We walk the
# WHOLE tree (unlike chunking, which stops at top-level blocks) and record:
#   symbols — every function/method/class: name, qualified name
#             (e.g. "SQLiteTaskStore.fail_task"), file, line range
#   calls   — every call site: which symbol it's inside (the caller),
#             the name being called (the callee), and the line
#
# Callees are matched by NAME only. Without type information we can't tell
# which of two methods called "run" a call like obj.run() refers to — the
# query functions flag that ambiguity instead of guessing.
#
# Stored in SQLite and updated incrementally by file mtime, alongside the
# vector index (see retrieval/indexing/factory.py).
# ------------------------------------------------------------------

# Nodes that define a named symbol. Superset of code_parser.BLOCK_NODE_TYPES:
# also catches structs/impls/traits so methods get a qualified parent name.
_SYMBOL_NODE_TYPES = {
    "function_definition", "function_declaration", "method_definition", "arrow_function",
    "class_definition", "class_declaration", "method_declaration", "constructor_declaration",
    "interface_declaration", "enum_declaration", "function_item", "struct_item", "impl_item",
    "trait_item", "enum_item", "class_specifier", "struct_specifier", "module", "class",
    "method", "singleton_method",
}
_CONTAINER_WORDS = ("class", "struct", "impl", "trait", "interface", "enum", "module")

# Nodes that are a call, across the bundled grammars.
_CALL_NODE_TYPES = {
    "call",                       # python, ruby
    "call_expression",            # js/ts, go, rust, c/c++, swift, kotlin
    "new_expression",             # js/ts
    "method_invocation",          # java
    "object_creation_expression", # java, c#
    "invocation_expression",      # c#
    "function_call_expression", "member_call_expression", "scoped_call_expression",  # php
}
_CALLEE_FIELDS = ("function", "name", "method", "constructor", "type")
_IDENTIFIER    = re.compile(r"[A-Za-z_$][\w$]*")


@dataclass
class Symbol:
    name: str
    qualname: str
    kind: str          # "class" | "function"
    start_line: int
    end_line: int


@dataclass
class Call:
    caller: str        # qualname of the enclosing symbol, or "<module>"
    callee: str        # bare name being called
    line: int


# ------------------------------------------------------------------
# Extraction
# ------------------------------------------------------------------

def _text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def _symbol_name(node, source: bytes) -> str | None:
    name_node = node.child_by_field_name("name") or node.child_by_field_name("type")
    if name_node is None:
        for child in node.children:
            if child.type in ("identifier", "name", "property_identifier", "type_identifier", "constant"):
                name_node = child
                break
    if name_node is not None:
        return _text(name_node, source)
    # const handler = () => {...}  /  { handler: () => {...} }
    parent = node.parent
    if node.type == "arrow_function" and parent is not None:
        key = parent.child_by_field_name("name") or parent.child_by_field_name("key")
        if key is not None:
            return _text(key, source)
    return None  # anonymous (e.g. an inline callback) — not worth indexing


def _callee_name(node, source: bytes) -> str | None:
    callee = None
    for field in _CALLEE_FIELDS:
        callee = node.child_by_field_name(field)
        if callee is not None:
            break
    if callee is None and node.named_children:
        callee = node.named_children[0]
    if callee is None:
        return None
    # "self.store.fail_task" → "fail_task"; "List<String>" → "List"; "f(x)(y)" → "f"
    head = re.split(r"[(<\[]", _text(callee, source), maxsplit=1)[0]
    names = _IDENTIFIER.findall(head)
    return names[-1] if names else None


def extract(source_text: str, language: str) -> tuple[list[Symbol], list[Call]]:
    """Parse source and return every symbol definition and call site in it."""
    source  = source_text.encode("utf-8")
    tree    = get_parser(language).parse(source)
    symbols: list[Symbol] = []
    calls:   list[Call]   = []

    # Iterative walk carrying the enclosing qualified name — avoids recursion
    # limits on deeply nested files.
    stack = [(tree.root_node, "")]
    while stack:
        node, scope = stack.pop()
        child_scope = scope

        # node.parent check: Python's root node is also called "module"
        if node.type in _SYMBOL_NODE_TYPES and node.parent is not None:
            name = _symbol_name(node, source)
            if name:
                qualname = f"{scope}.{name}" if scope else name
                kind = "class" if any(w in node.type for w in _CONTAINER_WORDS) else "function"
                symbols.append(Symbol(name, qualname, kind, node.start_point[0] + 1, node.end_point[0] + 1))
                child_scope = qualname

        elif node.type in _CALL_NODE_TYPES:
            callee = _callee_name(node, source)
            if callee:
                calls.append(Call(scope or "<module>", callee, node.start_point[0] + 1))

        stack.extend((child, child_scope) for child in reversed(node.children))

    return symbols, calls


# ------------------------------------------------------------------
# Storage
# ------------------------------------------------------------------

class SymbolStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS files (
                    file  TEXT PRIMARY KEY,
                    mtime REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS symbols (
                    file TEXT, name TEXT, qualname TEXT, kind TEXT, start_line INTEGER, end_line INTEGER
                );
                CREATE TABLE IF NOT EXISTS calls (
                    file TEXT, caller TEXT, callee TEXT, line INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_symbols_name     ON symbols(name);
                CREATE INDEX IF NOT EXISTS idx_symbols_qualname ON symbols(qualname);
                CREATE INDEX IF NOT EXISTS idx_calls_callee     ON calls(callee);
                CREATE INDEX IF NOT EXISTS idx_calls_caller     ON calls(caller);
            """)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Writes ──────────────────────────────────────────────────────

    def indexed_mtimes(self) -> dict[str, float]:
        with self._conn() as conn:
            return {r["file"]: r["mtime"] for r in conn.execute("SELECT file, mtime FROM files")}

    def remove_file(self, filepath: str) -> None:
        with self._conn() as conn:
            for table in ("files", "symbols", "calls"):
                conn.execute(f"DELETE FROM {table} WHERE file = ?", (filepath,))

    def index_file(self, filepath: str) -> int:
        """(Re)index one file. Returns the number of symbols found (0 for unsupported files)."""
        language = EXTENSION_TO_LANGUAGE.get(Path(filepath).suffix.lower())
        if language is None or not os.path.exists(filepath):
            self.remove_file(filepath)
            return 0
        mtime  = os.path.getmtime(filepath)
        source = Path(filepath).read_text(encoding="utf-8", errors="ignore")
        try:
            symbols, calls = extract(source, language)
        except Exception as e:
            logger.error(f"Symbol extraction failed for {filepath}: {e}")
            symbols, calls = [], []

        with self._conn() as conn:
            for table in ("files", "symbols", "calls"):
                conn.execute(f"DELETE FROM {table} WHERE file = ?", (filepath,))
            conn.execute("INSERT INTO files VALUES (?, ?)", (filepath, mtime))
            conn.executemany(
                "INSERT INTO symbols VALUES (?,?,?,?,?,?)",
                [(filepath, s.name, s.qualname, s.kind, s.start_line, s.end_line) for s in symbols],
            )
            conn.executemany(
                "INSERT INTO calls VALUES (?,?,?,?)",
                [(filepath, c.caller, c.callee, c.line) for c in calls],
            )
        return len(symbols)

    # ── Queries ─────────────────────────────────────────────────────

    def find_definitions(self, name: str) -> list[dict]:
        """Exact match on bare name or qualified name ("fail_task" or "SQLiteTaskStore.fail_task")."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM symbols WHERE name = ? OR qualname = ?
                   ORDER BY file, start_line""",
                (name, name),
            ).fetchall()
        return [dict(r) for r in rows]

    def find_callers(self, name: str) -> list[dict]:
        callee = name.split(".")[-1]
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT caller, file, line FROM calls WHERE callee = ? ORDER BY file, line", (callee,)
            ).fetchall()
        return [dict(r) for r in rows]

    def find_callees(self, name: str) -> list[dict]:
        """Calls made inside the symbol(s) whose qualified name is name or ends with ".name"."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT callee, MIN(line) AS line, file, caller FROM calls
                   WHERE caller = ? OR substr(caller, -length(?) - 1) = '.' || ?
                   GROUP BY caller, callee
                   ORDER BY file, line""",
                (name, name, name),
            ).fetchall()
            defined = {
                r["name"] for r in conn.execute(
                    f"SELECT DISTINCT name FROM symbols WHERE name IN ({','.join('?' * len(rows))})",
                    [r["callee"] for r in rows],
                )
            } if rows else set()
        return [dict(r) | {"in_repo": r["callee"] in defined} for r in rows]

    def stats(self) -> dict:
        with self._conn() as conn:
            return {
                table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("files", "symbols", "calls")
            }


_store: SymbolStore | None = None


def symbols_enabled() -> bool:
    return bool(config.get("symbols", {}).get("enabled", True))


def get_symbol_store() -> SymbolStore:
    global _store
    if _store is None:
        _store = SymbolStore(config.get("symbols", {}).get("db_path", ".repopilot/symbols.db"))
    return _store


# ------------------------------------------------------------------
# Indexing entry points (wired in via retrieval/indexing/factory.py)
# ------------------------------------------------------------------

def update_symbol_index(repo_path: str) -> None:
    """Incrementally sync the symbol index with repo_path (new/changed/deleted files)."""
    store   = get_symbol_store()
    current = {
        f: os.path.getmtime(f) for f in get_source_files(repo_path)
        if Path(f).suffix.lower() in EXTENSION_TO_LANGUAGE
    }
    indexed = store.indexed_mtimes()

    for filepath in set(indexed) - set(current):
        store.remove_file(filepath)
    changed = [f for f, mtime in current.items() if indexed.get(f) != mtime]
    for filepath in changed:
        store.index_file(filepath)

    counts = store.stats()
    logger.info(
        f"Symbol index updated — {len(changed)} files re-parsed | "
        f"{counts['symbols']} symbols, {counts['calls']} calls in {counts['files']} files"
    )
