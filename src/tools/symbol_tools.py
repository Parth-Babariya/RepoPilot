import os
from pathlib import Path

from langchain.tools import tool

from repopilot.tools.sources import add_source
from repopilot.retrieval.symbol_index import get_symbol_store
from repopilot.observability.logger import get_logger


logger = get_logger(__name__)

_MAX_SNIPPET_LINES = 80
_MAX_RESULTS       = 40


def _rel(path: str) -> str:
   """Show paths relative to the repo being analysed — shorter for the LLM and the user."""
   try:
       return os.path.relpath(path)
   except ValueError:  # different drive on Windows
       return path


def _snippet(path: str, start: int, end: int) -> str:
   try:
       lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
   except OSError:
       return "(file not readable)"
   stop = min(end, start + _MAX_SNIPPET_LINES - 1)
   body = "\n".join(lines[start - 1:stop])
   if stop < end:
       body += f"\n... ({end - stop} more lines)"
   return body


def _ambiguity_note(name: str) -> str:
   definitions = get_symbol_store().find_definitions(name.split(".")[-1])
   if len(definitions) <= 1:
       return ""
   places = ", ".join(f"{d['qualname']} ({_rel(d['file'])}:{d['start_line']})" for d in definitions[:8])
   return (
       f"\nNote: {len(definitions)} symbols are named '{name.split('.')[-1]}': {places}. "
       "Calls are matched by name, so some results may refer to a different one."
   )


@tool
def find_symbol(name: str) -> str:
   """
   Find where a function, method or class is defined and show its code.
   Use for exact lookups like "show me fail_task" or "where is SQLiteTaskStore defined".
   name can be a bare name ("fail_task") or qualified ("SQLiteTaskStore.fail_task").
   """
   logger.info(f"Tool called: find_symbol('{name}')")
   definitions = get_symbol_store().find_definitions(name)
   if not definitions:
       return f"No function, method or class named '{name}' was found. Try search_codebase instead."

   results = []
   for d in definitions[:5]:
       add_source(d["file"], d["start_line"], d["end_line"])
       results.append(
           f"{d['kind']} {d['qualname']} — {_rel(d['file'])} (lines {d['start_line']}-{d['end_line']})\n"
           f"{_snippet(d['file'], d['start_line'], d['end_line'])}"
       )
   if len(definitions) > 5:
       results.append(f"... and {len(definitions) - 5} more definitions with this name.")
   return "\n---\n".join(results)


@tool
def find_callers(name: str) -> str:
   """
   List every place in the codebase that calls a given function or method.
   Use for questions like "what calls fail_task?", "where is get_retriever used?",
   or "what happens before X runs?".
   """
   logger.info(f"Tool called: find_callers('{name}')")
   callers = get_symbol_store().find_callers(name)
   if not callers:
       return f"No calls to '{name}' were found in the indexed code."

   for c in callers[:_MAX_RESULTS]:
       add_source(c["file"], c["line"])
   lines = [f"{c['caller']} — {_rel(c['file'])}:{c['line']}" for c in callers[:_MAX_RESULTS]]
   if len(callers) > _MAX_RESULTS:
       lines.append(f"... and {len(callers) - _MAX_RESULTS} more call sites.")
   return f"{len(callers)} call site(s) of '{name}':\n" + "\n".join(lines) + _ambiguity_note(name)


@tool
def find_callees(name: str) -> str:
   """
   List the functions and methods that a given function or method calls.
   Use for questions like "what does _execute do step by step?" or
   "what happens when handle_plan_command runs?".
   """
   logger.info(f"Tool called: find_callees('{name}')")
   callees = get_symbol_store().find_callees(name)
   if not callees:
       return f"No calls found inside '{name}' (or no symbol with that name)."

   in_repo  = [c for c in callees if c["in_repo"]]
   for c in in_repo[:_MAX_RESULTS]:
       add_source(c["file"], c["line"])
   external = sorted({c["callee"] for c in callees if not c["in_repo"]})
   lines = [f"{c['callee']} — called from {c['caller']} at {_rel(c['file'])}:{c['line']}" for c in in_repo[:_MAX_RESULTS]]
   result = (
       f"'{name}' makes {len(in_repo)} call(s) whose name matches a function defined in this codebase "
       "(matched by name only — e.g. a dict's .get() also matches a method named get):\n"
       + "\n".join(lines or ["(none)"])
   )
   if external:
       result += f"\nIt also calls library/built-in functions: {', '.join(external[:30])}"
   return result
