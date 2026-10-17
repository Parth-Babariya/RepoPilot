import contextvars
import re
from contextlib import contextmanager

from langchain.tools import tool


from repopilot.tools.sources import add_source
from repopilot.retrieval.search.factory import get_retriever
from repopilot.observability.logger import get_logger


logger = get_logger(__name__)

# Queries already searched while answering the current question. In /eval the
# agent searched "_READ_ONLY_PROGRAMS" 12 times in one answer, getting the same
# results each time; a repeat now gets a reminder instead of the same chunks.
# Word order and punctuation are ignored ("_X =" repeats "_X").
_searched: contextvars.ContextVar[set | None] = contextvars.ContextVar("repopilot_searched", default=None)


@contextmanager
def search_memory():
    """Remember the searches made inside this block (one question and its revision)."""
    token = _searched.set(set())
    try:
        yield
    finally:
        _searched.reset(token)


def _query_key(query: str) -> str:
    return " ".join(sorted(set(re.findall(r"\w+", query.lower()))))


@tool
def search_codebase(query: str) -> str:
   """
   Search the codebase for relevant classes, functions or logic.
   Use this tool whenever you need to find code related to a question.
   """
   logger.info(f"Tool called: search_codebase with query: {query}")
   searched = _searched.get()
   if searched is not None:
       if _query_key(query) in searched:
           return (f'You already searched for "{query}" while answering this question; those results '
                   "are above. Search with different words (another name, the file, a caller), use "
                   "find_symbol for an exact function or class name, or answer with what you have.")
       searched.add(_query_key(query))
   retrieve = get_retriever()
   chunks = retrieve(query, k=5)
   if not chunks:
       return "No relevant code found."


   results = []
   for chunk in chunks:
       add_source(chunk["source"], chunk["start_line"], chunk["end_line"])
       results.append(
           f"File: {chunk['source']} (lines {chunk['start_line']}-{chunk['end_line']})\n"
           f"Type: {chunk['type']} — {chunk['name']}\n"
           f"Code:\n{chunk['content']}\n"
       )
   return "\n---\n".join(results)
