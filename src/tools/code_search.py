from langchain.tools import tool


from repopilot.tools.sources import add_source
from repopilot.retrieval.search.factory import get_retriever
from repopilot.observability.logger import get_logger


logger = get_logger(__name__)


@tool
def search_codebase(query: str) -> str:
   """
   Search the codebase for relevant classes, functions or logic.
   Use this tool whenever you need to find code related to a question.
   """
   logger.info(f"Tool called: search_codebase with query: {query}")
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
