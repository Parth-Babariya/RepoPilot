from langchain.agents import create_agent

from repopilot.llm.factory import get_llm
from repopilot.agent.tools import search_codebase
from repopilot.agent.symbol_tools import find_symbol, find_callers, find_callees
from repopilot.context.symbol_index import symbols_enabled
from repopilot.observability.logger import get_logger
from repopilot.tools.terminal_tools import run_command, run_in_directory
from repopilot.mcp_integration.mcp_client import get_mcp_tools
from repopilot.skills.skill_tools import load_skill, build_skills_prompt
from repopilot.memory.short_term import get_summarization_middleware

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a senior software engineer with deep knowledge of the codebase.
Always use the search_codebase tool before answering any question.
Reference specific file names, function names and line numbers in your answers.
If you cannot find the answer in the codebase, say so explicitly."""

SYMBOL_TOOLS_PROMPT = """Choosing a search tool:
- search_codebase: questions about behaviour or concepts ("how does caching work?").
- find_symbol: show where a specific function, method or class is defined, with its code.
- find_callers: "what calls X?", "where is X used?", "what happens before X?".
- find_callees: "what does X do step by step?", "what happens when X runs?".
Exact-name tools are precise; combine them with search_codebase when a question needs both."""

async def build_agent(checkpointer):
   """Create and return a LangChain agent with persistent memory."""
   llm = get_llm()
   mcp_tools = await get_mcp_tools()

   skills_prompt = build_skills_prompt()
   full_prompt = SYSTEM_PROMPT
   if symbols_enabled():
       full_prompt += "\n\n" + SYMBOL_TOOLS_PROMPT
   if skills_prompt:
       full_prompt += "\n\n" + skills_prompt

   tools = [
       search_codebase,
       load_skill,
       run_command,
       run_in_directory,
       *mcp_tools,
   ]
   if symbols_enabled():
       tools += [find_symbol, find_callers, find_callees]

   return create_agent(
       llm,
       tools=tools,
       system_prompt=full_prompt,
       checkpointer=checkpointer,
       # Summarise older messages once the conversation passes
       # memory.summarize_at_tokens so long sessions don't overflow the context window.
       middleware=[get_summarization_middleware()],
   )