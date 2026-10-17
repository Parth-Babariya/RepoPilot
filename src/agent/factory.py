from langchain.agents import create_agent

from repopilot.llm.factory import get_llm
from repopilot.agent.tools import search_codebase
from repopilot.agent.symbol_tools import find_symbol, find_callers, find_callees
from repopilot.context.symbol_index import symbols_enabled
from repopilot.observability.logger import get_logger
from repopilot.tools.terminal_tools import run_command
from repopilot.mcp_integration.mcp_client import get_mcp_tools
from repopilot.skills.skill_tools import build_skills_prompt, get_registry, load_skill
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

   full_prompt = SYSTEM_PROMPT
   # run_command stays for questions about history ("when did this change?" → git log);
   # the safety policy lets read-only commands through and asks about anything else.
   tools = [search_codebase, run_command, *mcp_tools]

   if symbols_enabled():
       full_prompt += "\n\n" + SYMBOL_TOOLS_PROMPT
       tools += [find_symbol, find_callers, find_callees]

   # Skills: only the names/descriptions go in the prompt; the agent loads a
   # skill's full instructions with load_skill when a question matches one.
   if get_registry().skill_names:
       full_prompt += "\n\n" + build_skills_prompt()
       tools.append(load_skill)

   return create_agent(
       llm,
       tools=tools,
       system_prompt=full_prompt,
       checkpointer=checkpointer,
       # Summarise older messages once the conversation passes
       # memory.summarize_at_tokens so long sessions don't overflow the context window.
       middleware=[get_summarization_middleware()],
   )