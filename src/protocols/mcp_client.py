import functools

from langchain_mcp_adapters.client import MultiServerMCPClient
from repopilot.protocols.mcp_config import load_mcp_configs
from repopilot.observability.logger import get_logger
from repopilot.guardrails.tool_safety import is_read_only_mcp_tool, is_secret_path, mcp_read_only


logger = get_logger(__name__)


def _strings(value) -> list[str]:
  if isinstance(value, str):
      return [value]
  if isinstance(value, (list, tuple)):
      return [s for item in value for s in _strings(item)]
  if isinstance(value, dict):
      return [s for item in value.values() for s in _strings(item)]
  return []


def _guard_secrets(tool):
  """
  Wrap an MCP tool so it refuses any argument naming a secret file such as .env,
  and so a server error comes back to the agent as the tool's result.

  MCP servers report failures as exceptions (GitHub: "Not Found: Resource not
  found" for a guessed repo or path). Raised, one failed lookup ends the whole
  answer — it did for 2 of 24 /eval questions. Returned, the agent reads the
  error and carries on with its other tools, as it does with any tool's "Error: ...".
  """
  original = tool.coroutine

  def result(message: str):
      return (message, None) if tool.response_format == "content_and_artifact" else message

  @functools.wraps(original)   # keeps the signature, so LangChain still injects `runtime`
  async def guarded(**kwargs):
      if any(is_secret_path(s) for s in _strings(kwargs)):
          logger.warning(f"Blocked MCP tool {tool.name}: argument names a secret file")
          return result("Error: access to secret files is not allowed")
      logger.info(f"MCP tool called: {tool.name}")
      try:
          return await original(**kwargs)
      except Exception as e:
          logger.warning(f"MCP tool {tool.name} failed: {e}")
          return result(f"Error from {tool.name}: {e}")

  return tool.model_copy(update={"coroutine": guarded})


async def get_mcp_tools() -> list:
  """
  Connect to all configured MCP servers and return their tools.
  Each server is optional: one that fails to start (e.g. Node.js missing,
  no GITHUB_TOKEN) is skipped with a warning instead of crashing startup.

  With safety.mcp_read_only (default), only tools that read are kept —
  RepoPilot answers questions, so it never needs to push files, open PRs
  or edit files through MCP, and a prompt injection can't make it.
  """
  configs = load_mcp_configs()
  logger.info(f"Connecting to MCP servers: {list(configs.keys())}")
  client = MultiServerMCPClient(configs)
  read_only = mcp_read_only()
  tools = []
  for name in configs:
      try:
          server_tools = await client.get_tools(server_name=name)
      except Exception as e:
          logger.warning(f"MCP server '{name}' unavailable, continuing without it: {e}")
          continue
      kept = [t for t in server_tools if not read_only or is_read_only_mcp_tool(t.name)]
      dropped = sorted(t.name for t in server_tools if t not in kept)
      # Tagged with their server, so an agent can offer a server's tools only when relevant
      tools.extend(_guard_secrets(t).model_copy(update={"metadata": {**(t.metadata or {}), "mcp_server": name}})
                   for t in kept)
      logger.info(f"Loaded {len(kept)} tools from MCP server '{name}'"
                  + (f" (read-only; skipped {dropped})" if dropped else ""))
  return tools
