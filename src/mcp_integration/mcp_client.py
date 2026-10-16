from langchain_mcp_adapters.client import MultiServerMCPClient
from repopilot.mcp_integration.mcp_config import load_mcp_configs
from repopilot.observability.logger import get_logger


logger = get_logger(__name__)


async def get_mcp_tools() -> list:
  """
  Connect to all configured MCP servers and return their tools.
  Each server is optional: one that fails to start (e.g. Node.js missing,
  no GITHUB_TOKEN) is skipped with a warning instead of crashing startup.
  """
  configs = load_mcp_configs()
  logger.info(f"Connecting to MCP servers: {list(configs.keys())}")
  client = MultiServerMCPClient(configs)
  tools = []
  for name in configs:
      try:
          server_tools = await client.get_tools(server_name=name)
          tools.extend(server_tools)
          logger.info(f"Loaded {len(server_tools)} tools from MCP server '{name}'")
      except Exception as e:
          logger.warning(f"MCP server '{name}' unavailable, continuing without it: {e}")
  return tools
