import os
import re
import json
from pathlib import Path


# Keys come from repopilot/.env, loaded by main.py. (This module used to call
# load_dotenv() itself, which read the .env of whatever repo is being analysed.)
_CONFIG_PATH = Path(__file__).parent.parent / "mcp_servers.json"


def load_mcp_configs() -> dict:
 """Return mcp_servers dict from mcp_servers.json with env vars resolved.


 CWD is injected at load time so ${CWD} in the config always resolves to the
 directory where RepoPilot was launched — not the package install location.
 """
 os.environ.setdefault("CWD", str(Path.cwd()))
 raw = json.loads(_CONFIG_PATH.read_text())
 # Replace ${VAR} placeholders in the config with actual env var values.
 # Values are JSON-escaped first: a Windows path like C:\Users\... pasted in
 # raw would contain invalid JSON escapes (\U, \D, ...) and break parsing.
 resolved = re.sub(
     r"\$\{(\w+)\}",
     lambda m: json.dumps(os.getenv(m.group(1), ""))[1:-1],
     json.dumps(raw),
 )
 return json.loads(resolved).get("mcp_servers", {})
