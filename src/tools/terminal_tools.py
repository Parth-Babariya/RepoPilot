import os
from langchain.tools import tool

from repopilot.guardrails.tool_safety import run_guarded


_TIMEOUT_SECONDS = 30


@tool
def run_command(command: str, directory: str = "") -> str:
   """Run a shell command and return its output. Times out after 30 seconds.
   Runs in the project folder, or in `directory` (a folder inside the project) if given.
   Common read-only, build and test commands run directly; others need the user's approval."""
   if directory and not os.path.isdir(directory):
       return f"Error: directory does not exist: {directory}"
   return run_guarded(command, cwd=directory or None, timeout=_TIMEOUT_SECONDS)[1]
