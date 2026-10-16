import os
from langchain.tools import tool

from repopilot.tools.safety import run_guarded


_TIMEOUT_SECONDS = 30


@tool
def run_command(command: str) -> str:
   """Run a shell command in the project folder and return its output. Times out after 30 seconds.
   Common read-only, build and test commands run directly; others need the user's approval."""
   return run_guarded(command, timeout=_TIMEOUT_SECONDS)[1]


@tool
def run_in_directory(command: str, directory: str) -> str:
   """Run a shell command inside a specific directory of the project. Times out after 30 seconds.
   Common read-only, build and test commands run directly; others need the user's approval."""
   if not directory or not directory.strip():
       return "Error: directory cannot be empty"
   if not os.path.isdir(directory):
       return f"Error: directory does not exist: {directory}"
   return run_guarded(command, cwd=directory, timeout=_TIMEOUT_SECONDS)[1]
