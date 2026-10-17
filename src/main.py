import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.prompt import Confirm, Prompt

from repopilot.config import config
from repopilot.retrieval.indexing.factory import get_indexer, get_index_inspector
from repopilot.services.llm_service import required_api_keys
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from repopilot.agents.qa_agent import build_agent
from repopilot.memory.conversation import get_checkpointer_db_path
from repopilot.workflows.ask_workflow import QueryResult, handle_query
from repopilot.tools.sources import format_sources
from repopilot.memory.session import get_current_session, new_session, switch_session
from repopilot.memory.semantic_cache import build_semantic_cache, get_repo_domain
from repopilot.observability.logger import enable_file_logging, get_logger
from repopilot.observability.usage import track
from repopilot.observability.stats import show_stats
from repopilot.evaluation.runner import run_eval
from repopilot.evaluation.judge_check import run_judge_check

from repopilot.workflows.plan_workflow import handle_plan_command
from repopilot.workflows.status import show_task_status
from repopilot.retrieval.indexing.watcher import start_watcher, stop_watcher
from repopilot.guardrails.tool_safety import set_confirm_handler

# .env lives in the project root (next to pyproject.toml) and wins over keys
# already set in Windows/shell environment variables.
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # src/main.py → root
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE, override=True)
# Google's docs call the key GEMINI_API_KEY; LangChain reads GOOGLE_API_KEY — accept either.
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

console = Console()
logger = get_logger("repopilot.main")   # not __name__: that is "__main__" under python -m


COMMANDS = [
    ("/ask <question>",      "ask a question about the codebase"),
    ("/plan <goal>",         "generate and execute a plan"),
    ("/task_status",         "show task progress for the latest project"),
    ("/stats",               "show token usage, cost and cache hit rate"),
    ("/eval [file]",         "score answer quality on a question set"),
    ("/judge_check [file]",  "label eval answers yourself and see how far the LLM judge agrees"),
    ("/show_index",          "show all chunks in the index"),
    ("/reindex",             "manually re-index current directory"),
    ("/new_session",         "start a fresh conversation"),
    ("/switch <session_id>", "resume a past session"),
    ("/session",             "show current session id"),
    ("/help",                "show this list"),
    ("/exit",                "quit"),
]


def print_answer(result: QueryResult) -> None:
    """Render the answer as Markdown (headings, lists, highlighted code) with its sources below."""
    console.print()
    console.print(Markdown(result.answer, code_theme="monokai"))
    if result.cached:
        console.print("\n[dim]⚡ Answered from the semantic cache[/dim]")
    elif result.sources:
        console.print("\n[dim]Sources:[/dim]")
        for line in format_sources(result.sources):
            console.print(f"[dim]  • {line}[/dim]", highlight=False)
    if result.check is not None:
        if result.check.passed:
            when = "after one revision" if result.revised else ""
            console.print(f"[dim]✓ Checked{' ' + when if when else ''}: {escape(result.check.summary)}[/dim]")
        else:
            console.print(f"[yellow]⚠ Not verified — {escape(result.check.summary)}[/yellow]")
    console.print()


def confirm_command(status=None):
    """
    Approval prompt for agent commands that aren't on the safety allowlist.
    Pauses the "Searching..." spinner (if any) while the user answers.
    """
    def ask(prompt: str) -> bool:
        if status is not None:
            status.stop()
        try:
            console.print()
            return Confirm.ask(f"[yellow]{escape(prompt)}[/yellow]", default=False)
        finally:
            if status is not None:
                status.start()
    return ask


def print_help() -> None:
    for usage, description in COMMANDS:
        # escape(): "[file]" in "/eval [file]" would otherwise be read as a Rich style tag and vanish
        console.print(f"  [bold]{escape(f'{usage:<22}')}[/bold] — {description}")


def check_environment() -> list[str]:
    """
    Return human-readable problems with the configuration (empty if OK), so a
    missing key fails fast with a clear message instead of mid-query.
    Local embeddings (fastembed) need no key.
    """
    problems = []
    try:
        needed = required_api_keys()
    except ValueError as e:
        return [str(e)]
    for key, reason in needed.items():
        if not os.getenv(key):
            problems.append(f"{key} is not set ({reason} in config.yaml)")
    if config["embeddings"]["provider"] == "openai" and not os.getenv("OPENAI_API_KEY"):
        problems.append("OPENAI_API_KEY is not set (embeddings.provider is 'openai' in config.yaml)")
    return sorted(set(problems))


def get_or_create_index():
    repo_path = str(Path.cwd())
    logger.info(f"Checking index for: {repo_path}")
    console.print(f"[dim]Checking index for {repo_path}...[/dim]")
    return get_indexer()(repo_path)


async def initialize(checkpointer):
    """Bootstrap the index, watcher, cache, agent (with MCP tools) and session before the REPL starts."""
    console.print(f"[dim]LLM: {config['llm']['provider']} / {config['llm']['model']}[/dim]")
    console.print(f"[dim]Embedder: {config['embeddings']['provider']} / {config['embeddings']['model']}[/dim]")

    repo_path = str(Path.cwd())
    index = get_or_create_index()

    semantic_cache = await build_semantic_cache()
    cache_domain = get_repo_domain(repo_path) if semantic_cache else None
    if semantic_cache is not None:
        console.print(f"[dim]Semantic cache: enabled (threshold={semantic_cache.threshold})[/dim]")
    else:
        reason = "turned off in config.yaml" if not config.get("semantic_cache", {}).get("enabled") \
            else "couldn't start it — is Redis running? See the log for details"
        console.print(f"[dim]Semantic cache: disabled ({reason})[/dim]")

    loop = asyncio.get_running_loop()

    def _invalidate_cache_on_change() -> None:
        if semantic_cache is not None:
            asyncio.run_coroutine_threadsafe(
                semantic_cache.invalidate_domain(cache_domain), loop
            )

    observer   = start_watcher(repo_path, on_change=_invalidate_cache_on_change)
    agent      = await build_agent(checkpointer)
    session_id = get_current_session()
    console.print(f"[dim]Session: {session_id}[/dim]")
    console.print(f"[green]✓ Ready[/green]\n")
    return index, agent, session_id, observer, semantic_cache, cache_domain


async def _run_async():
    logger.info("Starting RepoPilot")
    console.print("\n[bold blue]RepoPilot[/bold blue] — RAG-powered code assistant")

    problems = check_environment()
    if problems:
        console.print("[bold red]Can't start — configuration problems:[/bold red]")
        for problem in problems:
            console.print(f"  [red]•[/red] {problem}")
        console.print(f"[dim]Add the missing values to {ENV_FILE} (copy .env.example to start)[/dim]")
        sys.exit(1)

    async with AsyncSqliteSaver.from_conn_string(get_checkpointer_db_path()) as checkpointer:
        index, agent, session_id, observer, semantic_cache, cache_domain = await initialize(checkpointer)
        console.print("Type [bold]/help[/bold] for commands, [bold]/exit[/bold] to quit\n")

        try:
            while True:
                # lstrip BOM: PowerShell adds one to text piped into the program
                user_input = Prompt.ask("[bold green]>[/bold green]").lstrip("﻿").strip()

                if not user_input.strip():
                    continue
                if user_input.lower() in ("/exit", "/quit"):
                    logger.info("Shutting down")
                    console.print("[dim]Goodbye![/dim]")
                    break
                elif user_input.startswith("/ask "):
                    question = user_input.removeprefix("/ask ").strip()
                    logger.info(f"Ask command received: {question}")
                    with track("ask", session_id), console.status("[dim]Searching the codebase...[/dim]") as status:
                        set_confirm_handler(confirm_command(status))
                        try:
                            result = await handle_query(
                                agent, question, session_id,
                                semantic_cache=semantic_cache, cache_domain=cache_domain,
                            )
                        finally:
                            set_confirm_handler(None)
                    print_answer(result)
                elif user_input == "/reindex":
                    console.print("[dim]Re-indexing current directory...[/dim]")
                    get_or_create_index()
                    if semantic_cache is not None:
                        await semantic_cache.invalidate_domain(cache_domain)
                        console.print("[dim]Semantic cache invalidated for this repo.[/dim]")
                    console.print("[green]✓ Re-index complete.[/green]")
                elif user_input == "/new_session":
                    session_id = new_session()
                    console.print(f"[green]New session started: {session_id}[/green]")
                elif user_input.startswith("/switch "):
                    target = user_input.removeprefix("/switch ").strip()
                    session_id = switch_session(target)
                    console.print(f"[green]Switched to session: {session_id}[/green]")
                elif user_input == "/session":
                    console.print(f"[dim]Current session: {session_id}[/dim]")
                elif user_input == "/show_index":
                    logger.info("Showing index")
                    get_index_inspector()(index)
                elif user_input.startswith("/plan "):
                    goal = user_input.removeprefix("/plan ").strip()
                    logger.info(f"Plan command received: {goal}")
                    set_confirm_handler(confirm_command())
                    try:
                        with track("plan", session_id):
                            await handle_plan_command(goal)
                    finally:
                        set_confirm_handler(None)
                elif user_input == "/task_status":
                    show_task_status()
                elif user_input == "/stats":
                    show_stats(session_id)
                elif user_input == "/eval" or user_input.startswith("/eval "):
                    questions_file = user_input.removeprefix("/eval").strip() or None
                    await run_eval(agent, questions_file)
                elif user_input == "/judge_check" or user_input.startswith("/judge_check "):
                    run_judge_check(user_input.removeprefix("/judge_check").strip() or None)
                elif user_input == "/help":
                    print_help()
                else:
                    logger.warning(f"Unknown command received: {user_input}")
                    console.print("[yellow]Unknown command. Try:[/yellow]")
                    print_help()
        finally:
            stop_watcher(observer)

# Provider account problems (bad key, no credits, network) — shown as a short
# explanation instead of a traceback. Matched by class name so both the
# openai and anthropic SDKs are covered without importing either here.
_PROVIDER_ERRORS = {
    "AuthenticationError":   "The API key was rejected — check the key in your .env file.",
    "ClientError":           "The provider rejected the request — check the API key and llm.model in config.yaml.",
    "ChatGoogleGenerativeAIError": "Gemini rejected the request — check GOOGLE_API_KEY and llm.model (see `python -m repopilot.services.list_models`).",
    "PermissionDeniedError": "The API key doesn't have access to this model.",
    "RateLimitError":        "The provider refused the request: rate limit hit or no credits left on the account.",
    "NotFoundError":         "The model in config.yaml wasn't found — check llm.model / embeddings.model.",
    "APIConnectionError":    "Couldn't reach the provider — check your internet connection.",
}


def run():
    log_file = enable_file_logging()
    console.print(f"[dim]Logs: {log_file}[/dim]")
    try:
        asyncio.run(_run_async())
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")
    except Exception as e:
        hint = _PROVIDER_ERRORS.get(type(e).__name__)
        if hint is None or not type(e).__module__.startswith(("openai", "anthropic", "google", "langchain_google_genai", "groq")):
            raise
        logger.error(f"Provider error: {e}")
        console.print(f"\n[bold red]{hint}[/bold red]")
        console.print(f"[dim]{type(e).__name__}: {str(e)[:300]}[/dim]")
        sys.exit(1)

if __name__ == "__main__":
    run()