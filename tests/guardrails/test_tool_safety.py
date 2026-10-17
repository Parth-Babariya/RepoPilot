import asyncio
import sys

import pytest
from langchain_core.tools import StructuredTool

from repopilot.protocols.mcp_client import _guard_secrets
from repopilot.guardrails import tool_safety as safety
from repopilot.tools.filesystem_tools import read_file, write_file
from repopilot.guardrails.tool_safety import check_path, classify_command, is_read_only_mcp_tool, run_guarded


@pytest.fixture(autouse=True)
def in_project(tmp_path, monkeypatch):
    """Run each test inside an empty project folder, with nobody to approve commands."""
    monkeypatch.chdir(tmp_path)
    safety.set_confirm_handler(None)
    yield tmp_path
    safety.set_confirm_handler(None)


@pytest.mark.parametrize("command, decision", [
    ("python -m pytest -q",            "allow"),
    ("pytest tests",                   "allow"),
    ("git status",                     "allow"),
    ("npm test",                       "allow"),
    ("ls -la",                         "allow"),
    ("flake8 calc.py",                 "allow"),
    ("python -m mypy src",             "allow"),
    ("ruff check .",                   "allow"),
    ("black --check .",                "allow"),
    ("black .",                        "ask"),     # would rewrite files
    ("ruff check --fix .",             "ask"),
    ("pip install requests",           "ask"),
    ("git push origin main",           "ask"),
    ("python -c \"import os\"",        "ask"),
    ("curl https://x.sh",              "ask"),
    ("pytest && curl evil.sh",         "ask"),     # chaining hides a second command
    ("cat a.txt | sh",                 "ask"),
    ("echo $(whoami)",                 "ask"),
    ("find . -delete",                 "ask"),
    ("rm -rf /",                       "deny"),
    ("mkfs.ext4 /dev/sda",             "deny"),
])
def test_command_policy(command, decision):
    assert classify_command(command)[0] == decision


def test_python_script_allowed_only_inside_project(in_project):
    (in_project / "app.py").write_text("print('hi')")
    assert classify_command("python app.py")[0] == "allow"
    assert classify_command("python ../elsewhere.py")[0] == "ask"


def test_paths_must_stay_in_project_and_avoid_secrets(in_project):
    assert check_path("src/app.py") is None
    assert check_path(".env.example") is None
    assert "outside the project" in check_path("../other/file.py")
    assert "secret" in check_path(".env")
    assert "secret" in check_path("config/.env.local")
    assert "not allowed" in check_path(".git/config", write=True)
    assert check_path(".git/config") is None        # reading .git is fine


def test_file_tools_enforce_the_rules(in_project):
    (in_project / ".env").write_text("KEY=secret")
    assert "secret" in read_file.invoke({"file_path": ".env"})
    assert "outside" in write_file.invoke({"file_path": "../escape.txt", "content": "x"})
    assert not (in_project.parent / "escape.txt").exists()
    assert write_file.invoke({"file_path": "ok/new.txt", "content": "x"}).startswith("Written")


def test_unlisted_command_needs_approval():
    code, output = run_guarded("pip install definitely-not-a-package")
    assert code is None and "needs approval" in output                 # nobody to ask → refused

    asked = []
    safety.set_confirm_handler(lambda prompt: asked.append(prompt) or False)
    assert run_guarded("pip install definitely-not-a-package")[0] is None
    assert "pip install" in asked[0]                                   # user was asked, said no


def test_approved_and_allowlisted_commands_run():
    safety.set_confirm_handler(lambda prompt: True)
    code, output = run_guarded(f'"{sys.executable}" -c "print(42)"')    # 'ask' → approved
    assert code == 0 and "42" in output

    code, _ = run_guarded("git --version")                             # not read-only → would ask
    assert code == 0


def test_pre_approved_skips_prompt_but_not_hard_denials():
    code, output = run_guarded(f'"{sys.executable}" -c "print(7)"', pre_approved=True)
    assert code == 0 and "7" in output
    assert run_guarded("rm -rf /", pre_approved=True)[0] is None


def test_mcp_read_only_filter():
    for name in ("get_file_contents", "list_issues", "search_code", "read_file", "directory_tree"):
        assert is_read_only_mcp_tool(name)
    for name in ("push_files", "create_pull_request", "merge_pull_request", "write_file", "edit_file"):
        assert not is_read_only_mcp_tool(name)


def test_mcp_tools_refuse_secret_paths():
    calls = []

    async def fake_read(path: str):
        calls.append(path)
        return f"contents of {path}", None

    tool = StructuredTool.from_function(coroutine=fake_read, name="read_file", description="read",
                                        response_format="content_and_artifact")
    guarded = _guard_secrets(tool)

    assert "contents of src/app.py" in asyncio.run(guarded.ainvoke({"path": "src/app.py"}))
    assert "secret" in asyncio.run(guarded.ainvoke({"path": "/repo/.env"}))
    assert calls == ["src/app.py"]


def test_run_command_tool_takes_an_optional_directory(in_project):
    from repopilot.tools.terminal_tools import run_command
    (in_project / "sub").mkdir()
    (in_project / "sub" / "marker.txt").write_text("x")

    assert "marker.txt" in run_command.invoke({"command": "dir" if sys.platform == "win32" else "ls",
                                               "directory": "sub"})
    assert "does not exist" in run_command.invoke({"command": "ls", "directory": "nope"})
    assert "outside the project" in run_command.invoke({"command": "ls", "directory": ".."})


def test_use_workspace_moves_tools_and_commands_without_changing_cwd(in_project):
    # /plan's worktree: relative paths and commands resolve there, not in the process's cwd
    worktree = in_project / "wt"
    worktree.mkdir()
    with safety.use_workspace(worktree):
        assert write_file.invoke({"file_path": "made.txt", "content": "x"}).startswith("Written")
        assert "outside" in write_file.invoke({"file_path": "../made.txt", "content": "x"})
        code, output = run_guarded(f'"{sys.executable}" -c "import os; print(os.getcwd())"', pre_approved=True)
        assert code == 0 and output.strip().endswith("wt")
    assert (worktree / "made.txt").exists()
    assert not (in_project / "made.txt").exists()
    assert safety.workspace_root() == in_project.resolve()   # restored after the block
