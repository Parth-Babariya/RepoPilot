import shutil
import subprocess

import pytest

from repopilot.workflows import worktree as wt_module
from repopilot.workflows.worktree import apply_work, commit_work, open_worktree, remove_worktree

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def git(repo, *args) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo,
                          capture_output=True, text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "app.py").write_text("print('v1')\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_plan_work_stays_out_of_the_checkout_until_applied(repo):
    wt = open_worktree("abcdef12-0000", cwd=repo)
    assert wt.branch == "repopilot/plan-abcdef12"
    assert (wt.workspace / "app.py").exists()

    (wt.workspace / "app.py").write_text("print('v2')\n")
    (wt.workspace / "new.py").write_text("x = 1\n")
    (wt.workspace / "__pycache__").mkdir()
    (wt.workspace / "__pycache__" / "new.cpython-312.pyc").write_bytes(b"junk")

    assert (repo / "app.py").read_text() == "print('v1')\n"      # checkout untouched while it runs
    stat = commit_work(wt, "plan")
    assert "app.py" in stat and "new.py" in stat and "pyc" not in stat

    remove_worktree(wt, delete_branch=False)
    ok, _ = apply_work(wt)
    assert ok
    assert (repo / "app.py").read_text() == "print('v2')\n"
    assert set(git(repo, "diff", "--cached", "--name-only").split()) == {"app.py", "new.py"}   # staged, not committed
    assert git(repo, "log", "--oneline").count("\n") == 1


def test_resuming_reuses_the_same_worktree(repo):
    first = open_worktree("feedface-1", cwd=repo)
    (first.workspace / "partial.py").write_text("pass\n")
    again = open_worktree("feedface-1", cwd=repo)
    assert again.path == first.path and (again.workspace / "partial.py").exists()


def test_discard_deletes_the_branch(repo):
    wt = open_worktree("deadbeef-2", cwd=repo)
    (wt.workspace / "x.py").write_text("pass\n")
    commit_work(wt, "plan")
    remove_worktree(wt, delete_branch=True)
    assert not wt.path.exists()
    assert "repopilot/plan-deadbeef" not in git(repo, "branch")


def test_falls_back_to_in_place_without_git_or_when_disabled(tmp_path_factory, repo, monkeypatch):
    plain = tmp_path_factory.mktemp("plain")
    if subprocess.run(["git", "rev-parse"], cwd=plain, capture_output=True).returncode == 0:
        pytest.skip("tmp folder is inside a git repo")
    assert open_worktree("p1", cwd=plain) is None

    monkeypatch.setitem(wt_module.config, "tasks", {"isolation": "none"})
    assert open_worktree("p2", cwd=repo) is None
