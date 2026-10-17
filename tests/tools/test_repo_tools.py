import pytest

from repopilot.tools.repo_tools import read_lines


@pytest.fixture(autouse=True)
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text("\n".join(f"line {n}" for n in range(1, 301)))
    return tmp_path


def test_reads_just_the_requested_lines_with_numbers():
    text = read_lines.invoke({"file_path": "app.py", "start_line": 10, "end_line": 12})
    assert text.splitlines()[1:4] == ["10  line 10", "11  line 11", "12  line 12"]
    assert "continues to line 300" in text


def test_long_ranges_are_capped_and_bad_ranges_explained():
    text = read_lines.invoke({"file_path": "app.py", "start_line": 1, "end_line": 300})
    assert "lines 1-200" in text
    assert "only 300 lines" in read_lines.invoke({"file_path": "app.py", "start_line": 400, "end_line": 410})


def test_same_safety_rules_as_the_file_tools(project):
    (project / ".env").write_text("KEY=secret")
    assert "secret" in read_lines.invoke({"file_path": ".env", "start_line": 1, "end_line": 1}).lower()
    assert "KEY=secret" not in read_lines.invoke({"file_path": ".env", "start_line": 1, "end_line": 1})
    assert "outside" in read_lines.invoke({"file_path": "../x.py", "start_line": 1, "end_line": 1})
