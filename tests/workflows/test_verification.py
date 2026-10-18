import json
import sys

import pytest
from repopilot.guardrails import tool_safety as safety
from repopilot.workflows.verification import verify_task


@pytest.fixture(autouse=True)
def in_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    safety.set_confirm_handler(None)
    return tmp_path


def task(**overrides) -> dict:
    base = {
        "id": "task_001",
        "title": "T",
        "description": "Do it",
        "task_type": "implement",
        "output_files": "[]",
        "acceptance_criteria": "[]",
        "verify_command": None,
        "retry_count": 0,
        "max_retries": 3,
        "error": None,
    }
    return base | overrides


def test_missing_output_file_fails(in_project):
    result = verify_task(task(output_files=json.dumps(["app.py"])))
    assert not result.passed and "missing output files: app.py" in result.report


def test_syntax_error_fails_and_writes_no_pycache(in_project):
    (in_project / "app.py").write_text("def broken(:\n")
    result = verify_task(task(output_files=json.dumps(["app.py"])))
    assert not result.passed and "app.py does not parse" in result.report
    assert not (in_project / "__pycache__").exists()


def test_invalid_json_fails(in_project):
    (in_project / "data.json").write_text("{not json")
    assert not verify_task(task(output_files=json.dumps(["data.json"]))).passed


def test_verify_command_decides(in_project):
    (in_project / "app.py").write_text("print('ok')\n")
    files = json.dumps(["app.py"])
    passing = verify_task(task(output_files=files, verify_command=f'"{sys.executable}" -c "print(1)"'))
    failing = verify_task(
        task(
            output_files=files,
            verify_command=f'"{sys.executable}" -c "import sys; print(\'tests failed\'); sys.exit(1)"',
        )
    )

    assert passing.passed and "✓ all 1 output files exist" in passing.report
    assert not failing.passed and "exited 1" in failing.report and "tests failed" in failing.report


def test_no_checks_passes(in_project):
    assert verify_task(task()).passed
