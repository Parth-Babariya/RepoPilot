from repopilot.agents.task_agent import TOOLS_BY_TYPE, build_system_prompt, opening_message


def task(**overrides) -> dict:
    base = {"id": "task_001", "title": "T", "description": "Do it", "task_type": "implement",
            "output_files": '["app.py"]', "acceptance_criteria": '["works"]',
            "retry_count": 0, "max_retries": 3, "error": None}
    return base | overrides


def test_retry_prompt_includes_previous_failure():
    first = build_system_prompt(task(), [])
    retry = build_system_prompt(task(retry_count=1, error="Automatic checks failed:\n✗ `pytest` exited 1"), [])

    assert "PREVIOUS ATTEMPT FAILED" not in first
    assert "PREVIOUS ATTEMPT FAILED (this is attempt 2 of 3)" in retry
    assert "`pytest` exited 1" in retry


def test_prompt_includes_criteria_files_and_dependency_outputs():
    prompt = build_system_prompt(task(), [{"id": "task_000", "title": "Schema", "result": "- models.py"}])
    assert "  - app.py" in prompt and "  - works" in prompt
    assert "[task_000] Schema" in prompt and "- models.py" in prompt


def test_only_test_and_integrate_tasks_can_run_commands():
    can_run = {kind for kind, tools in TOOLS_BY_TYPE.items() if any(t.name == "run_command" for t in tools)}
    assert can_run == {"test", "integrate"}


def test_every_task_can_find_existing_code_read_only():
    for kind, tools in TOOLS_BY_TYPE.items():
        names = {t.name for t in tools}
        assert {"search_codebase", "find_symbol", "read_lines"} <= names, kind


def test_opening_message_depends_on_whether_there_is_code(tmp_path, monkeypatch):
    task = {"description": "Add retry logging.", "retry_count": 0}
    monkeypatch.chdir(tmp_path)
    assert "directory is empty" in opening_message(task)
    (tmp_path / "orchestrator.py").write_text("def run():\n    pass\n")
    assert "existing codebase" in opening_message(task)
    assert "previous attempt failed" in opening_message({**task, "retry_count": 1})
