from pathlib import Path

import pytest

from repopilot.evaluation.metrics import (
    context_recall, fact_recall, file_matches, load_questions, mean, reciprocal_rank, retrieval_rank,
)


def test_file_matches_handles_windows_paths_and_partial_names():
    assert file_matches(r"C:\proj\repopilot\tasks\task_store.py", "tasks/task_store.py")
    assert file_matches("/home/u/proj/main.py", "main.py")
    assert not file_matches("/home/u/proj/not_main.py", "main.py")   # must match a whole path part


def test_retrieval_rank_and_mrr():
    sources = ["/r/a.py", "/r/tasks/task_store.py", "/r/b.py"]
    assert retrieval_rank(sources, ["tasks/task_store.py"]) == 2
    assert retrieval_rank(sources, ["missing.py"]) is None
    assert reciprocal_rank(2) == 0.5 and reciprocal_rank(None) == 0.0


def test_context_and_fact_recall():
    outputs = [r"File: C:\r\tasks\task_store.py (lines 1-5)", "orchestrator.py:99"]
    assert context_recall(outputs, ["tasks/task_store.py", "main.py"]) == 0.5
    assert context_recall(outputs, []) is None

    assert fact_recall("It calls FAIL_TASK and retries.", ["fail_task", "max_retries"]) == 0.5
    assert fact_recall("anything", []) is None


def test_mean_ignores_missing_values():
    assert mean([1.0, None, 0.0]) == 0.5
    assert mean([None]) is None


def test_bundled_question_set_is_valid():
    path = Path(__file__).resolve().parents[2] / "data" / "evals" / "questions.yaml"
    questions = load_questions(path)
    assert len(questions) >= 20
    assert len({q.id for q in questions}) == len(questions)   # ids are unique
    # every expected file actually exists (paths are relative to the package folder;
    # retrieval matching is by path suffix, so they match from the project root too)
    package = path.parents[2] / "src"
    for q in questions:
        for expected in q.expected_files:
            assert (package / expected).exists(), f"{q.id}: {expected} does not exist"


def test_load_questions_rejects_missing_question(tmp_path):
    bad = tmp_path / "q.yaml"
    bad.write_text("- expected_files: [a.py]\n")
    with pytest.raises(ValueError):
        load_questions(bad)
