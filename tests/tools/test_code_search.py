import pytest
from repopilot.tools import code_search
from repopilot.tools.code_search import search_codebase, search_memory


@pytest.fixture
def retrievals(monkeypatch):
    calls = []

    def retrieve(query, k):
        calls.append(query)
        return [
            {
                "source": "a.py",
                "start_line": 1,
                "end_line": 2,
                "type": "module",
                "name": "_READ_ONLY_PROGRAMS",
                "content": "_READ_ONLY_PROGRAMS = {...}",
            }
        ]

    monkeypatch.setattr(code_search, "get_retriever", lambda: retrieve)
    return calls


def test_a_repeated_search_gets_a_reminder_instead_of_the_same_results(retrievals):
    # In /eval the agent searched "_READ_ONLY_PROGRAMS" 12 times in one answer
    with search_memory():
        assert "_READ_ONLY_PROGRAMS = {" in search_codebase.invoke({"query": "_READ_ONLY_PROGRAMS"})
        repeat = search_codebase.invoke({"query": "_READ_ONLY_PROGRAMS ="})  # same words
        assert "already searched" in repeat
        search_codebase.invoke({"query": "_ALWAYS_DENIED"})  # new words are fine
    assert retrievals == ["_READ_ONLY_PROGRAMS", "_ALWAYS_DENIED"]


def test_each_question_starts_with_a_fresh_memory(retrievals):
    for _ in range(2):
        with search_memory():
            search_codebase.invoke({"query": "cache"})
    search_codebase.invoke({"query": "cache"})  # outside a question: never blocked
    assert retrievals == ["cache", "cache", "cache"]
