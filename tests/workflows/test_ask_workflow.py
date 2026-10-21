import pytest
from repopilot.agents.qa_agent import SEARCH_RESULTS_MARKER
from repopilot.workflows import ask_workflow
from repopilot.workflows.ask_workflow import with_initial_search


class FakeSearch:
    def __init__(self, result="File: a.py (lines 1-3)\nCode:\ndef place_order(): ..."):
        self.result, self.queries = result, []

    def invoke(self, args):
        self.queries.append(args["query"])
        return self.result


@pytest.fixture
def search(monkeypatch):
    fake = FakeSearch()
    monkeypatch.setattr(ask_workflow, "search_codebase", fake)
    monkeypatch.setitem(ask_workflow.config, "ask", {"initial_search": True})
    return fake


def test_question_is_sent_with_its_search_results(search):
    message, found = with_initial_search("Where are orders placed?")
    assert message.startswith("Where are orders placed?")
    assert SEARCH_RESULTS_MARKER in message and "def place_order" in message
    assert found == [search.result]  # also evidence for the answer checks


def test_follow_ups_about_the_last_answer_are_not_searched(search):
    assert with_initial_search("Explain that in more detail", has_history=True) == ("Explain that in more detail", [])
    # a new question in an old session is still searched
    assert with_initial_search("How is the cache invalidated?", has_history=True)[1]
    assert search.queries == ["How is the cache invalidated?"]


def test_nothing_found_or_turned_off_sends_the_plain_question(search, monkeypatch):
    search.result = "No relevant code found."
    assert with_initial_search("q?") == ("q?", [])
    monkeypatch.setitem(ask_workflow.config, "ask", {"initial_search": False})
    search.result = "code"
    assert with_initial_search("q?") == ("q?", [])


def test_only_fully_checked_answers_are_cached(monkeypatch):
    from repopilot.workflows.answer_verification import AnswerCheck
    from repopilot.workflows.ask_workflow import cacheable

    monkeypatch.setitem(ask_workflow.config, "ask", {"verify": True})
    assert cacheable(AnswerCheck(True, "ok"), cut_short=False)
    assert not cacheable(AnswerCheck(False, "unsupported claims"), cut_short=False)
    assert not cacheable(AnswerCheck(True, "claims not checked", complete=False), cut_short=False)
    assert not cacheable(AnswerCheck(True, "ok"), cut_short=True)  # stopped by the token budget
    assert not cacheable(None, cut_short=False)  # checks should have run but didn't

    monkeypatch.setitem(ask_workflow.config, "ask", {"verify": False})
    assert cacheable(None, cut_short=False)  # nothing to check by choice
    assert not cacheable(None, cut_short=True)
