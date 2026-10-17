import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from repopilot.agents.answer_critic import AnswerVerdict
from repopilot.observability import usage
from repopilot.observability.usage import UsageStore
from repopilot.workflows import answer_verification as av
from repopilot.workflows.answer_verification import check_citations, verify_answer
from repopilot.workflows.ask_workflow import handle_query


@pytest.fixture(autouse=True)
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(usage, "_store", UsageStore(str(tmp_path / "usage.db")))
    monkeypatch.setitem(av.config, "ask", {"initial_search": False})   # tested in test_ask_workflow.py
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "orders.py").write_text("def place_order():\n    pass\n\n\ndef cancel_order():\n    pass\n")
    return tmp_path


def critic_says(monkeypatch, verdict: AnswerVerdict):
    calls = []

    async def fake(question, answer, evidence, unknown_names):
        calls.append(unknown_names)
        return verdict
    monkeypatch.setattr(av, "critique_answer", fake)
    return calls


def test_valid_citations_in_every_common_format():
    answer = ("`place_order` is in `src/orders.py:1`, `cancel_order` in src/orders.py (lines 5-6) "
              "and orders.py line 2. Docs: https://example.com/guide.py")
    report = check_citations(answer, [], evidence=[])
    assert report.problems == []
    assert report.checked == 3                        # the URL is not a citation


def test_missing_files_and_lines_past_the_end_are_problems():
    report = check_citations("See `src/payments.py` and `src/orders.py:40-45`.", [], evidence=[])
    assert any("payments.py" in p and "no such file" in p for p in report.problems)
    assert any("orders.py" in p and "only 6 lines" in p for p in report.problems)


def test_a_bare_file_name_can_be_a_plain_mention():
    # A real answer said "the repository does not contain a package.json file" — true,
    # and it must not fail the check. A path or line numbers still make it a citation.
    report = check_citations("This repo has no `package.json`, so `npm test` has nothing to run.", [], [])
    assert report.problems == []
    assert check_citations("See `package.json:3`.", [], []).problems


def test_code_names_missing_from_the_retrieved_code_are_reported():
    answer = "`place_order` calls `charge_card` via `PaymentGateway.charge`, see `/plan` and `True`."
    report = check_citations(answer, [], evidence=["def place_order(): ..."])
    assert report.unknown_names == ["charge_card", "PaymentGateway.charge"]   # not /plan or True


def test_citation_problems_skip_the_critic(monkeypatch):
    calls = critic_says(monkeypatch, AnswerVerdict(grounded=True, reason="ok"))
    check = asyncio.run(verify_answer("q", "It's in `src/nope.py`.", [], ["code"]))
    assert not check.passed and "nope.py" in check.feedback[0]
    assert calls == []


def test_critic_decides_when_citations_are_fine(monkeypatch):
    critic_says(monkeypatch, AnswerVerdict(grounded=False, unsupported_claims=["retries 5 times"], reason="x"))
    check = asyncio.run(verify_answer("q", "`src/orders.py` retries 5 times.", [], ["def place_order(): ..."]))
    assert not check.passed and check.feedback == ["retries 5 times"]


def test_verification_can_be_turned_off(monkeypatch):
    monkeypatch.setitem(av.config, "ask", {"verify": False})
    assert asyncio.run(verify_answer("q", "`src/nope.py`", [], [])) is None


class FakeAgent:
    """Answers from a script; every turn 'retrieves' orders.py."""

    def __init__(self, answers):
        self.answers, self.history, self.prompts = list(answers), [], []

    async def ainvoke(self, payload, config):
        text = payload["messages"][0]["content"]
        self.prompts.append(text)
        self.history += [HumanMessage(text),
                         ToolMessage(content="def place_order():\n    pass", tool_call_id="1"),
                         AIMessage(self.answers.pop(0))]
        return {"messages": list(self.history)}


def test_failed_answer_is_revised_once_with_the_reasons(monkeypatch):
    critic_says(monkeypatch, AnswerVerdict(grounded=True, reason="ok"))
    agent = FakeAgent(["It's in `src/order.py:3`.", "It's `place_order` in `src/orders.py:1`."])

    result = asyncio.run(handle_query(agent, "where are orders placed?", "t1"))

    assert result.revised and result.check.passed
    assert result.answer == "It's `place_order` in `src/orders.py:1`."
    assert "src/order.py" in agent.prompts[1]        # the revision prompt says what was wrong


def test_answer_still_failing_after_the_revision_is_flagged(monkeypatch):
    critic_says(monkeypatch, AnswerVerdict(grounded=False, unsupported_claims=["made up"], reason="x"))
    agent = FakeAgent(["first", "second"])

    result = asyncio.run(handle_query(agent, "q", "t2"))

    assert len(agent.prompts) == 2                   # ask.max_revisions = 1
    assert not result.check.passed and result.answer == "second"
