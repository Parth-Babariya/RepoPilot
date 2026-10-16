import asyncio

from repopilot.evaluation import judge
from repopilot.evaluation.judge import JudgeVerdict, judge_answer
from repopilot.evaluation.metrics import EvalQuestion


class _FakeJudgeLLM:
    """Stands in for the judge model: records the prompt and returns a fixed verdict."""

    def __init__(self, verdict: JudgeVerdict):
        self.verdict = verdict
        self.prompt = None

    def with_structured_output(self, schema):
        assert schema is JudgeVerdict
        return self

    async def ainvoke(self, prompt):
        self.prompt = prompt
        return self.verdict


def _run_judge(monkeypatch, verdict, tool_outputs):
    fake = _FakeJudgeLLM(verdict)
    monkeypatch.setattr(judge, "get_judge_llm", lambda feature, **kw: fake)
    question = EvalQuestion(id="q", question="What calls fail_task?", expected_facts=["_execute"])
    result = asyncio.run(judge_answer(question, "TaskOrchestrator._execute calls it.", tool_outputs))
    return result, fake.prompt


def test_judge_prompt_contains_question_facts_evidence_and_answer(monkeypatch):
    verdict = JudgeVerdict(faithfulness=0.9, correctness=0.8, reason="ok")
    result, prompt = _run_judge(monkeypatch, verdict, ["File: tasks/orchestrator.py ... fail_task(...)"])

    assert (result.faithfulness, result.correctness) == (0.9, 0.8)
    assert "What calls fail_task?" in prompt
    assert "- _execute" in prompt
    assert "tasks/orchestrator.py" in prompt
    assert "TaskOrchestrator._execute calls it." in prompt


def test_judge_clamps_scores_and_handles_no_evidence(monkeypatch):
    verdict = JudgeVerdict(faithfulness=1.7, correctness=-0.2, reason="out of range")
    result, prompt = _run_judge(monkeypatch, verdict, [])

    assert (result.faithfulness, result.correctness) == (1.0, 0.0)
    assert "(nothing retrieved)" in prompt
