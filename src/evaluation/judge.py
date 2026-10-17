from __future__ import annotations

from pydantic import BaseModel, Field

from repopilot.evaluation.metrics import EvalQuestion
from repopilot.services.llm_service import get_fallback_llm, get_judge_llm

# ------------------------------------------------------------------
# LLM-as-judge eval metrics — the counterpart to metrics.py.
#
# The deterministic metrics check *what* was retrieved and *which* facts
# were mentioned, but can't tell whether the answer is actually right or
# whether it invents things. A separate judge model (llm.judge_model, at
# temperature 0) reads the question, the code the agent retrieved and the
# answer, and scores:
#   faithfulness — are the claims supported by the retrieved code?
#   correctness  — does it fully and correctly answer the question?
# ------------------------------------------------------------------

_JUDGE_CONTEXT_CHARS = 12_000


class JudgeVerdict(BaseModel):
    faithfulness: float = Field(description="0-1: how fully the answer's claims are supported by the retrieved code")
    correctness:  float = Field(description="0-1: how well the answer correctly and completely answers the question")
    reason: str         = Field(description="One short sentence")


_JUDGE_PROMPT = """You grade answers from a code assistant.

QUESTION:
{question}

FACTS A CORRECT ANSWER SHOULD MENTION (may be empty):
{facts}

CODE THE ASSISTANT RETRIEVED (its only evidence):
{context}

ASSISTANT'S ANSWER:
{answer}

Score two things from 0 to 1:
- faithfulness: are the answer's claims supported by the retrieved code? Penalise
  invented functions, files or behaviour. An honest "I couldn't find it" is faithful.
- correctness: does it correctly and completely answer the question, covering the facts listed?"""


async def judge_answer(question: EvalQuestion, answer: str, tool_outputs: list[str]) -> JudgeVerdict:
    """Score one answer with the judge model. Its cost is tracked as feature "eval-judge"."""
    llm = get_judge_llm("eval-judge", temperature=0).with_structured_output(JudgeVerdict)
    if (fallback := get_fallback_llm("eval-judge", temperature=0)) is not None:
        llm = llm.with_fallbacks([fallback.with_structured_output(JudgeVerdict)])

    prompt = _JUDGE_PROMPT.format(
        question=question.question,
        facts="\n".join(f"- {f}" for f in question.expected_facts) or "(none)",
        context="\n---\n".join(tool_outputs)[:_JUDGE_CONTEXT_CHARS] or "(nothing retrieved)",
        answer=answer,
    )
    verdict = await llm.ainvoke(prompt)
    verdict.faithfulness = min(max(verdict.faithfulness, 0.0), 1.0)
    verdict.correctness  = min(max(verdict.correctness, 0.0), 1.0)
    return verdict
