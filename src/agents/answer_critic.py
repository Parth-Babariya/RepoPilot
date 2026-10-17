from __future__ import annotations

from pydantic import BaseModel, Field

from repopilot.services.llm_service import get_fallback_llm, get_judge_llm

# ------------------------------------------------------------------
# Answer critic — LLM-as-judge for /ask answers, before you see them.
# Runs after the citation checks (workflows/answer_verification.py) and
# decides one thing code can't: are the answer's claims about this codebase
# backed by the code the Q&A agent actually retrieved? A "no" sends the
# answer back to the agent once, with the unsupported claims listed.
# Same idea as the /plan critic, and the same cheap judge_model.
# ------------------------------------------------------------------

_EVIDENCE_CHARS = 12000   # retrieved code shown to the critic


class AnswerVerdict(BaseModel):
    grounded: bool
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Specific claims about this codebase that the retrieved code does not support (empty if grounded)",
    )
    reason: str = Field(description="One short sentence")


_PROMPT = """You check answers about a codebase before the user sees them.

QUESTION:
{question}

CODE THE ASSISTANT RETRIEVED WHILE ANSWERING:
{evidence}

NAMES IN THE ANSWER THAT DON'T APPEAR IN THAT CODE (may be fine, e.g. library names):
{unknown_names}

ANSWER:
{answer}

Is every specific claim the answer makes about THIS codebase — files, functions,
classes, values, what the code does and in what order — supported by the retrieved code?
- General programming knowledge and explanations are fine; only check claims about this code.
- An honest "I couldn't find it" is grounded.
- grounded=false only if some claim contradicts the code or has nothing in it to back it up.
  List each such claim briefly in unsupported_claims."""


async def critique_answer(question: str, answer: str, evidence: list[str], unknown_names: list[str]) -> AnswerVerdict:
    """Judge whether an /ask answer is grounded in the retrieved code (one judge_model call, feature "ask-verify")."""
    llm = get_judge_llm("ask-verify", temperature=0).with_structured_output(AnswerVerdict)
    if (fallback := get_fallback_llm("ask-verify", temperature=0)) is not None:
        llm = llm.with_fallbacks([fallback.with_structured_output(AnswerVerdict)])

    prompt = _PROMPT.format(
        question=question,
        evidence="\n---\n".join(evidence)[-_EVIDENCE_CHARS:] or "(nothing retrieved)",
        unknown_names=", ".join(unknown_names) or "(none)",
        answer=answer,
    )
    return await llm.ainvoke(prompt)
