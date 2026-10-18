from __future__ import annotations

import re

from pydantic import BaseModel, Field
from repopilot.services.llm_service import get_fallback_llm, get_judge_llm, with_schema

# ------------------------------------------------------------------
# Answer critic — LLM-as-judge for /ask answers, before you see them.
# Runs after the citation checks (workflows/answer_verification.py) and
# decides one thing code can't: are the answer's claims about this codebase
# backed by the code the Q&A agent actually retrieved? A "no" sends the
# answer back to the agent once, with the unsupported claims listed.
# Same idea as the /plan critic, and the same cheap judge_model.
# ------------------------------------------------------------------

_EVIDENCE_CHARS = 12000  # retrieved code shown to the critic


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


_TERM = re.compile(r"`([^`\n]{2,80})`|([A-Za-z_][\w.]{3,})")


def select_evidence(evidence: list[str], answer: str, limit: int = _EVIDENCE_CHARS) -> str:
    """
    The retrieved code most relevant to the answer, within `limit` characters.
    Tool outputs are split into their chunks (search results are separated by
    ---), duplicates from repeated searches are dropped, and chunks are kept in
    order of how many of the answer's terms they contain. Simply cutting the
    concatenation dropped the chunk that backed the answer, and the critic then
    flagged true claims as unsupported.
    """
    terms = {a or b for a, b in _TERM.findall(answer)}
    chunks = list(dict.fromkeys(c.strip() for text in evidence for c in text.split("\n---\n") if c.strip()))
    ranked = sorted(chunks, key=lambda c: -sum(term in c for term in terms))  # stable: ties keep retrieval order
    picked, size = [], 0
    for chunk in ranked:
        if size + len(chunk) > limit:
            continue
        picked.append(chunk)
        size += len(chunk) + 5
    return "\n---\n".join(picked)


async def critique_answer(question: str, answer: str, evidence: list[str], unknown_names: list[str]) -> AnswerVerdict:
    """Judge whether an /ask answer is grounded in the retrieved code (one judge_model call, feature "ask-verify")."""
    llm = with_schema(get_judge_llm("ask-verify", temperature=0), AnswerVerdict)
    if (fallback := get_fallback_llm("ask-verify", temperature=0)) is not None:
        llm = llm.with_fallbacks([with_schema(fallback, AnswerVerdict)])

    prompt = _PROMPT.format(
        question=question,
        evidence=select_evidence(evidence, answer) or "(nothing retrieved)",
        unknown_names=", ".join(unknown_names) or "(none)",
        answer=answer,
    )
    return await llm.ainvoke(prompt)
