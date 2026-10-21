from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from repopilot.agents.answer_critic import critique_answer
from repopilot.config import config
from repopilot.guardrails.tool_safety import workspace_root
from repopilot.observability.logger import get_logger
from repopilot.retrieval.indexing.code_parser import SKIP_DIRS
from repopilot.tools.sources import Source

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Answer verification — /ask's counterpart to /plan's verification.py
# ------------------------------------------------------------------
# Before an answer is shown:
#   1. Citation checks (code, free): every file the answer cites must exist,
#      and every cited line number must be inside that file.
#   2. Answer critic (one judge_model call): are the answer's claims about the
#      code backed by the code the agent retrieved?
# A failure in step 1 skips step 2 — the answer is wrong either way. The
# caller (ask_workflow) gives the agent one chance to revise.
# ------------------------------------------------------------------

_EXTENSIONS = (
    "py|pyi|js|jsx|ts|tsx|mjs|cjs|java|kt|go|rs|rb|php|c|h|cc|cpp|hpp|cs|swift|scala|"
    "sh|sql|html|css|vue|md|yaml|yml|toml|json|ini|cfg"
)
# path.ext, then optionally ":12", ":12-30", " (lines 12-30)", " line 12"
_CITATION = re.compile(
    rf"(?<![\w./-])((?:[\w.-]+/)*[\w-][\w.-]*\.(?:{_EXTENSIONS}))\b"
    r"(?:(?::|,?\s*\(?\s*lines?\s+)(\d+)(?:\s*[-–]\s*(\d+))?)?",
    re.IGNORECASE,
)
# `name`, `name()`, `Class.method` — only names that look like code identifiers
_CODE_NAME = re.compile(r"`([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)(?:\(\))?`")
_MAX_FILES_SCANNED = 20000


@dataclass
class CitationReport:
    checked: int = 0
    problems: list[str] = field(default_factory=list)  # objectively wrong: missing files, lines past the end
    unknown_names: list[str] = field(default_factory=list)  # code names not found in the retrieved code


@dataclass
class AnswerCheck:
    passed: bool
    summary: str  # one line shown under the answer
    feedback: list[str] = field(default_factory=list)  # what the agent must fix on a revision
    complete: bool = True  # False when a check that should have run was skipped (nothing retrieved, critic failed)


def _looks_like_code(name: str) -> bool:
    last = name.rsplit(".", 1)[-1]
    return "_" in last or "." in name or any(c.isupper() for c in last[1:])


class _RepoFiles:
    """Project files by relative path, listed once and only if a bare file name needs looking up."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._paths: list[str] | None = None

    def matching(self, cited: str) -> list[Path]:
        if self._paths is None:
            self._paths = []
            for folder, dirs, files in os.walk(self.root):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info")]
                rel = Path(folder).relative_to(self.root).as_posix()
                self._paths += [f if rel == "." else f"{rel}/{f}" for f in files]
                if len(self._paths) > _MAX_FILES_SCANNED:
                    break
        return [self.root / p for p in self._paths if p == cited or p.endswith("/" + cited)]


def _line_count(path: Path) -> int:
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def check_citations(answer: str, sources: list[Source], evidence: list[str]) -> CitationReport:
    """Check the answer's file and line references against the project on disk."""
    root, report = workspace_root(), CitationReport()
    repo_files = _RepoFiles(root)
    cited_files: list[Path] = []
    seen: set[tuple] = set()

    for match in _CITATION.finditer(answer.replace("\\", "/")):
        cited, start, end = match.group(1).removeprefix("./"), match.group(2), match.group(3)
        if (cited, start, end) in seen:
            continue
        seen.add((cited, start, end))
        report.checked += 1

        candidates = [root / cited] if (root / cited).is_file() else []
        candidates = candidates or [Path(s.path) for s in sources if Path(s.path).as_posix().endswith(cited)][:1]
        candidates = candidates or repo_files.matching(cited)
        if not candidates:
            # Only a real citation — a path or line numbers — must exist. A bare name can be
            # a plain mention: "this repo has no package.json" is true and must not fail.
            if "/" in cited or start:
                report.problems.append(f"`{cited}` is cited but no such file exists in the project")
            continue
        cited_files += candidates

        last = int(end or start or 0)
        if last and all(_line_count(p) < last for p in candidates):
            lines = max(_line_count(p) for p in candidates)
            report.problems.append(f"`{cited}` is cited at line {last}, but it has only {lines} lines")

    # Names the answer presents as code but that appear nowhere in what it looked at
    known = "\n".join(evidence)
    for path in dict.fromkeys(cited_files):
        try:
            known += "\n" + path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    for name in dict.fromkeys(_CODE_NAME.findall(answer)):
        if _looks_like_code(name) and name.rsplit(".", 1)[-1] not in known:
            report.unknown_names.append(name)
    return report


def _settings() -> dict:
    cfg = config.get("ask", {}) or {}
    return {
        "verify": bool(cfg.get("verify", True)),
        "verify_with_llm": bool(cfg.get("verify_with_llm", True)),
        "max_revisions": int(cfg.get("max_revisions", 1)),
    }


def max_revisions() -> int:
    return _settings()["max_revisions"]


def verification_enabled() -> bool:
    return _settings()["verify"]


async def verify_answer(question: str, answer: str, sources: list[Source], evidence: list[str]) -> AnswerCheck | None:
    """Run the checks on one answer. None when verification is turned off (ask.verify)."""
    settings = _settings()
    if not settings["verify"]:
        return None

    report = check_citations(answer, sources, evidence)
    logger.info(
        f"Citation check: {report.checked} checked, problems={report.problems}, unknown names={report.unknown_names}"
    )
    if report.problems:
        return AnswerCheck(False, "; ".join(report.problems), report.problems)

    refs = f"{report.checked} file reference{'s' if report.checked != 1 else ''} valid"
    if not settings["verify_with_llm"]:
        return AnswerCheck(True, refs)
    if not evidence:
        return AnswerCheck(True, f"{refs} · claims not checked (no code was retrieved)", complete=False)

    try:
        verdict = await critique_answer(question, answer, evidence, report.unknown_names)
    except Exception as e:
        # The check is a safety net — if it fails, show the answer rather than nothing.
        logger.warning(f"Answer critic failed: {e}")
        return AnswerCheck(True, f"{refs} · claims not checked (the answer critic failed)", complete=False)
    logger.info(
        f"Answer critic: grounded={verdict.grounded}, reason={verdict.reason}, unsupported={verdict.unsupported_claims}"
    )
    if verdict.grounded:
        return AnswerCheck(True, f"{refs} · claims supported by the retrieved code")
    claims = verdict.unsupported_claims or [verdict.reason]
    return AnswerCheck(False, f"unsupported claims: {'; '.join(claims)}", claims)
