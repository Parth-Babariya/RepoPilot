from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage
from repopilot.agents.qa_agent import SEARCH_RESULTS_MARKER, GitHubToolsOnDemand


@dataclass
class Tool:
    name: str


@dataclass
class Request:
    messages: list
    tools: list = field(default_factory=lambda: [Tool("search_codebase"), Tool("list_issues"), Tool("read_file")])

    def override(self, **changes):
        return Request(changes.get("messages", self.messages), changes.get("tools", self.tools))


def visible_tools(question: str) -> set[str]:
    middleware = GitHubToolsOnDemand({"list_issues"})
    request = Request([HumanMessage("earlier question about issues"), AIMessage("answer"), HumanMessage(question)])
    return {t.name for t in middleware.wrap_model_call(request, lambda r: r).tools}


def test_github_tools_are_hidden_for_questions_about_the_code():
    assert visible_tools("How does the cache work?") == {"search_codebase", "read_file"}


def test_github_tools_are_shown_for_questions_about_github():
    for question in ("List the open issues", "What changed in the last PR?", "Which commits touched main.py?"):
        assert "list_issues" in visible_tools(question)


def test_only_the_question_decides_not_the_attached_search_results():
    question = f"How is a task retried?\n\n{SEARCH_RESULTS_MARKER}; ...]\n\ndef commit(): ...  # issue tracker"
    assert "list_issues" not in visible_tools(question)
