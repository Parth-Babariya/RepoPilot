from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from repopilot.memory.conversation import TurnSafeSummarization


def turn(question: str, tool_calls: int) -> list:
    messages = [HumanMessage(question)]
    for i in range(tool_calls):
        call_id = f"{question}-{i}"
        messages += [
            AIMessage("", tool_calls=[{"name": "search_codebase", "args": {}, "id": call_id}]),
            ToolMessage("code", tool_call_id=call_id),
        ]
    return messages + [AIMessage("answer")]


def middleware(keep: int) -> TurnSafeSummarization:
    return TurnSafeSummarization(
        model=GenericFakeChatModel(messages=iter([])), trigger=("tokens", 10), keep=("messages", keep)
    )


def test_the_current_question_is_never_summarised_away():
    # 10 tool calls = 20 messages after the question: plain "keep 20" would cut it off
    messages = turn("old question", 3) + turn("what happens to blocked tasks?", 10)
    cutoff = middleware(keep=20)._determine_cutoff_index(messages)
    assert messages[cutoff].content == "what happens to blocked tasks?"  # earlier turns only


def test_earlier_turns_are_still_summarised():
    messages = turn("q1", 2) + turn("q2", 2) + turn("q3", 1)
    cutoff = middleware(keep=4)._determine_cutoff_index(messages)
    assert 0 < cutoff <= len(turn("q1", 2) + turn("q2", 2))
