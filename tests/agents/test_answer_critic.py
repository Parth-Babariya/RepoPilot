from repopilot.agents.answer_critic import select_evidence

DENY = 'File: tool_safety.py\nCode:\n_ALWAYS_DENIED = ("rm -rf /", "mkfs", "dd if=")'


def filler(n: int) -> str:
    return f"File: other_{n}.py\nCode:\n" + "x = 1\n" * 400


def test_keeps_the_chunk_the_answer_relies_on_even_when_it_came_first():
    # The backing chunk was retrieved first, then many unrelated ones — cutting
    # the joined text from either end would lose it.
    evidence = ["\n---\n".join([DENY, filler(1), filler(2)]), "\n---\n".join([filler(3), filler(4)])]
    picked = select_evidence(evidence, "`_ALWAYS_DENIED` refuses `mkfs` and `dd if=`.", limit=5000)
    assert "_ALWAYS_DENIED" in picked and len(picked) <= 5000


def test_repeated_searches_are_not_counted_twice():
    picked = select_evidence([DENY, DENY, DENY], "mkfs", limit=10_000)
    assert picked.count("_ALWAYS_DENIED") == 1
