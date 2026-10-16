from repopilot.context.retrievers import bm25
from repopilot.context.retrievers.bm25 import BM25Index, reciprocal_rank_fusion, tokenize


def chunk(source, name, content="", line=1):
    return {"source": source, "name": name, "content": content, "start_line": line,
            "end_line": line + 5, "type": "function"}


def test_tokenize_splits_identifiers_and_drops_stopwords():
    tokens = tokenize("What calls fail_task in TaskOrchestrator?")
    assert {"calls", "fail_task", "fail", "task", "taskorchestrator", "orchestrator"} <= set(tokens)
    assert "what" not in tokens and "in" not in tokens


def test_bm25_ranks_exact_identifier_match_first():
    docs = [
        "def complete_task(self, task_id): mark the task completed",
        "def fail_task(self, task_id, error): retry or mark failed",
        "def claim_task(self, task_id): set in progress",
    ]
    index = BM25Index([tokenize(d) for d in docs])
    assert index.top("fail_task", 3)[0][0] == 1


def test_rrf_rewards_chunks_found_by_both_methods():
    a, b, c = chunk("a.py", "a"), chunk("b.py", "b"), chunk("c.py", "c")
    fused = reciprocal_rank_fusion([[a, b], [c, b]])
    assert fused[0]["source"] == "b.py"                 # in both lists → ranked first
    assert {x["source"] for x in fused} == {"a.py", "b.py", "c.py"}


def test_keyword_index_rebuilds_after_invalidate():
    loads = []

    def load():
        loads.append(1)
        return [chunk("x.py", "fail_task", "def fail_task(): pass")]

    bm25.invalidate()
    assert bm25.search("fail_task", 5, load)[0]["source"] == "x.py"
    bm25.search("fail_task", 5, load)
    assert len(loads) == 1                              # cached
    bm25.invalidate()
    bm25.search("fail_task", 5, load)
    assert len(loads) == 2                              # rebuilt after an index change
