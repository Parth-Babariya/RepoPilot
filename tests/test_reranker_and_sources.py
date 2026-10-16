from repopilot.agent.sources import Source, add_source, collect_sources, format_sources
from repopilot.context.retrievers import reranker


def chunk(name: str, content: str = "") -> dict:
    return {"source": f"/repo/{name}.py", "name": name, "type": "function",
            "start_line": 1, "end_line": 2, "content": content}


class KeywordEncoder:
    """Stand-in cross-encoder: score = how often the query's words appear in the document."""
    def rerank(self, query, documents):
        words = query.lower().split()
        return [sum(doc.lower().count(w) for w in words) for doc in documents]


class BrokenEncoder:
    def rerank(self, query, documents):
        raise RuntimeError("model failed to load")


def test_rerank_reorders_by_score_and_keeps_k():
    chunks = [chunk("a", "nothing"), chunk("b", "retry retry"), chunk("c", "retry")]
    result = reranker.rerank("retry", chunks, k=2, encoder=KeywordEncoder())
    assert [c["name"] for c in result] == ["b", "c"]
    assert result[0]["rerank_score"] > result[1]["rerank_score"]


def test_rerank_falls_back_to_vector_order_on_error():
    chunks = [chunk("a"), chunk("b"), chunk("c")]
    assert reranker.rerank("q", chunks, k=2, encoder=BrokenEncoder()) == chunks[:2]


def test_with_reranking_overfetches_candidates(monkeypatch):
    monkeypatch.setattr(reranker, "rerank_config",
                        lambda: {"enabled": True, "model": "x", "candidates": 20})
    monkeypatch.setattr(reranker, "_get_encoder", lambda model: KeywordEncoder())
    requested = {}

    def fake_retrieve(query, k):
        requested["k"] = k
        return [chunk(f"c{i}", "retry" * (i % 3)) for i in range(k)]

    result = reranker.with_reranking(fake_retrieve)("retry", k=5)
    assert requested["k"] == 20
    assert len(result) == 5


def test_sources_are_collected_only_inside_collect_sources():
    add_source("/repo/ignored.py", 1)          # no-op: nothing is collecting
    with collect_sources() as sources:
        add_source("/repo/a.py", 10, 20)
    assert sources == [Source("/repo/a.py", 10, 20)]


def test_format_sources_merges_overlapping_ranges_per_file():
    sources = [
        Source("/repo/a.py", 10, 20), Source("/repo/b.py", 5, 5),
        Source("/repo/a.py", 15, 30), Source("/repo/a.py", 50, 50),
    ]
    lines = format_sources(sources)
    assert lines[0].endswith("a.py:10-30, 50")
    assert lines[1].endswith("b.py:5")


def test_format_sources_limits_files():
    sources = [Source(f"/repo/f{i}.py", 1, 1) for i in range(10)]
    lines = format_sources(sources, limit=3)
    assert len(lines) == 4 and lines[-1] == "... and 7 more files"
