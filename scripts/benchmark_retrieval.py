"""
Retrieval benchmark — reproduces the numbers in the README.

Scores search alone (no LLM, no API key, free) on the eval question set:
for each search setup, is a file that answers the question in the top k
results (hit@k), and how high (MRR)?

    python scripts/benchmark_retrieval.py              # from the project root
    python scripts/benchmark_retrieval.py --k 10

Run it in the repo the questions describe (RepoPilot itself, by default):
it first brings the index up to date, using the configured vector store
and embedding model.
"""
import argparse
import os
import time

from repopilot.config import config
from repopilot.evaluation.metrics import load_questions, mean, reciprocal_rank, retrieval_rank
from repopilot.retrieval.indexing.factory import get_indexer
from repopilot.retrieval.search.factory import get_retriever

SETUPS = [
    ("Vector only",                 "semantic", False),
    ("Vector + reranker",           "semantic", True),
    ("Hybrid (vector + BM25)",      "hybrid",   False),
    ("Hybrid + reranker",           "hybrid",   True),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default=config["eval"]["questions_file"])
    parser.add_argument("--k", type=int, default=int(config["eval"].get("top_k", 5)))
    args = parser.parse_args()

    questions = [q for q in load_questions(args.questions) if q.expected_files]
    print(f"Indexing {os.getcwd()} …", flush=True)
    get_indexer()(os.getcwd())   # absolute path, exactly as the app does

    print(f"\n{len(questions)} questions · {config['embeddings']['model']} · "
          f"{config['vector_store']['provider']}\n")
    print(f"{'Search':26} {'hit@' + str(args.k):>7} {'MRR':>6} {'s/query':>8}  missed")
    for label, mode, rerank in SETUPS:
        config["rag"]["mode"] = mode
        config["rerank"]["enabled"] = rerank
        retrieve = get_retriever()

        started = time.perf_counter()
        ranks = [retrieval_rank([c["source"] for c in retrieve(q.question, k=args.k)], q.expected_files)
                 for q in questions]
        per_query = (time.perf_counter() - started) / len(questions)

        hit = mean([float(r is not None) for r in ranks])
        mrr = mean([reciprocal_rank(r) for r in ranks])
        missed = ", ".join(q.id for q, r in zip(questions, ranks) if r is None) or "—"
        print(f"{label:26} {hit:>7.0%} {mrr:>6.2f} {per_query:>8.2f}  {missed}", flush=True)


if __name__ == "__main__":
    main()
