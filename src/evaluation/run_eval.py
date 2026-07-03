"""
run_eval.py — End-to-End Evaluation Runner
Chains retrieval_eval + generation_eval and logs everything to MLflow.

Usage:
    # Start MLflow server first
    mlflow server --host 0.0.0.0 --port 5000

    # Then run
    python src/evaluation/run_eval.py

By default uses Member 3's real FAISS vector store (CommentVectorStore) built
from data/processed/comments_enriched.csv. Pass use_mock=True to fall back to
the mock functions for smoke-testing without any data file.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.evaluation.retrieval_eval  import evaluate_retriever
from src.evaluation.generation_eval import evaluate_generation
from src.evaluation.mlflow_tracker  import setup_mlflow, log_retrieval_metrics, log_generation_metrics
from src.evaluation.ground_truth    import build_ground_truth, summarize_ground_truth
from src.rag.generator              import DEFAULT_DATA_PATH, get_vector_store, answer_with_context, _search


# ── Evaluation dataset ─────────────────────────────────────────────────────────
# These are the ground-truth QA pairs used for both retrieval and generation
# evaluation. The "relevant document IDs" are comment_ids from
# comments_enriched.csv that genuinely answer each question.
#
# IMPORTANT: update GROUND_TRUTH_IDS when Member 2 delivers the real dataset —
# the mock IDs below match the 10-row sample CSV only.

EVAL_QUERIES = [
    "What do viewers say about Ryan's storytelling style?",
    "What is the overall sentiment in the comments?",
    "What are the most common topics in negative comments?",
    "Which people and brands are most mentioned by viewers?",
    "Summarize what viewers think about Ryan Trahan's recent videos.",
]

# Built at runtime from real YouTube comment IDs in comments_enriched.csv
# (see src/evaluation/ground_truth.py). Placeholder ints {1,2,3…} are NOT used.
GROUND_TRUTH_IDS: list[set[str]] = []

# Ground-truth answers — reference summaries for generation metrics (ROUGE / BERTScore)
REFERENCE_ANSWERS = [
    "Viewers praise Ryan's storytelling as engaging, funny, and well-edited with a strong narrative arc.",
    "The overall sentiment is largely positive, with viewers expressing excitement and admiration.",
    "Negative comments frequently mention disappointment with sponsored content and video length.",
    "Ryan and Haley are the most mentioned people; YouTube and various brands appear frequently.",
    "Viewers are highly enthusiastic about Ryan's recent videos, praising his creativity and dedication.",
]


# ── Mock fallbacks (used when use_mock=True or no CSV present) ────────────────

def _mock_retrieve(query: str, k: int) -> list[int]:
    lookup = {
        "crash":    [3, 8],
        "photo":    [3, 8],
        "dark":     [4, 9],
        "mode":     [4, 9],
        "support":  [2, 7],
        "checkout": [1],
        "pricing":  [6, 10],
        "price":    [6, 10],
    }
    for kw, ids in lookup.items():
        if kw in query.lower():
            return ids[:k]
    return [1][:k]


def _mock_generate(query: str, context: str) -> str:
    if "crash" in query.lower() or "photo" in query.lower():
        return "The app crashes when uploading more than five photos at once."
    if "dark" in query.lower():
        return "Users love dark mode but want a scheduled auto-switch."
    if "support" in query.lower():
        return "Support takes three days to respond; calling directly is faster."
    if "checkout" in query.lower():
        return "The new checkout flow is much faster and users love it."
    return "Users find the pricing page unclear and loyalty point carry-over unconfirmed."


def _mock_context(query: str) -> str:
    contexts = {
        "crash":    "App crashed twice while uploading photos. Crashes happen mostly when uploading more than five photos at once.",
        "dark":     "The dark mode update looks great and is easy on the eyes. Love dark mode but wish there was a scheduled auto switch.",
        "support":  "Customer support took three days to respond to my refund request. Refund finally came through after I called support directly.",
        "checkout": "The new checkout flow on the app is so much faster than before, I love it.",
        "pricing":  "Not sure how the new pricing tiers work. Not sure if loyalty points carry over after the pricing change.",
    }
    for kw, ctx in contexts.items():
        if kw in query.lower():
            return ctx
    return "General comment context."


# ── Real FAISS retrieve + generate wrappers ───────────────────────────────────

def _faiss_retrieve_fn(path: str, mode: str = "hybrid"):
    """
    Returns a (query, k) -> list[str] function backed by CommentVectorStore.
    """
    store = get_vector_store(path)

    def _retrieve(query: str, k: int) -> list[str]:
        results = _search(store, mode, query, k=k)
        return [str(r["comment_id"]) for r in results]

    return _retrieve


def _faiss_generate_fn(path: str):
    """
    Returns a (query, context_str) → str function backed by Member 3's
    answer_with_context (MMR retrieval + DSPy getAnswer).
    The context_str argument is ignored here because answer_with_context
    does its own retrieval; we accept it to match the signature expected by
    run_full_evaluation.
    """
    def _generate(query: str, _context: str) -> str:
        result = answer_with_context(query, k=4, retrieval="hybrid", path=path)
        return result["answer"]

    return _generate


def _faiss_context_fn(path: str):
    """
    Returns a query → str function that pulls the top-4 MMR comments and
    joins them as a context string (used for faithfulness scoring).
    """
    store = get_vector_store(path)

    def _context(query: str) -> str:
        results = _search(store, "hybrid", query, k=4)
        return "\n".join(f"[{r['comment_id']}] {r['text']}" for r in results)

    return _context


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_full_evaluation(
    path: str = DEFAULT_DATA_PATH,
    use_mock: bool = False,
    vectorstore_name: str | None = None,
    fast: bool = True,
) -> dict:
    """
    Run retrieval + generation evaluation and log to MLflow.

    Parameters
    ----------
    path : str
        Path to comments_enriched.csv. Passed through to Member 3's
        get_vector_store so the same data file is used everywhere.
    use_mock : bool
        If True, use the simple mock retrieve/generate/context functions
        instead of the real FAISS pipeline. Useful for smoke-testing
        without a data file.
    vectorstore_name : str | None
        Label used in MLflow run names (defaults to "faiss" or "mock").
    """
    if use_mock:
        retrieve_fn = _mock_retrieve
        generate_fn = _mock_generate
        context_fn  = _mock_context
        label       = vectorstore_name or "mock"
        ground_truth = [
            {"1", "2"}, {"3", "4"}, {"5", "6"}, {"7", "8"}, {"9", "10"},
        ]
    else:
        retrieve_fn = _faiss_retrieve_fn(path)
        generate_fn = _faiss_generate_fn(path)
        context_fn  = _faiss_context_fn(path)
        label       = vectorstore_name or "faiss"
        ground_truth = build_ground_truth(path)
        summarize_ground_truth(EVAL_QUERIES, ground_truth)

    # ── 1. Setup MLflow ───────────────────────────────────────────────────────
    setup_mlflow("dspy")

    # ── 2. Retrieval evaluation ───────────────────────────────────────────────
    print("\n═══ RETRIEVAL EVALUATION ═══════════════════════════════")
    retrieval_metrics = evaluate_retriever(
        queries      = EVAL_QUERIES,
        ground_truth = ground_truth,
        retrieve_fn  = retrieve_fn,
        k_values     = [1, 3, 5],
    )
    log_retrieval_metrics(
        run_name = f"{label}_retrieval",
        metrics  = retrieval_metrics,
        params   = {
            "vectorstore":  label,
            "k_values":     "1,3,5",
            "data_path":    path,
            "retrieval":    "hybrid",
        },
    )

    # ── 3. Generate answers ───────────────────────────────────────────────────
    print("\n═══ GENERATING ANSWERS ══════════════════════════════════")
    contexts   = [context_fn(q)       for q in EVAL_QUERIES]
    hypotheses = [generate_fn(q, ctx) for q, ctx in zip(EVAL_QUERIES, contexts)]

    for q, h in zip(EVAL_QUERIES, hypotheses):
        print(f"  Q: {q}")
        print(f"  A: {h}\n")

    # ── 4. Generation evaluation ──────────────────────────────────────────────
    print("\n═══ GENERATION EVALUATION ═══════════════════════════════")
    gen_result = evaluate_generation(
        hypotheses = hypotheses,
        references = REFERENCE_ANSWERS,
        contexts   = contexts,
        skip_bert  = fast,
    )
    log_generation_metrics(
        run_name = f"{label}_generation",
        metrics  = gen_result.to_dict(),
    )

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print("\n═══ SUMMARY ═════════════════════════════════════════════")
    all_metrics = {**retrieval_metrics, **gen_result.to_dict()}
    for k, v in sorted(all_metrics.items()):
        print(f"  {k:<22} {v:.4f}")

    return {
        "retrieval": retrieval_metrics,
        "generation": gen_result.to_dict(),
        "all_metrics": all_metrics,
        "queries": EVAL_QUERIES,
        "hypotheses": hypotheses,
        "label": label,
        "ground_truth_sizes": [len(g) for g in ground_truth],
        "fast_mode": fast,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the full RAG evaluation pipeline.")
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock retrieve/generate functions instead of the real FAISS pipeline.",
    )
    parser.add_argument(
        "--path", default=DEFAULT_DATA_PATH,
        help="Path to comments_enriched.csv (default: %(default)s).",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Include BERTScore (slower, ~3-5 extra minutes on CPU).",
    )
    args = parser.parse_args()

    run_full_evaluation(path=args.path, use_mock=args.mock, fast=not args.full)
