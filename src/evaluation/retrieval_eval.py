"""
Retrieval Evaluation
Metrics: Hit@k, MRR (Mean Reciprocal Rank), NDCG@k

Works with both FAISS (via LangChain) and Weaviate vector stores.
Outputs a results dict that can be passed directly to mlflow_tracker.log_retrieval_metrics().
"""

from __future__ import annotations

import math
from typing import Callable


# ── Core metric functions ──────────────────────────────────────────────────────

def hit_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """
    Hit@k — 1.0 if any of the top-k retrieved items is relevant, else 0.0.

    Parameters
    ----------
    retrieved_ids : list[str]
        Ordered list of retrieved document IDs (most relevant first).
    relevant_ids : set[str]
        Ground-truth set of relevant document IDs for this query.
    k : int
        Cut-off rank.
    """
    return float(any(doc_id in relevant_ids for doc_id in retrieved_ids[:k]))


def reciprocal_rank(retrieved_ids: list, relevant_ids: set) -> float:
    """
    Reciprocal Rank for a single query.
    Returns 1/rank of the first relevant result, or 0 if none found.
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Discounted Cumulative Gain at k (binary relevance)."""
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)
    return dcg


def ndcg_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """
    Normalised DCG@k.
    IDCG is computed assuming all relevant docs occupy the top positions.
    """
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(r + 2) for r in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg_at_k(retrieved_ids, relevant_ids, k) / idcg


# ── Dataset-level evaluation ───────────────────────────────────────────────────

def evaluate_retriever(
    queries: list[str],
    ground_truth: list[set],
    retrieve_fn: Callable[[str, int], list],
    k_values: list[int] = (1, 3, 5),
) -> dict[str, float]:
    """
    Evaluate a retrieval function over a dataset.

    Parameters
    ----------
    queries : list[str]
        List of query strings.
    ground_truth : list[set[str]]
        Parallel list; ground_truth[i] is the set of relevant doc IDs for queries[i].
    retrieve_fn : Callable[[str, int], list[str]]
        Function (query, k) → ordered list of retrieved document IDs.
        Wrap your FAISS or Weaviate retriever to match this signature.
    k_values : list[int]
        Cut-off ranks to evaluate (default: 1, 3, 5).

    Returns
    -------
    dict[str, float]
        Flat dict suitable for MLflow: {"hit_at_1": …, "mrr": …, "ndcg_at_5": …, …}
    """
    max_k = max(k_values)
    hit_sums   = {k: 0.0 for k in k_values}
    ndcg_sums  = {k: 0.0 for k in k_values}
    mrr_sum    = 0.0
    n          = len(queries)

    for query, relevant in zip(queries, ground_truth):
        retrieved = [str(d) for d in retrieve_fn(query, max_k)]
        rel       = {str(d) for d in relevant}

        mrr_sum += reciprocal_rank(retrieved, rel)

        for k in k_values:
            hit_sums[k]  += hit_at_k(retrieved, rel, k)
            ndcg_sums[k] += ndcg_at_k(retrieved, rel, k)

    results: dict[str, float] = {"mrr": mrr_sum / n}
    for k in k_values:
        results[f"hit_at_{k}"]  = hit_sums[k]  / n
        results[f"ndcg_at_{k}"] = ndcg_sums[k] / n

    _print_results(results)
    return results


def _print_results(results: dict[str, float]) -> None:
    print("\n── Retrieval Evaluation Results ──────────────────────")
    for metric, value in sorted(results.items()):
        print(f"  {metric:<20} {value:.4f}")
    print("──────────────────────────────────────────────────────\n")


# ── Retriever wrappers ─────────────────────────────────────────────────────────

def faiss_retrieve_fn(vectorstore, search_type: str = "mmr"):
    """
    Returns a retrieve_fn compatible with evaluate_retriever() for a FAISS vectorstore.

    Usage:
        fn = faiss_retrieve_fn(vectorstore, search_type="mmr")
        results = evaluate_retriever(queries, ground_truth, fn)
    """
    def _retrieve(query: str, k: int) -> list[str]:
        if search_type == "mmr":
            docs = vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=k * 4)
        else:
            docs = vectorstore.similarity_search(query, k=k)
        # Use page_content as the ID (swap for metadata["id"] if you have one)
        return [doc.page_content[:80] for doc in docs]

    return _retrieve


def weaviate_retrieve_fn(collection, embeddings, threshold: float = 0.5):
    """
    Returns a retrieve_fn compatible with evaluate_retriever() for a Weaviate collection.
    """
    def _retrieve(query: str, k: int) -> list[str]:
        from weaviate.classes import query as wq
        vector = embeddings.embed_query(query)
        response = collection.query.near_vector(
            near_vector=vector,
            limit=k,
            certainty=threshold,
            return_metadata=wq.MetadataQuery(certainty=True),
        )
        return [obj.properties.get("text", "")[:80] for obj in response.objects]

    return _retrieve


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Demo with dummy data — replace with real vectorstore and ground truth
    dummy_corpus = {
        "doc_a": "Large Language Models are neural networks trained on text.",
        "doc_b": "RAG combines retrieval with generation.",
        "doc_c": "FAISS is a library for efficient similarity search.",
        "doc_d": "Weaviate is a vector database with semantic search.",
        "doc_e": "BERTopic uses BERT embeddings for topic modelling.",
    }

    dummy_queries      = ["What is a language model?", "How does RAG work?"]
    dummy_ground_truth = [{"doc_a"}, {"doc_b"}]

    # Fake retriever that always returns all doc keys in order
    def dummy_retrieve(query: str, k: int) -> list[str]:
        return list(dummy_corpus.keys())[:k]

    results = evaluate_retriever(
        queries       = dummy_queries,
        ground_truth  = dummy_ground_truth,
        retrieve_fn   = dummy_retrieve,
        k_values      = [1, 3, 5],
    )
    print("Demo results:", results)
