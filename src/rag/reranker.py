from __future__ import annotations

"""
RAG Reranker — Cross-encoder reranking
Directly mirrors professor's reRankCrossEncoder() from 5_Retrieval.py:

    from sentence_transformers import CrossEncoder
    def reRankCrossEncoder(question, chunks, model_name, top_k=10):
        reranker = CrossEncoder(model_name)
        pairs = [[question, chunk] for chunk in chunks]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
"""

from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


_reranker_cache: dict[str, CrossEncoder] = {}


def _get_reranker(model_name: str) -> CrossEncoder:
    if model_name not in _reranker_cache:
        print(f"Loading cross-encoder: {model_name}")
        _reranker_cache[model_name] = CrossEncoder(model_name)
    return _reranker_cache[model_name]


def rerank_cross_encoder(
    question: str,
    docs: List[Document],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 5,
) -> List[Tuple[Document, float]]:
    """
    Cross-encoder reranking.
    Directly implements professor's reRankCrossEncoder() from 5_Retrieval.py.

    Args:
        question:   user query
        docs:       list of retrieved Document objects
        model_name: cross-encoder model (professor uses ms-marco-MiniLM-L6-v2)
        top_k:      number of top results to return

    Returns:
        List of (Document, score) tuples sorted by relevance descending.
    """
    if not docs:
        return []

    reranker = _get_reranker(model_name)

    # Build question-chunk pairs exactly as professor does
    chunks = [doc.page_content for doc in docs]
    pairs = [[question, chunk] for chunk in chunks]

    scores = reranker.predict(pairs)

    # Sort by score descending — professor's exact pattern
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def rerank_and_extract(
    question: str,
    docs: List[Document],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 5,
) -> List[Document]:
    """
    Convenience wrapper — returns only the reranked Document objects (no scores).
    """
    ranked = rerank_cross_encoder(question, docs, model_name, top_k)
    return [doc for doc, _ in ranked]
