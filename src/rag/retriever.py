from __future__ import annotations

"""
RAG Retriever — YouTube Comments
Implements all retrieval modes from professor's 5_Retrieval.py:
  - Semantic      : FAISS similarity_search
  - MMR           : vectorstore.as_retriever(search_type="mmr")  [Task 5]
  - Hybrid        : BM25 + semantic combined with RRF            [Task 6]
  - HyDE          : Hypothetical Document Embedding              [professor's HyDE()]
  - MultiHop      : iterative query expansion                    [Task 7]
"""

import os
import pickle
from pathlib import Path
from typing import List, Tuple

import dspy
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ── path helper ───────────────────────────────────────────────────────────

def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


# ── load FAISS + BM25 ────────────────────────────────────────────────────

def load_faiss(faiss_dir: str, embedder) -> FAISS:
    path = str(resolve_path(faiss_dir))
    vectorstore = FAISS.load_local(
        path, embedder, allow_dangerous_deserialization=True
    )
    print(f"FAISS index loaded from: {path}")
    return vectorstore


def load_bm25(bm25_path: str) -> Tuple:
    path = resolve_path(bm25_path)
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"BM25 index loaded from: {path}")
    return data["bm25"], data["docs"]


# ── retrieval methods ─────────────────────────────────────────────────────

def semantic_retrieval(
    vectorstore: FAISS,
    query: str,
    k: int = 20,
) -> List[Document]:
    """
    Semantic retrieval using FAISS similarity search.
    Mirrors professor's 4_Vectorization_FAISS.py:
        docs = vectorstore.similarity_search(query, k=5)
    """
    return vectorstore.similarity_search(query, k=k)


def mmr_retrieval(
    vectorstore: FAISS,
    query: str,
    k: int = 20,
    fetch_k: int = 50,
    lambda_mult: float = 0.3,
) -> List[Document]:
    """
    MMR retrieval — balances relevance with diversity.
    Mirrors professor's 5_Retrieval.py mmrRetrieval():
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 2, "fetch_k": 20, "lambda_mult": 0.3}
        )
    lambda_mult: 0 = max diversity, 1 = max relevance. Professor uses 0.3.
    """
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
        },
    )
    return retriever.invoke(query)


def bm25_retrieval(
    bm25,
    bm25_docs: List[Document],
    query: str,
    k: int = 20,
) -> List[Document]:
    """
    Lexical BM25 retrieval.
    Supports hybrid retrieval (professor's Task 6).
    """
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [bm25_docs[i] for i in top_indices]


def hybrid_retrieval(
    vectorstore: FAISS,
    bm25,
    bm25_docs: List[Document],
    query: str,
    k: int = 20,
    alpha: float = 0.5,
) -> List[Document]:
    """
    Hybrid retrieval: BM25 + semantic combined using Reciprocal Rank Fusion (RRF).
    Implements professor's Task 6 from 5_Retrieval.py.
    alpha: weight for semantic score (1-alpha = BM25 weight).
    """
    semantic_docs = semantic_retrieval(vectorstore, query, k=k)
    bm25_docs_results = bm25_retrieval(bm25, bm25_docs, query, k=k)

    # Reciprocal Rank Fusion
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(semantic_docs):
        key = doc.page_content[:100]
        rrf_scores[key] = rrf_scores.get(key, 0) + alpha * (1 / (rank + 60))
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_docs_results):
        key = doc.page_content[:100]
        rrf_scores[key] = rrf_scores.get(key, 0) + (1 - alpha) * (1 / (rank + 60))
        doc_map[key] = doc

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked[:k]]


# ── DSPy signatures for HyDE and MultiHop (professor's 5_Retrieval.py) ───

class HyDESignature(dspy.Signature):
    """Provide a short answer or brief explanation."""
    question: str = dspy.InputField(desc="question or a concept")
    answer: str = dspy.OutputField(desc="short answer or better explanation")


class KeyphraseSignature(dspy.Signature):
    """Extract the most important keywords from the text relevant to the query."""
    text: str = dspy.InputField(desc="text to extract keywords from")
    query: str = dspy.InputField(desc="original user query for relevance context")
    keywords: str = dspy.OutputField(desc="comma-separated list of key phrases")


class QueryRefinementSignature(dspy.Signature):
    """Refine the search query using the answer and keywords."""
    original_query: str = dspy.InputField(desc="original user question")
    answer: str = dspy.InputField(desc="answer from previous retrieval step")
    keywords: str = dspy.InputField(desc="key phrases extracted from the answer")
    refined_query: str = dspy.OutputField(desc="improved search query")


def hyde_retrieval(
    vectorstore: FAISS,
    query: str,
    lm: dspy.LM,
    k: int = 20,
    n_hypothetical_docs: int = 1,
) -> List[Document]:
    """
    HyDE: Hypothetical Document Embedding.
    Mirrors professor's HyDE() function in 5_Retrieval.py exactly:
        gen_answer = dspy.Predict(getHyDE)
        result = gen_answer(question=question)
        final_q = reRankCrossEncoder(org_question, docs, model, 1)[0][0]
    """
    gen_answer = dspy.Predict(HyDESignature)
    hypothetical_docs = []
    current_question = query

    with dspy.context(lm=lm):
        for _ in range(n_hypothetical_docs):
            result = gen_answer(question=current_question)
            hypothetical_docs.append(result.answer)
            current_question = result.answer

    # Use best hypothetical doc to retrieve
    best_hyp = hypothetical_docs[-1]
    return semantic_retrieval(vectorstore, best_hyp, k=k)


def multihop_retrieval(
    vectorstore: FAISS,
    query: str,
    lm: dspy.LM,
    k: int = 20,
    hops: int = 2,
) -> List[Document]:
    """
    MultiHop query expansion.
    Implements professor's Task 7 from 5_Retrieval.py:
        0. send question to vectorDB → get top N docs
        1. send question to LLM → get answer
        2. extract key phrases from answer relevant to query
        3. regenerate refined query from keyphrases + answer
        4. repeat M times, accumulate docs, dedup, return top k
    """
    gen_answer = dspy.Predict(HyDESignature)
    extract_keys = dspy.Predict(KeyphraseSignature)
    refine_query = dspy.Predict(QueryRefinementSignature)

    all_docs: list[Document] = []
    seen_content: set[str] = set()
    current_query = query

    with dspy.context(lm=lm):
        for hop in range(hops):
            # Step 0: retrieve
            hop_docs = semantic_retrieval(vectorstore, current_query, k=k)
            for doc in hop_docs:
                key = doc.page_content[:100]
                if key not in seen_content:
                    seen_content.add(key)
                    all_docs.append(doc)

            # Step 1: get answer
            answer = gen_answer(question=current_query).answer

            # Step 2: extract keywords
            context_text = " ".join([d.page_content[:200] for d in hop_docs[:3]])
            keywords = extract_keys(text=context_text, query=current_query).keywords

            # Step 3: refine query for next hop
            current_query = refine_query(
                original_query=query,
                answer=answer,
                keywords=keywords,
            ).refined_query

    return all_docs[:k]


# ── unified retrieve interface ────────────────────────────────────────────

def retrieve(
    mode: str,
    query: str,
    vectorstore: FAISS,
    bm25=None,
    bm25_docs: List[Document] = None,
    lm: dspy.LM = None,
    k: int = 20,
    **kwargs,
) -> List[Document]:
    """
    Single entry point for all retrieval modes.
    mode options: 'semantic', 'mmr', 'hybrid', 'hyde', 'multihop'
    """
    if mode == "semantic":
        return semantic_retrieval(vectorstore, query, k=k)

    elif mode == "mmr":
        return mmr_retrieval(
            vectorstore, query, k=k,
            fetch_k=kwargs.get("fetch_k", 50),
            lambda_mult=kwargs.get("lambda_mult", 0.3),
        )

    elif mode == "hybrid":
        if bm25 is None or bm25_docs is None:
            raise ValueError("hybrid mode requires bm25 and bm25_docs.")
        return hybrid_retrieval(
            vectorstore, bm25, bm25_docs, query, k=k,
            alpha=kwargs.get("alpha", 0.5),
        )

    elif mode == "hyde":
        if lm is None:
            raise ValueError("hyde mode requires an lm (dspy.LM).")
        return hyde_retrieval(
            vectorstore, query, lm, k=k,
            n_hypothetical_docs=kwargs.get("n_hypothetical_docs", 1),
        )

    elif mode == "multihop":
        if lm is None:
            raise ValueError("multihop mode requires an lm (dspy.LM).")
        return multihop_retrieval(
            vectorstore, query, lm, k=k,
            hops=kwargs.get("hops", 2),
        )

    else:
        raise ValueError(f"Unknown retrieval mode: '{mode}'. "
                         f"Choose from: semantic, mmr, hybrid, hyde, multihop.")
