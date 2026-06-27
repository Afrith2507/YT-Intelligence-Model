"""
src/rag/generator.py

This is the direct continuation of the professor's 7_Generator.py demo,
adapted to run over our own data instead of the W8L7 FAISS store / Weaviate
collection:

  - 7_Generator.py:  FAISS.load_local(...) + retrievers.mmrRetrieval(...)
    -> here:         CommentVectorStore.mmr_search(...) over comments_enriched.csv

  - 7_Generator.py:  retrievers.semanticRetrieval(collection, question)  (Weaviate)
    -> here:         CommentVectorStore.similarity_search(...)  (FAISS only --
       we dropped the Weaviate path since it needs a separate running
       server; FAISS alone covers the same retrieval need for this project)

  - 7_Generator.py:  retrievers.HyDE(question, collection, lm_2, 1)
    -> here:         CommentVectorStore.hyde_search(...), using the
       `getHyDE` DSPy signature from prompts.py

  - 7_Generator.py:  class getAnswer(dspy.Signature): ...
                      gen_answer = dspy.Predict(getAnswer)
                      result = gen_answer(question=question, context=context)
    -> here:         same getAnswer signature (copied verbatim into
       prompts.py), same dspy.Predict call pattern, wrapped in
       generate_answer() below

  - 7_Generator.py Task 9 option A ("ask an LLM to summarize the text then
    use it as context")
    -> here:         answer_with_summarized_context()

Embedding backend defaults to TF-IDF (no network required) so this module
can be developed and graded without a local Ollama/Weaviate instance
running. Set RAG_EMBED_BACKEND=ollama (and have Ollama serving
`embeddinggemma`, exactly like 7_Generator.py) to switch to the same
embeddings the professor used. Same idea for generation: if the configured
Ollama model can't be reached, generate_* functions fall back to a simple
extractive response instead of crashing, so the rest of the agent
(router/tools/orchestrator) keeps working end-to-end.
"""

import os
import json
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import dspy
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

from src.rag.prompts import getAnswer, getSummary, getHyDE, getInsight, getIntent

# --------------------------------------------------------------------------
# Config -- override any of these with environment variables, no code edits
# needed to point this at a real local Ollama instance.
# --------------------------------------------------------------------------
OLLAMA_API_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
GEN_MODEL = os.environ.get("RAG_GEN_MODEL", "ollama/hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M")
EMBED_BACKEND = os.environ.get("RAG_EMBED_BACKEND", "tfidf")  # "tfidf" or "ollama"
OLLAMA_EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "embeddinggemma")
DEFAULT_DATA_PATH = os.environ.get("ENRICHED_CSV_PATH", "data/processed/comments_enriched.csv")

# LM is created lazily so importing this module never fails when Ollama is offline.
_lm = None
_LM_WARNED = False


def _get_lm():
    """Return the DSPy LM, creating it on first use. Returns None if unavailable."""
    global _lm, _LM_WARNED
    if _lm is not None:
        return _lm
    try:
        _lm = dspy.LM(GEN_MODEL, api_base=OLLAMA_API_BASE, api_key=None, temperature=0.9)
        return _lm
    except Exception as exc:
        if not _LM_WARNED:
            print(
                f"[generator] could not initialise local LM ({exc}); "
                f"generation will use extractive fallbacks. "
                f"Start Ollama at {OLLAMA_API_BASE} with model {GEN_MODEL} for real answers."
            )
            _LM_WARNED = True
        return None


def _predict_safe(predictor, output_field, fallback_value, **kwargs):
    """Run a dspy.Predict call under the configured LM. If the local LLM
    can't be reached (e.g. Ollama isn't running), warn once and fall back to
    a simple non-LLM response instead of raising, so callers higher up
    (tools/orchestrator) never crash because of an unavailable model."""
    global _LM_WARNED
    if predictor is None:
        return fallback_value
    lm = _get_lm()
    if lm is None:
        return fallback_value
    try:
        with dspy.context(lm=lm):
            result = predictor(**kwargs)
        return getattr(result, output_field)
    except Exception as exc:
        if not _LM_WARNED:
            print(
                f"[generator] local LM unavailable ({exc}); using a simple "
                f"fallback response instead. Start Ollama at {OLLAMA_API_BASE} "
                f"with model {GEN_MODEL} to get real generations."
            )
            _LM_WARNED = True
        return fallback_value


def _extractive_fallback(text, max_chars=280):
    text = " ".join(str(text).split())
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def _extractive_answer(question: str, results: list) -> str:
    """Build a readable extractive answer from retrieved comments without an LLM.
    Called only when Ollama is offline."""
    if not results:
        return "No relevant comments found for that query."

    # Sentiment tally
    counts: dict = {}
    for r in results:
        s = str(r.get("sentiment") or "neutral").strip().lower()
        counts[s] = counts.get(s, 0) + 1
    total = len(results)

    sent_parts = []
    for label, clr in [("positive", "positive"), ("negative", "negative"), ("neutral", "neutral")]:
        if label in counts:
            pct = counts[label] / total * 100
            sent_parts.append(f"{pct:.0f}% {label}")

    sentiment_line = (
        "Sentiment across retrieved comments: " + ", ".join(sent_parts) + "."
        if sent_parts else ""
    )

    # Pick the top 3 longest (usually most informative) comments
    top = sorted(results, key=lambda r: len(str(r.get("text", ""))), reverse=True)[:3]
    quote_lines = []
    for r in top:
        txt = str(r.get("text", "")).strip()
        if txt:
            quote_lines.append(f'• "{txt}"')

    quotes = "\n".join(quote_lines)

    return (
        f"⚠ Local LLM unavailable — showing an extractive answer instead.\n\n"
        f"{sentiment_line}\n\n"
        f"Most representative comments:\n{quotes}"
    )


# --------------------------------------------------------------------------
# Embedding backends
# --------------------------------------------------------------------------
class TfidfEmbeddings:
    """Offline embedding backend, no network or local server required.
    Default choice so the pipeline is runnable/testable anywhere."""

    def __init__(self, max_features=2048):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self._fitted = False
        self.matrix = None

    def fit(self, texts):
        self.matrix = self.vectorizer.fit_transform(texts).toarray().astype("float32")
        self._fitted = True
        return self.matrix

    def embed(self, texts):
        if not self._fitted:
            raise RuntimeError("TfidfEmbeddings must be fit on the corpus before embedding queries")
        return self.vectorizer.transform(texts).toarray().astype("float32")


class SentenceTransformerEmbeddings:
    """Lightweight semantic embeddings using sentence-transformers.
    Uses all-MiniLM-L6-v2 (~80MB, runs on CPU, no Ollama needed).
    Far better semantic quality than TF-IDF without the RAM cost of large LLMs."""

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        print(f"[embedder] loading {self.MODEL_NAME} (downloads ~80MB on first run) …")
        self._model = SentenceTransformer(self.MODEL_NAME)

    def _encode(self, texts):
        return self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype("float32")

    def fit(self, texts):
        return self._encode(texts)

    def embed(self, texts):
        return self._encode(texts)


class OllamaEmbeddingsBackend:
    """Matches the professor's `OllamaEmbeddings(model="embeddinggemma")`
    usage in 7_Generator.py exactly. Requires a running local Ollama server
    with that embedding model pulled."""

    def __init__(self, model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_API_BASE):
        from langchain_ollama import OllamaEmbeddings
        self._embedder = OllamaEmbeddings(model=model, base_url=base_url)

    def fit(self, texts):
        return np.array(self._embedder.embed_documents(texts), dtype="float32")

    def embed(self, texts):
        return np.array(self._embedder.embed_documents(texts), dtype="float32")


def _make_embedder(kind=EMBED_BACKEND):
    if kind == "ollama":
        return OllamaEmbeddingsBackend()
    if kind == "sentence-transformers":
        return SentenceTransformerEmbeddings()
    return TfidfEmbeddings()


# --------------------------------------------------------------------------
# Vector store over comments_enriched.csv
# --------------------------------------------------------------------------
class CommentVectorStore:
    """FAISS-backed semantic index over the enriched comments dataset.
    The comments themselves are the retrieval corpus for this project, so
    this is built directly from comments_enriched.csv rather than a
    separate document store like in 7_Generator.py's FAISS/Weaviate demo."""

    def __init__(self, df, text_col="text", embedder=None):
        self.df = df.reset_index(drop=True)
        self.text_col = text_col
        self.embedder = embedder or _make_embedder()
        texts = self.df[text_col].astype(str).tolist()
        vectors = self.embedder.fit(texts)
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def _to_results(self, idxs, scores):
        results = []
        for idx, score in zip(idxs, scores):
            idx = int(idx)
            if idx < 0:
                continue
            row = self.df.iloc[idx]
            raw_id = row["comment_id"] if "comment_id" in row else idx
            results.append(
                {
                    "comment_id": int(raw_id) if hasattr(raw_id, "item") or isinstance(raw_id, (int, np.integer)) else raw_id,
                    "text": row[self.text_col],
                    "score": float(score),
                    "sentiment": row.get("sentiment"),
                }
            )
        return results

    def similarity_search(self, query, k=4):
        """Plain top-k semantic search -- equivalent to
        retrievers.semanticRetrieval in 7_Generator.py."""
        qvec = self.embedder.embed([query])
        faiss.normalize_L2(qvec)
        scores, idxs = self.index.search(qvec, min(k, len(self.df)))
        return self._to_results(idxs[0], scores[0])

    def mmr_search(self, query, k=4, fetch_k=12, lambda_mult=0.5):
        """Maximal Marginal Relevance search -- equivalent to
        retrievers.mmrRetrieval in 7_Generator.py. Fetches a larger
        candidate pool, then greedily picks results that are relevant to the
        query AND diverse from each other, so the LLM doesn't just get k
        near-duplicate comments."""
        fetch_k = min(fetch_k, len(self.df))
        qvec = self.embedder.embed([query])
        faiss.normalize_L2(qvec)
        scores, idxs = self.index.search(qvec, fetch_k)
        idxs, scores = idxs[0], scores[0]
        valid = idxs >= 0
        idxs, scores = idxs[valid], scores[valid]

        cand_texts = self.df.iloc[idxs][self.text_col].tolist()
        cand_vecs = self.embedder.embed(cand_texts)
        faiss.normalize_L2(cand_vecs)

        selected = []
        remaining = list(range(len(idxs)))
        while remaining and len(selected) < min(k, len(idxs)):
            if not selected:
                best = max(remaining, key=lambda i: scores[i])
            else:
                def mmr_score(i):
                    diversity = max(float(cand_vecs[i] @ cand_vecs[j]) for j in selected)
                    return lambda_mult * scores[i] - (1 - lambda_mult) * diversity

                best = max(remaining, key=mmr_score)
            selected.append(best)
            remaining.remove(best)

        return self._to_results(idxs[selected], scores[selected])

    def hyde_search(self, query, k=4):
        """HyDE retrieval -- equivalent to the
        `retrievers.HyDE(question, collection, lm_2, 1)` call in
        7_Generator.py: generate a hypothetical answer with the LLM first,
        embed THAT instead of the raw question, then search for real
        comments near it. Useful when the literal query wording doesn't
        overlap much with how people actually phrase comments."""
        hypothetical = generate_hypothetical_answer(query)
        return self.similarity_search(hypothetical, k=k)


_vector_store_cache = {}


def load_comments(path=DEFAULT_DATA_PATH):
    return pd.read_csv(path)


def get_vector_store(path=DEFAULT_DATA_PATH, embedder_kind=EMBED_BACKEND):
    """Build (or reuse a cached) vector store for the given CSV path."""
    key = (path, embedder_kind)
    if key not in _vector_store_cache:
        df = load_comments(path)
        _vector_store_cache[key] = CommentVectorStore(df, embedder=_make_embedder(embedder_kind))
    return _vector_store_cache[key]


# --------------------------------------------------------------------------
# DSPy generation -- same call pattern as 7_Generator.py's gen_answer
# --------------------------------------------------------------------------
try:
    gen_answer = dspy.Predict(getAnswer)
    gen_summary = dspy.Predict(getSummary)
    gen_hyde = dspy.Predict(getHyDE)
    gen_insight = dspy.Predict(getInsight)
    gen_intent = dspy.Predict(getIntent)
except Exception as _e:
    print(f"[generator] could not create DSPy predictors ({_e}); LLM generation disabled.")
    gen_answer = gen_summary = gen_hyde = gen_insight = gen_intent = None


def generate_answer(question, context, results=None):
    """Direct equivalent of 7_Generator.py's:
    result = gen_answer(question=question, context=context); result.answer
    Falls back to an extractive summary when Ollama is offline.
    """
    fallback = _extractive_answer(question, results or [])
    return _predict_safe(gen_answer, "answer", fallback, question=question, context=context)


def generate_summary(text):
    """Task 9 option A: summarize text with an LLM."""
    return _predict_safe(gen_summary, "summary", _extractive_fallback(text), text=text)


def generate_hypothetical_answer(question):
    """Used internally by CommentVectorStore.hyde_search."""
    return _predict_safe(gen_hyde, "hypothetical_answer", question, question=question)


def generate_insight(question, stats):
    """Used by the agent's sentiment/topic/entity insight tools to turn
    aggregate statistics into a natural-language answer."""
    stats_str = stats if isinstance(stats, str) else json.dumps(stats, default=str)
    fallback = f"Based on the data: {stats_str}"
    return _predict_safe(gen_insight, "insight", fallback, question=question, stats=stats_str)


def _format_context(results):
    return "\n".join(f"[{r['comment_id']}] {r['text']}" for r in results)


def answer_with_context(question, k=4, retrieval="mmr", path=DEFAULT_DATA_PATH):
    """End-to-end QA: retrieve relevant comments, concatenate as context
    (same as `context="".join(docs[1])` in 7_Generator.py, just joined with
    citation tags instead of raw concatenation), then generate an answer."""
    store = get_vector_store(path)
    search_fn = {"mmr": store.mmr_search, "similarity": store.similarity_search, "hyde": store.hyde_search}[retrieval]
    results = search_fn(question, k=k)
    context = _format_context(results)
    answer = generate_answer(question, context, results=results)
    return {"answer": answer, "citations": [r["comment_id"] for r in results], "retrieved": results}


def answer_with_summarized_context(question, k=4, path=DEFAULT_DATA_PATH):
    """Task 9 option A end-to-end: retrieve, summarize the retrieved
    comments first, then answer using the summary as context instead of the
    raw concatenated text."""
    store = get_vector_store(path)
    results = store.mmr_search(question, k=k)
    raw_context = _format_context(results)
    summary = generate_summary(raw_context)
    answer = generate_answer(question, summary, results=results)
    return {"answer": answer, "summary": summary, "citations": [r["comment_id"] for r in results], "retrieved": results}


def summarize_query(query, k=6, path=DEFAULT_DATA_PATH):
    """Used by the agent's 'summarize' intent: retrieve comments relevant to
    the query/topic and summarize them."""
    store = get_vector_store(path)
    results = store.mmr_search(query, k=k)
    raw_context = _format_context(results)
    summary = generate_summary(raw_context)
    return {"summary": summary, "citations": [r["comment_id"] for r in results], "retrieved": results}


if __name__ == "__main__":
    # Same shape as the bottom half of 7_Generator.py: pick a question, run
    # retrieval, generate and print an answer.
    question = "What do viewers say about Ryan Trahan's storytelling style?"
    out = answer_with_context(question, k=4, retrieval="mmr")
    print(f"Q: {question}")
    print(f"A: {out['answer']}")
    print(f"citations: {out['citations']}")
