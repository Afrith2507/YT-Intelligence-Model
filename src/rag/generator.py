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
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _bootstrap_env() -> None:
    """Load .env from project root so Groq works even without start.ps1."""
    try:
        from dotenv import load_dotenv
        load_dotenv(_PROJECT_ROOT / ".env", override=False)
    except ImportError:
        pass
    # Defaults when launched via streamlit run directly
    os.environ.setdefault("RAG_GEN_MODEL", "groq/llama-3.1-8b-instant")
    os.environ.setdefault("RAG_EMBED_BACKEND", "sentence-transformers")


_bootstrap_env()

import numpy as np
import pandas as pd
import dspy
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

from src.rag.prompts import getAnswer, getSummary, getHyDE, getInsight, getIntent

# --------------------------------------------------------------------------
# Config -- override with environment variables / .env
# --------------------------------------------------------------------------
OLLAMA_API_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "embeddinggemma")
DEFAULT_DATA_PATH = os.environ.get(
    "ENRICHED_CSV_PATH",
    str(_PROJECT_ROOT / "data" / "processed" / "comments_enriched.csv"),
)


def _gen_model() -> str:
    explicit = os.environ.get("RAG_GEN_MODEL", "").strip()
    if explicit:
        return explicit
    if os.environ.get("GROQ_API_KEY", "").strip():
        return "groq/llama-3.1-8b-instant"
    return "ollama/hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M"


def _embed_backend() -> str:
    return os.environ.get("RAG_EMBED_BACKEND", "sentence-transformers")


# LM is created lazily so importing this module never fails when offline.
_lm = None
_LM_WARNED = False
_LAST_LM_ERROR: str | None = None


def llm_status() -> dict:
    """Status dict for dashboard sidebar."""
    _bootstrap_env()
    key = os.environ.get("GROQ_API_KEY", "").strip()
    model = _gen_model()
    if model.startswith("groq/") and not key:
        return {
            "ok": False,
            "model": model,
            "message": "Add GROQ_API_KEY to .env in the project folder, then restart.",
        }
    lm = _get_lm()
    if lm is None:
        return {
            "ok": False,
            "model": model,
            "message": _LAST_LM_ERROR or "Could not connect to the language model.",
        }
    return {"ok": True, "model": model, "message": "Groq connected"}


def _get_lm():
    """Return the DSPy LM, creating it on first use. Returns None if unavailable."""
    global _lm, _LM_WARNED, _LAST_LM_ERROR
    if _lm is not None:
        return _lm

    _bootstrap_env()
    model = _gen_model()

    try:
        is_groq = model.startswith("groq/")
        is_ollama = model.startswith("ollama")

        api_key = None
        api_base = None

        if is_groq:
            api_key = os.environ.get("GROQ_API_KEY", "").strip()
            if not api_key:
                _LAST_LM_ERROR = "GROQ_API_KEY is missing from .env"
                if not _LM_WARNED:
                    print(f"[generator] {_LAST_LM_ERROR}")
                    _LM_WARNED = True
                return None
            os.environ["GROQ_API_KEY"] = api_key
        elif is_ollama:
            api_base = OLLAMA_API_BASE

        kwargs: dict = dict(temperature=0.3)
        if api_base:
            kwargs["api_base"] = api_base
        if api_key:
            kwargs["api_key"] = api_key

        _lm = dspy.LM(model, **kwargs)
        print(f"[generator] LM initialised: {model}")
        _LAST_LM_ERROR = None
        return _lm
    except Exception as exc:
        _LAST_LM_ERROR = str(exc)
        if not _LM_WARNED:
            print(f"[generator] could not initialise LM ({exc})")
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
        _LAST_LM_ERROR = str(exc)
        if not _LM_WARNED:
            print(f"[generator] generation failed ({exc})")
            _LM_WARNED = True
        return fallback_value


def _llm_fallback_prefix() -> str:
    st = llm_status()
    if not st["ok"]:
        return f"⚠ {st['message']}\n\nShowing retrieved comments instead:\n\n"
    return "⚠ Generation failed. Showing retrieved comments instead:\n\n"


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
        f"{_llm_fallback_prefix()}"
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


def _make_embedder(kind=None):
    if kind is None:
        kind = _embed_backend()
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
        self._build_bm25(texts)

    def _build_bm25(self, texts: list[str]) -> None:
        from rank_bm25 import BM25Okapi
        self._corpus_tokens = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def _ensure_bm25(self) -> None:
        if getattr(self, "_bm25", None) is None:
            texts = self.df[self.text_col].astype(str).tolist()
            self._build_bm25(texts)

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

    def bm25_search(self, query, k=4):
        """Lexical BM25 over comment text (professor Task 6 lexical leg)."""
        self._ensure_bm25()
        tokenized = query.lower().split()
        scores = self._bm25.get_scores(tokenized)
        top_idx = np.argsort(scores)[::-1][: min(k, len(scores))]
        return self._to_results(top_idx, scores[top_idx])

    def hybrid_search(self, query, k=4, fetch_k=20, alpha=0.5):
        """BM25 + FAISS semantic fused with Reciprocal Rank Fusion (Task 6)."""
        fetch_k = min(fetch_k, len(self.df))
        semantic = self.similarity_search(query, k=fetch_k)
        lexical  = self.bm25_search(query, k=fetch_k)

        rrf_scores: dict[str, float] = {}
        by_id: dict[str, dict] = {}

        for rank, row in enumerate(semantic):
            cid = str(row["comment_id"])
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + alpha * (1.0 / (rank + 60))
            by_id[cid] = row

        for rank, row in enumerate(lexical):
            cid = str(row["comment_id"])
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 - alpha) * (1.0 / (rank + 60))
            by_id.setdefault(cid, row)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [by_id[cid] for cid, _ in ranked]


RETRIEVAL_MODES = {
    "hybrid":     "hybrid_search",
    "mmr":        "mmr_search",
    "similarity": "similarity_search",
    "bm25":       "bm25_search",
    "hyde":       "hyde_search",
}


def _search(store: CommentVectorStore, mode: str, query: str, k: int) -> list:
    fn_name = RETRIEVAL_MODES.get(mode, "mmr_search")
    fn = getattr(store, fn_name)
    return fn(query, k=k)


_vector_store_cache = {}


def load_comments(path=DEFAULT_DATA_PATH):
    return pd.read_csv(path)


def _index_cache_path(csv_path: str, embedder_kind: str) -> str:
    """Return a deterministic path for the persisted FAISS + metadata cache."""
    import hashlib
    key = f"{csv_path}:{embedder_kind}"
    h   = hashlib.md5(key.encode()).hexdigest()[:10]
    cache_dir = os.path.join(os.path.dirname(csv_path), ".vector_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"store_{h}.pkl")


def get_vector_store(path=DEFAULT_DATA_PATH, embedder_kind=None):
    if embedder_kind is None:
        embedder_kind = _embed_backend()
    """Build (or reuse a cached) vector store for the given CSV path.

    On first call the embeddings are computed and the FAISS index is saved to
    data/processed/.vector_cache/ so subsequent runs (including dashboard
    restarts) skip re-encoding the entire dataset and load in ~1 second.
    """
    import pickle

    key = (path, embedder_kind)
    if key in _vector_store_cache:
        return _vector_store_cache[key]

    cache_file = _index_cache_path(path, embedder_kind)

    if os.path.isfile(cache_file):
        print(f"[generator] loading cached vector store from {cache_file} …")
        try:
            with open(cache_file, "rb") as f:
                store = pickle.load(f)
            _vector_store_cache[key] = store
            return store
        except Exception as e:
            print(f"[generator] cache load failed ({e}), rebuilding …")

    print(f"[generator] building vector store (first run, this takes a minute) …")
    df    = load_comments(path)
    store = CommentVectorStore(df, embedder=_make_embedder(embedder_kind))

    try:
        with open(cache_file, "wb") as f:
            pickle.dump(store, f)
        print(f"[generator] vector store cached to {cache_file}")
    except Exception as e:
        print(f"[generator] could not save cache ({e}), continuing without cache.")

    _vector_store_cache[key] = store
    return store


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


def _rerank_results(question: str, results: list, top_k: int) -> list:
    """Rerank retrieved comment dicts using a cross-encoder. Returns top_k."""
    try:
        from sentence_transformers import CrossEncoder
        _RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        if not hasattr(_rerank_results, "_model"):
            print(f"[reranker] loading {_RERANK_MODEL} …")
            _rerank_results._model = CrossEncoder(_RERANK_MODEL)
        pairs  = [[question, r["text"]] for r in results]
        scores = _rerank_results._model.predict(pairs)
        ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        return [r for r, _ in ranked[:top_k]]
    except Exception as e:
        print(f"[reranker] skipped ({e})")
        return results[:top_k]


def answer_with_context(question, k=4, retrieval="hybrid", path=DEFAULT_DATA_PATH):
    """End-to-end QA: retrieve -> rerank -> generate answer."""
    store = get_vector_store(path)
    fetch_k = min(k * 3, 20)
    results = _search(store, retrieval, question, k=fetch_k)
    results = _rerank_results(question, results, top_k=k)
    context = _format_context(results)
    answer  = generate_answer(question, context, results=results)
    return {"answer": answer, "citations": [r["comment_id"] for r in results], "retrieved": results}


def answer_with_summarized_context(question, k=4, path=DEFAULT_DATA_PATH):
    """Task 9 option A end-to-end: retrieve, summarize the retrieved
    comments first, then answer using the summary as context instead of the
    raw concatenated text."""
    store = get_vector_store(path)
    results = _search(store, "mmr", question, k=k)
    raw_context = _format_context(results)
    summary = generate_summary(raw_context)
    answer = generate_answer(question, summary, results=results)
    return {"answer": answer, "summary": summary, "citations": [r["comment_id"] for r in results], "retrieved": results}


def summarize_query(query, k=6, path=DEFAULT_DATA_PATH):
    """Used by the agent's 'summarize' intent: retrieve comments relevant to
    the query/topic and summarize them."""
    store = get_vector_store(path)
    results = _search(store, "hybrid", query, k=k)
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
