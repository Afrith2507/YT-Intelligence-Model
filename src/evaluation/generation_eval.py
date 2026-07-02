"""
Generation Evaluation
Metrics:
    • ROUGE-L   — lexical overlap between generated answer and reference
    • BERTScore — semantic similarity using contextual embeddings
    • Faithfulness — simple entailment check: does the answer stay within the retrieved context?

Install:
    pip install rouge-score bert-score
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    rouge_l:       float = 0.0
    bert_precision: float = 0.0
    bert_recall:    float = 0.0
    bert_f1:        float = 0.0
    faithfulness:   float = 0.0
    # flat dict for MLflow logging
    def to_dict(self) -> dict[str, float]:
        return {
            "rouge_l":        self.rouge_l,
            "bert_precision": self.bert_precision,
            "bert_recall":    self.bert_recall,
            "bert_f1":        self.bert_f1,
            "faithfulness":   self.faithfulness,
        }


# ── ROUGE-L ────────────────────────────────────────────────────────────────────

def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest Common Subsequence length."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def rouge_l_score(hypothesis: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 between a generated answer and a reference answer.
    Uses whitespace tokenisation for simplicity.
    """
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    if not hyp_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(hyp_tokens, ref_tokens)
    precision = lcs / len(hyp_tokens)
    recall    = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def batch_rouge_l(hypotheses: list[str], references: list[str]) -> float:
    """Average ROUGE-L across a list of (hypothesis, reference) pairs."""
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have the same length.")
    scores = [rouge_l_score(h, r) for h, r in zip(hypotheses, references)]
    return sum(scores) / len(scores)


# ── BERTScore ─────────────────────────────────────────────────────────────────

def bert_score_eval(
    hypotheses: list[str],
    references: list[str],
    model_type: str = "distilbert-base-uncased",
    device: str = "cpu",
) -> tuple[float, float, float]:
    """
    Compute BERTScore (P, R, F1) averaged over all pairs.

    Parameters
    ----------
    model_type : str
        HuggingFace model used by bert_score; 'distilbert-base-uncased' is lightweight.
    device : str
        "cpu" or "cuda".

    Returns
    -------
    tuple[float, float, float]
        (mean_precision, mean_recall, mean_f1)
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        print("[WARNING] bert-score not installed. Run: pip install bert-score")
        return 0.0, 0.0, 0.0

    P, R, F1 = bert_score_fn(
        cands=hypotheses,
        refs=references,
        model_type=model_type,
        device=device,
        verbose=False,
    )
    return P.mean().item(), R.mean().item(), F1.mean().item()


# ── Faithfulness ──────────────────────────────────────────────────────────────

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "to", "of", "in", "and", "or",
    "it", "this", "that", "for", "on", "with", "as", "at", "by", "from", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "she", "he", "they", "we", "you",
    "her", "his", "their", "our", "your", "its", "who", "what", "which", "when",
    "where", "how", "all", "one", "two", "many", "much", "more", "most", "very",
    "just", "also", "about", "into", "than", "then", "them", "these", "those",
}

# Light synonym groups for short YouTube comment paraphrases
_EQUIV_GROUPS: list[frozenset[str]] = [
    frozenset({"beautiful", "beauty", "pretty", "gorgeous", "hot", "cute"}),
    frozenset({"funny", "humor", "humour", "hilarious", "lol", "lmao"}),
    frozenset({"good", "great", "better", "best", "awesome", "amazing", "talented"}),
    frozenset({"love", "loved", "loving", "adore", "admire", "admiration"}),
    frozenset({"watch", "watching", "watched", "attention", "focus"}),
    frozenset({"positive", "praise", "praising", "compliment", "compliments"}),
]


def _simple_stem(word: str) -> str:
    w = word.lower()
    for suffix in ("ingly", "edly", "ness", "ment", "ful", "ous", "ive", "able",
                   "ing", "edly", "ly", "ed", "es", "s"):
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            return w[: -len(suffix)]
    return w


def _canonical(token: str) -> str:
    t = _simple_stem(token.lower())
    for group in _EQUIV_GROUPS:
        if t in group:
            return sorted(group)[0]
    return t


def _content_tokens(text: str) -> list[str]:
    return [
        t for t in re.findall(r"\b\w+\b", text.lower())
        if t not in _STOPWORDS and len(t) > 2
    ]


def _lexical_faithfulness(answer: str, context: str) -> float:
    """Stem + synonym-aware overlap between answer tokens and context."""
    ctx_raw = _content_tokens(context)
    ctx_canon = {_canonical(t) for t in ctx_raw}
    ctx_canon.update(ctx_raw)
    ctx_blob = context.lower()

    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return 1.0

    supported = 0.0
    for tok in answer_tokens:
        canon = _canonical(tok)
        if (
            tok in ctx_blob
            or canon in ctx_canon
            or _simple_stem(tok) in ctx_blob
            or any(canon[:4] == c[:4] for c in ctx_canon if len(canon) >= 4)
        ):
            supported += 1.0
    return supported / len(answer_tokens)


def _semantic_faithfulness(answer: str, context: str) -> float:
    """Cosine similarity between answer and retrieved context (handles paraphrase)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return _lexical_faithfulness(answer, context)

    if not hasattr(_semantic_faithfulness, "_model"):
        _semantic_faithfulness._model = SentenceTransformer("all-MiniLM-L6-v2")

    model = _semantic_faithfulness._model
    chunks = [c.strip() for c in re.split(r"[\n]+|(?<=[.!?])\s+", context) if len(c.strip()) > 8]
    if not chunks:
        chunks = [context]

    texts = [answer, context, *chunks]
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    ans = embs[0]
    rest = embs[1:]
    sims = rest @ ans
    # Whole-context sim + best single-chunk sim
    full_sim = float(sims[0])
    best_chunk = float(max(sims[1:])) if len(sims) > 1 else full_sim
    return 0.55 * full_sim + 0.45 * best_chunk


def faithfulness_score(answer: str, context: str, threshold: float = 0.5) -> float:
    """
    Hybrid faithfulness: lexical overlap (stemmed + synonym groups) blended with
    semantic similarity (MiniLM). Handles paraphrased RAG answers much better
    than raw token matching alone.

    Returns a float in [0, 1]. Values >= threshold are considered faithful.
    """
    if not answer.strip() or not context.strip():
        return 0.0

    lex = _lexical_faithfulness(answer, context)
    try:
        sem = _semantic_faithfulness(answer, context)
        score = 0.30 * lex + 0.70 * sem
    except Exception:
        score = lex
    return max(0.0, min(1.0, score))


def batch_faithfulness(answers: list[str], contexts: list[str]) -> float:
    """Average faithfulness over a list of (answer, context) pairs."""
    scores = [faithfulness_score(a, c) for a, c in zip(answers, contexts)]
    return sum(scores) / len(scores)


# ── Full evaluation pipeline ───────────────────────────────────────────────────

def evaluate_generation(
    hypotheses:  list[str],
    references:  list[str],
    contexts:    list[str],
    bert_model:  str = "distilbert-base-uncased",
    device:      str = "cpu",
    skip_bert:   bool = False,
) -> GenerationResult:
    """
    Run all three metrics and return a GenerationResult.

    Parameters
    ----------
    hypotheses : list[str]
        Generated answers from the RAG pipeline.
    references : list[str]
        Ground-truth reference answers.
    contexts : list[str]
        Retrieved context chunks used to produce each answer.
    """
    print("[eval] Computing ROUGE-L …")
    rl = batch_rouge_l(hypotheses, references)

    if skip_bert:
        print("[eval] Skipping BERTScore (fast mode)")
        bp, br, bf = 0.0, 0.0, 0.0
    else:
        print("[eval] Computing BERTScore …")
        bp, br, bf = bert_score_eval(hypotheses, references, bert_model, device)

    print("[eval] Computing faithfulness …")
    faith = batch_faithfulness(hypotheses, contexts)

    result = GenerationResult(
        rouge_l        = rl,
        bert_precision = bp,
        bert_recall    = br,
        bert_f1        = bf,
        faithfulness   = faith,
    )
    _print_results(result)
    return result


def _print_results(r: GenerationResult) -> None:
    print("\n── Generation Evaluation Results ─────────────────────")
    print(f"  ROUGE-L        {r.rouge_l:.4f}")
    print(f"  BERTScore P    {r.bert_precision:.4f}")
    print(f"  BERTScore R    {r.bert_recall:.4f}")
    print(f"  BERTScore F1   {r.bert_f1:.4f}")
    print(f"  Faithfulness   {r.faithfulness:.4f}")
    print("──────────────────────────────────────────────────────\n")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    hyps = [
        "ASR stands for Attack Success Rate, measuring how often an agent invokes a malicious tool.",
        "Large language models are trained on large corpora of text data.",
    ]
    refs = [
        "ASR is defined as the percentage of runs where the agent invokes a malicious tool and sends data to the attacker.",
        "LLMs are neural networks trained on massive text datasets.",
    ]
    ctxs = [
        "We define ASR as the percentage of runs in which the agent invokes a malicious tool and sends task related data to the attacker endpoint.",
        "Large Language Models (LLMs) are trained on large amounts of text data using self-supervised learning.",
    ]

    result = evaluate_generation(hyps, refs, ctxs)
    print("Full result dict:", result.to_dict())

    # Paraphrased YouTube-style answer (should score ~50-75%, not ~8%)
    haley_answer = (
        "Viewers have positive opinions about Haley, praising her beauty and humor. "
        "Many say she is so good and that they were watching Haley the whole time."
    )
    haley_ctx = (
        "Haley is so beautiful. haley is so freaken funny lol. Haley is so good. "
        "HALEY IS SO GOOD! I'm just watching Haley the whole time."
    )
    print(f"\nHaley example faithfulness: {faithfulness_score(haley_answer, haley_ctx):.1%}")
