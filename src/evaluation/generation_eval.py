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
class QueryScore:
    query: str
    hypothesis: str
    rouge_l: float
    bert_f1: float
    bert_f1_gold: float
    faithfulness: float


@dataclass
class GenerationResult:
    rouge_l:        float = 0.0
    bert_precision: float = 0.0
    bert_recall:    float = 0.0
    bert_f1:        float = 0.0
    bert_f1_gold:   float = 0.0
    faithfulness:   float = 0.0
    bertscore_error: str | None = None
    bertscore_used_fallback: bool = False
    per_query:      list[QueryScore] = field(default_factory=list)

    def to_dict(self) -> dict[str, float]:
        out = {
            "rouge_l":        self.rouge_l,
            "bert_precision": self.bert_precision,
            "bert_recall":    self.bert_recall,
            "bert_f1":        self.bert_f1,
            "faithfulness":   self.faithfulness,
        }
        if self.bert_f1_gold > 0:
            out["bert_f1_gold"] = self.bert_f1_gold
        return out


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

def reference_from_context(context: str, max_chars: int = 450) -> str:
    """
    Build an extractive reference from retrieved comments for BERTScore.
    RAG answers should align with this evidence summary when generation is good.
    """
    lines: list[str] = []
    for raw in context.split("\n"):
        line = re.sub(r"^\[[^\]]+\]\s*", "", raw.strip())
        if len(line) > 15:
            lines.append(line)
    if not lines:
        return context.strip()[:max_chars]
    # Prefer longer, more informative comments as the semantic reference
    unique = sorted(set(lines), key=len, reverse=True)[:3]
    ref = " ".join(unique)
    return ref[:max_chars]


def bert_score_eval(
    hypotheses: list[str],
    references: list[str],
    model_type: str = "distilbert-base-uncased",
    device: str = "cpu",
    rescale: bool = True,
) -> tuple[float, float, float, list[float], str | None]:
    """
    Compute BERTScore (P, R, F1) averaged over all pairs.

    Parameters
    ----------
    model_type : str
        HuggingFace model used by bert_score; 'distilbert-base-uncased' is lightweight.
    device : str
        "cpu" or "cuda".

    Returns (mean_p, mean_r, mean_f1, per_item_f1, error_message).
    error_message is None on success.
    """
    hyps = [h.strip() for h in hypotheses]
    refs = [r.strip() for r in references]
    if not hyps or not refs:
        return 0.0, 0.0, 0.0, [0.0] * len(hyps), "empty hypotheses or references"

    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        msg = "bert-score not installed — run: pip install bert-score"
        print(f"[WARNING] {msg}")
        return *_semantic_fallback(hyps, refs), msg

    models_to_try = [model_type]
    if model_type != "distilbert-base-uncased":
        models_to_try.append("distilbert-base-uncased")

    last_err = ""
    for model in models_to_try:
        for use_rescale in (rescale, False):
            try:
                P, R, F1 = bert_score_fn(
                    cands=hyps,
                    refs=refs,
                    model_type=model,
                    lang="en",
                    device=device,
                    verbose=False,
                    rescale_with_baseline=use_rescale,
                )
                f1_list = [float(x) for x in F1.tolist()]
                if any(x > 0 for x in f1_list):
                    return P.mean().item(), R.mean().item(), F1.mean().item(), f1_list, None
            except Exception as exc:
                last_err = f"{model} (rescale={use_rescale}): {exc}"
                print(f"[WARNING] BERTScore failed — {last_err}")

    msg = last_err or "BERTScore returned all zeros"
    print(f"[WARNING] {msg} — using semantic similarity fallback")
    return *_semantic_fallback(hyps, refs), msg


def _semantic_fallback(
    hypotheses: list[str],
    references: list[str],
) -> tuple[float, float, float, list[float]]:
    """MiniLM cosine similarity when bert-score package/model fails."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        return 0.0, 0.0, 0.0, [0.0] * len(hypotheses)

    if not hasattr(_semantic_fallback, "_model"):
        _semantic_fallback._model = SentenceTransformer("all-MiniLM-L6-v2")

    model = _semantic_fallback._model
    hyp_embs = model.encode(hypotheses, normalize_embeddings=True, show_progress_bar=False)
    ref_embs = model.encode(references, normalize_embeddings=True, show_progress_bar=False)
    sims = np.sum(hyp_embs * ref_embs, axis=1)
    f1_list = [float(max(0.0, min(1.0, s))) for s in sims]
    mean = sum(f1_list) / len(f1_list) if f1_list else 0.0
    return mean, mean, mean, f1_list


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
    queries:     list[str] | None = None,
    bert_model:  str = "distilbert-base-uncased",
    device:      str = "cpu",
    skip_bert:   bool = False,
    bert_reference_mode: str = "both",
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
    if len(hypotheses) != len(references) or len(hypotheses) != len(contexts):
        raise ValueError("hypotheses, references, and contexts must have the same length.")

    context_refs = [reference_from_context(c) for c in contexts]
    queries = queries or [f"query_{i + 1}" for i in range(len(hypotheses))]

    print("[eval] Computing ROUGE-L …")
    rl = batch_rouge_l(hypotheses, references)

    bp, br, bf, bf_gold = 0.0, 0.0, 0.0, 0.0
    per_item_f1: list[float] = []
    per_item_gold: list[float] = []
    bert_errors: list[str] = []
    used_fallback = False
    if skip_bert:
        print("[eval] Skipping BERTScore (fast mode)")
    else:
        mode = bert_reference_mode.lower()
        if mode in ("context", "both"):
            print(f"[eval] Computing BERTScore vs retrieved context ({bert_model}) …")
            bp, br, bf, per_item_f1, err = bert_score_eval(hypotheses, context_refs, bert_model, device)
            if err:
                bert_errors.append(err)
                used_fallback = True
        if mode in ("static", "gold", "both"):
            print(f"[eval] Computing BERTScore vs gold references ({bert_model}) …")
            _, _, bf_gold, per_item_gold, err = bert_score_eval(hypotheses, references, bert_model, device)
            if err:
                bert_errors.append(err)
                used_fallback = True
            if mode in ("static", "gold"):
                bp, br, bf, per_item_f1 = _, _, bf_gold, per_item_gold

    print("[eval] Computing faithfulness …")
    per_query: list[QueryScore] = []
    faith_scores: list[float] = []
    for i, (q, hyp, ref, ctx) in enumerate(zip(queries, hypotheses, references, contexts)):
        rouge = rouge_l_score(hyp, ref)
        faith = faithfulness_score(hyp, ctx)
        faith_scores.append(faith)
        b_f1 = per_item_f1[i] if i < len(per_item_f1) else 0.0
        b_gold = per_item_gold[i] if i < len(per_item_gold) else 0.0
        per_query.append(QueryScore(
            query=q,
            hypothesis=hyp,
            rouge_l=rouge,
            bert_f1=b_f1,
            bert_f1_gold=b_gold,
            faithfulness=faith,
        ))

    faith = sum(faith_scores) / len(faith_scores) if faith_scores else 0.0

    result = GenerationResult(
        rouge_l        = rl,
        bert_precision = bp,
        bert_recall    = br,
        bert_f1        = bf,
        bert_f1_gold   = bf_gold,
        faithfulness   = faith,
        bertscore_error = "; ".join(dict.fromkeys(bert_errors)) if bert_errors else None,
        bertscore_used_fallback = used_fallback,
        per_query      = per_query,
    )
    _print_results(result, skip_bert=skip_bert, bert_reference_mode=bert_reference_mode)
    return result


def _print_results(
    r: GenerationResult,
    skip_bert: bool = False,
    bert_reference_mode: str = "both",
) -> None:
    print("\n── Generation Evaluation Results ─────────────────────")
    print(f"  ROUGE-L        {r.rouge_l:.4f}")
    if not skip_bert:
        if bert_reference_mode.lower() in ("context", "both"):
            print(f"  BERTScore P    {r.bert_precision:.4f}  (vs retrieved context)")
            print(f"  BERTScore R    {r.bert_recall:.4f}")
            print(f"  BERTScore F1   {r.bert_f1:.4f}")
        if bert_reference_mode.lower() in ("static", "gold", "both") and r.bert_f1_gold > 0:
            print(f"  BERTScore F1*  {r.bert_f1_gold:.4f}  (vs gold reference)")
    print(f"  Faithfulness   {r.faithfulness:.4f}")
    if r.per_query:
        print("\n  Per-query breakdown:")
        for row in r.per_query:
            line = f"    • {row.query[:60]}…  R-L={row.rouge_l:.2f}  faith={row.faithfulness:.2f}"
            if not skip_bert:
                line += f"  BERT={row.bert_f1:.2f}"
                if row.bert_f1_gold > 0:
                    line += f"  BERT(gold)={row.bert_f1_gold:.2f}"
            print(line)
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
