"""
src/nlp/keywords.py
--------------------
Keyword extraction using two complementary approaches:

1. KeyBERT  – embedding-based, captures semantic relevance.
2. YAKE     – statistical, language-agnostic, fast.

Adds to each comment row:
  - keybert_keywords   : JSON list of top-5 keyword strings (KeyBERT)
  - yake_keywords      : JSON list of top-5 keyword strings (YAKE)
  - combined_keywords  : JSON deduplicated union list (for RAG retrieval)

Design notes
------------
- Both extractors are initialised once and reused (singleton pattern).
- Short comments (< MIN_WORDS words) gracefully return empty lists.
- All keyword columns are stored as JSON strings for CSV portability.

Author : Member 2 – NLP Enrichment Layer
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd
import yake
from keybert import KeyBERT

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s – %(message)s"))
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
TOP_N: int = 5           # keywords to extract per comment
MIN_WORDS: int = 4       # skip extraction for very short comments
NGRAM_RANGE: tuple[int, int] = (1, 2)   # unigrams + bigrams

# YAKE language-specific settings
YAKE_LANGUAGE: str = "en"
YAKE_MAX_NGRAM: int = 2
YAKE_DEDUP_LIM: float = 0.9
YAKE_WINDOWS: int = 1

# ---------------------------------------------------------------------------
# Singleton model instances
# ---------------------------------------------------------------------------
_KEYBERT_MODEL: KeyBERT | None = None
_YAKE_EXTRACTOR: Any | None = None


def _get_keybert(model_name: str = "all-MiniLM-L6-v2") -> KeyBERT:
    """Load (or return cached) KeyBERT model."""
    global _KEYBERT_MODEL
    if _KEYBERT_MODEL is None:
        logger.info("Initialising KeyBERT with sentence-transformer: %s", model_name)
        _KEYBERT_MODEL = KeyBERT(model=model_name)
    return _KEYBERT_MODEL


def _get_yake() -> Any:
    """Load (or return cached) YAKE extractor."""
    global _YAKE_EXTRACTOR
    if _YAKE_EXTRACTOR is None:
        logger.info("Initialising YAKE extractor …")
        _YAKE_EXTRACTOR = yake.KeywordExtractor(
            lan=YAKE_LANGUAGE,
            n=YAKE_MAX_NGRAM,
            dedupLim=YAKE_DEDUP_LIM,
            windowsSize=YAKE_WINDOWS,
            top=TOP_N,
        )
    return _YAKE_EXTRACTOR


# ---------------------------------------------------------------------------
# Per-comment extraction functions
# ---------------------------------------------------------------------------

def _is_too_short(text: str) -> bool:
    return len(text.split()) < MIN_WORDS


def extract_keybert_keywords(
    text: str,
    kw_model: KeyBERT,
    top_n: int = TOP_N,
    ngram_range: tuple[int, int] = NGRAM_RANGE,
    stop_words: str = "english",
) -> list[str]:
    """
    Extract top-N keywords from a single comment using KeyBERT.
    """
    if not isinstance(text, str) or _is_too_short(text.strip()):
        return []
    try:
        results = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=ngram_range,
            stop_words=stop_words,
            top_n=top_n,
        )
        return [kw for kw, _score in results]
    except Exception as exc:           # noqa: BLE001
        logger.warning("KeyBERT failed for '%s…': %s", text[:60], exc)
        return []


def extract_yake_keywords(
    text: str,
    extractor: Any,
    top_n: int = TOP_N,
) -> list[str]:
    """
    Extract top-N keywords from a single comment using YAKE.
    """
    if not isinstance(text, str) or _is_too_short(text.strip()):
        return []
    try:
        results = extractor.extract_keywords(text)
        return [kw for kw, _score in results[:top_n]]
    except Exception as exc:           # noqa: BLE001
        logger.warning("YAKE failed for '%s…': %s", text[:60], exc)
        return []


def combine_keywords(
    keybert_kws: list[str],
    yake_kws: list[str],
) -> list[str]:
    """
    Deduplicated union of KeyBERT and YAKE keywords (case-insensitive).
    KeyBERT results are listed first (semantic priority).
    """
    seen: set[str] = set()
    combined: list[str] = []
    for kw in keybert_kws + yake_kws:
        lower = kw.lower().strip()
        if lower and lower not in seen:
            seen.add(lower)
            combined.append(kw)
    return combined


# ---------------------------------------------------------------------------
# DataFrame-level enrichment
# ---------------------------------------------------------------------------

def run_keyword_extraction_on_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    keybert_model_name: str = "all-MiniLM-L6-v2",
    top_n: int = TOP_N,
    ngram_range: tuple[int, int] = NGRAM_RANGE,
) -> pd.DataFrame:
    """
    Run KeyBERT and YAKE over every row, adding three keyword columns:
    ``keybert_keywords``, ``yake_keywords``, ``combined_keywords``.
    """
    kw_model = _get_keybert(keybert_model_name)
    yake_extractor = _get_yake()

    df = df.copy()
    texts: list[str] = df[text_col].fillna("").astype(str).tolist()
    total = len(texts)

    logger.info("Extracting keywords from %d comments …", total)

    kb_results: list[list[str]] = []
    yk_results: list[list[str]] = []

    for i, text in enumerate(texts):
        if i % 500 == 0 and i > 0:
            logger.info("  … %d / %d done", i, total)
        kb_results.append(
            extract_keybert_keywords(text, kw_model, top_n=top_n, ngram_range=ngram_range)
        )
        yk_results.append(
            extract_yake_keywords(text, yake_extractor, top_n=top_n)
        )

    combined_results = [
        combine_keywords(kb, yk) for kb, yk in zip(kb_results, yk_results)
    ]

    df["keybert_keywords"]  = [json.dumps(kws, ensure_ascii=False) for kws in kb_results]
    df["yake_keywords"]     = [json.dumps(kws, ensure_ascii=False) for kws in yk_results]
    df["combined_keywords"] = [json.dumps(kws, ensure_ascii=False) for kws in combined_results]

    logger.info(
        "Keyword extraction complete.  "
        "Avg KeyBERT kws: %.2f  |  Avg YAKE kws: %.2f  |  Avg combined: %.2f",
        sum(len(k) for k in kb_results) / max(total, 1),
        sum(len(k) for k in yk_results) / max(total, 1),
        sum(len(k) for k in combined_results) / max(total, 1),
    )
    return df


# ---------------------------------------------------------------------------
# Convenience helpers for downstream consumers (Member 3 / Member 4)
# ---------------------------------------------------------------------------

def parse_keywords(keywords_json: str) -> list[str]:
    """
    Deserialise any keyword column value back to a Python list.

    Usage (in RAG / Dashboard):
    >>> kws = parse_keywords(row["combined_keywords"])
    """
    try:
        return json.loads(keywords_json) if keywords_json else []
    except json.JSONDecodeError:
        return []


def keywords_as_string(keywords_json: str, sep: str = ", ") -> str:
    """
    Convert JSON keyword list to a flat string (useful for BM25 / prompt injection).

    >>> keywords_as_string('["machine learning", "python"]')
    'machine learning, python'
    """
    return sep.join(parse_keywords(keywords_json))
