"""
src/nlp/topics.py
------------------
Topic modelling with BERTopic.

Runs four separate BERTopic models:
  1. overall_topics     — trained on all comments
  2. positive_topics    — trained on positive-sentiment comments only
  3. negative_topics    — trained on negative-sentiment comments only
  4. neutral_topics     — trained on neutral-sentiment comments only

Adds to each comment row:
  - overall_topic_id     (int)   : topic index from the overall model (-1 = outlier)
  - overall_topic_name   (str)   : human-readable label
  - sentiment_topic_id   (int)   : topic index from the matching sentiment model
  - sentiment_topic_name (str)   : human-readable label

Topic Name Format
-----------------
  "Topic 3: python · tutorial · beginner"
  "Topic -1: Outlier"

Design notes
------------
- BERTopic handles its own sentence embedding internally (uses all-MiniLM-L6-v2
  by default, matching KeyBERT so embeddings can be reused in future).
- UMAP + HDBSCAN parameters are tuned for short social-media text.
- For very small corpora (< MIN_DOCS), the model degrades gracefully to
  a single "Uncategorised" topic.
- All labels are safe for downstream DataFrame joins and dashboard display.

Author : Member 2 – NLP Enrichment Layer
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

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
# Configuration
# ---------------------------------------------------------------------------
MIN_DOCS: int = 10          # minimum comments needed to run BERTopic
MIN_TOPIC_SIZE: int = 5     # HDBSCAN min_cluster_size
OUTLIER_TOPIC_ID: int = -1
OUTLIER_TOPIC_NAME: str = "Topic -1: Outlier"
UNCATEGORISED_TOPIC_NAME: str = "Topic 0: Uncategorised"

SentimentLabel = Literal["positive", "negative", "neutral"]
SENTIMENT_LABELS: list[SentimentLabel] = ["positive", "negative", "neutral"]


# ---------------------------------------------------------------------------
# BERTopic factory
# ---------------------------------------------------------------------------

def _build_bertopic_model(
    n_neighbors: int = 10,
    n_components: int = 5,
    min_dist: float = 0.0,
    min_topic_size: int = MIN_TOPIC_SIZE,
    nr_topics: int | str | None = "auto",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> BERTopic:
    """
    Construct a BERTopic model with explicit UMAP + HDBSCAN parameters.
    Parameters are tuned for short (YouTube comment) texts.
    """
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
        low_memory=True,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=nr_topics,
        verbose=False,
        calculate_probabilities=False,
    )


# ---------------------------------------------------------------------------
# Topic-label helper
# ---------------------------------------------------------------------------

def _make_topic_label(topic_id: int, topic_info_row: pd.Series | None) -> str:
    """Build a human-readable label string."""
    if topic_id == OUTLIER_TOPIC_ID:
        return OUTLIER_TOPIC_NAME
    if topic_info_row is None:
        return f"Topic {topic_id}: Uncategorised"
    raw_name: str = str(topic_info_row.get("Name", ""))
    parts = raw_name.split("_")
    words = [p for p in parts[1:] if p] if len(parts) > 1 else parts
    label_words = " · ".join(words[:5]) if words else "unnamed"
    return f"Topic {topic_id}: {label_words}"


# ---------------------------------------------------------------------------
# Single-corpus modelling
# ---------------------------------------------------------------------------

def fit_bertopic(
    texts: list[str],
    label_prefix: str = "overall",
) -> tuple[list[int], list[str], BERTopic | None]:
    """
    Fit BERTopic on a list of texts.

    Returns
    -------
    topic_ids : list[int]
    topic_names : list[str]
    model : BERTopic | None
    """
    n = len(texts)
    if n < MIN_DOCS:
        logger.warning(
            "[%s] Only %d documents — skipping BERTopic (need ≥ %d). "
            "Assigning 'Uncategorised'.",
            label_prefix, n, MIN_DOCS,
        )
        return (
            [0] * n,
            [UNCATEGORISED_TOPIC_NAME] * n,
            None,
        )

    logger.info("[%s] Fitting BERTopic on %d documents …", label_prefix, n)
    model = _build_bertopic_model()

    try:
        topics, _ = model.fit_transform(texts)
    except Exception as exc:
        logger.error("[%s] BERTopic fit_transform failed: %s", label_prefix, exc)
        return (
            [OUTLIER_TOPIC_ID] * n,
            [OUTLIER_TOPIC_NAME] * n,
            None,
        )

    try:
        topic_info = model.get_topic_info().set_index("Topic")
    except Exception:
        topic_info = pd.DataFrame()

    topic_names: list[str] = []
    for tid in topics:
        row = topic_info.loc[tid] if (not topic_info.empty and tid in topic_info.index) else None
        topic_names.append(_make_topic_label(tid, row))

    unique_topics = len(set(t for t in topics if t != OUTLIER_TOPIC_ID))
    outlier_pct = 100.0 * sum(1 for t in topics if t == OUTLIER_TOPIC_ID) / max(n, 1)
    logger.info(
        "[%s] Done.  Unique topics: %d  |  Outlier %%: %.1f",
        label_prefix, unique_topics, outlier_pct,
    )
    return list(topics), topic_names, model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_topic_modeling_on_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    sentiment_col: str = "sentiment",
) -> pd.DataFrame:
    """
    Run BERTopic across the full corpus and per-sentiment slices.

    Appends four columns to ``df``:
      - ``overall_topic_id``
      - ``overall_topic_name``
      - ``sentiment_topic_id``
      - ``sentiment_topic_name``
    """
    df = df.copy()
    texts_all: list[str] = df[text_col].fillna("").astype(str).tolist()

    # 1. Overall topic model
    overall_ids, overall_names, _overall_model = fit_bertopic(
        texts_all, label_prefix="overall"
    )
    df["overall_topic_id"]   = overall_ids
    df["overall_topic_name"] = overall_names

    # 2. Per-sentiment topic models
    df["sentiment_topic_id"]   = OUTLIER_TOPIC_ID
    df["sentiment_topic_name"] = OUTLIER_TOPIC_NAME

    for sentiment in SENTIMENT_LABELS:
        mask = df[sentiment_col].str.lower() == sentiment
        subset_idx = df.index[mask].tolist()
        subset_texts = df.loc[mask, text_col].fillna("").astype(str).tolist()

        if not subset_texts:
            logger.info("No '%s' comments found — skipping.", sentiment)
            continue

        s_ids, s_names, _model = fit_bertopic(
            subset_texts, label_prefix=f"sentiment:{sentiment}"
        )

        for row_idx, tid, tname in zip(subset_idx, s_ids, s_names):
            df.at[row_idx, "sentiment_topic_id"]   = tid
            df.at[row_idx, "sentiment_topic_name"] = tname

    df["overall_topic_id"]   = df["overall_topic_id"].fillna(OUTLIER_TOPIC_ID).astype(int)
    df["sentiment_topic_id"] = df["sentiment_topic_id"].fillna(OUTLIER_TOPIC_ID).astype(int)

    logger.info(
        "Topic modelling complete.  Overall topic distribution:\n%s",
        df["overall_topic_name"].value_counts().head(10).to_string(),
    )
    return df


# ---------------------------------------------------------------------------
# Convenience helpers for downstream consumers (Member 3 / Member 4)
# ---------------------------------------------------------------------------

def get_topic_summary(df: pd.DataFrame, topic_col: str = "overall_topic_name") -> pd.DataFrame:
    """
    Return a summary DataFrame of topic frequencies.
    Columns: ``topic_name``, ``comment_count``, ``pct``
    Compatible with pandas >= 2.0 (value_counts().reset_index() column naming changed).
    """
    counts = df[topic_col].value_counts().reset_index()
    # pandas < 2.0: columns are ["index", topic_col]
    # pandas >= 2.0: columns are [topic_col, "count"]
    if "count" in counts.columns:
        counts.columns = ["topic_name", "comment_count"]
    else:
        counts.columns = ["topic_name", "comment_count"]
    counts["pct"] = (counts["comment_count"] / counts["comment_count"].sum() * 100).round(2)
    return counts


def filter_comments_by_topic(
    df: pd.DataFrame,
    topic_name: str,
    topic_col: str = "overall_topic_name",
) -> pd.DataFrame:
    """
    Return all rows belonging to a given topic (partial match, case-insensitive).
    """
    mask = df[topic_col].str.contains(topic_name, case=False, na=False)
    return df[mask].copy()
