"""
src/agent/tools.py

One tool function per agent intent. `qa_tool` and `summarize_tool` are
generation-based (RAG over comments_enriched.csv, via src/rag/generator.py).
`sentiment_insight_tool`, `topic_insight_tool`, and `entity_insight_tool`
are mostly direct pandas aggregations over the enriched dataset -- they pull
real numbers first, then optionally phrase them in natural language with
`generator.generate_insight`, so the answer is always grounded in the
actual data rather than something the LLM invented.

Expected schema of data/processed/comments_enriched.csv (Member 2's output).
This is documented here because Member 3 doesn't generate this file, just
consumes it -- if Member 2's real column names differ, update COLUMNS below
rather than touching the tool logic:

    comment_id            int, unique id per comment
    text                  str, cleaned comment text
    sentiment             str, one of positive / negative / neutral
    entities               str, JSON list of {"text": ..., "label": ...}
    keywords               str, JSON list of up to 5 keyword strings
    topic_id               int, overall BERTopic topic id
    topic_label             str, overall BERTopic topic label
    topic_id_sentiment       int, topic id from the per-sentiment BERTopic model
    topic_label_sentiment    str, label from the per-sentiment BERTopic model
"""

import json
import ast
import re

import pandas as pd

from src.rag.generator import (
    DEFAULT_DATA_PATH,
    load_comments,
    answer_with_context,
    summarize_query,
    generate_insight,
)

COLUMNS = {
    "id": "comment_id",
    "text": "text",
    "sentiment": "sentiment",
    "entities": "entities",
    "keywords": "keywords",
    "topic_id": "topic_id",
    "topic_label": "topic_label",
    "topic_id_sentiment": "topic_id_sentiment",
    "topic_label_sentiment": "topic_label_sentiment",
}

_STOPWORDS = {
    "the", "and", "are", "for", "what", "which", "about", "people", "with",
    "that", "this", "have", "does", "did", "most", "common", "many", "much",
    "comments", "comment", "users", "customers", "feel", "think", "saying",
}


def _parse_listish(value):
    """entities/keywords columns may come in as JSON, Python-literal, or
    already-parsed lists depending on how Member 2 saved the CSV -- handle
    all three instead of assuming one."""
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(value)
        except Exception:
            continue
    return []


def _extract_keywords_from_query(query):
    words = [w.strip("?.,!").lower() for w in query.split()]
    return [w for w in words if len(w) >= 4 and w not in _STOPWORDS]


def _filter_by_query(df, query, text_col):
    """Narrow the dataset to rows that look relevant to the query (e.g.
    'sentiment about checkout flow' -> only rows mentioning 'checkout' or
    'flow'). Falls back to the full dataframe if nothing matches, so a tool
    never returns an empty result just because the keyword filter missed."""
    kws = _extract_keywords_from_query(query)
    if not kws:
        return df
    mask = df[text_col].astype(str).str.lower().apply(lambda t: any(k in t for k in kws))
    filtered = df[mask]
    return filtered if len(filtered) > 0 else df


def _detect_sentiment_filter(query):
    """Detect an explicit sentiment qualifier in the query (e.g. 'topics in
    NEGATIVE comments') so callers can filter by the structured sentiment
    column directly, rather than relying on the word 'negative' happening
    to appear in the comment text itself (it usually won't)."""
    text = query.lower()
    for sentiment in ("positive", "negative", "neutral"):
        if re.search(rf"\b{sentiment}\b", text):
            return sentiment
    return None


def _apply_sentiment_filter(df, query, sentiment_col):
    sentiment = _detect_sentiment_filter(query)
    if sentiment and sentiment_col in df.columns:
        narrowed = df[df[sentiment_col] == sentiment]
        if len(narrowed) > 0:
            return narrowed
    return df


def _load(path=DEFAULT_DATA_PATH, df=None):
    return df if df is not None else load_comments(path)


# --------------------------------------------------------------------------
# qa
# --------------------------------------------------------------------------
def qa_tool(query, k=4, retrieval="hybrid", path=DEFAULT_DATA_PATH):
    result = answer_with_context(query, k=k, retrieval=retrieval, path=path)
    return {
        "intent": "qa",
        "answer": result["answer"],
        "citations": result["citations"],
        "raw": {"retrieved": result["retrieved"]},
    }


# --------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------
def summarize_tool(query, k=6, path=DEFAULT_DATA_PATH):
    result = summarize_query(query, k=k, path=path)
    return {
        "intent": "summarize",
        "answer": result["summary"],
        "citations": result["citations"],
        "raw": {"retrieved": result["retrieved"]},
    }


# --------------------------------------------------------------------------
# sentiment_insight
# --------------------------------------------------------------------------
def sentiment_insight_tool(query, df=None, path=DEFAULT_DATA_PATH):
    df = _load(path, df)
    text_col, sent_col, id_col = COLUMNS["text"], COLUMNS["sentiment"], COLUMNS["id"]
    subset = _filter_by_query(df, query, text_col)

    counts = subset[sent_col].value_counts().to_dict()
    total = int(sum(counts.values())) or 1
    breakdown = {k: {"count": int(v), "pct": round(100 * v / total, 1)} for k, v in counts.items()}
    example_ids = {
        sent: subset[subset[sent_col] == sent][id_col].head(2).tolist() for sent in counts
    }

    stats = {"matched_comments": int(len(subset)), "breakdown": breakdown, "examples": example_ids}
    insight = generate_insight(query, stats)
    citations = [cid for ids in example_ids.values() for cid in ids]
    return {"intent": "sentiment_insight", "answer": insight, "citations": citations, "raw": stats}


# --------------------------------------------------------------------------
# topic_insight
# --------------------------------------------------------------------------
def topic_insight_tool(query, df=None, top_n=5, path=DEFAULT_DATA_PATH):
    df = _load(path, df)
    text_col = COLUMNS["text"]
    label_col, sent_label_col, id_col = COLUMNS["topic_label"], COLUMNS["topic_label_sentiment"], COLUMNS["id"]
    subset = _apply_sentiment_filter(df, query, COLUMNS["sentiment"])
    subset = _filter_by_query(subset, query, text_col)

    overall = subset[label_col].value_counts().head(top_n)
    overall_topics = [{"topic": t, "count": int(c)} for t, c in overall.items()]

    per_sentiment = {}
    if COLUMNS["sentiment"] in subset.columns and sent_label_col in subset.columns:
        for sentiment, group in subset.groupby(COLUMNS["sentiment"]):
            top = group[sent_label_col].value_counts().head(top_n)
            per_sentiment[sentiment] = [{"topic": t, "count": int(c)} for t, c in top.items()]

    example_ids = subset[id_col].head(5).tolist()
    stats = {"matched_comments": int(len(subset)), "top_topics_overall": overall_topics, "top_topics_by_sentiment": per_sentiment}
    insight = generate_insight(query, stats)
    return {"intent": "topic_insight", "answer": insight, "citations": example_ids, "raw": stats}


# --------------------------------------------------------------------------
# entity_insight
# --------------------------------------------------------------------------
def entity_insight_tool(query, df=None, top_n=5, path=DEFAULT_DATA_PATH):
    df = _load(path, df)
    text_col, ent_col, id_col = COLUMNS["text"], COLUMNS["entities"], COLUMNS["id"]
    subset = _apply_sentiment_filter(df, query, COLUMNS["sentiment"])
    subset = _filter_by_query(subset, query, text_col)

    if ent_col not in subset.columns:
        stats = {"matched_comments": int(len(subset)), "top_entities": []}
        return {"intent": "entity_insight", "answer": generate_insight(query, stats),
                "citations": [], "raw": stats}

    # Vectorized: explode entity lists into one row per entity, count, then join
    # back to the original ids — much faster than iterrows on large DataFrames.
    parsed = subset[[id_col, ent_col]].copy()
    parsed["_ents"] = parsed[ent_col].apply(_parse_listish)
    exploded = parsed.explode("_ents").dropna(subset=["_ents"])
    exploded = exploded[exploded["_ents"].apply(lambda e: bool(e))]

    def _ent_name(e):
        if isinstance(e, dict):
            return e.get("text", "")
        return str(e)

    exploded["_name"] = exploded["_ents"].apply(_ent_name)
    exploded = exploded[exploded["_name"].str.strip() != ""]

    counts       = exploded["_name"].value_counts()
    top_entities = counts.head(top_n)

    top_entities_fmt = [{"entity": name, "mentions": int(cnt)}
                        for name, cnt in top_entities.items()]

    # Gather up to 2 comment ids per top entity
    citations = []
    for name in top_entities.index:
        ids = exploded.loc[exploded["_name"] == name, id_col].head(2).tolist()
        citations.extend(ids)

    stats = {"matched_comments": int(len(subset)), "top_entities": top_entities_fmt}
    insight = generate_insight(query, stats)
    return {"intent": "entity_insight", "answer": insight, "citations": citations, "raw": stats}


TOOL_REGISTRY = {
    "qa": qa_tool,
    "summarize": summarize_tool,
    "sentiment_insight": sentiment_insight_tool,
    "topic_insight": topic_insight_tool,
    "entity_insight": entity_insight_tool,
}
