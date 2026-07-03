"""
Build retrieval ground-truth comment_id sets from comments_enriched.csv.

Each eval query gets a set of YouTube comment IDs whose text matches
query-specific keywords (and optional sentiment filters). This replaces
the old placeholder IDs {1, 2, 3…} which never existed in the real dataset.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# One spec per EVAL_QUERIES entry in run_eval.py (same order)
QUERY_SPECS: list[dict] = [
    {
        "keywords": [
            "story", "storytelling", "narrative", "edit", "edited", "editing",
            "entertaining", "funny", "plot", "cinematic", "video style",
        ],
    },
    {
        "keywords": [
            "love", "great", "amazing", "best", "awesome", "good", "fire",
            "hate", "worst", "bad", "terrible", "mid", "boring",
        ],
    },
    {
        "keywords": [
            "sponsor", "ad", "long", "boring", "clickbait", "disappoint",
            "annoying", "mid", "bad", "worst", "hate", "skip",
        ],
        "sentiment": "negative",
    },
    {
        "keywords": [
            "ryan", "haley", "youtube", "sponsor", "brand", "mrbeast",
            "apple", "nike", "google", "netflix",
        ],
    },
    {
        "keywords": [
            "ryan", "video", "vlog", "upload", "episode", "series",
            "recent", "latest", "new video", "trahan",
        ],
    },
]


def build_ground_truth(
    path: str | Path,
    specs: list[dict] | None = None,
    max_ids: int = 150,
) -> list[set[str]]:
    """
    Return parallel list of relevant comment_id sets (YouTube string IDs).
    """
    specs = specs or QUERY_SPECS
    path = Path(path)
    df = pd.read_csv(path, usecols=["comment_id", "text", "sentiment"], low_memory=False)
    df["comment_id"] = df["comment_id"].astype(str)
    text = df["text"].astype(str).str.lower()

    ground: list[set[str]] = []
    for spec in specs:
        mask = pd.Series(True, index=df.index)
        keywords = spec.get("keywords", [])

        if keywords:
            pat = "|".join(re.escape(k.lower()) for k in keywords)
            mask &= text.str.contains(pat, regex=True, na=False)

        if spec.get("sentiment"):
            mask &= df["sentiment"].astype(str).str.lower() == spec["sentiment"].lower()

        if spec.get("min_len"):
            mask &= text.str.len() >= int(spec["min_len"])

        ids = set(df.loc[mask, "comment_id"].drop_duplicates().head(max_ids).tolist())

        # Fallback: widen with single-keyword matches if too strict
        if len(ids) < 10 and keywords:
            for kw in keywords:
                kw_mask = text.str.contains(re.escape(kw.lower()), regex=True, na=False)
                extra = df.loc[kw_mask, "comment_id"].head(max_ids)
                ids.update(extra.astype(str).tolist())
                if len(ids) >= 10:
                    break

        ground.append(ids)

    return ground


def summarize_ground_truth(queries: list[str], ground: list[set[str]]) -> None:
    print("\n── Ground-truth relevance sets ───────────────────────")
    for q, ids in zip(queries, ground):
        print(f"  Q: {q}")
        print(f"     {len(ids)} relevant comment IDs")
    print("──────────────────────────────────────────────────────\n")
