"""
patch_sentiment.py
------------------
Fixes comments_enriched.csv where sentiment is all "unknown".

Runs VADER (rule-based, no model download, < 1 min on 40k rows) and
optionally RoBERTa (transformer, ~20 min, better accuracy).

Usage:
    python patch_sentiment.py                  # VADER only (fast)
    python patch_sentiment.py --roberta        # VADER + RoBERTa (accurate)
    python patch_sentiment.py --dry-run        # preview first 20 rows, no write
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parent
ENRICHED = ROOT / "data" / "processed" / "comments_enriched.csv"


def vader_label(score: float) -> str:
    if score >= 0.05:  return "positive"
    if score <= -0.05: return "negative"
    return "neutral"


def run_vader(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Running VADER on {len(df):,} comments …")
    analyzer = SentimentIntensityAnalyzer()
    texts  = df["text"].fillna("").astype(str).tolist()
    scores = [analyzer.polarity_scores(t)["compound"] for t in tqdm(texts)]
    df["sentiment_vader_score"] = [round(s, 4) for s in scores]
    df["sentiment_vader"]       = [vader_label(s) for s in scores]
    return df


def run_roberta(df: pd.DataFrame, batch_size: int = 64) -> pd.DataFrame:
    print("Loading RoBERTa model (first run ~500 MB download) …")
    from transformers import pipeline as hf_pipeline
    classifier = hf_pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        max_length=128,
        truncation=True,
        device=-1,
    )
    texts = df["text"].fillna("").astype(str).tolist()
    labels, scores = [], []
    print(f"Running RoBERTa on {len(texts):,} comments (batch {batch_size}) …")
    for i in tqdm(range(0, len(texts), batch_size)):
        for r in classifier(texts[i:i+batch_size]):
            lbl = r["label"].lower()
            labels.append("positive" if "pos" in lbl else ("negative" if "neg" in lbl else "neutral"))
            scores.append(round(r["score"], 4))
    df["sentiment_roberta"]       = labels
    df["sentiment_roberta_score"] = scores
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   default=str(ENRICHED))
    parser.add_argument("--output",  default=str(ENRICHED), help="defaults to same file (in-place patch)")
    parser.add_argument("--roberta", action="store_true",   help="also run RoBERTa for higher accuracy")
    parser.add_argument("--dry-run", action="store_true",   help="print sample output, don't write")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"ERROR: file not found: {path}"); sys.exit(1)

    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from {path.name}")

    # Show current distribution
    if "sentiment" in df.columns:
        print(f"\nCurrent sentiment distribution:\n{df['sentiment'].value_counts().to_string()}\n")

    # Check text column
    non_empty = df["text"].fillna("").astype(str).str.strip().ne("").sum()
    print(f"Non-empty text rows: {non_empty:,} / {len(df):,}")
    if non_empty == 0:
        print("ERROR: text column is empty — check your CSV."); sys.exit(1)

    # Run VADER
    df = run_vader(df)

    # Optionally run RoBERTa
    roberta_ok = False
    if args.roberta:
        try:
            df = run_roberta(df)
            roberta_ok = True
        except Exception as e:
            print(f"RoBERTa failed ({e}) — using VADER only.")

    # Set final sentiment column
    if roberta_ok:
        df["sentiment"]       = df["sentiment_roberta"]
        df["sentiment_score"] = df["sentiment_roberta_score"]
        method = "RoBERTa"
    else:
        df["sentiment"]       = df["sentiment_vader"]
        df["sentiment_score"] = df["sentiment_vader_score"]
        method = "VADER"

    print(f"\nNew sentiment distribution ({method}):")
    print(df["sentiment"].value_counts().to_string())

    if args.dry_run:
        print("\n[dry-run] First 20 rows:")
        print(df[["comment_id", "text", "sentiment", "sentiment_score"]].head(20).to_string())
        print("\nNo file written (--dry-run).")
        return

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\nSaved → {out}  ({len(df):,} rows)")
    print("\nDone. Restart the dashboard to see updated sentiment charts.")


if __name__ == "__main__":
    main()
