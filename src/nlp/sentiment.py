from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dual sentiment analysis (VADER + RoBERTa) on cleaned comments."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/comments_clean.csv",
        help="Path to cleaned comments CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/comments_with_sentiment.csv",
        help="Path to save output CSV with sentiment columns.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of the text column in the input CSV.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for RoBERTa inference.",
    )
    parser.add_argument(
        "--skip-roberta",
        action="store_true",
        help="Skip RoBERTa and only run VADER (faster, no GPU needed).",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    project_root = Path(__file__).resolve().parents[2]
    return project_root / path


def label_vader(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    return "neutral"


def run_vader(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    print("Running VADER sentiment...")
    analyzer = SentimentIntensityAnalyzer()
    scores = df[text_col].astype(str).apply(
        lambda t: analyzer.polarity_scores(t)["compound"]
    )
    df["sentiment_vader_score"] = scores.round(4)
    df["sentiment_vader"] = scores.apply(label_vader)
    return df


def run_roberta(df: pd.DataFrame, text_col: str, batch_size: int) -> pd.DataFrame:
    print("Loading RoBERTa model (first run downloads ~500MB)...")
    from transformers import pipeline

    classifier = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        max_length=128,
        truncation=True,
        device=-1,
    )

    texts = df[text_col].fillna("").astype(str).tolist()
    n = len(texts)
    labels: list[str]  = ["neutral"] * n
    scores: list[float] = [0.0] * n

    def _map_label(raw: str) -> str:
        raw = raw.lower()
        if "pos" in raw: return "positive"
        if "neg" in raw: return "negative"
        return "neutral"

    print(f"Running RoBERTa on {n:,} comments (batch size {batch_size}) …")
    failed_batches = 0

    for i in tqdm(range(0, n, batch_size)):
        batch      = texts[i : i + batch_size]
        batch_size_ = len(batch)
        try:
            results = classifier(batch)
            for j, r in enumerate(results):
                labels[i + j] = _map_label(r["label"])
                scores[i + j] = round(r["score"], 4)
        except Exception as exc:
            # Batch failed: fall back to per-item inference so we recover
            # as many rows as possible instead of losing the whole batch.
            failed_batches += 1
            for j, text in enumerate(batch):
                try:
                    r = classifier([text])[0]
                    labels[i + j] = _map_label(r["label"])
                    scores[i + j] = round(r["score"], 4)
                except Exception:
                    # Single item failed — leave as VADER fallback values ("neutral", 0.0)
                    pass

    if failed_batches:
        print(f"  ⚠ {failed_batches} batch(es) needed per-item fallback during RoBERTa inference.")

    df["sentiment_roberta"]       = labels
    df["sentiment_roberta_score"] = scores
    return df


def add_final_sentiment(df: pd.DataFrame, roberta_available: bool) -> pd.DataFrame:
    # Use RoBERTa as primary if available, otherwise fall back to VADER.
    if roberta_available and "sentiment_roberta" in df.columns:
        df["sentiment"] = df["sentiment_roberta"]
        df["sentiment_score"] = df["sentiment_roberta_score"]
    else:
        df["sentiment"] = df["sentiment_vader"]
        df["sentiment_score"] = df["sentiment_vader_score"]
    return df


def print_summary(df: pd.DataFrame) -> None:
    print("\nSentiment distribution (VADER):")
    print(df["sentiment_vader"].value_counts().to_string())

    if "sentiment_roberta" in df.columns:
        print("\nSentiment distribution (RoBERTa):")
        print(df["sentiment_roberta"].value_counts().to_string())

        agree = (df["sentiment_vader"] == df["sentiment_roberta"]).sum()
        total = len(df)
        print(f"\nVADER vs RoBERTa agreement: {agree}/{total} ({100*agree/total:.1f}%)")

    print("\nFinal sentiment column distribution:")
    print(df["sentiment"].value_counts().to_string())


def main() -> None:
    args = parse_args()

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} comments from {input_path}")

    if args.text_column not in df.columns:
        raise ValueError(f"Text column '{args.text_column}' not found in dataset.")

    df = run_vader(df, args.text_column)

    roberta_ran = False
    if not args.skip_roberta:
        try:
            df = run_roberta(df, args.text_column, args.batch_size)
            roberta_ran = True
        except Exception as e:
            print(f"RoBERTa failed ({e}). Falling back to VADER only.")

    df = add_final_sentiment(df, roberta_ran)

    print_summary(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
