from __future__ import annotations

"""
src/pipeline/run_full_pipeline.py

Full NLP enrichment pipeline.  Produces comments_enriched.csv.

Pipeline
--------
  1. Sentiment  (VADER + RoBERTa)     → comments_with_sentiment.csv
  2. NER + Keywords + Topics          → comments_enriched.csv
     (Member 2's enrich.py)

Usage (from project root)
-------------------------
  python src/pipeline/run_full_pipeline.py
  python src/pipeline/run_full_pipeline.py --skip-roberta
  python src/pipeline/run_full_pipeline.py --start-from enrich   # resume after sentiment
"""

import argparse
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def header(msg: str) -> None:
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


# ── Step 1: Sentiment ─────────────────────────────────────────────────────

def run_sentiment(input_path: Path, output_path: Path, skip_roberta: bool) -> None:
    header("Step 1/2 — Sentiment Analysis (VADER + RoBERTa)")
    from src.nlp.sentiment import run_vader, run_roberta, add_final_sentiment, print_summary

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} comments.")

    df = run_vader(df, "text")
    roberta_ran = False
    if not skip_roberta:
        try:
            df = run_roberta(df, "text", batch_size=64)
            roberta_ran = True
        except Exception as e:
            print(f"RoBERTa skipped ({e}). Using VADER only.")
    df = add_final_sentiment(df, roberta_ran)
    print_summary(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nSaved → {output_path.name}")


# ── Step 2: NER + Keywords + Topics (Member 2) ───────────────────────────

def run_enrich(input_path: Path, output_path: Path, spacy_model: str,
               keybert_model: str, top_n: int, batch_size: int) -> None:
    header("Step 2/2 — NER + Keywords + Topics (Member 2's enrich.py)")
    from src.nlp.enrich import run_enrichment_pipeline, save_enriched_csv

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} comments.")

    enriched = run_enrichment_pipeline(
        df,
        text_col="text",
        sentiment_col="sentiment",
        spacy_model=spacy_model,
        keybert_model=keybert_model,
        top_n_keywords=top_n,
        ner_batch_size=batch_size,
    )
    save_enriched_csv(enriched, output_path)


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full NLP pipeline and produce comments_enriched.csv."
    )
    parser.add_argument("--input",   default="data/processed/comments_clean.csv")
    parser.add_argument("--output",  default="data/processed/comments_enriched.csv")
    parser.add_argument("--skip-roberta",  action="store_true")
    parser.add_argument("--spacy-model",   default="en_core_web_sm")
    parser.add_argument("--keybert-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--top-n-keywords", type=int, default=5)
    parser.add_argument("--ner-batch-size", type=int, default=64)
    parser.add_argument(
        "--start-from",
        choices=["sentiment", "enrich", "topics"],
        default="sentiment",
        help="Resume from this stage (skip earlier completed steps).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path   = resolve(args.input)
    output_path  = resolve(args.output)
    sentiment_path = output_path.parent / "comments_with_sentiment.csv"

    keywords_path = output_path.parent / "comments_with_keywords.csv"

    if args.start_from != "topics" and not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if args.start_from == "topics":
        # Resume right before BERTopic — reads the existing keywords file
        if not keywords_path.exists():
            raise FileNotFoundError(
                f"Cannot resume from topics: {keywords_path} not found."
            )
        header("Step 2/2 — Topics only (resuming from comments_with_keywords.csv)")
        from src.nlp.topics import run_topic_modeling_on_dataframe
        from src.nlp.enrich import _add_compatibility_aliases, save_enriched_csv
        df = pd.read_csv(keywords_path)
        print(f"Loaded {len(df)} comments.")

        # Patch in sentiment column if missing
        if "sentiment" not in df.columns:
            print("'sentiment' column missing — trying to patch from earlier files…")
            patched = False
            for fallback in [
                output_path.parent / "comments_with_ner.csv",
                output_path.parent / "comments_with_sentiment.csv",
            ]:
                if fallback.exists():
                    try:
                        src_df = pd.read_csv(fallback)
                        if "sentiment" in src_df.columns and "comment_id" in src_df.columns:
                            df = df.merge(
                                src_df[["comment_id", "sentiment"]],
                                on="comment_id", how="left"
                            )
                            print(f"  Merged sentiment from {fallback.name}")
                            patched = True
                            break
                    except Exception:
                        pass
            if not patched:
                print("  No sentiment source found — running VADER as fallback …")
                try:
                    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                    _va = SentimentIntensityAnalyzer()
                    def _vlabel(t):
                        s = _va.polarity_scores(str(t))["compound"]
                        return "positive" if s >= 0.05 else ("negative" if s <= -0.05 else "neutral")
                    df["sentiment"] = df["text"].fillna("").apply(_vlabel)
                    print(f"  VADER done. Distribution:\n{df['sentiment'].value_counts().to_string()}")
                except Exception as _ve:
                    print(f"  VADER also failed ({_ve}) — setting 'neutral' as safe default.")
                    df["sentiment"] = "neutral"
        df = run_topic_modeling_on_dataframe(df, text_col="text", sentiment_col="sentiment")
        df = _add_compatibility_aliases(df)
        save_enriched_csv(df, output_path)

    else:
        if args.start_from == "sentiment":
            run_sentiment(input_path, sentiment_path, args.skip_roberta)
        else:
            print(f"Skipping sentiment (using existing {sentiment_path.name})")
            if not sentiment_path.exists():
                raise FileNotFoundError(
                    f"Cannot resume: {sentiment_path} not found. "
                    f"Remove --start-from to run from the beginning."
                )

        run_enrich(
            sentiment_path, output_path,
            spacy_model=args.spacy_model,
            keybert_model=args.keybert_model,
            top_n=args.top_n_keywords,
            batch_size=args.ner_batch_size,
        )

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Output: {output_path}")
    print(f"\nNext — build FAISS + BM25 indexes:")
    print(f"  python src/rag/indexer.py --chunk-strategy propositional")


if __name__ == "__main__":
    main()
