"""
src/nlp/enrich.py
------------------
NLP enrichment pipeline (Member 2).

Execution order
---------------
  1. Load cleaned comments CSV (with sentiment already added by sentiment.py).
  2. Run NER          → entities, entity_count
  3. Run keywords     → keybert_keywords, yake_keywords, combined_keywords
  4. Run topics       → overall_topic_*, sentiment_topic_*
  5. Add column aliases for downstream compatibility.
  6. Save to data/processed/comments_enriched.csv

Column aliases added at the end
--------------------------------
  combined_keywords  → keywords          (for generator.py / tools.py)
  overall_topic_id   → topic_id          (for tools.py / dashboard.py)
  overall_topic_name → topic_label       (for tools.py / dashboard.py)
  sentiment_topic_id → topic_id_sentiment
  sentiment_topic_name → topic_label_sentiment

Usage
-----
  python src/nlp/enrich.py --input data/processed/comments_with_sentiment.csv
  python src/nlp/enrich.py --input data/processed/comments_with_sentiment.csv --output data/processed/comments_enriched.csv

Author : Member 2 – NLP Enrichment Layer
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from src.nlp.ner import run_ner_on_dataframe
from src.nlp.keywords import run_keyword_extraction_on_dataframe
from src.nlp.topics import run_topic_modeling_on_dataframe

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required input columns
# ---------------------------------------------------------------------------
REQUIRED_INPUT_COLS: list[str] = ["text", "sentiment"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns: {missing}. "
            f"Run sentiment.py first to add sentiment columns."
        )


# ---------------------------------------------------------------------------
# Column aliases — bridges Member 2's column names to downstream consumers
# ---------------------------------------------------------------------------

def _add_compatibility_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add alias columns so Member 3's tools.py and Member 4's dashboard.py
    work without changes.  Original columns are kept alongside aliases.
    """
    alias_map = {
        "combined_keywords":   "keywords",
        "overall_topic_id":    "topic_id",
        "overall_topic_name":  "topic_label",
        "sentiment_topic_id":  "topic_id_sentiment",
        "sentiment_topic_name": "topic_label_sentiment",
    }
    for src, dst in alias_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
    return df


# ---------------------------------------------------------------------------
# Core pipeline function
# ---------------------------------------------------------------------------

def run_enrichment_pipeline(
    df: pd.DataFrame,
    text_col: str = "text",
    sentiment_col: str = "sentiment",
    spacy_model: str = "en_core_web_sm",
    keybert_model: str = "all-MiniLM-L6-v2",
    ner_batch_size: int = 64,
    top_n_keywords: int = 5,
) -> pd.DataFrame:
    """
    Full NLP enrichment pipeline (NER + Keywords + Topics).
    Expects sentiment to already be present in the DataFrame.

    Returns
    -------
    pd.DataFrame
        Fully enriched DataFrame with all NLP columns + downstream aliases.
    """
    _validate_input(df)
    total_start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("NLP ENRICHMENT PIPELINE STARTED  (%d comments)", len(df))
    logger.info("=" * 60)

    # ── Step 1: NER ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    logger.info("STEP 1/3 — Named Entity Recognition")
    df = run_ner_on_dataframe(
        df,
        text_col=text_col,
        model_name=spacy_model,
        batch_size=ner_batch_size,
    )
    logger.info("  ✓ NER done in %.1f s", time.perf_counter() - t0)

    # ── Step 2: Keyword Extraction ─────────────────────────────────────────
    t0 = time.perf_counter()
    logger.info("STEP 2/3 — Keyword Extraction (KeyBERT + YAKE)")
    df = run_keyword_extraction_on_dataframe(
        df,
        text_col=text_col,
        keybert_model_name=keybert_model,
        top_n=top_n_keywords,
    )
    logger.info("  ✓ Keywords done in %.1f s", time.perf_counter() - t0)

    # ── Step 3: Topic Modelling ────────────────────────────────────────────
    t0 = time.perf_counter()
    logger.info("STEP 3/3 — Topic Modelling (BERTopic × 4 models)")
    df = run_topic_modeling_on_dataframe(
        df,
        text_col=text_col,
        sentiment_col=sentiment_col,
    )
    logger.info("  ✓ Topics done in %.1f s", time.perf_counter() - t0)

    # ── Add downstream-compatible column aliases ───────────────────────────
    df = _add_compatibility_aliases(df)

    total_elapsed = time.perf_counter() - total_start
    logger.info("=" * 60)
    logger.info(
        "PIPELINE COMPLETE  |  %d rows  |  %d columns  |  %.1f s total",
        len(df), len(df.columns), total_elapsed,
    )
    logger.info("=" * 60)
    return df


# ---------------------------------------------------------------------------
# Save utility
# ---------------------------------------------------------------------------

def save_enriched_csv(
    df: pd.DataFrame,
    output_path: str | Path = "data/processed/comments_enriched.csv",
) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Saved enriched CSV → %s  (%d rows)", path, len(df))
    return path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NLP enrichment pipeline on sentiment-annotated YouTube comments."
    )
    parser.add_argument(
        "--input", "-i",
        default="data/processed/comments_with_sentiment.csv",
        help="Path to comments CSV with sentiment column already present.",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/processed/comments_enriched.csv",
    )
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--keybert-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info("Loading input CSV: %s", input_path)
    df = pd.read_csv(input_path)

    enriched = run_enrichment_pipeline(
        df,
        spacy_model=args.spacy_model,
        keybert_model=args.keybert_model,
        top_n_keywords=args.top_n,
        ner_batch_size=args.batch_size,
    )
    save_enriched_csv(enriched, args.output)


if __name__ == "__main__":
    main()
