from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean raw YouTube comments and save a processed CSV."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/final_dataset.csv",
        help="Input CSV path (absolute or relative to project root).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/comments_clean.csv",
        help="Output CSV path (absolute or relative to project root).",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="comment_id",
        help="Unique comment ID column name. Falls back to 'id' if missing.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Text column to clean.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=3,
        help="Drop rows with fewer words than this threshold.",
    )
    parser.add_argument(
        "--english-only",
        action="store_true",
        help="Enable lightweight English heuristic filter.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    project_root = Path(__file__).resolve().parents[1]
    return project_root / path


def normalize_text(text: str) -> str:
    text = str(text)
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_emoji_or_symbol_heavy(text: str) -> bool:
    cleaned = re.sub(r"\s+", "", text)
    if not cleaned:
        return True
    alpha_num_count = len(re.findall(r"[A-Za-z0-9]", cleaned))
    # Mostly symbols/emojis/punctuation => likely not useful for NLP pipeline.
    return alpha_num_count / max(len(cleaned), 1) < 0.20


def has_excessive_repetition(text: str) -> bool:
    # Flags patterns like "loooool", "!!!!!", "hahahahahahahaha".
    repeated_char = re.search(r"(.)\1{7,}", text)
    repeated_token = re.search(r"\b(\w+)(?:\s+\1){5,}\b", text.lower())
    return bool(repeated_char or repeated_token)


def is_likely_english(text: str) -> bool:
    alpha_tokens = re.findall(r"[A-Za-z]+", text.lower())
    if not alpha_tokens:
        return False

    common_english = {
        "the",
        "and",
        "is",
        "to",
        "it",
        "this",
        "that",
        "you",
        "for",
        "with",
        "on",
        "was",
        "are",
        "his",
        "he",
        "i",
        "of",
        "in",
    }
    hits = sum(1 for token in alpha_tokens if token in common_english)
    alpha_ratio = len("".join(alpha_tokens)) / max(len(re.sub(r"\s+", "", text)), 1)

    # Heuristic: decent alphabetic ratio + at least one common English token.
    return alpha_ratio >= 0.45 and hits >= 1


def detect_id_column(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    if "id" in df.columns:
        return "id"
    raise ValueError(
        f"Could not find ID column '{preferred}' or fallback 'id' in dataset columns."
    )


def run_preprocessing(args: argparse.Namespace) -> Tuple[pd.DataFrame, dict]:
    input_path = resolve_path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    initial_rows = len(df)

    if args.text_column not in df.columns:
        raise ValueError(f"Text column '{args.text_column}' not found in input dataset.")

    id_column = detect_id_column(df, args.id_column)

    # Keep text as string and clean format.
    df[args.text_column] = df[args.text_column].astype(str).map(normalize_text)

    # Remove null/empty-like rows.
    df = df[df[args.text_column].notna()].copy()
    df = df[df[args.text_column].str.strip() != ""].copy()
    after_empty_filter = len(df)

    # Drop duplicate comment IDs and duplicate raw text.
    df = df.drop_duplicates(subset=[id_column]).copy()
    df = df.drop_duplicates(subset=[args.text_column]).copy()
    after_dedup = len(df)

    # Basic spam / low-signal removal.
    word_counts = df[args.text_column].str.split().str.len()
    df = df[word_counts >= args.min_words].copy()
    df = df[~df[args.text_column].map(is_emoji_or_symbol_heavy)].copy()
    df = df[~df[args.text_column].map(has_excessive_repetition)].copy()
    after_noise_filter = len(df)

    if args.english_only:
        df = df[df[args.text_column].map(is_likely_english)].copy()
    after_language_filter = len(df)

    stats = {
        "initial_rows": initial_rows,
        "after_empty_filter": after_empty_filter,
        "after_dedup": after_dedup,
        "after_noise_filter": after_noise_filter,
        "after_language_filter": after_language_filter,
    }
    return df.reset_index(drop=True), stats


def main() -> None:
    args = parse_args()
    cleaned_df, stats = run_preprocessing(args)

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False, encoding="utf-8")

    print("Preprocessing complete.")
    print(f"Input rows: {stats['initial_rows']}")
    print(f"After empty filter: {stats['after_empty_filter']}")
    print(f"After dedup: {stats['after_dedup']}")
    print(f"After noise filter: {stats['after_noise_filter']}")
    print(f"After language filter: {stats['after_language_filter']}")
    print(f"Saved cleaned dataset to: {output_path}")


if __name__ == "__main__":
    main()
