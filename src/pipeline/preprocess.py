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
        default="data/raw/merged/final_dataset.csv",
        help="Input CSV path (absolute or relative to project root).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/comments_clean.csv",
        help="Output CSV path (absolute or relative to project root).",
    )
    parser.add_argument(
        "--append-to",
        type=str,
        default=None,
        help=(
            "Path to existing cleaned CSV. New cleaned rows will be appended "
            "and deduplicated against it instead of overwriting."
        ),
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default=None,
        help="Unique comment ID column. Auto-detected if not set.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default=None,
        help="Text column to clean. Auto-detected if not set.",
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
    project_root = Path(__file__).resolve().parents[2]
    return project_root / path


ID_CANDIDATES = ["comment_id", "id", "commentId", "comment_Id"]
TEXT_CANDIDATES = ["text", "comment", "body", "content", "comment_text", "textDisplay"]


def detect_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise ValueError(
        f"Could not auto-detect {label} column. "
        f"Available columns: {list(df.columns)}. "
        f"Use --{label}-column to specify it manually."
    )


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
    return alpha_num_count / max(len(cleaned), 1) < 0.20


def has_excessive_repetition(text: str) -> bool:
    repeated_char = re.search(r"(.)\1{7,}", text)
    repeated_token = re.search(r"\b(\w+)(?:\s+\1){5,}\b", text.lower())
    return bool(repeated_char or repeated_token)


def is_likely_english(text: str) -> bool:
    alpha_tokens = re.findall(r"[A-Za-z]+", text.lower())
    if not alpha_tokens:
        return False
    common_english = {
        "the", "and", "is", "to", "it", "this", "that", "you",
        "for", "with", "on", "was", "are", "his", "he", "i", "of", "in",
        "a", "my", "so", "im", "its", "like", "just", "love", "ryan",
    }
    hits = sum(1 for token in alpha_tokens if token in common_english)
    alpha_ratio = len("".join(alpha_tokens)) / max(len(re.sub(r"\s+", "", text)), 1)
    return alpha_ratio >= 0.40 and hits >= 1


def run_preprocessing(
    df: pd.DataFrame,
    id_column: str,
    text_column: str,
    min_words: int,
    english_only: bool,
) -> Tuple[pd.DataFrame, dict]:
    initial_rows = len(df)

    df[text_column] = df[text_column].astype(str).map(normalize_text)

    df = df[df[text_column].notna()].copy()
    df = df[df[text_column].str.strip().str.lower() != "nan"].copy()
    df = df[df[text_column].str.strip() != ""].copy()
    after_empty_filter = len(df)

    df = df.drop_duplicates(subset=[id_column]).copy()
    df = df.drop_duplicates(subset=[text_column]).copy()
    after_dedup = len(df)

    word_counts = df[text_column].str.split().str.len()
    df = df[word_counts >= min_words].copy()
    df = df[~df[text_column].map(is_emoji_or_symbol_heavy)].copy()
    df = df[~df[text_column].map(has_excessive_repetition)].copy()
    after_noise_filter = len(df)

    if english_only:
        df = df[df[text_column].map(is_likely_english)].copy()
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

    input_path = resolve_path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path.name}")
    print(f"Columns detected: {list(df.columns)}")

    id_col = args.id_column or detect_column(df, ID_CANDIDATES, "id")
    text_col = args.text_column or detect_column(df, TEXT_CANDIDATES, "text")
    print(f"Using ID column: '{id_col}' | Text column: '{text_col}'")

    cleaned_df, stats = run_preprocessing(
        df=df,
        id_column=id_col,
        text_column=text_col,
        min_words=args.min_words,
        english_only=args.english_only,
    )

    if args.append_to:
        append_path = resolve_path(args.append_to)
        if append_path.exists():
            existing = pd.read_csv(append_path)
            before_append = len(existing)
            combined = pd.concat([existing, cleaned_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=[id_col]).reset_index(drop=True)
            combined = combined.drop_duplicates(subset=[text_col]).reset_index(drop=True)
            print(f"\nAppend mode: existing={before_append}, new={len(cleaned_df)}, after dedup={len(combined)}")
            cleaned_df = combined
        else:
            print(f"Append target not found at {append_path} — saving as new file.")

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False, encoding="utf-8")

    print("\nPreprocessing complete.")
    print(f"  Input rows:              {stats['initial_rows']}")
    print(f"  After empty filter:      {stats['after_empty_filter']}")
    print(f"  After dedup:             {stats['after_dedup']}")
    print(f"  After noise filter:      {stats['after_noise_filter']}")
    print(f"  After language filter:   {stats['after_language_filter']}")
    print(f"  Final saved rows:        {len(cleaned_df)}")
    print(f"  Saved to:                {output_path}")


if __name__ == "__main__":
    main()
