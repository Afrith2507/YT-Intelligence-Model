from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge member datasets by stacking rows and removing duplicate comments."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of input CSV paths (absolute or relative to project root).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/merged/final_dataset.csv",
        help="Output CSV path (absolute or relative to project root).",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="comment_id",
        help="Unique ID column used for de-duplication. Falls back to 'id' if missing.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    project_root = Path(__file__).resolve().parents[1]
    return project_root / path


def detect_id_column(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    if "id" in df.columns:
        return "id"
    raise ValueError(
        f"Could not find ID column '{preferred}' or fallback 'id' in dataframe columns."
    )


def load_dataframes(input_paths: List[Path]) -> List[pd.DataFrame]:
    dfs = []
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        df = pd.read_csv(path)
        df["source_file"] = path.name
        dfs.append(df)
    return dfs


def main() -> None:
    args = parse_args()

    input_paths = [resolve_path(p) for p in args.inputs]
    output_path = resolve_path(args.output)

    dfs = load_dataframes(input_paths)
    merged = pd.concat(dfs, ignore_index=True)
    before_dedup = len(merged)

    id_column = detect_id_column(merged, args.id_column)
    merged = merged.drop_duplicates(subset=[id_column]).reset_index(drop=True)
    after_dedup = len(merged)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8")

    print("Merge complete.")
    print(f"Input files: {len(input_paths)}")
    print(f"Rows before dedup: {before_dedup}")
    print(f"Rows after dedup: {after_dedup}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
