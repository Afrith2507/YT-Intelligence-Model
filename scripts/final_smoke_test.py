"""Quick pre-push smoke test. Run: python scripts/final_smoke_test.py"""
from __future__ import annotations

import os
import sys
import py_compile

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

FAIL = 0


def check(label: str, fn) -> None:
    global FAIL
    try:
        fn()
        print(f"  OK  {label}")
    except Exception as e:
        FAIL += 1
        print(f"  FAIL {label}: {e}")


def main() -> None:
    print("=== Youtube Comments Intelligence — smoke test ===\n")

    check("dashboard.py compiles", lambda: py_compile.compile("app/dashboard.py", doraise=True))

    csv = os.path.join("data", "processed", "comments_enriched.csv")

    def _csv_exists() -> None:
        if not os.path.isfile(csv):
            raise FileNotFoundError(csv)

    check("enriched CSV exists", _csv_exists)

    def _load_csv():
        import pandas as pd
        df = pd.read_csv(csv, nrows=50)
        assert len(df) > 0
        assert "text" in df.columns

    check("CSV loads (pandas)", _load_csv)

    def _imports():
        from src.rag.prompts import getAnswer  # noqa: F401
        from src.evaluation.run_eval import run_full_evaluation  # noqa: F401

    check("core modules import", _imports)

    print()
    if FAIL:
        print(f"FAILED — {FAIL} check(s)")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
