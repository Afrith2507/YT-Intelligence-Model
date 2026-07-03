"""
organize_data.py
Run once from the project root to move files into the correct directories.

    python scripts/organize_data.py
"""
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def move(src_rel: str, dst_rel: str):
    src = BASE / src_rel
    dst = BASE / dst_rel
    if not src.exists():
        print(f"  [skip]  {src_rel}  (not found)")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    print(f"  [moved] {src_rel}  →  {dst_rel}")


def main():
    print("=" * 55)
    print("  DATA ORGANISATION")
    print("=" * 55)

    print("\n[1] Moving unclean files to data/raw/")
    move("data/final_1.csv", "data/raw/final_1.csv")
    move("data/final_2.csv", "data/raw/final_2.csv")
    move("data/final_unclean.csv", "data/raw/final_unclean.csv")

    print("\n[2] Creating missing directories")
    for d in ["data/raw/merged", "data/vectorstore"]:
        path = BASE / d
        path.mkdir(parents=True, exist_ok=True)
        gk = path / ".gitkeep"
        if not gk.exists():
            gk.touch()
        print(f"  [ok]    {d}/")

    print("\n[3] Removing ghost files")
    ghost = BASE / "data" / "raw" / ".csv"
    if ghost.exists():
        ghost.unlink()
        print("  [removed] data/raw/.csv")
    else:
        print("  [skip]  data/raw/.csv  (not found)")

    print("\n" + "=" * 55)
    print("  DONE — data directory is now clean.")
    print("=" * 55)


if __name__ == "__main__":
    main()
