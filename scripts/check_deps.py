"""Quick dependency check for start.ps1"""
import sys

MODS = ["pandas", "streamlit", "plotly", "faiss", "dspy", "sentence_transformers"]
missing = []
for m in MODS:
    try:
        __import__(m)
    except ImportError:
        missing.append(m)

if missing:
    print("MISSING:", ", ".join(missing))
    print("Run: .\\start.ps1 -Install")
    sys.exit(1)

print("   All core packages OK")
