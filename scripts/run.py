"""
run.py — minimal install + dashboard launch.
Usage: python scripts/run.py
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "streamlit", "pandas", "plotly", "scikit-learn",
    "numpy", "faiss-cpu", "dspy-ai", "vaderSentiment",
    "transformers", "torch", "sentence-transformers",
]

print("Installing dependencies …")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "--quiet", *REQUIRED]
)

print("\nLaunching dashboard …")
os.chdir(ROOT)
os.execv(
    sys.executable,
    [sys.executable, "-m", "streamlit", "run", "app/dashboard.py",
     "--server.headless", "true"],
)
