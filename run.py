"""
run.py — one command to install deps and launch the dashboard.
Usage:  python run.py
"""
import subprocess, sys, os

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
os.execv(
    sys.executable,
    [sys.executable, "-m", "streamlit", "run", "app/dashboard.py",
     "--server.headless", "true"],
)
