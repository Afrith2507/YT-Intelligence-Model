# Grader / professor setup

After clone, the LLM works without adding API keys manually.

## Steps

**Important:** Run commands in **PowerShell** or **Terminal** — do not double-click `start.ps1` (it opens as a text file on some PCs).

```powershell
git clone https://github.com/Afrith2507/YT-Intelligence-Model.git
cd YT-Intelligence-Model
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python install.py
.\start.ps1
```

**Or** double-click `start.bat` after install.

**Or:**

```powershell
python -m streamlit run app/dashboard.py
```

Open http://localhost:8501

## What is already in the repo

- `.env` — Groq and YouTube API keys (Ask / Summarize / evaluation)
- `data/processed/comments_enriched.csv` — full enriched dataset
- `data/raw/*.csv` — raw member scrapes

## Test the LLM

1. Go to **Ask** tab
2. Question: `What do viewers say about Ryan's livestreams?`
3. You should get an answer with source comments

## Test evaluation

Analytics tab → check **Include BERTScore** → Run evaluation

Or terminal:

```powershell
pip install bert-score
python src/evaluation/run_eval.py --bertscore
```
