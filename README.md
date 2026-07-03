# Youtube Comments Intelligence

**CSCI370 Group Project | Spring 2026 | University of Wollongong in Dubai**

End-to-end NLP and RAG system for Ryan Trahan YouTube comments. The pipeline scrapes and enriches comments, then serves grounded Q&A, summaries, and audience insights through a Streamlit dashboard.

**Repository:** https://github.com/Afrith2507/YT-Intelligence-Model

---

## Features

- **Data pipeline:** scrape, merge, preprocess (`src/pipeline/`)
- **NLP enrichment:** VADER + RoBERTa sentiment, spaCy NER, KeyBERT + YAKE keywords, BERTopic (`src/nlp/`)
- **Hybrid RAG:** FAISS + BM25 + RRF, cross-encoder reranking, DSPy + Groq generation (`src/rag/`)
- **Agent routing:** QA, summarize, sentiment/topic/entity insight (`src/agent/`)
- **Evaluation:** Hit@k, MRR, NDCG, ROUGE-L, BERTScore, faithfulness + MLflow (`src/evaluation/`)
- **Dashboard:** analytics charts, Ask, Summarize, Evaluation tabs (`app/dashboard.py`)

---

## Prerequisites

- Python 3.10+
- Git
- PowerShell (Windows launcher)
- API keys in `.env`:
  - `GROQ_API_KEY` (required for Ask / Summarize)
  - `YOUTUBE_API_KEY` (required only for scraping)

**Dataset:** `data/processed/comments_enriched.csv` is required locally to run the dashboard. Data files are excluded from Git (see `.gitignore`). Share datasets within the team outside GitHub.

---

## Quick Start

```powershell
git clone https://github.com/Afrith2507/YT-Intelligence-Model.git
cd YT-Intelligence-Model

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python install.py
copy .env.example .env
```

Edit `.env` and add your API keys. Place `comments_enriched.csv` in `data/processed/` (or run the enrichment pipeline below).

```powershell
.\start.ps1
```

Open http://localhost:8501

Optional:

```powershell
.\start.ps1 -Install    # pip install then start
.\start.ps1 -Check      # dependency + smoke tests first
python src/evaluation/run_eval.py
```

---

## Full Pipeline (from raw data)

```powershell
# 1. Scrape (requires YOUTUBE_API_KEY)
python src/pipeline/scrape_youtube.py

# 2. Merge member CSVs
python src/pipeline/merge_member_datasets.py --inputs data/raw/member_submissions/*.csv --output data/raw/merged/final_dataset.csv

# 3. Preprocess
python src/pipeline/preprocess.py

# 4. NLP enrichment (sentiment, NER, keywords, topics)
python src/pipeline/run_full_pipeline.py

# 5. Build FAISS + BM25 index
python src/rag/indexer.py

# 6. Launch dashboard
.\start.ps1
```

Configuration: `config/config.yaml`

---

## Project Structure

```text
.
├── app/
│   └── dashboard.py              # Streamlit UI
├── config/
│   └── config.yaml               # Central settings
├── data/
│   ├── eval/
│   │   └── test_queries.json     # Evaluation queries (tracked in Git)
│   ├── processed/                # Enriched CSV (local only, gitignored)
│   ├── raw/                      # Scrapes and member submissions (gitignored)
│   └── vectorstore/              # FAISS index (gitignored)
├── docs/
│   ├── scope.md
│   ├── team_data_protocol.md
│   └── CSCI370_Project_Report.md
├── scripts/                      # Utilities and smoke tests
├── src/
│   ├── agent/                    # Intent router and orchestrator
│   ├── evaluation/               # Metrics and MLflow tracking
│   ├── nlp/                      # Sentiment, NER, keywords, topics
│   ├── pipeline/                 # Scrape, merge, preprocess, enrich
│   └── rag/                      # Index, retrieve, rerank, generate
├── .env.example
├── install.py
├── requirements.txt
└── start.ps1
```

---

## Team

| Member | Area |
|--------|------|
| Afrith | Dashboard, RAG, agent |
| Daniel | Pipeline, config, data collection |
| Feliks | NLP enrichment |
| Eshfar | Evaluation, MLflow |

See `docs/team_data_protocol.md` for data sharing rules.

---

## Notes

- Do not commit `.env`, CSV datasets, or `mlflow.db`.
- Default LLM: Groq `llama-3.1-8b-instant` (set in `start.ps1` and `.env`).
- Default retrieval mode: hybrid (FAISS + BM25 + RRF).
