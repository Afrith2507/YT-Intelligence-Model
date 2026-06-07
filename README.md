# YouTube Intelligence Engine — Ryan Trahan Audience Analytics

**CSCI370 Group Project | Spring 2026**

An end-to-end NLP and RAG system that transforms YouTube comments into structured insights, grounded answers, and interactive analytics — powered by dual sentiment models, BERTopic, hybrid retrieval, LLM agents, and MLflow monitoring.

---

## System Architecture

```
YouTube API
    │
    ▼
┌──────────────┐
│   Scraper    │  per member: 10k comments
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Merge +     │  deduplicated master dataset
│  Preprocess  │
└──────┬───────┘
       │
       ├──────────────────────────────────────────────────────┐
       ▼                                                      ▼
┌──────────────┐                                   ┌──────────────────┐
│  NLP Layer   │                                   │   RAG Indexer    │
│              │                                   │                  │
│  Sentiment   │  VADER + RoBERTa                  │  Embeddings      │
│  NER         │  spaCy en_core_web_sm             │  ChromaDB        │
│  Keywords    │  KeyBERT + YAKE ensemble           │  BM25 Index      │
│  Topics      │  BERTopic (overall + per-sentiment)│                  │
└──────┬───────┘                                   └────────┬─────────┘
       │                                                    │
       └──────────────────────────┬─────────────────────────┘
                                  ▼
                        ┌──────────────────┐
                        │  Agent Router    │
                        │                  │
                        │  Intent: qa      │
                        │         summary  │
                        │         insight  │
                        │         entity   │
                        │         topic    │
                        └────────┬─────────┘
                                 │
                    ┌────────────┼────────────────┐
                    ▼            ▼                 ▼
             ┌──────────┐ ┌──────────┐    ┌──────────────┐
             │ Retriever│ │ Retriever│    │  Analytics   │
             │ semantic │ │  BM25    │    │  Tools       │
             │  + RRF   │ │ hybrid   │    └──────────────┘
             └────┬─────┘ └────┬─────┘
                  └─────┬──────┘
                        ▼
                 ┌──────────────┐
                 │  Reranker    │  cross-encoder top-5
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │  LLM (GPT)   │  structured prompts + citations
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │  Dashboard   │  Streamlit multi-tab
                 └──────────────┘
                        │
                 ┌──────────────┐
                 │   MLflow     │  experiment tracking
                 └──────────────┘
```

---

## Repository Structure

```text
csci370-youtube-intelligence/
│
├── README.md
├── requirements.txt
├── config.yaml                          ← centralized config for all modules
├── .env.example                         ← API key template
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── member_submissions/          ← one 10k CSV per member
│   │   └── merged/                      ← combined master dataset
│   ├── processed/
│   │   ├── comments_clean.csv
│   │   ├── comments_enriched.csv        ← all NLP outputs merged
│   │   └── topic_model/                 ← saved BERTopic artifacts
│   ├── vectorstore/                     ← ChromaDB persistent store
│   └── eval/
│       └── test_queries.json            ← evaluation query-answer pairs
│
├── src/
│   ├── pipeline/
│   │   ├── scrape_youtube.py            ← YouTube Data API scraper
│   │   ├── merge_member_datasets.py     ← stack + dedup member CSVs
│   │   ├── preprocess.py                ← full cleaning pipeline
│   │   └── run_full_pipeline.py         ← runs everything in order
│   │
│   ├── nlp/
│   │   ├── sentiment.py                 ← VADER + RoBERTa dual inference
│   │   ├── ner.py                       ← spaCy NER + entity aggregation
│   │   ├── keywords.py                  ← KeyBERT + YAKE ensemble
│   │   └── topics.py                    ← BERTopic overall + per sentiment
│   │
│   ├── rag/
│   │   ├── indexer.py                   ← chunk + embed + store ChromaDB + BM25
│   │   ├── retriever.py                 ← semantic / BM25 / hybrid RRF
│   │   ├── reranker.py                  ← cross-encoder reranking
│   │   ├── generator.py                 ← LLM QA + summarization with citations
│   │   └── prompts.py                   ← all structured prompt templates
│   │
│   ├── agent/
│   │   ├── router.py                    ← intent classifier + query routing
│   │   ├── tools.py                     ← QA, Summary, Sentiment, Topic, Entity tools
│   │   └── orchestrator.py              ← agent loop + tool dispatch
│   │
│   └── evaluation/
│       ├── retrieval_eval.py            ← Hit@k, MRR, NDCG@k
│       ├── generation_eval.py           ← ROUGE-L, BERTScore, faithfulness check
│       ├── mlflow_tracker.py            ← experiment logging wrapper
│       └── run_eval.py                  ← full evaluation runner
│
├── notebooks/
│   ├── 01_eda.ipynb                     ← comment distribution, length, likes
│   ├── 02_sentiment_analysis.ipynb      ← VADER vs RoBERTa comparison + charts
│   ├── 03_ner_keywords.ipynb            ← entity frequency, keyword clouds
│   ├── 04_topic_modeling.ipynb          ← BERTopic visualizations, per-sentiment topics
│   └── 05_rag_evaluation.ipynb          ← retrieval comparison, answer quality
│
├── app/
│   ├── dashboard.py                     ← Streamlit entry point
│   └── pages/
│       ├── 01_analytics.py              ← sentiment/topics/NER charts
│       ├── 02_qa.py                     ← RAG QA with cited sources
│       ├── 03_summarize.py              ← video/topic summarization
│       └── 04_retrieval_compare.py      ← side-by-side retrieval mode comparison
│
├── tests/
│   ├── test_preprocess.py
│   ├── test_retriever.py
│   └── test_router.py
│
└── docs/
    ├── team_data_protocol.md
    └── evaluation_results.md
```

---

## Setup

### 1) Clone the repo
```powershell
git clone https://github.com/<your-username>/csci370-youtube-intelligence.git
cd csci370-youtube-intelligence
```

### 2) Create and activate virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies
```powershell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4) Set environment variables
Copy `.env.example` to `.env` and fill in your keys:
```env
YOUTUBE_API_KEY=your_youtube_api_key
OPENAI_API_KEY=your_openai_api_key
```

---

## Running the Pipeline

### Step 1 — Data collection (each member runs independently)
```powershell
python .\src\pipeline\scrape_youtube.py --mode standard --channel-query "Ryan Trahan" --max-videos 40 --max-comments-per-video 300 --output "data/raw/member_submissions/member_a_10k.csv"
```

### Step 2 — Merge member datasets
```powershell
python .\src\pipeline\merge_member_datasets.py --inputs "data/raw/member_submissions/member_a_10k.csv" "data/raw/member_submissions/member_b_10k.csv" --output "data/raw/merged/final_dataset.csv"
```

### Step 3 — Preprocess
```powershell
python .\src\pipeline\preprocess.py --input "data/raw/merged/final_dataset.csv" --output "data/processed/comments_clean.csv" --english-only
```

### Step 4 — Run full NLP pipeline
```powershell
python .\src\pipeline\run_full_pipeline.py
```

This runs sentiment, NER, keywords, and topic modeling and saves `data/processed/comments_enriched.csv`.

### Step 5 — Build RAG index
```powershell
python .\src\rag\indexer.py
```

### Step 6 — Run evaluation
```powershell
python .\src\evaluation\run_eval.py
```

### Step 7 — Launch dashboard
```powershell
streamlit run app/dashboard.py
```

---

## NLP Components

| Component | Models / Libraries | Output |
|---|---|---|
| Sentiment | VADER + `cardiffnlp/twitter-roberta-base-sentiment-latest` | `sentiment_vader`, `sentiment_roberta`, `sentiment_score` |
| NER | spaCy `en_core_web_sm` | Entities per comment (PERSON, ORG, PRODUCT, ...) |
| Keywords | KeyBERT + YAKE ensemble (deduplicated) | Top-5 keywords per comment |
| Topic Modeling | BERTopic (overall + per sentiment) | `topic_id`, topic label, coherence score |

---

## RAG Pipeline

| Stage | Method |
|---|---|
| Chunking | One comment = one chunk + metadata (video_id, sentiment, topic) |
| Embedding | `all-MiniLM-L6-v2` via sentence-transformers |
| Vector store | ChromaDB (persistent) |
| Lexical index | BM25 (rank-bm25) |
| Retrieval modes | Semantic / BM25 / Hybrid RRF |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` (top-20 → top-5) |
| Generation | GPT-4o-mini with structured prompts + citations |

---

## Agent Routing

| Intent | Routed To |
|---|---|
| `qa` | Retriever → Reranker → LLM QA tool |
| `summarize` | Retriever → LLM Summarization tool |
| `sentiment_insight` | Sentiment analytics tool |
| `topic_insight` | Topic analytics tool |
| `entity_insight` | NER aggregation tool |

---

## Evaluation Metrics

| Type | Metrics |
|---|---|
| Retrieval | Hit@1, Hit@3, Hit@5, MRR, NDCG@5 |
| Generation | ROUGE-L, BERTScore, Faithfulness score |
| System | Latency per query, retrieval mode comparison |
| Monitored via | MLflow (all experiments tracked and logged) |

---

## Team Contribution

Each member independently collects 10,000 comments.

| Member | File | Scope |
|---|---|---|
| Member A | `member_a_10k.csv` | Recent videos |
| Member B | `member_b_10k.csv` | Most viewed videos |
| Member C | `member_c_10k.csv` | Viral/trending videos |

Commit regularly with descriptive messages:
- `data: member_a collects 10k from recent videos`
- `nlp: add RoBERTa sentiment to enriched dataset`
- `rag: add hybrid retrieval with RRF fusion`
- `eval: log retrieval metrics to MLflow`

---

## Deliverables Checklist

- [ ] Dataset (10k per member with evidence)
- [ ] Preprocessing pipeline with quality report
- [ ] Dual sentiment analysis (VADER + RoBERTa)
- [ ] NER with entity aggregation
- [ ] Keyword extraction (KeyBERT + YAKE)
- [ ] Topic modeling (BERTopic overall + per sentiment)
- [ ] RAG indexing (semantic + BM25 + hybrid)
- [ ] Cross-encoder reranking
- [ ] LLM QA + summarization with citations
- [ ] Agent routing with 5 intent classes
- [ ] Evaluation (Hit@k, MRR, NDCG, ROUGE, BERTScore, faithfulness)
- [ ] MLflow experiment tracking
- [ ] Multi-tab Streamlit dashboard
- [ ] Jupyter notebooks with visualizations
- [ ] Report (8-section structure)
- [ ] GitHub repo with clear commit history
- [ ] Presentation (10 min demo + 5 min QA)
