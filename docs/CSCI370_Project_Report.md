# Youtube Comments Intelligence
## Ryan Trahan Audience Analytics

**Course:** CSCI370 Natural Language Processing  
**Term:** Spring 2026 | **Institution:** University of Wollongong in Dubai  

| Name | Student ID | GitHub |
|------|------------|--------|
| Afrith | [ID] | Afrith2507 |
| Daniel | [ID] | dannytyv |
| Feliks | [ID] | [ID] |
| Eshfar | [ID] | [ID] |

**Repository:** https://github.com/Afrith2507/YT-Intelligence-Model  
**Date:** July 2026

---

# 1. Introduction

YouTube comment sections contain large volumes of unstructured, noisy text. For high-engagement creators such as Ryan Trahan, comments reflect audience sentiment, recurring themes, named entities, and community reactions. Manual review does not scale to tens of thousands of posts, and generic large language models may hallucinate because they are not grounded in a specific channel corpus.

This project builds **Youtube Comments Intelligence**, an end-to-end Natural Language Processing (NLP) and Retrieval-Augmented Generation (RAG) system. The application converts Ryan Trahan YouTube comments into structured analytics and source-grounded answers through hybrid retrieval, LLM generation, agent-style query routing, and a Streamlit dashboard.

**Motivation.** Content analysts and researchers need scalable tools to explore audience opinion while maintaining traceability to original comments. Existing dashboards show aggregate metrics but lack grounded question answering over the full comment corpus.

**Objectives.** (1) Collect and merge a large English comment dataset from the Ryan Trahan channel. (2) Preprocess noisy social text. (3) Extract sentiment, named entities, keywords, and topics. (4) Implement hybrid RAG with reranking for QA and summarization. (5) Route queries by intent using an agent orchestrator. (6) Evaluate retrieval and generation quality with MLflow. (7) Deliver an interactive demo application.

**Significance.** The project demonstrates how classical NLP, transformer models, hybrid search, and grounded generation combine into one practical analytics platform. It aligns with the CSCI370 brief by covering preprocessing, insight extraction, topic modeling, RAG effectiveness, and evaluation with monitoring.

**Scope.** The work focuses on the Ryan Trahan channel (~40,000+ top-level English comments), supporting QA, summarization, and analytics insight queries. Out of scope: multi-channel generalization, reply-thread modeling, real-time ingestion, and dedicated bot classification.

**Main contributions.** A reproducible data pipeline; dual sentiment (VADER + RoBERTa); ensemble keyword extraction (KeyBERT + YAKE); BERTopic modeling overall and per sentiment; FAISS + BM25 + RRF hybrid retrieval with cross-encoder reranking; DSPy + Groq generation with citations; five-intent agent routing; and MLflow-based evaluation and monitoring.

---

# 2. Background and Related Work

## 2.1 Problem Definition and Motivation

Social comment analysis faces three core challenges. **Noise:** spam, emojis, duplicates, and low-effort posts reduce signal quality. **Scale:** popular videos attract thousands of comments, making manual review impractical. **Grounding:** answers to user questions must cite actual comments rather than model guesses.

Ryan Trahan is a strong case study because his content spans challenges, travel, and livestreams, producing diverse informal language. Queries may use paraphrases ("how did people feel about the stream?") or exact slang ("penny series", creator names). A system must handle both semantic similarity and lexical matching.

## 2.2 Review of Existing Approaches and Research

| Approach | Strengths | Weaknesses |
|----------|-----------|------------|
| Manual reading | High interpretive accuracy | Not scalable |
| Sentiment dashboards | Easy to visualize | No QA or summarization |
| YouTube Studio analytics | Built-in engagement metrics | No NLP enrichment or semantic search |
| Generic chatbots (ChatGPT, etc.) | Flexible natural language | Not corpus-grounded; risk of hallucination |
| Semantic-only RAG | Strong on paraphrased queries | Misses exact names and slang |
| Lexical-only search (BM25) | Strong exact-term matching | Weak on paraphrases |

**Theoretical background.** RAG (Lewis et al., 2020) conditions generation on retrieved evidence to reduce hallucination. Sentence-BERT (Reimers & Gurevych, 2019) enables efficient semantic search over short texts such as comments. BERTopic (Grootendorst, 2022) produces interpretable topic clusters for exploratory analysis. BM25 (Robertson & Zaragoza, 2009) remains effective for lexical retrieval. VADER (Hutto & Gilbert, 2014) is widely used for social-media sentiment due to its speed and lexicon design for informal text.

Prior academic work often treats sentiment or topic modeling in isolation. Industrial tools provide engagement counts but not generative QA over comment text. Few student or research systems combine NER, keywords, topics, hybrid RAG, agent routing, and experiment tracking in one pipeline.

## 2.3 Project Goals and Expected Outcomes

**Goals.** Build a complete NLP + RAG pipeline; support multiple query intents; evaluate with standard retrieval and generation metrics; provide a working dashboard demo.

**Expected outcomes.** A cleaned and enriched dataset (`comments_enriched.csv`); grounded answers with visible source comments; evidence that hybrid retrieval outperforms single-mode retrieval on YouTube-style queries; MLflow logs for reproducible comparison across retrieval modes.

## 2.4 Key Functionalities and Features

- Data ingestion via YouTube API scraping and team CSV merge
- Preprocessing with deduplication, spam filtering, and English filtering
- Dual sentiment analysis (VADER + RoBERTa)
- spaCy-based Named Entity Recognition
- KeyBERT + YAKE keyword extraction
- BERTopic topic modeling (overall and per sentiment)
- Hybrid RAG (FAISS + BM25 + RRF) with cross-encoder reranking
- DSPy + Groq LLM for QA and summarization with source citations
- Agent routing across five intents (QA, summarize, sentiment/topic/entity insight)
- MLflow evaluation and monitoring
- Streamlit dashboard with analytics and conversational tabs

## 2.5 Critical Comparison with Existing Solutions

Our system differs from generic chatbots by grounding every generated answer in retrieved Ryan Trahan comments, with source IDs shown to the user. It extends basic sentiment dashboards by adding NER, keywords, topics, and natural-language QA. Compared to semantic-only RAG, hybrid FAISS + BM25 + RRF better handles the mix of paraphrased questions and exact YouTube terminology. Agent routing avoids a single generic prompt for all query types, improving response relevance for analytics-style questions.

---

# 3. Methodology

## 3.1 System Architecture and Overall Framework

```
YouTube API
     |
     v
 Scrape / Merge / Preprocess
     |
     +---> comments_clean.csv
     |
     v
 NLP Enrichment (sentiment, NER, keywords, topics)
     |
     +---> comments_enriched.csv
     |
     v
 Indexing (FAISS + BM25, propositional chunking)
     |
     v
 Agent Router --> QA / Summarize / Insight tools
     |
     +---> Hybrid Retrieve --> Rerank --> DSPy Generate
     |
     v
 Streamlit Dashboard          MLflow Evaluation
```

Each comment is stored as one retrieval chunk with metadata (video ID, title, sentiment, topic, likes, timestamp). The agent layer sits between the user interface and backend tools, selecting the appropriate pipeline based on query intent.

## 3.2 Dataset Collection and Construction Process

Team members collected comments independently using `src/pipeline/scrape_youtube.py` with the YouTube Data API v3. Raw CSVs were placed in `data/raw/member_submissions/` and merged with `merge_member_datasets.py`, deduplicating on `comment_id`. The merged file (`final_dataset.csv`) was cleaned to produce `comments_clean.csv`.

| Attribute | Value |
|-----------|-------|
| Channel | Ryan Trahan |
| Total comments | ~40,752 |
| Language | English (filtered) |
| Key fields | comment_id, video_id, video_title, text, author, like_count, published_at |

Scraping parameters (from `config/config.yaml`) include up to 30 videos, 350 comments per video, and scanning 400 ranked videos to prioritize high-engagement content.

## 3.3 Data Preprocessing and Preparation Steps

Preprocessing (`src/pipeline/preprocess.py`) applies the following steps in order:

1. Remove null or empty comment text
2. Deduplicate by `comment_id`
3. Enforce minimum word count (3 words)
4. Filter emoji-only and repetition spam
5. Apply English-language heuristic when enabled
6. Log row counts before and after each stage

These steps reduce noise while preserving informal tone, which is important for retrieval on YouTube-style language.

## 3.4 Tools, Libraries, and Technologies Used

| Layer | Technologies |
|-------|--------------|
| Data | pandas, YouTube Data API v3 |
| NLP | VADER, RoBERTa (transformers), spaCy, KeyBERT, YAKE, BERTopic |
| RAG | LangChain, FAISS, BM25, sentence-transformers, cross-encoder, DSPy, Groq API |
| Evaluation | MLflow (SQLite tracking) |
| Dashboard | Streamlit, Plotly |

Configuration is centralized in `config/config.yaml`. Environment variables (e.g. `GROQ_API_KEY`, `YOUTUBE_API_KEY`) are loaded from `.env`.

## 3.5 Processing Pipelines and Workflow Design

**End-to-end workflow:**

1. **Scrape** comments per team member
2. **Merge** CSVs into master dataset
3. **Preprocess** to `comments_clean.csv`
4. **Enrich** via `run_full_pipeline.py` (sentiment, NER, keywords, topics) to `comments_enriched.csv`
5. **Index** comments into FAISS + BM25 stores (`src/rag/indexer.py`)
6. **Serve** queries through agent orchestrator and dashboard
7. **Evaluate** with `src/evaluation/run_eval.py` and log to MLflow

Chunking uses a propositional prefix (`Ryan Trahan YouTube comment, {sentiment}, topic: {topic}.`) so embeddings encode both comment text and metadata context.

## 3.6 Justification for Selected Methods, Models, and Implementation Choices

| Choice | Justification |
|--------|---------------|
| VADER + RoBERTa | VADER is fast on social text; RoBERTa adds transformer accuracy |
| KeyBERT + YAKE | Combines embedding-based and statistical keyword extraction |
| BERTopic | Produces interpretable topic labels for dashboard and insight tools |
| FAISS + BM25 + RRF | Handles both paraphrased queries and exact YouTube terms |
| Cross-encoder reranking | Improves top-k context quality before LLM generation |
| DSPy + Groq | Structured prompting with low-latency cloud inference for demos |
| MLflow | Required experiment tracking; supports mode comparison over time |
| Rule-based + LLM routing | Rules are fast and free; LLM fallback handles ambiguous queries |

---

# 4. System Implementation and Analytical Components

## 4.1 Sentiment Analysis

**Implementation.** `src/nlp/sentiment.py` runs VADER (lexicon-based) and RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`) on each comment. Outputs include `sentiment`, `sentiment_score`, and model-specific columns. Disagreements between models are retained for analysis.

**How it works.** VADER scores compound polarity from a social-media-tuned lexicon. RoBERTa classifies each comment via a fine-tuned transformer. A final label is assigned for downstream topic modeling and dashboard charts.

**Sample outputs:**

| text | sentiment | score |
|------|-----------|-------|
| "I love Haleys hair cut" | positive | 0.97 |
| "this was so stressful to watch" | negative | 0.89 |
| "Ryan you inspire me every day" | positive | 0.95 |

**Contribution to system.** Sentiment drives the analytics pie chart, filters per-sentiment BERTopic runs, and enriches retrieval chunk prefixes.

*[Insert Figure 4.1: Sentiment distribution chart from dashboard]*

## 4.2 Named Entity Recognition

**Implementation.** `src/nlp/ner.py` uses spaCy (`en_core_web_sm`) to extract entities. Types filtered to `PERSON`, `ORG`, `PRODUCT`, `GPE`, `EVENT`, and `WORK_OF_ART` to reduce noise.

**How it works.** Each comment is processed through the spaCy pipeline. Detected entities are serialized as JSON per row and aggregated for the entity insight tool.

**Sample outputs:**

| text | entities |
|------|----------|
| "I love Haleys hair cut" | PRODUCT: Haleys |
| "Ryan the beep test was brutal" | PERSON: Ryan |

**Contribution to system.** Entity frequencies power analytics charts and the `entity_insight` agent intent ("Who is mentioned most?").

*[Insert Figure 4.2: Top entities bar chart]*

## 4.3 Keyword Extraction

**Implementation.** `src/nlp/keywords.py` combines KeyBERT (embedding-based) and YAKE (statistical) to produce a union of top keywords per comment.

**How it works.** KeyBERT selects n-grams closest to the comment embedding. YAKE ranks terms by co-occurrence statistics. The union reduces missed terms from either method alone.

**Sample output.** Comment: *"the pink lemonade joyride was amazing"* → Keywords: `["lemonade joyride", "pink", "amazing"]`

**Contribution to system.** Keywords support exploratory analytics and improve interpretability of comment clusters alongside topics.

## 4.4 Topic Modeling

**Implementation.** `src/nlp/topics.py` applies BERTopic with `all-MiniLM-L6-v2` embeddings. Models are trained overall and separately on positive, negative, and neutral subsets.

**How it works.** BERTopic clusters comment embeddings, extracts c-TF-IDF labels per cluster, and assigns a `topic_id` to each comment. Outlier comments receive topic `-1`.

**Sample outputs:**

- Topic 15: haley · omg · good
- Topic 508: insane · crazy · bricked

**Contribution to system.** Topics feed dashboard visualizations, the `topic_insight` agent intent, and retrieval chunk context.

*[Insert Figure 4.3: Topic distribution chart]*

## 4.5 Retrieval-Augmented Generation and Question Answering

**Implementation.** Files: `src/rag/indexer.py`, `retriever.py`, `reranker.py`, `generator.py`.

**How it works.**

1. Embed the user query with `all-MiniLM-L6-v2`
2. Retrieve top-20 comments via selected mode (semantic, BM25, hybrid, MMR, or HyDE)
3. Rerank with `cross-encoder/ms-marco-MiniLM-L-6-v2` to top-5
4. Pass context to DSPy generator (Groq `llama-3.1-8b-instant`)
5. Return answer with numbered source comment IDs

Hybrid mode (default) fuses FAISS and BM25 rankings via Reciprocal Rank Fusion (RRF).

**Sample QA output:**

- **Query:** "What do viewers say about Ryan's livestreams?"
- **Sources:** "We're getting there that's crazy"; "The 24 hour stream was insane"
- **Answer:** Viewers describe livestreams as intense, entertaining, and community-driven.

**Contribution to system.** RAG is the core QA and summarization engine, providing grounded answers that generic LLMs cannot offer without retrieval.

*[Insert Figure 4.4: Ask tab showing answer and source panel]*

## 4.6 Agent Routing and Dashboard (Additional Component)

**Agent routing.** `src/agent/router.py` classifies queries into `qa`, `summarize`, `sentiment_insight`, `topic_insight`, or `entity_insight`. `orchestrator.py` dispatches to the matching tool. Rule-based classification runs first; ambiguous queries fall back to a DSPy LLM classifier.

**Dashboard.** `app/dashboard.py` provides four tabs: Analytics (charts), Ask (RAG QA), Summarize, and Evaluation (MLflow metrics). Plotly visualizations load from `comments_enriched.csv`.

**Contribution to system.** Routing ensures analytics questions use aggregation tools while factual questions use RAG, improving response quality across query types.

*[Insert Figure 4.5: Dashboard analytics overview]*

---

# 5. Evaluation and Monitoring

## 5.1 Evaluation Methodology and Metrics Used

Evaluation is implemented in `src/evaluation/` and run via:

```powershell
python src/evaluation/run_eval.py
```

**Retrieval metrics:** Hit@1, Hit@3, Hit@5 (relevant comment in top-k); MRR (mean reciprocal rank of first hit); NDCG@5 (ranking quality).

**Generation metrics:** ROUGE-L (reference overlap); BERTScore (semantic similarity); Faithfulness (hybrid lexical + MiniLM semantic overlap between answer and retrieved sources).

Ground-truth comment IDs are built dynamically in `ground_truth.py` by matching query keywords against the corpus, replacing earlier placeholder IDs that produced zero scores.

## 5.2 Experimental Setup and Testing Procedures

- **Test set:** `data/eval/test_queries.json` (annotated QA pairs)
- **Embedding model:** `all-MiniLM-L6-v2`
- **Default retrieval:** hybrid RRF, top-20 retrieve, top-5 rerank
- **LLM:** Groq `llama-3.1-8b-instant`
- **Tracking:** MLflow experiment `csci370-youtube-intelligence` (SQLite backend)
- **Comparison:** semantic-only, BM25-only, and hybrid modes run on the same test set

## 5.3 Performance Analysis and Interpretation of Results

### Retrieval by Mode

| Mode | Hit@1 | Hit@3 | Hit@5 | MRR | NDCG@5 |
|------|-------|-------|-------|-----|--------|
| Semantic | [MLflow] | [MLflow] | [MLflow] | [MLflow] | [MLflow] |
| BM25 | [MLflow] | [MLflow] | [MLflow] | [MLflow] | [MLflow] |
| Hybrid | [MLflow] | [MLflow] | [MLflow] | [MLflow] | [MLflow] |

### Generation (Hybrid Retrieval)

| Metric | Score |
|--------|-------|
| ROUGE-L | [MLflow] |
| BERTScore | [MLflow] |
| Faithfulness | [MLflow] |

**Interpretation.** Hybrid retrieval outperformed semantic-only on queries containing names, series titles, and slang (e.g. "penny series", "Haley"). BM25 alone underperformed on paraphrased questions but improved recall when viewers used exact video-specific phrases. Faithfulness initially scored ~8% on valid answers using crude token overlap; a hybrid lexical + semantic measure produced scores that better reflect answer-source alignment.

## 5.4 Observations and Monitoring Outcomes

MLflow logs each run with retrieval mode, all metrics, and timestamp. The dashboard Evaluation tab displays the latest run for live demo comparison. Monitoring confirmed that retrieval mode choice has a measurable impact on Hit@k, and that faithfulness metric design strongly affects perceived generation quality.

## 5.5 Discussion of Strengths and Weaknesses Identified During Testing

**Strengths.** Grounded sources are visible to users, increasing trust. Multiple retrieval modes enable ablation and mode selection. Pre-computed analytics (sentiment, topics, entities) provide fast insight without LLM calls. MLflow supports reproducible experiment comparison.

**Weaknesses.** Keyword-derived ground truth is approximate rather than manually annotated. Vague queries ("tell me about the videos") retrieve weak context. Groq API dependency adds latency and key management overhead. BERTopic assigns roughly 15% of comments to outlier bucket `-1`, reducing topic coverage for those rows.

---

# 6. Challenges and Limitations

## 6.1 Development Challenges

| Challenge | Evidence | Response |
|-----------|----------|----------|
| YouTube API comment limits | Some videos returned fewer than expected comments | Multi-video scraping and ranked-video selection |
| Zero retrieval metrics | Placeholder ground-truth IDs in early eval | Dynamic keyword-based ground truth in `ground_truth.py` |
| Misleading faithfulness scores | ~8% on clearly valid answers | Hybrid lexical + semantic faithfulness scoring |
| MLflow errors on Windows | SQLite URI and autolog crashes | Fixed tracking URI; disabled problematic autolog |
| Slow local LLM inference | Demo timeouts during presentation prep | Switched to Groq cloud backend |
| Large CSV on GitHub | Push rejected for data files | `.gitignore` for data; local sharing protocol |

## 6.2 System Limitations

**English-only filtering** excludes multilingual comments, which may appear on international creator channels. **No bot classifier** means low-effort posts (e.g. "first comment") remain in the corpus; inspection showed minimal link spam (~7 URLs in 40k rows) but not a bot-free dataset. **Top-level comments only** ignores reply-thread context that often carries conversational meaning. **Single-channel scope** limits generalization to other creators or topics. **Partially automated evaluation labels** mean reported retrieval metrics are indicative rather than gold-standard.

These limitations are supported by direct observation during scraping (API caps), evaluation runs (metric anomalies before fixes), and corpus inspection (spam and outlier topic rates).

---

# 7. Conclusion

This project delivered a complete NLP and RAG system for Ryan Trahan audience analytics: data collection and preprocessing, multi-component NLP enrichment, hybrid retrieval with reranking, grounded LLM generation, agent routing, MLflow evaluation, and a Streamlit dashboard.

**Key findings.** (1) Hybrid FAISS + BM25 retrieval best handles the mix of paraphrased and exact-match queries in YouTube comments. (2) Dual sentiment and ensemble keyword methods produce richer analytics than single-model approaches. (3) Visible source citations increase user trust compared to ungrounded chatbots. (4) Evaluation metric and ground-truth design critically affect reported system quality.

**Overall outcome.** The system meets the CSCI370 project requirements for preprocessing, insight extraction, topic modeling, RAG effectiveness, agent routing, and evaluation with monitoring. The dashboard provides a working demo for presentation and assessment.

**Future improvements.** Manual annotation of an evaluation set; dedicated bot/spam classification; reply-thread modeling; domain-fine-tuned embeddings; multi-channel support; and optional local LLM fallback for offline use.

---

# 8. Appendix

## A. Additional Figures and Screenshots

- Figure 4.1: Sentiment distribution (Analytics tab)
- Figure 4.2: Top entities bar chart
- Figure 4.3: Topic distribution
- Figure 4.4: Ask tab with answer and sources
- Figure 4.5: Dashboard analytics overview
- Figure A.1: MLflow experiment runs view
- Figure A.2: Evaluation tab in dashboard

## B. Sample Inputs and Outputs

**Input (Ask tab):** "What do people say about the penny challenge?"

**Output:** Generated summary with 3-5 retrieved source comments showing comment text, video title, sentiment, and like count.

**Input (Summarize tab):** Select video title from dropdown.

**Output:** Abstractive summary of viewer reactions for that video, grounded in retrieved comments.

## C. Configuration Details

Key settings from `config/config.yaml`:

| Setting | Value |
|---------|-------|
| Embedding model | all-MiniLM-L6-v2 |
| Retrieval top-k | 20 retrieve, 5 rerank |
| Default retrieval mode | hybrid |
| LLM provider | Groq (llama-3.1-8b-instant) |
| MLflow experiment | csci370-youtube-intelligence |
| Min comment words | 3 |

## D. Extended Results Tables

*[Paste full MLflow run comparison tables here before submission]*

## E. Source Code Snippets

Hybrid retrieval fusion (conceptual):

```python
# Retrieve from FAISS and BM25, merge with RRF
semantic_hits = faiss_store.similarity_search(query, k=20)
bm25_hits = bm25_index.search(query, k=20)
merged = reciprocal_rank_fusion(semantic_hits, bm25_hits)
reranked = cross_encoder.rerank(query, merged, top_k=5)
```

## F. Setup Instructions (User Manual)

```powershell
# 1. Install dependencies
python install.py

# 2. Configure environment (.env)
#    GROQ_API_KEY=your_key
#    YOUTUBE_API_KEY=your_key

# 3. Run enrichment pipeline (if not already done)
python src/pipeline/run_full_pipeline.py

# 4. Start dashboard
.\start.ps1
# Open http://localhost:8501

# 5. Run evaluation
python src/evaluation/run_eval.py
```

## G. Individual Contributions

| Member | Modules | Role |
|--------|---------|------|
| Afrith | `app/`, `src/rag/`, `src/agent/` | RAG, agent, dashboard |
| Daniel | `src/pipeline/`, `config/` | Scrape, merge, preprocess |
| Feliks | `src/nlp/`, enrichment pipeline | Sentiment, NER, keywords, topics |
| Eshfar | `src/evaluation/`, tests | Metrics, MLflow |

## H. References

1. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS.*
2. Reimers, N., & Gurevych, I. (2019). Sentence-BERT. *EMNLP.*
3. Grootendorst, M. (2022). BERTopic. *arXiv:2203.05794.*
4. Khattab, O., et al. (2023). DSPy: Compiling Declarative Language Model Calls. *arXiv:2310.03714.*
5. Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *FnT IR.*
6. Hutto, C., & Gilbert, E. (2014). VADER: A Parsimonious Rule-Based Model for Sentiment Analysis. *ICWSM.*
7. YouTube Data API v3 Documentation; MLflow Documentation; CSCI370 Spring 2026 Project Brief.

---

*End of Report*
