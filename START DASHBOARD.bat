@echo off
cd /d "C:\Users\Afrith\Desktop\Afrith\UOWD\CSCI370\Project"
set RAG_GEN_MODEL=ollama/llama3.2:3b
set RAG_EMBED_BACKEND=sentence-transformers
python -m streamlit run app/dashboard.py
pause
