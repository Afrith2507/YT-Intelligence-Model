$env:RAG_GEN_MODEL     = "ollama/llama3.2:3b"
$env:RAG_EMBED_BACKEND = "sentence-transformers"
python -m streamlit run app/dashboard.py
