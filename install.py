"""
install.py  —  Install every package the project needs, in the right order.
Run:
    python install.py
"""
import subprocess
import sys

def pip(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *packages])

print("\n========================================")
print("  CSCI370 Project — Full Installer")
print("========================================\n")

print("[1/9] Core data libraries")
pip("pandas", "numpy", "tqdm", "python-dotenv")

print("\n[2/9] YouTube scraping")
pip("google-api-python-client")

print("\n[3/9] NLP — core")
pip("nltk", "vaderSentiment")
pip("spacy")
subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

print("\n[4/9] NLP — transformers & sentence models")
pip("transformers", "torch", "sentence-transformers")

print("\n[5/9] NLP — keywords & topic modelling")
pip("keybert", "yake", "bertopic", "umap-learn", "hdbscan")

print("\n[6/9] LangChain stack (RAG / chunking / embeddings)")
pip(
    "langchain",
    "langchain-community",
    "langchain-core",
    "langchain-text-splitters",
    "langchain-ollama",
    "langchain-huggingface",
)

print("\n[7/9] Vector stores & lexical search")
pip("faiss-cpu", "rank-bm25")

print("\n[8/9] Generation — DSPy + OpenAI")
pip("dspy-ai", "openai")

print("\n[9/9] Evaluation & dashboard")
pip("rouge-score", "bert-score", "mlflow", "streamlit", "plotly", "requests")

print("\n========================================")
print("  ALL PACKAGES INSTALLED SUCCESSFULLY")
print("========================================")
print("\nYou can now run:")
print("  python src/rag/indexer.py --chunk-strategy propositional")
