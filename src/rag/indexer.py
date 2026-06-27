from __future__ import annotations

"""
RAG Indexer — YouTube Comments
Implements all 3 chunking strategies from professor's 2_Chunking.py:
  1. Sliding window  (RecursiveCharacterTextSplitter)
  2. Propositional   (prefix context to each chunk)
  3. Structured      (keep full metadata per chunk)
Then embeds and stores in FAISS (professor's 4_Vectorization_FAISS.py).
"""

import argparse
import os
import pickle
from pathlib import Path
from typing import List, Dict

import pandas as pd
from tqdm import tqdm

# LangChain - chunking (professor's 2_Chunking.py)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain - FAISS (professor's 4_Vectorization_FAISS.py)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # needed on some systems (professor's line)


# ── helpers ───────────────────────────────────────────────────────────────

def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


def get_embedder(provider: str, ollama_model: str, st_model: str, ollama_base_url: str):
    """
    Returns embedding model.
    Provider 'ollama' matches professor's OllamaEmbeddings(model='embeddinggemma').
    Provider 'sentence-transformers' is the cloud fallback.
    """
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        print(f"Using Ollama embeddings: {ollama_model}")
        return OllamaEmbeddings(model=ollama_model, base_url=ollama_base_url)
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        print(f"Using sentence-transformers embeddings: {st_model}")
        return HuggingFaceEmbeddings(model_name=st_model)


# ── chunking strategies (professor's 2_Chunking.py) ───────────────────────

def sliding_window_chunks(
    df: pd.DataFrame,
    text_col: str,
    chunk_size: int = 300,
    chunk_overlap: int = 60,
) -> List[Document]:
    """
    Strategy 1: Sliding window using RecursiveCharacterTextSplitter.
    Mirrors professor's 2_Chunking.py section 2.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    docs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Sliding window chunks"):
        text = str(row[text_col])
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "chunk_strategy": "sliding_window",
                    "chunk_index": i,
                    "comment_id": str(row.get("comment_id", row.get("id", ""))),
                    "video_id": str(row.get("video_id", "")),
                    "video_title": str(row.get("video_title", "")),
                    "sentiment": str(row.get("sentiment", "")),
                    "topic_id": str(row.get("topic_id", "")),
                    "published_at": str(row.get("published_at", "")),
                    "like_count": int(float(row.get("like_count") or 0)),
                },
            ))
    return docs


def propositional_chunks(
    df: pd.DataFrame,
    text_col: str,
    chunk_size: int = 300,
    chunk_overlap: int = 60,
) -> List[Document]:
    """
    Strategy 2: Propositional chunking.
    Adds topic/sentiment context prefix to each chunk.
    Mirrors professor's 2_Chunking.py section 3 — she prepended 'Large Language Model,'
    to all chunks. Here we prepend meaningful metadata instead.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    docs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Propositional chunks"):
        text = str(row[text_col])
        sentiment = str(row.get("sentiment", ""))
        topic_id = str(row.get("topic_id", ""))
        video_title = str(row.get("video_title", ""))

        prefix = f"Ryan Trahan YouTube comment"
        if sentiment:
            prefix += f", sentiment: {sentiment}"
        if topic_id and topic_id not in ("", "nan", "-1"):
            prefix += f", topic: {topic_id}"
        if video_title:
            prefix += f", video: {video_title}"
        prefix += ". "

        base_chunks = splitter.split_text(text)
        propositional = [prefix + chunk for chunk in base_chunks]

        for i, chunk in enumerate(propositional):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "chunk_strategy": "propositional",
                    "chunk_index": i,
                    "comment_id": str(row.get("comment_id", row.get("id", ""))),
                    "video_id": str(row.get("video_id", "")),
                    "video_title": video_title,
                    "sentiment": sentiment,
                    "topic_id": topic_id,
                    "published_at": str(row.get("published_at", "")),
                    "like_count": int(float(row.get("like_count") or 0)),
                },
            ))
    return docs


def structured_chunks(df: pd.DataFrame, text_col: str) -> List[Document]:
    """
    Strategy 3: Structured chunking — one document per comment, full metadata.
    Mirrors professor's 2_Chunking.py section 4 (structured_sents with par_num, sent_id).
    Each comment is treated as a unit with all its structured context preserved.
    """
    docs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Structured chunks"):
        text = str(row[text_col])
        docs.append(Document(
            page_content=text,
            metadata={
                "chunk_strategy": "structured",
                "comment_id": str(row.get("comment_id", row.get("id", ""))),
                "video_id": str(row.get("video_id", "")),
                "video_title": str(row.get("video_title", "")),
                "sentiment": str(row.get("sentiment", "")),
                "sentiment_score": float(row.get("sentiment_score", 0.0)),
                "topic_id": str(row.get("topic_id", "")),
                "published_at": str(row.get("published_at", "")),
                "like_count": int(float(row.get("like_count") or 0)),
                "author": str(row.get("author", "")),
            },
        ))
    return docs


CHUNK_STRATEGIES = {
    "sliding_window": sliding_window_chunks,
    "propositional": propositional_chunks,
    "structured": structured_chunks,
}


# ── BM25 index ────────────────────────────────────────────────────────────

def build_bm25_index(docs: List[Document], save_path: Path) -> None:
    from rank_bm25 import BM25Okapi
    corpus = [doc.page_content.lower().split() for doc in docs]
    bm25 = BM25Okapi(corpus)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": docs}, f)
    print(f"BM25 index saved to: {save_path}")


# ── main ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FAISS vector store and BM25 index from cleaned comments."
    )
    parser.add_argument("--input", type=str, default="data/processed/comments_enriched.csv")
    parser.add_argument("--faiss-dir", type=str, default="data/vectorstore/faiss_store")
    parser.add_argument("--bm25-path", type=str, default="data/vectorstore/bm25_index.pkl")
    parser.add_argument(
        "--chunk-strategy",
        choices=["sliding_window", "propositional", "structured"],
        default="propositional",
        help="Chunking strategy to use.",
    )
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--chunk-overlap", type=int, default=60)
    parser.add_argument(
        "--embedding-provider",
        choices=["sentence-transformers", "ollama"],
        default="sentence-transformers",
    )
    parser.add_argument("--ollama-model", type=str, default="embeddinggemma")
    parser.add_argument("--ollama-base-url", type=str, default="http://localhost:11434")
    parser.add_argument("--st-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = resolve_path(args.input)
    faiss_dir = resolve_path(args.faiss_dir)
    bm25_path = resolve_path(args.bm25_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} comments from {input_path.name}")

    # Build chunks using selected strategy
    strategy_fn = CHUNK_STRATEGIES[args.chunk_strategy]
    if args.chunk_strategy == "structured":
        docs = strategy_fn(df, args.text_column)
    else:
        docs = strategy_fn(df, args.text_column, args.chunk_size, args.chunk_overlap)

    print(f"Built {len(docs)} chunks using '{args.chunk_strategy}' strategy.")

    # Load embedding model
    embedder = get_embedder(
        provider=args.embedding_provider,
        ollama_model=args.ollama_model,
        st_model=args.st_model,
        ollama_base_url=args.ollama_base_url,
    )

    # Build FAISS index in batches (professor's 4_Vectorization_FAISS.py)
    print(f"Building FAISS index ({len(docs)} documents)...")
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    vectorstore = None
    for i in tqdm(range(0, len(texts), args.batch_size), desc="Indexing batches"):
        batch_texts = texts[i: i + args.batch_size]
        batch_metas = metadatas[i: i + args.batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_texts(batch_texts, embedder, metadatas=batch_metas)
        else:
            vectorstore.add_texts(batch_texts, metadatas=batch_metas)

    faiss_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(faiss_dir))
    print(f"FAISS index saved to: {faiss_dir}")

    # Build BM25 index alongside FAISS
    build_bm25_index(docs, bm25_path)

    print("\nIndexing complete.")
    print(f"  Documents indexed:  {len(docs)}")
    print(f"  Chunk strategy:     {args.chunk_strategy}")
    print(f"  Embedding provider: {args.embedding_provider}")
    print(f"  FAISS store:        {faiss_dir}")
    print(f"  BM25 index:         {bm25_path}")


if __name__ == "__main__":
    main()
