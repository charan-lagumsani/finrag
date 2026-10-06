"""Chunk documents and build a FAISS index over their embeddings."""
from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .embeddings import Embedder

INDEX_PATH = Path("index.faiss")
META_PATH = Path("index_meta.pkl")


def load_documents(corpus_dir: Path) -> list[dict]:
    docs = []
    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        # first non-empty line is the title
        title = next((ln.strip() for ln in text.splitlines() if ln.strip()), path.stem)
        docs.append({"doc_id": path.stem, "title": title, "text": text})
    return docs


def chunk_documents(docs: list[dict], chunk_size: int = 600, overlap: int = 80) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in docs:
        for i, piece in enumerate(splitter.split_text(doc["text"])):
            chunks.append(
                {
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "chunk_id": f"{doc['doc_id']}::c{i}",
                    "text": piece,
                }
            )
    return chunks


def build_index(
    corpus_dir: Path = Path("data/corpus"),
    backend: str = "tfidf",
    index_path: Path = INDEX_PATH,
    meta_path: Path = META_PATH,
) -> dict:
    docs = load_documents(corpus_dir)
    if not docs:
        raise SystemExit(f"No .txt documents found in {corpus_dir}")
    chunks = chunk_documents(docs)
    texts = [c["text"] for c in chunks]

    embedder = Embedder(backend=backend).fit(texts)
    vectors = embedder.encode(texts).astype(np.float32)

    index = faiss.IndexFlatIP(vectors.shape[1])  # inner product = cosine (normalized)
    index.add(vectors)

    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as fh:
        pickle.dump({"chunks": chunks, "backend": embedder.backend}, fh)

    return {
        "docs": len(docs),
        "chunks": len(chunks),
        "dim": vectors.shape[1],
        "backend": embedder.backend,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", default="tfidf", choices=["tfidf", "sbert"])
    args = parser.parse_args()
    stats = build_index(backend=args.embedder)
    print(f"Built index: {stats}")
