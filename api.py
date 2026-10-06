"""FastAPI query service over the FAISS index.

GET /query?q=...&k=3   -> extractive answer + cited chunks
GET /stats             -> index statistics
GET /health            -> liveness
"""
from __future__ import annotations

from fastapi import FastAPI, Query

from src.retriever import Retriever

app = FastAPI(title="finrag", version="0.1.0")
retriever = Retriever()


def synthesize(question: str, chunks: list[dict]) -> str:
    """Extractive answer: the top chunk verbatim.
    Swap this function for an LLM call to get abstractive answers."""
    if not chunks:
        return "No relevant documents found."
    return chunks[0]["text"].strip()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/stats")
def stats() -> dict:
    return {
        "chunks": len(retriever.chunks),
        "docs": len({c["doc_id"] for c in retriever.chunks}),
        "backend": retriever.embedder.backend,
    }


@app.get("/query")
def query(q: str = Query(..., min_length=2), k: int = Query(3, ge=1, le=10)) -> dict:
    chunks = retriever.search(q, k=k)
    return {
        "question": q,
        "answer": synthesize(q, chunks),
        "faithfulness_note": "extractive — answer is quoted from the cited chunk",
        "sources": [
            {
                "doc_id": c["doc_id"],
                "title": c["title"],
                "chunk_id": c["chunk_id"],
                "score": round(c["score"], 3),
            }
            for c in chunks
        ],
    }
