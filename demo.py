"""End-to-end demo: build index -> answer sample questions -> show citations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.index import build_index
from src.retriever import INDEX_PATH, META_PATH, Retriever

QUESTIONS = [
    "What drove the increase in net interest income?",
    "What does Regulation Best Interest require of broker-dealers?",
    "How did machine learning improve demand forecasting?",
    "What is the Federal Reserve's stance on interest rates?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", default="tfidf", choices=["tfidf", "sbert"])
    args = parser.parse_args()

    # Fresh build each demo run for reproducibility
    for p in (INDEX_PATH, META_PATH):
        if p.exists():
            p.unlink()
    stats = build_index(backend=args.embedder)
    print(f"Index built: {json.dumps(stats, indent=2)}\n")

    retriever = Retriever()
    for q in QUESTIONS:
        print(f"Q: {q}")
        for hit in retriever.search(q, k=2):
            snippet = " ".join(hit["text"].split())[:220]
            print(f"  → [{hit['doc_id']} :: {hit['chunk_id'].split('::')[1]} | "
                  f"score {hit['score']:.3f}]")
            print(f"    {snippet}...")
        print()


if __name__ == "__main__":
    main()
