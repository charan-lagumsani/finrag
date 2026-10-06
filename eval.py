"""Offline evaluation: retrieval quality + answer faithfulness.

Metrics:
  recall@k — fraction of questions whose expected doc appears in top-k
  MRR      — mean reciprocal rank of the expected doc
  faithfulness — fraction of answer sentences lexically grounded in
                 retrieved chunks (in the spirit of RAGAS faithfulness)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.retriever import Retriever

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "was", "were", "are", "be", "by", "as", "at", "it", "its",
    "that", "this", "from", "which", "what", "how", "does", "did",
}


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS}


def faithfulness(answer: str, chunks: list[dict]) -> float:
    """Each answer sentence must share content tokens with some chunk."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sentences:
        return 0.0
    chunk_tokens = [tokens(c["text"]) for c in chunks]
    grounded = 0
    for sent in sentences:
        st = tokens(sent)
        if not st:
            grounded += 1
            continue
        if any(len(st & ct) / len(st) >= 0.5 for ct in chunk_tokens):
            grounded += 1
    return grounded / len(sentences)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", default="tfidf", choices=["tfidf", "sbert"])
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    retriever = Retriever(backend=args.embedder)
    eval_path = Path("eval/questions.jsonl")
    questions = [json.loads(line) for line in eval_path.read_text().splitlines()]

    recalls, mrrs, faiths = [], [], []
    for item in questions:
        hits = retriever.search(item["question"], k=args.k)
        doc_ids = [h["doc_id"] for h in hits]
        hit = item["expected_doc"] in doc_ids
        recalls.append(float(hit))
        if hit:
            rank = doc_ids.index(item["expected_doc"]) + 1
            mrrs.append(1.0 / rank)
        else:
            mrrs.append(0.0)
        answer = hits[0]["text"] if hits else ""
        faiths.append(faithfulness(answer, hits))

    n = len(questions)
    print(f"evaluated {n} questions (k={args.k}, backend={retriever.embedder.backend})")
    print(f"  recall@{args.k}:  {sum(recalls) / n:.3f}")
    print(f"  MRR:            {sum(mrrs) / n:.3f}")
    print(f"  faithfulness:   {sum(faiths) / n:.3f}")


if __name__ == "__main__":
    main()
