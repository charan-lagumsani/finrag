"""Load a persisted FAISS index and run similarity search."""
from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np

from .embeddings import Embedder
from .index import INDEX_PATH, META_PATH


class Retriever:
    def __init__(
        self,
        index_path: Path = INDEX_PATH,
        meta_path: Path = META_PATH,
        corpus_dir: Path = Path("data/corpus"),
        backend: str | None = None,
    ):
        if not index_path.exists():
            from .index import build_index

            print("Index not found — building...")
            info = build_index(corpus_dir=corpus_dir, backend=backend or "tfidf")
            backend = info["backend"]
        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as fh:
            meta = pickle.load(fh)
        self.chunks: list[dict] = meta["chunks"]
        backend = backend or meta.get("backend", "tfidf")

        # Re-fit the embedder on chunk texts so queries map to the same space.
        self.embedder = Embedder(backend=backend).fit([c["text"] for c in self.chunks])

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = self.embedder.encode([query]).astype(np.float32)
        scores, idxs = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results
