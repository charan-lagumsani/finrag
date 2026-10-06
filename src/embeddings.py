"""Pluggable embedding backends with a single interface.

Default backend is TF-IDF (scikit-learn): zero downloads, fully offline,
deterministic. Pass backend="sbert" to use sentence-transformers
(all-MiniLM-L6-v2) when installed; falls back to TF-IDF with a warning if
the model cannot be loaded.
"""
from __future__ import annotations

import warnings

import numpy as np


class Embedder:
    def __init__(self, backend: str = "tfidf", model_name: str = "all-MiniLM-L6-v2"):
        self.requested = backend
        self.backend = backend
        self._vectorizer = None
        self._model = None
        if backend == "sbert":
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(model_name)
            except Exception as exc:  # model missing / no torch / offline
                warnings.warn(
                    f"sbert backend unavailable ({exc}); falling back to tfidf"
                )
                self.backend = "tfidf"

    # -- fitting (corpus-dependent for tfidf) --------------------------------
    def fit(self, texts: list[str]) -> "Embedder":
        if self.backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(
                lowercase=True, ngram_range=(1, 2), max_features=20000
            )
            self._vectorizer.fit(texts)
        return self

    # -- encoding -------------------------------------------------------------
    def encode(self, texts: list[str]) -> np.ndarray:
        if self.backend == "sbert" and self._model is not None:
            vecs = self._model.encode(texts, normalize_embeddings=True)
            return np.asarray(vecs, dtype=np.float32)
        if self._vectorizer is None:
            raise RuntimeError("Embedder.fit() must be called before encode()")
        mat = self._vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms  # L2-normalized -> cosine via inner product

    @property
    def dim(self) -> int:
        if self.backend == "sbert" and self._model is not None:
            return int(self._model.get_sentence_embedding_dimension())
        if self._vectorizer is not None:
            return len(self._vectorizer.vocabulary_)
        raise RuntimeError("Embedder not fitted yet")
