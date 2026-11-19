"""TF-IDF retriever (offline, deterministic). Swap in embeddings if desired."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .ingest import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("No text found in the uploaded course material.")
        self.chunks = chunks
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.matrix = self.vec.fit_transform([c.text for c in chunks])

    def search(self, query: str, k: int = 6) -> list[Hit]:
        q = self.vec.transform([query])
        sims = (self.matrix @ q.T).toarray().ravel()
        order = np.argsort(-sims)[:k]
        return [Hit(self.chunks[i], float(sims[i])) for i in order if sims[i] > 0]

    def topic_coverage(self, topic: str) -> float:
        """Best chunk similarity for the topic: a low value means the notes don't cover it."""
        hits = self.search(topic, 1)
        return hits[0].score if hits else 0.0
