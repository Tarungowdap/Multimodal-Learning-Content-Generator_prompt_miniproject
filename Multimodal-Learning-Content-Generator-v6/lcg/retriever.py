"""Hybrid retrieval for the RAG pipeline.

Dense    : all-MiniLM-L6-v2 sentence embeddings (ONNX, runs locally, no key) stored in a persistent
           ChromaDB vector database (cosine distance, HNSW index) under ./vectorstore.
Keyword  : TF-IDF (1-2 grams) cosine similarity, which catches exact terms, formulas and names.
Fusion   : Reciprocal Rank Fusion (RRF, k=60) of both rankings.

The same uploaded notes map to the same collection (content hash), so embeddings are computed once.
If ChromaDB or the embedding model is unavailable (e.g. the first run without internet), retrieval
falls back to TF-IDF only and says so in the pipeline log.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .ingest import Chunk

VECTOR_DIR = Path(os.getenv("LCG_VECTOR_DIR", "vectorstore"))
EMBED_MODEL = "all-MiniLM-L6-v2"
RRF_K = 60
DENSE_ON_TOPIC = 0.30      # cosine similarity above which the notes are considered to cover a topic
TFIDF_ON_TOPIC = 0.05


@dataclass
class Hit:
    chunk: Chunk
    score: float            # dense cosine similarity when available, else TF-IDF similarity


class Retriever:
    def __init__(self, chunks: list[Chunk], use_dense: bool = True):
        if not chunks:
            raise ValueError("No text found in the uploaded course material.")
        self.chunks = chunks
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.matrix = self.vec.fit_transform([c.text for c in chunks])
        self.collection = None
        self.mode = "TF-IDF keyword search only"
        if use_dense:
            try:
                self.collection = self._open_collection()
                self.mode = f"hybrid: ChromaDB vector store ({EMBED_MODEL} embeddings) + TF-IDF, fused with RRF"
            except Exception as e:  # missing package, model download blocked, disk issues ...
                self.mode = f"TF-IDF keyword search only (vector store unavailable: {type(e).__name__}: {str(e)[:120]})"

    # ------------------------------------------------------------------------------ vector store
    def _open_collection(self):
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        from chromadb.utils import embedding_functions

        VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(VECTOR_DIR),
                                           settings=ChromaSettings(anonymized_telemetry=False))
        digest = hashlib.sha1("\x1e".join(c.source + "\x1f" + c.text for c in self.chunks).encode()).hexdigest()[:16]
        col = client.get_or_create_collection(
            name=f"notes_{digest}", metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_functions.DefaultEmbeddingFunction())   # ONNX all-MiniLM-L6-v2
        if col.count() != len(self.chunks):
            ids = [str(i) for i in range(len(self.chunks))]
            if col.count():
                col.delete(ids=col.get()["ids"])
            col.add(ids=ids, documents=[c.text for c in self.chunks],
                    metadatas=[{"source": c.source, "chunk_id": c.id} for c in self.chunks])
        return col

    def _dense(self, query: str, n: int) -> list[tuple[int, float]]:
        if self.collection is None:
            return []
        n = min(n, len(self.chunks))
        res = self.collection.query(query_texts=[query], n_results=n, include=["distances"])
        return [(int(i), 1.0 - float(d)) for i, d in zip(res["ids"][0], res["distances"][0])]

    def _keyword(self, query: str, n: int) -> list[tuple[int, float]]:
        sims = (self.matrix @ self.vec.transform([query]).T).toarray().ravel()
        order = np.argsort(-sims)[:n]
        return [(int(i), float(sims[i])) for i in order if sims[i] > 0]

    # ------------------------------------------------------------------------------ public API
    def search(self, query: str, k: int = 6) -> list[Hit]:
        pool = max(k * 3, 10)
        dense, kw = self._dense(query, pool), self._keyword(query, pool)
        if not dense:
            return [Hit(self.chunks[i], s) for i, s in kw[:k]]
        fused: dict[int, float] = {}
        for ranking in (dense, kw):
            for rank, (i, _) in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        dense_sim = dict(dense)
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(self.chunks[i], dense_sim.get(i, 0.0)) for i in best]

    def covers(self, topic: str) -> bool:
        """Is the topic actually in the notes? (semantic OR keyword evidence)."""
        kw = self._keyword(topic, 1)
        if kw and kw[0][1] >= TFIDF_ON_TOPIC:
            return True
        dense = self._dense(topic, 1)
        return bool(dense) and dense[0][1] >= DENSE_ON_TOPIC

    def topic_coverage(self, topic: str) -> float:
        """Best similarity for the topic (kept for compatibility)."""
        kw = self._keyword(topic, 1)
        dense = self._dense(topic, 1)
        return max(kw[0][1] if kw else 0.0, dense[0][1] if dense else 0.0)
