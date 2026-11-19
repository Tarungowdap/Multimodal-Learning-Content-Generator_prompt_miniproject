"""Suggest topics from the uploaded document so the user never has to guess what the notes cover."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from .ingest import Chunk


def suggest_topics(chunks: list[Chunk], k: int = 8) -> list[str]:
    if len(chunks) < 2:
        texts = [c.text for c in chunks]
    else:
        texts = [c.text for c in chunks]
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True,
                              token_pattern=r"(?u)\b[A-Za-z][A-Za-z\-]{3,}\b", max_features=4000)
        m = vec.fit_transform(texts)
    except ValueError:
        return []
    scores = m.sum(axis=0).A1
    terms = vec.get_feature_names_out()
    out: list[str] = []
    for i in scores.argsort()[::-1]:
        t = terms[i]
        if any(t in o or o in t for o in out):   # skip near-duplicates ("gradient" / "gradient descent")
            continue
        out.append(t)
        if len(out) >= k:
            break
    return out
