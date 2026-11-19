"""Quiz correctness verification against the source material.

A question PASSES only if all of these hold:
  1. its `source_quote` really occurs in the uploaded notes (anti-hallucination, fuzzy match),
  2. a *blind* judge, shown only retrieved evidence (never the answer key), picks the same option,
  3. the schema constraints hold (4 unique options, valid answer index) - enforced by pydantic.
Failed questions are regenerated (up to N rounds) and finally dropped, never shipped.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from .ingest import Chunk, split_sentences
from .retriever import Retriever
from .schemas import Question, QuestionCheck, VerificationReport


def _norm(t: str) -> str:
    t = re.sub(r"[^\w .]+", "", re.sub(r"\s+", " ", t.casefold()))
    # Keep decimal points: 3.0 and 30 must remain different amounts.
    return re.sub(r"(?<!\d)\.|\.(?!\d)", "", t).strip()


def quote_evidence(quote: str, chunks: list[Chunk]) -> tuple[bool, float]:
    """Return (found, score). Exact normalised substring => 1.0, else best sentence/window similarity."""
    nq = _norm(quote)
    if not nq:
        return False, 0.0
    best = 0.0
    quote_words = nq.split()
    negations = {"no", "not", "never", "neither", "without", "cannot", "cant", "doesnt", "isnt", "dont"}
    def protected(words):
        return ({w for w in words if w in negations}, re.findall(r"\d+(?:\.\d+)?", " ".join(words)))

    for c in chunks:
        nc = _norm(c.text)
        if f" {nq} " in f" {nc} ":
            return True, 1.0
        for s in split_sentences(c.text):
            words = _norm(s).split()
            # Compare contiguous windows. Scattered matching letters can otherwise
            # make invented quotes look grounded in a much longer sentence.
            for size in range(max(1, len(quote_words) - 2), min(len(words), len(quote_words) + 2) + 1):
                for start in range(len(words) - size + 1):
                    window = words[start:start + size]
                    if protected(window) != protected(quote_words):
                        continue
                    best = max(best, SequenceMatcher(None, nq, " ".join(window), autojunk=False).ratio())
    return best >= 0.88, round(best, 3)


def verify_question(i: int, q: Question, retriever: Retriever, generator, k: int = 4) -> QuestionCheck:
    all_chunks = retriever.chunks
    found, score = quote_evidence(q.source_quote, all_chunks)
    # Evidence for the blind judge: retrieved with the QUESTION ONLY + the chunk holding the quote.
    ev = [h.chunk for h in retriever.search(q.question.replace("Fill in the blank: ", ""), k)]
    if found:
        qh = retriever.search(q.source_quote, 1)
        if qh and qh[0].chunk not in ev:
            ev.append(qh[0].chunk)
    pick = generator.judge(q, ev)
    supported = (pick == q.answer_index) if pick is not None else None
    passed = found and supported is True
    note = []
    if not found:
        note.append("quote not found in notes")
    if supported is None:
        note.append("evidence did not settle the answer")
    elif supported is False:
        note.append("blind judge disagrees with key")
    return QuestionCheck(index=i, quote_found=found, answer_supported=supported, evidence_score=score,
                         passed=passed, notes="; ".join(note) or "verified")


def verify_and_repair(topic: str, questions: list[Question], target: int, retriever: Retriever,
                      generator, context: list[Chunk], max_rounds: int = 2, log=lambda m: None):
    """Verify, regenerate failures, drop what still fails. Returns (final_questions, report)."""
    good: list[Question] = []
    checks: list[QuestionCheck] = []
    pending = questions
    rounds = 0
    dropped = 0
    while True:
        rounds += 1
        for q in pending:
            c = verify_question(len(good), q, retriever, generator)
            if c.passed and q.question not in {g.question for g in good}:
                good.append(q)
                checks.append(c.model_copy(update={"index": len(good) - 1}))
            else:
                dropped += 1
                log(f"  ✗ rejected: {q.question[:70]}… ({c.notes})")
        if len(good) >= target or rounds > max_rounds:
            break
        need = target - len(good)
        log(f"  ↻ repair round {rounds}: regenerating {need} question(s)")
        try:
            pending = generator.make_quiz(topic, context, need + 1, avoid=[g.question for g in good])
        except Exception as e:  # network/LLM failure should not crash the lesson
            log(f"  ! regeneration failed: {e}")
            break
    return good[:target], VerificationReport(checks=checks[:target], attempts=rounds, dropped=dropped)
