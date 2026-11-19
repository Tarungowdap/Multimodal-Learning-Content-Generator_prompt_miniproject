"""Text generators. `LLMGenerator` = prompt chain over an LLM; `ExtractiveGenerator` = offline fallback.

Both expose the same interface so the pipeline and the verifier don't care which is used.
"""
from __future__ import annotations

import random
import re
from collections import Counter
from typing import Optional

from pydantic import BaseModel, Field

from .ingest import Chunk, split_sentences
from .llm import LLMClient, generate_json
from .schemas import Question, Quiz, Scene, ScenePlan, Summary

STOP = set("""a an the and or of to in on for with as by at from is are was were be been being this that these those
it its into than then so such can could may might will would should which who whom whose what when where why how
not no nor but if else also very more most other some any each both between during over under about after before
while their there they them we our you your he she his her i""".split())


def fmt_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.id}] {c.text}" for c in chunks)


def shuffle_options(q: Question, rng: random.Random) -> Question:
    """LLMs love putting the key in slot B; randomise to remove positional bias."""
    correct = q.options[q.answer_index]
    opts = q.options[:]
    rng.shuffle(opts)
    return q.model_copy(update={"options": opts, "answer_index": opts.index(correct)})


# ----------------------------------------------------------------------------- LLM chain
SYSTEM = ("You are a careful teaching assistant. Use ONLY the supplied course notes. "
          "Never add outside facts. Reply with a single JSON object and nothing else.")


class _Judge(BaseModel):
    answer_index: Optional[int] = Field(default=None, ge=0, le=3)
    supported: bool = True


class LLMGenerator:
    name = "llm"

    def __init__(self, client: LLMClient, seed: int = 42):
        self.client = client
        self.rng = random.Random(seed)

    def summarize(self, topic: str, chunks: list[Chunk]) -> Summary:
        prompt = f"""Course notes:
{fmt_context(chunks)}

TASK: Write a lesson summary on "{topic}" for a student, grounded strictly in the notes.
JSON schema: {{"title": str, "summary": str (120-200 words, plain prose),
"key_points": [str, ... 4-6 short bullet sentences]}}"""
        return generate_json(self.client, SYSTEM, prompt, Summary)

    def make_quiz(self, topic: str, chunks: list[Chunk], n: int, avoid: list[str] | None = None) -> list[Question]:
        avoid_txt = ("\nDo NOT repeat these questions:\n- " + "\n- ".join(avoid)) if avoid else ""
        prompt = f"""Course notes:
{fmt_context(chunks)}

TASK: Write {n} multiple-choice questions on "{topic}" that can be answered from the notes alone.{avoid_txt}
Rules:
- exactly 4 options, exactly one correct, plausible distractors, no "all of the above"
- "answer_index" is the 0-based index of the correct option
- "source_quote" MUST be copied VERBATIM from the notes (one sentence or fragment) and must prove the answer
- "explanation" is one sentence
- mix difficulty ("easy" | "medium" | "hard")
JSON schema: {{"topic": str, "questions": [{{"question": str, "options": [str,str,str,str],
"answer_index": int, "explanation": str, "source_quote": str, "difficulty": str}}]}}"""
        quiz = generate_json(self.client, SYSTEM, prompt, Quiz, temperature=0.4)
        return [shuffle_options(q, self.rng) for q in quiz.questions]

    def plan_scenes(self, topic: str, summary: Summary) -> list[Scene]:
        prompt = f"""Lesson summary for "{topic}":
{summary.summary}

TASK: Plan a 3-scene explainer video (total narration <= 24 words, so it fits in 8 seconds).
Each scene: "caption" (<=8 words), "image_subject" (a concrete, drawable visual, no text in the image),
"narration" (one short sentence, <=8 words).
JSON schema: {{"scenes": [{{"caption": str, "image_subject": str, "narration": str}}]}}"""
        return generate_json(self.client, SYSTEM, prompt, ScenePlan).scenes

    def judge(self, q: Question, evidence: list[Chunk]) -> Optional[int]:
        """Blind check: answer the question from evidence only, WITHOUT seeing the key."""
        opts = "\n".join(f"{i}. {o}" for i, o in enumerate(q.options))
        prompt = f"""Evidence:
{fmt_context(evidence)}

Question: {q.question}
Options:
{opts}

Using ONLY the evidence, which option is correct? If the evidence does not settle it, set
"supported" to false. JSON: {{"answer_index": int|null, "supported": bool}}"""
        try:
            r = generate_json(self.client, SYSTEM, prompt, _Judge, retries=1, temperature=0.0)
        except Exception:
            return None
        return r.answer_index if r.supported else None


# ----------------------------------------------------------------------------- offline fallback
def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z][A-Za-z\-']+|\d+", text.lower()) if t not in STOP]


class ExtractiveGenerator:
    """No-API fallback: extractive summary + cloze (fill-in-the-blank) MCQs built from the notes."""

    name = "offline-extractive"

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _sentences(self, chunks: list[Chunk]) -> list[str]:
        seen, out = set(), []
        for c in chunks:
            for s in split_sentences(c.text):
                if s not in seen and 6 <= len(s.split()) <= 40:
                    seen.add(s)
                    out.append(s)
        return out

    def _rank(self, topic: str, sents: list[str]) -> list[str]:
        tset, freq = set(tokens(topic)), Counter(t for s in sents for t in tokens(s))
        def score(s):
            tk = tokens(s)
            return (3 * len(tset & set(tk)) + sum(freq[t] for t in set(tk)) / (len(tk) + 1))
        return sorted(sents, key=score, reverse=True)

    def summarize(self, topic: str, chunks: list[Chunk]) -> Summary:
        ranked = self._rank(topic, self._sentences(chunks))
        if len(ranked) < 2 or len(" ".join(ranked)) < 30:
            raise ValueError("Offline generation needs at least two readable sentences of 6–40 words. "
                             "Add more detailed notes or choose a text provider with an API key.")
        top = ranked[:6]
        order = {s: i for i, s in enumerate(self._sentences(chunks))}
        top.sort(key=lambda s: order.get(s, 0))
        return Summary(title=topic.title(), summary=" ".join(top[:5]),
                       key_points=[s if len(s) < 160 else s[:157] + "..." for s in ranked[:5]])

    def _candidates(self, sents: list[str]) -> Counter:
        cand: Counter = Counter()
        for s in sents:
            for m in re.findall(r"\b[A-Z][a-zA-Z\-]{3,}\b|\b[a-z]{6,}\b", s):
                if m.lower() not in STOP:
                    cand[m] += 1
        return cand

    def make_quiz(self, topic: str, chunks: list[Chunk], n: int, avoid: list[str] | None = None) -> list[Question]:
        sents = self._rank(topic, self._sentences(chunks))
        cand = self._candidates(sents)
        avoid = set(avoid or [])
        out: list[Question] = []
        for s in sents:
            words = [w for w in re.findall(r"\b[A-Za-z][A-Za-z\-]{4,}\b", s) if w in cand and w.lower() not in STOP]
            if not words:
                continue
            # key term: appears rarely elsewhere (specific) and is not the first word
            ans = min(words, key=lambda w: (cand[w], -len(w)))
            stem = re.sub(rf"\b{re.escape(ans)}\b", "_____", s, count=1)
            if "Fill in the blank: " + stem in avoid or stem.strip().startswith("_____") or len(s.split()) < 8:
                continue
            pool = [w for w in cand if w.lower() != ans.lower() and w.lower() not in s.lower()
                    and w[0].isupper() == ans[0].isupper() and abs(len(w) - len(ans)) <= 5]
            if len(pool) < 3:
                continue
            self.rng.shuffle(pool)
            # distractors that look like the answer (same ending / similar length) are harder & fairer
            pool.sort(key=lambda w: (-(w[-2:].lower() == ans[-2:].lower()), abs(len(w) - len(ans))))
            opts = [ans] + pool[:3]
            self.rng.shuffle(opts)
            try:
                q = Question(question="Fill in the blank: " + stem, options=opts, answer_index=opts.index(ans),
                             explanation=f"The notes state: \"{s[:140]}\"", source_quote=s,
                             difficulty="easy")
            except Exception:
                continue
            out.append(q)
            if len(out) >= n:
                break
        return out

    def plan_scenes(self, topic: str, summary: Summary) -> list[Scene]:
        cands = [p.strip() for p in summary.key_points if len(p.split()) >= 5 and not p.endswith("...")]
        cands.sort(key=lambda x: len(x.split()))          # shortest = most caption-friendly
        scenes = []
        for p in cands[:3]:
            w = p.rstrip(".").split()
            cap = " ".join(w[:8]) + ("…" if len(w) > 8 else "")
            # Keep complete statements; the video renderer fits long audio to its duration.
            nar = p if p.endswith((".", "!", "?")) else p + "."
            scenes.append(Scene(caption=cap, image_subject=f"{topic} - {' '.join(w[:10])}", narration=nar, detail=p))
        while len(scenes) < 2:  # tiny notes: still need a minimum of 2 scenes
            scenes.append(Scene(caption=topic.title(), image_subject=topic, narration=f"Let us learn about {topic}."))
        return scenes

    def judge(self, q: Question, evidence: list[Chunk]) -> Optional[int]:
        """Lexical blind judge: which option, when put into the stem, best matches an evidence sentence?"""
        sents = [s for c in evidence for s in split_sentences(c.text)]
        if not sents:
            return None
        stem = q.question.replace("Fill in the blank: ", "")
        scores = []
        for o in q.options:
            filled = tokens(stem.replace("_____", o)) if "_____" in stem else tokens(stem + " " + o)
            fs = set(filled)
            best = max((len(fs & set(tokens(s))) / (len(fs) or 1) for s in sents), default=0)
            scores.append(best)
        top = max(scores)
        if scores.count(top) > 1 or top < 0.5:
            return None
        return scores.index(top)
