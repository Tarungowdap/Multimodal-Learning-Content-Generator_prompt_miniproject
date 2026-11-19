"""Prompt-chaining orchestrator.

 retrieve -> summarise -> quiz -> VERIFY(+repair) -> scene plan -> images -> video
Each step consumes the previous step's validated output.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable
from uuid import uuid4

from .config import OUTPUT_DIR, Settings
from .generators import ExtractiveGenerator, LLMGenerator
from .images import generate_image
from .ingest import Chunk
from .llm import LLMClient, LLMError
from .retriever import Retriever
from .schemas import LessonPackage, Quiz
from .verify import verify_and_repair
from . import local_t2v
from .t2v import build_video_prompt, generate_clip
from .video import fit_narration, make_t2v_video, make_video, narrate

MIN_COVERAGE = 0.05  # below this TF-IDF similarity the topic is considered absent from the notes


def make_generator(s: Settings):
    p = s.resolve_llm()
    if p not in ("offline", "ollama") and not s.key(p):   # chosen provider but no key -> offline, no crash
        p = "offline-nokey"
    if p in ("offline", "offline-nokey"):
        return ExtractiveGenerator(s.seed), p
    if p == "ollama":
        return LLMGenerator(LLMClient("ollama", "", s.ollama_model or None), s.seed), p
    return LLMGenerator(LLMClient(p, s.key(p)), s.seed), p


def slug(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")[:40] or "lesson"


def run_pipeline(topic: str, chunks: list[Chunk], s: Settings, progress: Callable[[float, str], None] = lambda p, m: None,
                 make_media: bool = True) -> LessonPackage:
    log: list[str] = []

    def L(msg: str, p: float | None = None):
        log.append(msg)
        progress(p if p is not None else -1, msg)

    retr = Retriever(chunks)
    L(f"Retrieval: {retr.mode}")
    if not retr.covers(topic):
        raise ValueError(f"The uploaded notes do not appear to cover “{topic}”. Try a topic from your material.")
    gen, provider = make_generator(s)
    if provider == "offline-nokey":
        L("⚠ No API key found for the chosen text model -> OFFLINE mode (basic fill-in-the-blank quiz). "
          "Add a free Gemini key for real questions.")
    L(f"Provider: {provider} · {len(chunks)} chunks indexed", 0.05)

    # 1. retrieval
    hits = retr.search(topic, s.top_k)
    ctx = [h.chunk for h in hits]
    L(f"① Retrieved {len(ctx)} passages (top score {hits[0].score:.2f})", 0.12)

    # 2. summary  (falls back to the extractive generator if the LLM fails)
    try:
        summary = gen.summarize(topic, ctx)
    except LLMError as e:
        L(f"⚠ LLM call failed: {e}\n  -> falling back to OFFLINE mode (basic quiz).")
        gen, provider = ExtractiveGenerator(s.seed), "offline"
        summary = gen.summarize(topic, ctx)
    L("② Summary generated", 0.25)

    # 3. quiz
    try:
        raw_qs = gen.make_quiz(topic, ctx, s.num_questions + 1)
    except LLMError as e:
        L(f"⚠ Quiz generation failed: {e} -> falling back to offline generation.")
        gen, provider = ExtractiveGenerator(s.seed), "offline"
        summary = gen.summarize(topic, ctx)
        raw_qs = gen.make_quiz(topic, ctx, s.num_questions + 1)
    L(f"③ Draft quiz: {len(raw_qs)} questions → verifying against the notes", 0.4)

    # 4. verification + repair
    qs, report = verify_and_repair(topic, raw_qs, s.num_questions, retr, gen, ctx, s.max_repair_rounds, L)
    if not qs:
        raise ValueError("No quiz question could be verified against the notes. Add more detailed material.")
    quiz = Quiz(topic=topic, questions=qs)  # final schema validation of the shipped JSON
    L(f"④ Verified {len(qs)}/{s.num_questions} questions · rejected {report.dropped}", 0.55)

    # 5. scenes
    try:
        scenes = gen.plan_scenes(topic, summary)
    except LLMError as e:
        L(f"⚠ Scene planning failed: {e} -> using an offline scene plan.")
        scenes = ExtractiveGenerator(s.seed).plan_scenes(topic, summary)
    # Narration must fit the 10 s clip at a calm, natural pace: shorten the script, never speed it up.
    fitted = fit_narration([sc.narration for sc in scenes], s.narration_wpm)
    scenes = [sc.model_copy(update={"narration": n}) for sc, n in zip(scenes, fitted)]
    L(f"⑤ Planned {len(scenes)}-scene explainer", 0.62)

    pkg = LessonPackage(topic=topic, llm_provider=provider, summary=summary, quiz=quiz, verification=report,
                        scenes=scenes, sources=sorted({c.source for c in ctx}), log=log)

    if make_media:
        # Separate runs (including simultaneous browser sessions) must not overwrite media.
        out = OUTPUT_DIR / slug(topic) / uuid4().hex
        paths = []
        sources = []
        for i, sc in enumerate(scenes):
            detail = sc.detail or sc.narration
            path, src = generate_image(sc.image_subject, topic, i, out, s, L, heading=sc.caption,
                                       detail=detail, total=len(scenes))
            paths.append(path)
            sources.append(src)
            progress(0.62 + 0.2 * (i + 1) / len(scenes), f"⑥ Illustration {i + 1}/{len(scenes)}")
        pkg.image_paths = [str(p) for p in paths]
        pkg.image_sources = sources
        allow_online = s.llm_provider != "offline"
        # Record the narration once, measured against the clip length (shortened, never sped up).
        audio, spoken = narrate([sc.narration for sc in scenes], out / "narration.mp3", s.narration_wpm, L,
                                allow_online=allow_online)
        scenes = [sc.model_copy(update={"narration": n}) for sc, n in zip(scenes, spoken)]
        pkg.scenes = scenes
        narration = " ".join(spoken)
        captions = [sc.caption for sc in scenes]
        vid = None
        # ⑦a text-to-video model: LOCAL AnimateDiff-Lightning (no key) first, then optional online models
        backends = s.resolve_video(allow_online)
        for b in backends:
            progress(0.84, f"⑦ Generating video clip with {b} (this can take a few minutes)…")
            try:
                raw = out / f"t2v_{b}_raw.mp4"
                if b == "local":
                    _, prompts = local_t2v.generate_clip(
                        scenes, topic, raw, seed=s.seed, log=L,
                        progress=lambda i, n: progress(0.84 + 0.1 * i / n, f"⑦ Local video: scene {i}/{n} generated"))
                    pkg.video_prompt = "\n".join(prompts)
                else:
                    pkg.video_prompt = build_video_prompt(topic, scenes)
                    raw.write_bytes(generate_clip(pkg.video_prompt, b, s.key(b), L, seed=s.seed))
                vid, dur = make_t2v_video(raw, captions, narration, out / "explainer.mp4", L,
                                          allow_online=allow_online, wpm=s.narration_wpm, audio=audio)
                pkg.video_source = f"text-to-video ({b})"
                break
            except Exception as e:  # quota, safety block, timeout, undecodable clip …
                L(f"  video: {b} text-to-video failed ({str(e)[:160]}); trying next option")
        if vid is None:
            # ⑦b fallback: illustrations as a Ken-Burns image sequence with narration
            if s.video_backend != "slideshow" and not backends:
                why = "local video model not installed (run setup_local_video.bat)"
                L(f"  video: {why} -> image sequence with narration")
            vid, dur = make_video(paths, captions, narration, out / "explainer.mp4", L, allow_online=allow_online,
                                  wpm=s.narration_wpm, audio=audio)
            pkg.video_source = "image sequence"
        pkg.video_path = str(vid)
        L(f"⑦ Video rendered via {pkg.video_source} ({dur:.1f}s)", 0.95)
        pkg.log = list(log)
        (out / "lesson.json").write_text(pkg.model_dump_json(indent=2), encoding="utf-8")
        (out / "quiz.json").write_text(quiz.model_dump_json(indent=2), encoding="utf-8")
    progress(1.0, "Done")
    pkg.log = list(log)
    return pkg
