# Runtime check and corrections

Reviewed: 8 November 2026 (Asia/Kolkata).
Input: `Multimodal-Learning-Content-Generator (2).zip`.

The corrected project runs locally on Windows with CPython 3.13.9 and Streamlit 1.67.0. All 40 automated tests passed in the 8 November review (62 after the 18 November update below). The browser workflow was also exercised: upload notes, choose a topic, generate a package, answer the quiz, play the video and download quiz JSON. Testing reduces risk; it cannot establish that every possible input and external service will be error-free.

## Validation

| Check | Result |
|---|---|
| Full test suite | `40 passed` |
| Dependency compatibility | `pip check`: no broken requirements |
| Python compilation | app, library modules, views and tests compile |
| Existing original non-media tests | 18 passed; the initial 11 added regression cases exposed failures in the original source |
| Photosynthesis sample, offline | 5 verified questions, 3 images, 8.0-second MP4 |
| Machine-learning sample, offline | 5 verified questions, 3 images, 8.0-second MP4 |
| Upload readers | TXT, MD, DOCX and text-based PDF tested |
| Corrupt PDF | App shows an error and disables generation |
| Browser quiz | All five sample answers submitted; score 5/5 |
| Browser video | Duration 8 seconds; playing; readyState 4; no media error |
| Browser export | quiz.json downloaded and parsed; 5 questions |
| Long narration | 14-second test audio fitted into a 10-second MP4; decoded audio checked |

## Corrections

1. Validate quiz-option uniqueness and emptiness after removing A/B/C/D labels. Previously duplicate answers could survive validation.
2. Parse model JSON using Python's JSON decoder so escaped quotes, backslashes and nested delimiters work.
3. Tighten fuzzy evidence matching to contiguous windows. Preserve decimal numbers and reject the tested changes to numbers and negation.
4. Respect the offline quiz repair avoid-list so accepted questions are not repeatedly generated.
5. Convert empty or malformed provider replies into controlled LLM errors.
6. Recover from quiz-generation and scene-planning API failures, with logged offline fallbacks.
7. Skip online speech requests when Text LLM = offline. Online speech has a connection/read timeout; local speech subprocesses have a timeout.
8. Fit long narration to the video's duration rather than cutting off the end. Reject empty or mismatched image/caption inputs.
9. Keep complete sentences in offline narration instead of cutting each one off after nine words.
10. Save each generation in its own run directory to prevent repeated topics or simultaneous sessions from overwriting earlier media.
11. Include image and video events in both the displayed and saved pipeline logs.
12. Catch unreadable uploads; clear old lesson results when uploaded documents change or generation restarts; reset prior quiz answers.
13. Expose the OpenAI image key field when a different text provider or offline text is selected.
14. Give results tabs a stable key. Browser testing reproduced a reset to Summary when choosing quiz answers; the corrected quiz retains its tab and scores correctly.
15. Add pinned dependency versions, current Windows setup instructions, a Windows launcher and regression tests. Refresh the sample quiz and offline video from the corrected generator.

## Run the corrected project

Extract `Multimodal-Learning-Content-Generator-checked.zip`. Open the extracted project folder and double-click `run_windows.bat` with Python installed, or run these PowerShell commands there:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

First test: upload `sample_data/photosynthesis.md`, set Text LLM to `offline`, set Image model to `placeholder`, choose `photosynthesis`, and generate. Outputs go to `outputs/<topic>/<run-id>/`.

Run checks with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

## Verification limits

- No API credentials were supplied. Live authenticated Anthropic, OpenAI, Gemini and Hugging Face calls were not exercised. Provider error handling was tested with simulated replies. Pollinations retry/fallback behavior was tested with simulated responses, not a live image-generation request.
- The existing Claude and Gemini default model identifiers were checked against [Anthropic's model documentation](https://platform.claude.com/docs/en/models/overview) and [Google's Gemini 3.8 Flash documentation](https://aistudio.google.com/docs/models/gemini-3.8-flash). This does not verify account-specific access, billing, rate limits or service uptime.
- Offline runs in this environment used infographic images and silent audio, with captions. Audible offline narration needs espeak/espeak-ng installed. Online mode can use gTTS when reachable.
- Offline summaries and quizzes are extractive English-language fallbacks. Short notes, scanned PDFs without a text layer and notes without enough suitable sentences may need more material or a text provider. No OCR was added.
- Automated evidence matching and a model judge cannot guarantee educational correctness. A human should review material intended for assessment. TF-IDF retrieval remains lexical and may miss paraphrased topics.
- Public deployment, browser compatibility beyond the tested local browser, and concurrent load were not tested. Unique output directories prevent overwrite collisions; they are not an authenticated access-control system.
- Online image requests retain the original retry policy and may be slow when services are unavailable. Use placeholder images for an immediate offline demonstration.

The input ZIP was left unchanged. The corrected archive excludes Git history, local environments, caches and secrets.

---

# Update — 18 November 2026: prompt-chain fixes and text-to-video

## Prompt-chain fixes

1. **One bad question no longer discards the batch.** Quiz items are validated one at a time (`llm.generate_json_items`). Valid questions are kept, invalid ones are logged (`✗ invalid JSON item dropped: …`), and the model is re-asked only when no item is valid. Difficulty labels such as "Medium" or unknown values no longer invalidate a question.
2. **Rejected questions are fed back.** Repair rounds now list both accepted and *rejected* questions under "Do NOT repeat or rephrase", so the model does not regenerate known failures. Duplicates are logged as duplicates.
3. **Prompt-injection hardening.** Notes, summaries and judge evidence are wrapped in `<notes>`, `<summary>` and `<evidence>` tags; closing tags inside uploaded text are removed so the fence cannot be broken. The system prompt says text inside those tags is data, not instructions. The user's topic is sanitised (quotes, braces, tags and newlines removed, 120-character cap) before interpolation.

## Text-to-video module (`lcg/t2v.py`)

- **Gemini Omni Flash** (`gemini-omni-1.1-flash`) through the Gemini Interactions API, inline or URI delivery. Uses `GEMINI_API_KEY`. Veo 3.1 Preview was shut down on 22 Oct 2026 and Sora 2 on 24 Sep 2026, so neither is used.
- **Hugging Face Inference Providers** (default `Wan-AI/Wan2.2-TI2V-5B`) using `HF_TOKEN`. The live provider is resolved at runtime (fal-ai queue protocol or hf-inference). Implemented with `requests`; `huggingface_hub` was not added because its `click` requirement conflicts with gTTS.
- A standardised video prompt: planned shots + the shared `VIDEO_STYLE` + negative cues (no on-screen text, no dialogue).
- `video.make_t2v_video` fits any clip to **5–10 s** (trim, or smooth reverse loop for short clips), crops to 16:9 at 960×540, burns in captions and replaces the audio with narration.
- Sidebar **Video model**: auto | gemini | huggingface | slideshow. Failures fall back to the image sequence and the reason is logged and shown in the Video tab.

## Validation

| Check | Result |
|---|---|
| Full test suite (Linux, CPython 3.13.18, pinned lock file) | `66 passed` (26 new) |
| `pip check` | no broken requirements; no new dependencies |
| Offline pipeline, both samples | 5/5 verified questions, 8.0 s MP4 |
| Text-to-video fitting | 2 s, 8 s and 14 s clips at 640×360, 1280×720 and 480×480 → 5–10 s, 960×540, audio track present |
| Gemini / Hugging Face protocols | request shape, polling, download and error paths tested with simulated responses |
| Browser, Video model = gemini | video request attempted, failure logged, image-sequence fallback played (8 s, no media error) |

**Not verified live:** no Gemini or Hugging Face credentials were available and those hosts are blocked from the test environment, so real text-to-video generation, quotas, pricing and the exact current Hugging Face provider mapping for Wan 2.2 were not exercised. Run once with a real key before submission.

## Free text-to-video backend (added 18 November 2026)

- **NovAI `cogvideox-flash`** (`novai` in the Video model menu, `NOVAI_API_KEY`). Free with email sign-up at aiapi-pro.com, no card. NovAI's free page states 5 free video generations/day within 10 free-model calls/day and 5 requests/min, resetting 00:00 UTC+8. A 429 is reported as "daily limit reached" and the app falls back to the image sequence.
- Protocol (from aiapi-pro.com/docs): `POST /v1/video/generations` → poll `GET /v1/video/generations/{id}?model=…` until `succeeded` → download `video_url`. Prompts are compacted to ≤500 characters for this model class.
- Auto order: gemini → huggingface → novai → image sequence.
- Not verified live: the free quota and output quality are NovAI's own claims, and its host is blocked from the test environment. Tested with simulated responses only.

## Timeout fixes (19 November 2026)

- Text LLM timeout raised from 90 s to 180 s (`LLM_TIMEOUT`). A Gemini timeout, 429 (no free-tier quota for that model), 500 or 503 now tries the next Gemini model (`gemini-3.8-flash` → `3.6-flash` → `3.5-flash-lite` → `3.1-flash-lite`) instead of dropping to offline mode. An invalid key fails at once. Later calls in the same run start with the model that answered.
- NovAI submit timeout raised from 60 s to 300 s (`NOVAI_SUBMIT_TIMEOUT`) and never auto-retried, so a timeout cannot spend a second free video. Slow status checks keep polling. Pasted keys are stripped of spaces.
- Test suite: `71 passed`.

## v6 (19 November 2026): real local models, no API keys

- Retrieval: hybrid dense + keyword. all-MiniLM-L6-v2 embeddings in a persistent ChromaDB vector store, fused with TF-IDF by Reciprocal Rank Fusion; on-topic check uses semantic or keyword evidence. Falls back to TF-IDF only if the vector store cannot load (logged).
- Text: Ollama provider (local, no key), auto-selected when no API key is set.
- Video: local AnimateDiff-Lightning text-to-video (one shot per scene), fitted to 5-10 s with smooth slow motion and blended frames; setup_local_video.bat installs PyTorch (CUDA if an NVIDIA GPU is present) and downloads the weights.
- Narration: offline pyttsx3 voice at 140 wpm (adjustable). The script is measured after recording and shortened until it fits; speech is never sped up. gTTS removed (its click pin conflicted with diffusers).
- Verified here: the AnimateDiff code path against diffusers 0.43 with a tiny locally-built model, narration timing with a real offline voice, Windows wheels for every pinned package. Not verifiable here: downloading/running the full-size video and embedding models (blocked from this environment) - they run on your PC.
