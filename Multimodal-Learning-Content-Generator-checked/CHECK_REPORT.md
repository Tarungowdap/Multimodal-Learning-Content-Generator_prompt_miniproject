# Runtime check and corrections

Reviewed: 8 November 2026 (Asia/Kolkata).
Input: `Multimodal-Learning-Content-Generator (2).zip`.

The corrected project runs locally on Windows with CPython 3.13.9 and Streamlit 1.67.0. All 40 automated tests pass. The browser workflow was also exercised: upload notes, choose a topic, generate a package, answer the quiz, play the video and download quiz JSON. Testing reduces risk; it cannot establish that every possible input and external service will be error-free.

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
