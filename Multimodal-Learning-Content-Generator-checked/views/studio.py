"""Studio page: upload course material -> summary, verified quiz, illustrations, video."""
from pathlib import Path
import hashlib

import streamlit as st

from lcg.config import Settings
from lcg.ingest import build_chunks, read_document
from lcg.pipeline import run_pipeline
from lcg.topics import suggest_topics

st.page_link("views/home.py", label="← Back to home")
st.title("🎓 Lesson Studio")
st.caption("Course notes → RAG → summary · verified quiz · illustrations · short explainer video")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("1 · Course material")
    uploads = st.file_uploader("Upload notes (PDF, DOCX, TXT, MD)", type=["pdf", "docx", "txt", "md"],
                               accept_multiple_files=True)

    st.header("2 · Models")
    provider = st.selectbox("Text LLM", ["auto", "anthropic", "openai", "gemini", "offline"],
                            help="'auto' picks the first provider with an API key; 'offline' needs no key.")
    keys = {}
    if provider in ("auto", "anthropic", "openai", "gemini"):
        for p in ("anthropic", "openai", "gemini"):
            if provider in ("auto", p):
                keys[p] = st.text_input(f"{p.title()} API key (optional if set in env)", type="password", key=f"k_{p}")
    img_backend = st.selectbox("Image model", ["auto", "pollinations", "openai", "huggingface", "placeholder"],
                               help="auto: OpenAI → HuggingFace FLUX → Pollinations → readable infographic slides")
    if img_backend in ("auto", "openai") and "openai" not in keys:
        keys["openai"] = st.text_input("OpenAI image API key (optional if set in env)", type="password", key="k_openai_image")
    if img_backend in ("auto", "huggingface"):
        keys["huggingface"] = st.text_input("HuggingFace token (optional)", type="password", key="k_hf")
    if img_backend in ("auto", "pollinations"):
        keys["pollinations"] = st.text_input("Pollinations key (optional, removes rate limit)", type="password", key="k_poll")
    n_q = st.slider("Quiz questions", 3, 10, 5)
    st.caption("For fully offline use, select Text LLM = offline and Image model = placeholder. "
               "Offline narration uses espeak if installed; otherwise the video has captions and silent audio.")

st.subheader("3 · Topic")


@st.cache_data(show_spinner="Reading your document…")
def _load(files):
    docs = [(n, read_document(n, d)) for n, d in files]
    return docs, suggest_topics(build_chunks(docs))


docs, suggestions = ([], [])
fingerprint = tuple((f.name, hashlib.sha256(f.getvalue()).hexdigest()) for f in (uploads or []))
if fingerprint != st.session_state.get("document_fingerprint"):
    st.session_state.document_fingerprint = fingerprint
    st.session_state.pop("pkg", None)
    st.session_state.pop("answers", None)

if uploads:
    try:
        docs, suggestions = _load(tuple((f.name, f.getvalue()) for f in uploads))
    except Exception as e:
        st.error(f"Could not read the uploaded document. Check that it is a valid, unencrypted file. ({e})")
    words = sum(len(t.split()) for _, t in docs)
    if words < 50:
        st.error("Almost no text could be read from this file. Scanned PDFs have no text layer, so try a text-based PDF.")
        docs = []
    else:
        st.success(f"Read {len(uploads)} file(s), about {words:,} words.")
else:
    st.info("⬅ Upload your PDF, DOCX, TXT or MD file in the sidebar to begin.")

pick = st.pills("Suggested topics from your document", suggestions, selection_mode="single") if suggestions else None
typed = st.text_input("Topic (optional: pick a suggestion, type your own, or leave empty for the whole document)",
                      placeholder="e.g. a chapter or concept from your document")
topic = (typed or pick or "").strip()
go = st.button("✨ Generate lesson package", type="primary", disabled=not docs)

# ------------------------------------------------------------------ generate
if go:
    st.session_state.pop("pkg", None)
    st.session_state.pop("answers", None)
    for key in list(st.session_state):
        if key.startswith("q") and key[1:].isdigit():
            del st.session_state[key]
    chunks = build_chunks(docs)
    if not topic:   # no topic given -> cover the document's main subject
        topic = " ".join(suggestions[:3]) or "main ideas"
    settings = Settings(llm_provider=provider, image_backend=img_backend,
                        api_keys={k: v for k, v in keys.items() if v}, num_questions=n_q)
    bar, status = st.progress(0.0), st.empty()

    def progress(p, msg):
        if p >= 0:
            bar.progress(min(1.0, p))
        status.info(msg)

    try:
        with st.spinner("Running the generation chain…"):
            st.session_state.pkg = run_pipeline(topic, chunks, settings, progress)
        status.success("Lesson package ready")
    except Exception as e:
        status.error(str(e))

# ------------------------------------------------------------------ results
pkg = st.session_state.get("pkg")
if pkg:
    if pkg.llm_provider.startswith("offline"):
        st.warning("⚠ Offline mode was used, so the quiz is basic fill-in-the-blank and the summary is extracted "
                   "sentences. For real MCQs get a free Gemini key at aistudio.google.com, choose **gemini** in the "
                   "sidebar and paste the key. Details are in the Pipeline log tab.")
    if pkg.image_sources and all(x == "infographic" for x in pkg.image_sources):
        st.info("ℹ No image model was reachable, so lesson slides were drawn instead. See the Pipeline log for why.")
    t_sum, t_quiz, t_img, t_vid, t_dl, t_log = st.tabs(
        ["📝 Summary", "❓ Quiz", "🖼️ Illustrations", "🎬 Video", "⬇️ Export", "🔍 Pipeline log"],
        key="lesson_tabs")

    with t_sum:
        st.header(pkg.summary.title)
        st.write(pkg.summary.summary)
        st.markdown("**Key points**\n" + "\n".join(f"- {k}" for k in pkg.summary.key_points))
        st.caption("Sources: " + ", ".join(pkg.sources) + f" · generated by **{pkg.llm_provider}**")

    with t_quiz:
        v = pkg.verification
        st.success(f"✅ {len(v.checks)} questions verified against your notes "
                   f"({v.pass_rate:.0%} pass rate · {v.dropped} rejected · {v.attempts} round(s))")
        answers = st.session_state.setdefault("answers", {})
        for i, q in enumerate(pkg.quiz.questions):
            st.markdown(f"**Q{i + 1}. {q.question}**  `{q.difficulty}`")
            answers[i] = st.radio("Your answer", q.options, index=None, key=f"q{i}", label_visibility="collapsed")
            with st.expander("Evidence & verification"):
                c = v.checks[i]
                st.write(f"Source quote: “{q.source_quote}”")
                st.write(f"Quote found in notes: **{c.quote_found}** (similarity {c.evidence_score}) · "
                         f"Blind-judge agrees with key: **{c.answer_supported}**")
        if st.button("Submit answers"):
            score = 0
            for i, q in enumerate(pkg.quiz.questions):
                ok = answers.get(i) == q.correct_option
                score += ok
                (st.success if ok else st.error)(
                    f"Q{i + 1}: {'Correct' if ok else 'Incorrect'} — answer: **{q.correct_option}**. {q.explanation}")
            st.metric("Score", f"{score}/{len(pkg.quiz.questions)}")

    with t_img:
        cols = st.columns(len(pkg.image_paths))
        for col, p, sc in zip(cols, pkg.image_paths, pkg.scenes):
            col.image(p, caption=sc.caption, width="stretch")

    with t_vid:
        if pkg.video_path:
            st.video(pkg.video_path)
            st.caption("Narration: " + " ".join(s.narration for s in pkg.scenes))

    with t_dl:
        st.download_button("quiz.json", pkg.quiz.model_dump_json(indent=2), "quiz.json", "application/json")
        st.download_button("lesson.json (full package)", pkg.model_dump_json(indent=2), "lesson.json", "application/json")
        if pkg.video_path:
            st.download_button("explainer.mp4", Path(pkg.video_path).read_bytes(), "explainer.mp4", "video/mp4")
        st.json(pkg.quiz.model_dump(), expanded=False)

    with t_log:
        st.code("\n".join(pkg.log))
