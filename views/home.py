"""Intro page: 3D graduation cap + button to the studio."""
import streamlit as st

from lcg.ui import html_frame

left, right = st.columns([1.05, 1], vertical_alignment="center")
with left:
    st.markdown('<p class="big">Turn any course notes into a complete lesson</p>'
                '<p class="lead">Upload a PDF and get a grounded summary, a quiz verified against your material, '
                'illustrations and a short narrated explainer video.</p>', unsafe_allow_html=True)
    if st.button("Get started  →", type="primary"):
        st.switch_page("views/studio.py")
with right:
    html_frame("cap.html", 440)

st.markdown("&nbsp;")
cards = [("📄", "RAG over your notes", "PDF, DOCX, TXT or MD is chunked and searched, so every answer is grounded in your own material."),
         ("❓", "Verified quiz", "Multiple-choice JSON, validated, then checked: the evidence quote must exist and a blind judge must agree."),
         ("🖼️", "Consistent illustrations", "Text-to-image with one standard style prompt, so every picture looks like part of the same lesson."),
         ("🎬", "5–10 s explainer", "Image sequence with captions and narration, rendered as a downloadable MP4.")]
for col, (ic, t, d) in zip(st.columns(4), cards):
    col.markdown(f'<div class="glass"><div class="ic">{ic}</div><b>{t}</b><p>{d}</p></div>', unsafe_allow_html=True)

st.markdown("&nbsp;")
st.subheader("How it works")
for col, (n, t) in zip(st.columns(4), [("1", "Upload your notes"), ("2", "Pick or type a topic"),
                                       ("3", "Generate the lesson package"), ("4", "Take the quiz, watch the video")]):
    col.markdown(f'<span class="step">{n}</span>{t}', unsafe_allow_html=True)
