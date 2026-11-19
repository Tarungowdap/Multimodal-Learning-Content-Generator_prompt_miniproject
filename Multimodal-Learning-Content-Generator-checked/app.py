"""Entry point / router.   Run:  streamlit run app.py

  views/home.py    intro page with the 3D graduation cap
  views/studio.py  upload notes -> summary, verified quiz, illustrations, video
"""
import streamlit as st

from lcg.ui import inject_css, load_secrets

st.set_page_config(page_title="Multimodal Learning-Content Generator", page_icon="🎓", layout="wide")
load_secrets()
inject_css()

pages = [
    st.Page("views/home.py", title="Home", icon="🏠", default=True),
    st.Page("views/studio.py", title="Lesson Studio", icon="🎓"),
]
st.navigation(pages, position="hidden").run()   # navigation is done with our own buttons
