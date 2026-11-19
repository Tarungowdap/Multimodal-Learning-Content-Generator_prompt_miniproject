"""Shared Streamlit UI helpers (theme, secrets, 3D iframe)."""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parents[1] / "assets"

CSS = """<style>
.stApp{background:radial-gradient(1200px 650px at 85% -10%,#1e1b4b 0,#070b1a 62%)}
h1,h2,h3{background:linear-gradient(90deg,#5eead4,#a78bfa);-webkit-background-clip:text;background-clip:text;color:transparent!important}
[data-testid="stSidebar"]{background:rgba(255,255,255,.04);backdrop-filter:blur(14px)}
div.stButton>button,a[data-testid="stPageLink-NavLink"]{border:0;border-radius:14px;font-weight:700;transition:transform .15s}
div.stButton>button[kind="primary"]{background:linear-gradient(90deg,#5eead4,#a78bfa);color:#06121a;padding:.7rem 1.6rem;font-size:1.05rem}
div.stButton>button:hover{transform:translateY(-2px) scale(1.03)}
[data-testid="stTabs"] [role="tabpanel"]{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.14);border-radius:18px;padding:18px;margin-top:10px}
img,video{border-radius:14px}
.glass{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.15);border-radius:20px;padding:20px;height:100%;
backdrop-filter:blur(14px);box-shadow:0 18px 40px rgba(0,0,0,.28);transition:transform .2s}
.glass:hover{transform:translateY(-6px) rotateX(3deg)}
.glass .ic{font-size:34px}.glass b{font-size:1.05rem}.glass p{color:#a5b0d6;margin:.3rem 0 0;font-size:.93rem}
.big{font-size:clamp(2.2rem,5vw,3.6rem);font-weight:800;line-height:1.1;margin:0 0 .6rem;
background:linear-gradient(90deg,#5eead4,#a78bfa);-webkit-background-clip:text;background-clip:text;color:transparent}
.lead{color:#cbd5f5;font-size:1.15rem;max-width:34rem}
.step{display:inline-block;width:34px;height:34px;line-height:34px;text-align:center;border-radius:50%;
background:linear-gradient(90deg,#5eead4,#a78bfa);color:#06121a;font-weight:800;margin-right:8px}
</style>"""


def load_secrets() -> None:
    """Streamlit Cloud / HF Spaces secrets -> environment variables."""
    try:
        for k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN", "POLLINATIONS_API_KEY"):
            if k in st.secrets:
                os.environ.setdefault(k, st.secrets[k])
    except Exception:
        pass


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def html_frame(name: str, height: int) -> None:
    html = (ASSETS / name).read_text(encoding="utf-8")
    if hasattr(st, "iframe"):      # newer Streamlit (components.v1.html is being retired)
        st.iframe(html, height=height)
    else:  # pragma: no cover
        import streamlit.components.v1 as components
        components.html(html, height=height)
