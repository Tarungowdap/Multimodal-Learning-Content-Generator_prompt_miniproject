from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_home_then_studio_navigation():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception and [b.label for b in at.button] == ["Get started  →"]
    at.button[0].click().run()
    assert not at.exception
    assert [t.value for t in at.title] == ["🎓 Lesson Studio"]
    # nothing is pre-filled: generation is disabled until a document is uploaded
    assert all(b.disabled for b in at.button if "Generate" in b.label)
