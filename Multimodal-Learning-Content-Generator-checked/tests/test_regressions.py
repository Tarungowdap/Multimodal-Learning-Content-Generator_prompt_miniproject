"""Regression cases found during the runtime review. No external API keys needed."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from lcg.config import Settings
from lcg.generators import ExtractiveGenerator
from lcg.ingest import Chunk, build_chunks, read_path
from lcg.llm import LLMClient, LLMError, extract_json
from lcg.schemas import Question
from lcg.verify import quote_evidence


def question(**changes):
    fields = dict(question="Which option completes the sentence?",
                  options=["A) same", "B) same", "C) third", "D) fourth"],
                  answer_index=0, explanation="The notes support this answer.",
                  source_quote="This is a sentence from the notes.")
    return Question(**(fields | changes))


def test_options_are_validated_after_removing_labels():
    with pytest.raises(ValidationError):
        question()
    with pytest.raises(ValidationError):
        question(options=["A) ", "B) second", "C) third", "D) fourth"])


def test_json_escaped_quotes_and_delimiters():
    value = {"quote": 'He said "}" and then "[".', "path": "a\\b"}
    assert extract_json("Here is JSON:\n" + json.dumps(value) + "\nDone") == value


@pytest.mark.parametrize("quote", [
    "Chlorophyll does not absorb red and blue light.",
    "The Calvin cycle uses 30 ATP molecules to produce sugar.",
])
def test_fuzzy_evidence_cannot_change_negation_or_numbers(quote):
    chunks = [Chunk("n#0", "n", "Chlorophyll absorbs red and blue light. "
                    "The Calvin cycle uses 3 ATP molecules to produce sugar.")]
    assert not quote_evidence(quote, chunks)[0]


def test_evidence_preserves_decimal_values():
    chunks = [Chunk("n#0", "n", "The reaction uses 30 grams of glucose in this experiment.")]
    assert not quote_evidence("The reaction uses 3.0 grams of glucose in this experiment.", chunks)[0]


def test_offline_quiz_respects_avoid_list():
    root = Path(__file__).resolve().parents[1]
    chunks = build_chunks([("notes", read_path(root / "sample_data/photosynthesis.md"))])
    gen = ExtractiveGenerator()
    first = gen.make_quiz("photosynthesis", chunks, 2)
    avoided = [q.question for q in first]
    assert first
    assert all(q.question not in avoided for q in gen.make_quiz("photosynthesis", chunks, 10, avoid=avoided))


def test_offline_scene_narration_keeps_complete_facts():
    root = Path(__file__).resolve().parents[1]
    chunks = build_chunks([("notes", read_path(root / "sample_data/photosynthesis.md"))])
    gen = ExtractiveGenerator()
    scenes = gen.plan_scenes("photosynthesis", gen.summarize("photosynthesis", chunks))
    assert any("fix carbon dioxide into glucose" in scene.narration for scene in scenes)
    assert all(not scene.narration.endswith("...") for scene in scenes)


@pytest.mark.parametrize("provider,payload", [
    ("openai", {"choices": []}),
    ("gemini", {"candidates": []}),
    ("anthropic", {"content": []}),
])
def test_empty_provider_response_is_a_controlled_error(monkeypatch, provider, payload):
    class Response:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return payload
    monkeypatch.setattr("lcg.llm.requests.post", lambda *a, **kw: Response())
    with pytest.raises(LLMError):
        LLMClient(provider, "fake-key").complete("system", "prompt")


@pytest.mark.parametrize("stage", ["quiz", "scenes"])
def test_pipeline_recovers_from_late_llm_failure(monkeypatch, stage):
    import lcg.pipeline as pipeline
    class Generator(ExtractiveGenerator):
        def make_quiz(self, *args, **kwargs):
            if stage == "quiz":
                raise LLMError("service unavailable")
            return super().make_quiz(*args, **kwargs)
        def plan_scenes(self, *args, **kwargs):
            if stage == "scenes":
                raise LLMError("service unavailable")
            return super().plan_scenes(*args, **kwargs)
    root = Path(__file__).resolve().parents[1]
    chunks = build_chunks([("notes", read_path(root / "sample_data/photosynthesis.md"))])
    monkeypatch.setattr(pipeline, "make_generator", lambda s: (Generator(), "openai"))
    package = pipeline.run_pipeline("photosynthesis", chunks, Settings(), make_media=False)
    assert package.quiz.questions and package.scenes
    assert any("offline" in line.lower() for line in package.log)


def test_offline_narration_never_calls_gtts(tmp_path, monkeypatch):
    import gtts
    import lcg.video as video
    def forbidden(*args, **kwargs):
        pytest.fail("offline narration contacted gTTS")
    monkeypatch.setattr(gtts, "gTTS", forbidden)
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    assert video.synthesize("A lesson.", tmp_path / "audio.mp3", allow_online=False) is None


def test_repeated_runs_keep_media_and_complete_logs(tmp_path, monkeypatch):
    import lcg.pipeline as pipeline
    root = Path(__file__).resolve().parents[1]
    chunks = build_chunks([("notes", read_path(root / "sample_data/photosynthesis.md"))])
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path)
    def video(images, captions, narration, path, log, **kwargs):
        path.write_bytes(b"test-video")
        return path, 8.0
    monkeypatch.setattr(pipeline, "make_video", video)
    settings = Settings(llm_provider="offline", image_backend="placeholder")
    first = pipeline.run_pipeline("photosynthesis", chunks, settings)
    second = pipeline.run_pipeline("photosynthesis", chunks, settings)
    assert first.video_path != second.video_path
    assert Path(first.video_path).exists() and Path(second.video_path).exists()
    assert "Video rendered" in first.log[-1]
    saved = json.loads((Path(first.video_path).parent / "lesson.json").read_text(encoding="utf-8"))
    assert saved["log"] == first.log


def test_long_narration_is_fitted_to_video(tmp_path, monkeypatch):
    import wave
    import numpy as np
    from PIL import Image
    import lcg.video as video
    def synthesize(text, out, log, **kwargs):
        path = out.with_suffix(".wav")
        rate = 16000
        t = np.arange(rate * 14) / rate
        # Speech substitute: a non-silent tone lasting longer than the video limit.
        samples = (np.sin(2 * np.pi * 440 * t) * 12000).astype("<i2")
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(rate)
            audio.writeframes(samples.tobytes())
        return path
    monkeypatch.setattr(video, "synthesize", synthesize)
    image = tmp_path / "image.png"
    Image.new("RGB", (768, 432), "teal").save(image)
    log = []
    path, duration = video.make_video([image], ["A lesson"], "Narration", tmp_path / "clip.mp4", log.append)
    assert duration == 10.0
    assert 9.9 <= video.audio_duration(path) <= 10.1
    assert any("speech speed" in line for line in log)
    import subprocess
    result = subprocess.run([video.FFMPEG, "-i", str(path), "-vn", "-f", "s16le", "-ac", "1",
                             "-ar", "16000", "pipe:1"], capture_output=True, check=True)
    decoded = np.frombuffer(result.stdout, dtype="<i2")
    assert np.abs(decoded[8 * 16000:9 * 16000].astype(float)).mean() > 1000


@pytest.mark.parametrize("extension", ["txt", "md", "docx", "pdf"])
def test_supported_document_formats(extension):
    import io
    from lcg.ingest import read_document
    text = "Photosynthesis turns light energy into chemical energy."
    if extension in ("txt", "md"):
        raw = text.encode("utf-8")
    elif extension == "docx":
        from docx import Document
        document, buf = Document(), io.BytesIO()
        document.add_paragraph(text)
        document.save(buf)
        raw = buf.getvalue()
    else:
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
        writer, buf = PdfWriter(), io.BytesIO()
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = stream
        writer.write(buf)
        raw = buf.getvalue()
    assert text in read_document(f"notes.{extension}", raw)


def test_corrupt_upload_is_reported_without_crashing(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    class Upload:
        name = "broken.pdf"
        def getvalue(self):
            return b"This is not a PDF file."
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: [Upload()])
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "app.py"), default_timeout=60).run()
    app.button[0].click().run()
    assert not app.exception
    assert any("Could not read" in error.value for error in app.error)
    assert all(button.disabled for button in app.button if "Generate" in button.label)


def test_quiz_tab_keeps_identity_after_answer(tmp_path, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from PIL import Image
    import lcg.pipeline as pipeline
    root = Path(__file__).resolve().parents[1]
    class Upload:
        name = "photosynthesis.md"
        def getvalue(self):
            return (root / "sample_data/photosynthesis.md").read_bytes()
    chunks = build_chunks([("notes", read_path(root / "sample_data/photosynthesis.md"))])
    package = pipeline.run_pipeline("photosynthesis", chunks, Settings(llm_provider="offline"), make_media=False)
    for i, scene in enumerate(package.scenes):
        path = tmp_path / f"scene{i}.png"
        Image.new("RGB", (768, 432), "teal").save(path)
        package.image_paths.append(str(path))
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: [Upload()])
    monkeypatch.setattr(pipeline, "run_pipeline", lambda *a, **kw: package)
    app = AppTest.from_file(str(root / "app.py"), default_timeout=60).run()
    app.button[0].click().run()
    next(b for b in app.button if "Generate" in b.label).click().run()
    assert not app.exception
    before = app.get("tab_container")[0].proto.id
    app.radio[0].set_value(package.quiz.questions[0].correct_option).run()
    assert not app.exception
    assert before and before == app.get("tab_container")[0].proto.id
    assert app.radio[0].value == package.quiz.questions[0].correct_option
