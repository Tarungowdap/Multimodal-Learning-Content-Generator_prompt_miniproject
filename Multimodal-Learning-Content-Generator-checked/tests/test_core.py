import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from lcg.config import Settings
from lcg.generators import ExtractiveGenerator, LLMGenerator
from lcg.ingest import build_chunks, read_path
from lcg.llm import LLMClient, extract_json, generate_json
from lcg.retriever import Retriever
from lcg.schemas import Question, Quiz
from lcg.verify import quote_evidence, verify_and_repair

NOTES = Path(__file__).resolve().parents[1] / "sample_data" / "photosynthesis.md"


@pytest.fixture(scope="module")
def chunks():
    return build_chunks([("photosynthesis.md", read_path(NOTES))])


def good_q(**kw):
    d = dict(question="Which pigment absorbs red and blue light?", options=["Chlorophyll", "Keratin", "Melanin", "Hemoglobin"],
             answer_index=0, explanation="Chlorophyll absorbs red and blue light.",
             source_quote="Chloroplasts contain a green pigment named chlorophyll, which absorbs red and blue light")
    d.update(kw)
    return Question(**d)


# ---------------------------------------------------------------- schema validation
def test_schema_accepts_valid():
    assert good_q().correct_option == "Chlorophyll"


@pytest.mark.parametrize("bad", [
    dict(options=["a", "b", "c"]),                       # 3 options
    dict(options=["a", "a", "b", "c"]),                  # duplicates
    dict(answer_index=4),                                # out of range
    dict(source_quote=""),                               # no evidence
])
def test_schema_rejects_invalid(bad):
    with pytest.raises(ValidationError):
        good_q(**bad)


def test_extract_json_handles_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"a": [1, {"b": "}"}]}\n```\nbye') == {"a": [1, {"b": "}"}]}


# ---------------------------------------------------------------- retrieval
def test_retrieval_finds_relevant_chunk(chunks):
    hit = Retriever(chunks).search("calvin cycle stroma", 1)[0]
    assert "Calvin cycle" in hit.chunk.text


def test_topic_coverage_low_for_offtopic(chunks):
    r = Retriever(chunks)
    assert r.topic_coverage("calvin cycle") > 0.05 > r.topic_coverage("medieval french poetry")


# ---------------------------------------------------------------- verification
def test_quote_check_exact_and_fuzzy_and_fake(chunks):
    assert quote_evidence("chlorophyll, which absorbs red and blue light", chunks)[0]
    assert quote_evidence("Chlorophyl which absorbs red and blue light", chunks)[0]          # typo tolerated
    assert not quote_evidence("Mitochondria are the powerhouse of the cell and make ATP", chunks)[0]


class FakeGen:
    """Stub generator: first batch is bad (fake quote / wrong key), repair batch is good."""
    def __init__(self):
        self.calls = 0
        self.real = ExtractiveGenerator()

    def make_quiz(self, topic, ctx, n, avoid=None):
        self.calls += 1
        return [good_q()] if self.calls > 1 else [
            good_q(source_quote="Mitochondria are the powerhouse of the whole cell"),   # hallucinated evidence
            good_q(question="Which pigment reflects green light in leaves?", answer_index=1,
                   options=["Chlorophyll", "Keratin", "Melanin", "Hemoglobin"]),         # wrong key
        ]

    def judge(self, q, ev):
        return self.real.judge(q, ev) if False else (0 if "pigment" in q.question else None)


def test_verifier_rejects_bad_and_repairs(chunks):
    r, gen = Retriever(chunks), FakeGen()
    bad = gen.make_quiz("photosynthesis", chunks, 2)
    qs, rep = verify_and_repair("photosynthesis", bad, 1, r, gen, chunks, max_rounds=2)
    assert len(qs) == 1 and qs[0].source_quote.startswith("Chloroplasts")
    assert rep.dropped == 2 and rep.pass_rate == 1.0 and rep.attempts == 2


# ---------------------------------------------------------------- LLM JSON path (mocked, no network)
class MockClient(LLMClient):
    def __init__(self, replies):
        self.replies, self.n = replies, 0

    def complete(self, system, prompt, **kw):
        r = self.replies[min(self.n, len(self.replies) - 1)]
        self.n += 1
        return r


def test_generate_json_self_repairs():
    valid = Quiz(topic="t", questions=[good_q()]).model_dump_json()
    c = MockClient(["not json at all", '{"topic": "t", "questions": []}', "```json\n" + valid + "\n```"])
    assert generate_json(c, "s", "p", Quiz, retries=2).questions[0].correct_option == "Chlorophyll"
    assert c.n == 3


def test_llm_generator_shuffles_but_keeps_key(chunks):
    reply = Quiz(topic="t", questions=[good_q()]).model_dump_json()
    g = LLMGenerator(MockClient([reply]), seed=1)
    q = g.make_quiz("photosynthesis", chunks, 1)[0]
    assert q.correct_option == "Chlorophyll"


# ---------------------------------------------------------------- offline end-to-end incl. media constraints
def test_full_pipeline_offline(tmp_path, monkeypatch, chunks):
    import lcg.pipeline as P
    from lcg.video import audio_duration
    monkeypatch.setattr(P, "OUTPUT_DIR", tmp_path)
    s = Settings(llm_provider="offline", image_backend="placeholder", num_questions=4)
    pkg = P.run_pipeline("photosynthesis", chunks, s)
    assert len(pkg.quiz.questions) == 4 and pkg.verification.pass_rate == 1.0
    assert len(pkg.image_paths) == len(pkg.scenes) >= 2 and all(Path(p).exists() for p in pkg.image_paths)
    assert 5.0 <= audio_duration(Path(pkg.video_path)) <= 10.5          # 5-10 s requirement
    Quiz.model_validate(json.loads((Path(pkg.video_path).parent / "quiz.json").read_text()))


def test_pipeline_refuses_offtopic(chunks):
    import lcg.pipeline as P
    with pytest.raises(ValueError):
        P.run_pipeline("medieval french poetry", chunks, Settings(llm_provider="offline"), make_media=False)


# ---------------------------------------------------------------- image backends
class _Resp:
    def __init__(self, code, content=b"", ctype="image/png"):
        self.status_code, self.content, self.headers = code, content, {"content-type": ctype}


def test_pollinations_retries_after_rate_limit(monkeypatch):
    import io
    from PIL import Image
    import lcg.images as I
    buf = io.BytesIO(); Image.new("RGB", (300, 300), "red").save(buf, "PNG")
    png = buf.getvalue() + b"0" * 6000
    calls = iter([_Resp(429, b"", "text/plain"), _Resp(403, b"", "text/plain"), _Resp(200, png)])
    monkeypatch.setattr(I.requests, "get", lambda *a, **k: next(calls))
    monkeypatch.setattr(I.time, "sleep", lambda s: None)
    assert I._pollinations("a cell", 1, None) == png


def test_images_fall_back_to_readable_infographic(tmp_path, monkeypatch):
    import lcg.images as I
    monkeypatch.setattr(I.requests, "get", lambda *a, **k: (_ for _ in ()).throw(I.requests.ConnectionError("down")))
    monkeypatch.setattr(I.time, "sleep", lambda s: None)
    path, src = I.generate_image("cell", "photosynthesis", 0, tmp_path, Settings(image_backend="pollinations"),
                                 heading="Light reactions", detail="Light energy splits water in the thylakoid.")
    assert src == "infographic" and path.exists()


def test_no_key_falls_back_to_offline_with_warning():
    import lcg.pipeline as P
    gen, name = P.make_generator(Settings(llm_provider="gemini"))
    assert name == "offline-nokey"


def test_topic_suggestions_come_from_document(chunks):
    from lcg.topics import suggest_topics
    t = suggest_topics(chunks)
    assert t and any("photosynthesis" in x or "calvin" in x or "chloroplast" in x for x in t)
