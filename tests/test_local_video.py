"""Local (no-key) text-to-video, calm narration and the Ollama text model. No network needed."""
import importlib.util
import json
from pathlib import Path

import pytest

from lcg import video
from lcg.config import Settings
from lcg.ingest import build_chunks, read_path
from lcg.schemas import Scene

ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "diffusers", "transformers"))
SCENES = [Scene(caption="Light hits the leaf", image_subject="sunlight falling on a green leaf", narration="Light reaches the leaf."),
          Scene(caption="Glucose is made", image_subject="glucose molecules forming in a plant cell", narration="The plant makes sugar."),
          Scene(caption="Oxygen is released", image_subject="oxygen bubbles rising from a water plant", narration="Oxygen goes into the air.")]


# ------------------------------------------------------------------ calm narration
def test_narration_fits_ten_seconds_at_calm_pace():
    long = ["Photosynthesis is the process by which green plants, algae and some bacteria convert light energy "
            "into chemical energy stored in glucose.", "Short one.",
            "The Calvin cycle uses ATP and NADPH to fix carbon dioxide into glucose, and it does not require light."]
    out = video.fit_narration(long, wpm=140)
    words = sum(len(x.split()) for x in out)
    assert words <= video.word_budget(140) and words * 60 / 140 <= video.MAX_S - 0.6
    assert all(x.endswith(".") for x in out)
    assert not any(x.rstrip(".").split()[-1].lower() in video._WEAK_END for x in out)   # no "...by which."
    assert out[1] == "Short one."
    assert video.word_budget(110) < video.word_budget(170)            # slower speech -> shorter script


def test_speech_rate_is_passed_to_the_voice(tmp_path, monkeypatch):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"RIFF" + b"\0" * 2000)
        return type("R", (), {"returncode": 0, "stderr": b""})()
    monkeypatch.setattr(video.subprocess, "run", run)
    out = video.synthesize("Hello there.", tmp_path / "n.mp3", allow_online=False, wpm=120)
    assert out.suffix == ".wav" and calls[0][-2] == "120"              # pyttsx3 rate = 120 words/minute


def test_mux_never_speeds_speech_up(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(video.subprocess, "run", lambda cmd, **k: seen.append(cmd) or type("R", (), {"returncode": 0})())
    video._mux(tmp_path / "v.mp4", tmp_path / "a.wav", 14.0, 10.0, tmp_path / "o.mp4", lambda m: None)
    af = seen[0][seen[0].index("-af") + 1]
    assert "atempo" not in af and "afade" in af
    seen.clear()
    video._mux(tmp_path / "v.mp4", tmp_path / "a.wav", 9.0, 9.6, tmp_path / "o.mp4", lambda m: None)
    assert "atempo" not in seen[0][seen[0].index("-af") + 1]


# ------------------------------------------------------------------ local text-to-video
def _tiny_pipeline_dir(tmp: Path):
    """A tiny, randomly initialised AnimateDiff pipeline built locally (same code path as the real model)."""
    import torch
    from diffusers import AnimateDiffPipeline, AutoencoderKL, DDIMScheduler, MotionAdapter, UNet2DConditionModel
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import CLIPTextConfig, CLIPTextModel, PreTrainedTokenizerFast

    torch.manual_seed(0)
    kw = dict(block_out_channels=(32, 64), motion_layers_per_block=2, motion_norm_num_groups=2, motion_num_attention_heads=4)
    unet = UNet2DConditionModel(block_out_channels=(32, 64), layers_per_block=2, sample_size=32, in_channels=4,
                                out_channels=4, down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                                up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=32, norm_num_groups=2)
    vae = AutoencoderKL(block_out_channels=[32, 64], in_channels=3, out_channels=3, latent_channels=4, norm_num_groups=2,
                        down_block_types=["DownEncoderBlock2D"] * 2, up_block_types=["UpDecoderBlock2D"] * 2)
    te = CLIPTextModel(CLIPTextConfig(bos_token_id=0, eos_token_id=2, pad_token_id=1, hidden_size=32, intermediate_size=37,
                                      num_attention_heads=4, num_hidden_layers=2, vocab_size=1000))
    tk = Tokenizer(models.WordLevel({"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", pad_token="<pad>",
                                  unk_token="<unk>", model_max_length=77)
    sched = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="linear", clip_sample=False)
    AnimateDiffPipeline(unet=unet, vae=vae, text_encoder=te, tokenizer=tok, scheduler=sched,
                        motion_adapter=MotionAdapter(**kw)).save_pretrained(tmp / "base")
    save_file(MotionAdapter(**kw).state_dict(), str(tmp / "lightning.safetensors"))

    class TinyAdapter(MotionAdapter):
        def __init__(self, **_):
            super().__init__(**kw)
    return tmp / "base", tmp / "lightning.safetensors", TinyAdapter


@pytest.mark.skipif(not HAS_TORCH, reason="local video extras (torch, diffusers) not installed")
def test_local_model_generates_a_real_clip(tmp_path, monkeypatch):
    import diffusers
    import huggingface_hub
    from lcg import local_t2v as L
    base, ckpt, tiny = _tiny_pipeline_dir(tmp_path)
    monkeypatch.setattr(diffusers, "MotionAdapter", tiny)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, f: str(ckpt))
    monkeypatch.setattr(L, "BASE", str(base))
    monkeypatch.setattr(L, "WIDTH", 64)
    monkeypatch.setattr(L, "HEIGHT", 64)
    monkeypatch.setattr(L, "_PIPE", None)
    logs, prog = [], []
    path, prompts = L.generate_clip(SCENES, "photosynthesis", tmp_path / "raw.mp4", log=logs.append,
                                    progress=lambda i, n: prog.append(i))
    frames, fps = video.read_native(path)
    assert fps == L.CLIP_FPS and len(frames) == 3 * L.FRAMES                 # one 16-frame shot per scene
    assert prog == [1, 2, 3] and all(L.LOCAL_STYLE in p for p in prompts)
    assert prompts[0].startswith("sunlight falling on a green leaf")
    out, total = video.make_t2v_video(path, [s.caption for s in SCENES], "Light. Sugar. Oxygen.",
                                      tmp_path / "final.mp4", logs.append, allow_online=False)
    assert 5.0 <= total <= 10.0 and 5.0 <= video.audio_duration(out) <= 10.1
    L._PIPE = None


def test_local_unavailable_is_reported(monkeypatch):
    from lcg import local_t2v as L
    monkeypatch.setattr(L, "status", lambda: {"installed": False, "reason": "torch/diffusers not installed",
                                              "device": "none", "gpu": "", "vram_gb": 0})
    monkeypatch.setattr(L, "_PIPE", None)
    with pytest.raises(L.LocalT2VUnavailable, match="not installed"):
        L.load_pipeline()


def test_pipeline_uses_local_video_without_any_key(tmp_path, monkeypatch):
    import lcg.config as cfg
    import lcg.pipeline as pipeline
    from lcg import local_t2v
    import imageio
    import numpy as np
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "HF_TOKEN", "NOVAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(cfg, "local_video_installed", lambda: True)
    monkeypatch.setattr(cfg, "ollama_models", lambda timeout=1.5: [])
    monkeypatch.setattr(video, "synthesize", lambda *a, **k: None)

    def fake_clip(scenes, topic, out, seed=42, log=print, progress=lambda i, n: None):
        with imageio.get_writer(str(out), fps=8, codec="libx264", macro_block_size=1, pixelformat="yuv420p") as w:
            for k in range(16 * len(scenes)):
                w.append_data(np.full((384, 640, 3), (k * 5) % 255, np.uint8))
        progress(len(scenes), len(scenes))
        return out, [local_t2v.scene_prompt(s, topic) for s in scenes]
    monkeypatch.setattr(local_t2v, "generate_clip", fake_clip)
    chunks = build_chunks([("n", read_path(ROOT / "sample_data/photosynthesis.md"))])
    pkg = pipeline.run_pipeline("photosynthesis", chunks, Settings(llm_provider="auto", image_backend="placeholder"))
    assert pkg.llm_provider == "offline" and pkg.video_source == "text-to-video (local)"
    assert pkg.video_prompt.count(local_t2v.LOCAL_STYLE) == len(pkg.scenes)
    spoken = " ".join(s.narration for s in pkg.scenes)
    assert len(spoken.split()) <= video.word_budget()                        # fits 10 s at a calm pace
    assert 5.0 <= video.audio_duration(Path(pkg.video_path)) <= 10.1


# ------------------------------------------------------------------ Ollama (local text model, no key)
class _Resp:
    def __init__(self, code, js):
        self.status_code, self._js, self.ok = code, js, code < 400

    def json(self):
        return self._js

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(str(self.status_code), response=self)


def test_ollama_client_and_auto_selection(monkeypatch):
    import lcg.config as cfg
    import lcg.llm as llm
    sent = []
    monkeypatch.setattr(llm, "ollama_models", lambda timeout=1.5: ["qwen2.5:7b"])
    monkeypatch.setattr(llm.requests, "post", lambda url, json=None, timeout=None: sent.append((url, json)) or _Resp(
        200, {"message": {"content": '<think>hmm</think>{"ok": true}'}}))
    c = llm.LLMClient("ollama", "")
    assert c.model == "qwen2.5:7b" and c.complete("s", "p") == '{"ok": true}'
    assert sent[0][0].endswith("/api/chat") and sent[0][1]["format"] == "json" and sent[0][1]["stream"] is False
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(cfg, "ollama_models", lambda timeout=1.5: ["llama3.2:3b"])
    assert Settings().resolve_llm() == "ollama"
    monkeypatch.setattr(cfg, "ollama_models", lambda timeout=1.5: [])
    assert Settings().resolve_llm() == "offline"


def test_ollama_missing_model_message(monkeypatch):
    import lcg.llm as llm
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: _Resp(404, {"error": "model not found"}))
    with pytest.raises(llm.LLMError, match="ollama pull"):
        llm.LLMClient("ollama", "", "qwen2.5:7b").complete("s", "p")
    monkeypatch.setattr(llm, "ollama_models", lambda timeout=1.5: [])
    with pytest.raises(llm.LLMError, match="not running"):
        llm.LLMClient("ollama", "")
