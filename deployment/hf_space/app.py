"""Hugging Face Space entrypoint for AuralGuard.

Serves BOTH:
  * a JSON API   ->  POST /api/detect   (used by the Next.js UI)
  * a Gradio UI  ->  mounted at /        (used by casual visitors)

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 7860
On HF Spaces (Docker SDK) the Dockerfile launches exactly this.

The model checkpoint is downloaded from the Hub at startup (set MODEL_REPO /
MODEL_FILE env vars) or loaded from a local path (CKPT_PATH). If no checkpoint is
available the service boots in DEMO mode and returns a clearly-labelled random score,
so the UI wiring can be tested before the model is trained.
"""

from __future__ import annotations

import io
import os
import tempfile

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

MODEL_VERSION = os.environ.get("MODEL_VERSION", "auralguard-v1.0")
CKPT_PATH = os.environ.get("CKPT_PATH")
MODEL_REPO = os.environ.get("MODEL_REPO")            # e.g. "youruser/auralguard"
MODEL_FILE = os.environ.get("MODEL_FILE", "best.ckpt")
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))
MAX_MB = float(os.environ.get("MAX_UPLOAD_MB", "25"))
ALLOWED = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".webm"}

app = FastAPI(title="AuralGuard — AI-Generated Speech Detector", version=MODEL_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

_detector = None
_demo_mode = False


def _get_detector():
    """Lazy-load the model once."""
    global _detector, _demo_mode
    if _detector is not None or _demo_mode:
        return _detector
    ckpt = CKPT_PATH
    try:
        if ckpt is None and MODEL_REPO:
            from huggingface_hub import hf_hub_download

            ckpt = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
        if ckpt and os.path.exists(ckpt):
            from auralguard.inference.predict import Detector

            _detector = Detector(ckpt, threshold=THRESHOLD)
            return _detector
    except Exception as e:  # pragma: no cover
        print(f"[warn] could not load model, entering DEMO mode: {e}")
    _demo_mode = True
    return None


def _read_audio(raw: bytes, suffix: str) -> np.ndarray:
    import soundfile as sf

    try:
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception:
        # fall back via a temp file (handles mp3/m4a/webm through libsndfile/ffmpeg)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(raw)
            tmp = f.name
        import librosa

        wav, sr = librosa.load(tmp, sr=None, mono=True)
        os.unlink(tmp)
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav.astype(np.float32)


def _demo_result(wav: np.ndarray) -> dict:
    rng = np.random.default_rng(int(abs(wav[:1000].sum() * 1e4)) % (2**32))
    p = float(rng.uniform(0.1, 0.9))
    return {
        "verdict": "ai_generated" if p >= THRESHOLD else "human",
        "p_ai_generated": round(p, 4),
        "confidence": "low",
        "score": round(float(np.log(p / (1 - p))), 4),
        "threshold": THRESHOLD,
        "windows": 1,
        "model_version": f"{MODEL_VERSION} (DEMO — no trained weights loaded)",
        "sample_rate": 16000,
        "duration_s": round(len(wav) / 16000, 2),
        "demo": True,
    }


@app.get("/api/health")
def health():
    det = _get_detector()
    return {"status": "ok", "demo_mode": _demo_mode, "model_version": MODEL_VERSION,
            "model_loaded": det is not None}


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix and suffix not in ALLOWED:
        raise HTTPException(415, f"Unsupported format {suffix}. Allowed: {sorted(ALLOWED)}")
    raw = await file.read()
    if len(raw) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (> {MAX_MB} MB).")
    try:
        wav = _read_audio(raw, suffix or ".wav")
    except Exception as e:
        raise HTTPException(400, f"Could not decode audio: {e}")
    if wav.size < 1600:  # < 0.1 s
        raise HTTPException(400, "Audio too short (need at least ~0.1 s).")

    det = _get_detector()
    result = _demo_result(wav) if det is None else det.predict_array(wav)
    return JSONResponse(result)


# --- Gradio UI mounted at "/" ------------------------------------------------
def _build_gradio():
    import gradio as gr

    def infer(audio_path):
        if audio_path is None:
            return {"error": "no audio"}
        det = _get_detector()
        if det is None:
            wav = _read_audio(open(audio_path, "rb").read(), os.path.splitext(audio_path)[1])
            return _demo_result(wav)
        return det.predict_file(audio_path)

    with gr.Blocks(title="AuralGuard") as demo:
        gr.Markdown(
            "# 🛡️ AuralGuard\n"
            "Upload or record speech to estimate whether it is **AI-generated**.\n"
            "*Probabilistic tool — not a sole basis for high-stakes decisions.*"
        )
        inp = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio")
        out = gr.JSON(label="Result")
        gr.Button("Detect").click(infer, inputs=inp, outputs=out)
    return demo


try:
    import gradio as gr

    app = gr.mount_gradio_app(app, _build_gradio(), path="/")
except Exception as e:  # pragma: no cover
    print(f"[warn] Gradio UI not mounted: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
