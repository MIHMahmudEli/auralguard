"""Gradio demo for AuralGuard — deployed on Hugging Face Spaces."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np

CKPT_PATH = os.environ.get("CKPT_PATH", "best.ckpt")

# Lazy-import so the app loads fast even without torch at import time
detector = None


def get_detector():
    global detector
    if detector is None:
        from auralguard.inference.predict import Detector
        detector = Detector(CKPT_PATH)
    return detector


def predict(audio_path: str) -> tuple[str, float, str, float, dict]:
    det = get_detector()
    result = det.predict_file(audio_path)
    label = "AI-GENERATED" if result["verdict"] == "ai_generated" else "HUMAN"
    color = "#ff4444" if result["verdict"] == "ai_generated" else "#44bb44"

    # Build per-window score chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 2.5))
    n_windows = result.get("windows", 1)
    if n_windows > 1:
        det2 = get_detector()
        import torch, soundfile as sf, librosa
        wav, sr = sf.read(audio_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        window_scores = []
        w_len = int(4.0 * 16000)
        h_len = int(2.0 * 16000)
        if len(wav) > w_len:
            for start in range(0, len(wav) - w_len + 1, h_len):
                x = torch.from_numpy(wav[start:start + w_len]).unsqueeze(0)
                out = det2.model(x.to(det2.device))
                window_scores.append(float(out["score"].item()))
        else:
            window_scores = [result["score"]]
        ax.plot(window_scores, marker="o", color=color, linewidth=1.5)
        ax.axhline(y=det2.threshold, color="gray", linestyle="--", alpha=0.7, label="Threshold")
        ax.set_xlabel("Window")
        ax.set_ylabel("Spoof Score")
        ax.set_title("Per-Window Scores")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, f"Score: {result['score']:.4f}", ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    chart_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    fig.savefig(chart_path, bbox_inches="tight")
    plt.close(fig)

    return label, result["p_ai_generated"], result["confidence"], result["score"], chart_path


with gr.Blocks(title="AuralGuard — AI-Generated Speech Detector", css="""
    .green { color: #44bb44 !important; font-weight: bold; }
    .red { color: #ff4444 !important; font-weight: bold; }
""") as demo:
    gr.Markdown(
        "# AuralGuard\n"
        "### AI-Generated Speech Detection\n\n"
        "Upload an audio file to check if it's human speech or AI-generated.\n\n"
        "Supported formats: WAV, MP3, FLAC, OGG, M4A"
    )

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Upload Audio")
        with gr.Column():
            verdict = gr.Label(label="Verdict", value="—")
            confidence = gr.Label(label="Confidence", value="—")
            score = gr.Number(label="Raw Spoof Score", value=None)
            p_ai = gr.Number(label="P(AI-Generated)", value=None)

    chart = gr.Image(label="Per-Window Score Breakdown")

    audio_input.change(
        fn=predict,
        inputs=audio_input,
        outputs=[verdict, p_ai, confidence, score, chart],
    )

    gr.Markdown(
        "---\n"
        "* **Score > threshold → AI-Generated** | Score < threshold → Human\n"
        "* **Confidence** is based on margin from decision boundary\n"
        "* Model: AuralGuard — [GitHub](https://github.com/MIHMahmudEli/auralguard)"
    )


if __name__ == "__main__":
    demo.launch()
