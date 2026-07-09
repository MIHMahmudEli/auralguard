"""Single-file / batch inference.

Loads a checkpoint, runs the sliding-window scoring used at deployment time, and
returns a calibrated verdict. Shared by the CLI, the FastAPI service, and tests.
"""

from __future__ import annotations

import argparse
import json


import numpy as np
import torch

from ..models import build_model
from ..utils.logging import get_logger

logger = get_logger(__name__)

WINDOW_S = 4.0
HOP_S = 2.0
SR = 16000


class Detector:
    def __init__(self, ckpt_path: str, device: str | None = None, temperature: float = 1.0,
                 threshold: float = 0.5):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        cfg = ckpt["cfg"]
        self.model = build_model(cfg["model"]).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.temperature = temperature
        self.threshold = threshold
        self.model_version = cfg.get("experiment_name", "auralguard")

    @staticmethod
    def _load_wav(path: str) -> np.ndarray:
        import soundfile as sf

        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        return wav.astype(np.float32)

    def _windows(self, wav: np.ndarray):
        w = int(WINDOW_S * SR)
        h = int(HOP_S * SR)
        if len(wav) <= w:
            reps = int(np.ceil(w / len(wav)))
            yield np.tile(wav, reps)[:w]
            return
        for start in range(0, len(wav) - w + 1, h):
            yield wav[start:start + w]

    @torch.no_grad()
    def predict_array(self, wav: np.ndarray) -> dict:
        scores = []
        for win in self._windows(wav):
            x = torch.from_numpy(win).unsqueeze(0).to(self.device)
            out = self.model(x)
            scores.append(float(out["score"].item()))
        score = float(np.mean(scores))
        p_ai = 1.0 / (1.0 + np.exp(-score / max(self.temperature, 1e-6)))
        verdict = "ai_generated" if p_ai >= self.threshold else "human"
        margin = abs(p_ai - self.threshold)
        confidence = "high" if margin > 0.35 else "medium" if margin > 0.15 else "low"
        return {
            "verdict": verdict,
            "p_ai_generated": round(p_ai, 4),
            "confidence": confidence,
            "score": round(score, 4),
            "threshold": self.threshold,
            "windows": len(scores),
            "model_version": self.model_version,
            "sample_rate": SR,
            "duration_s": round(len(wav) / SR, 2),
        }

    def predict_file(self, path: str) -> dict:
        return self.predict_array(self._load_wav(path))


def main():
    ap = argparse.ArgumentParser(description="Detect AI-generated speech in an audio file.")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    det = Detector(args.ckpt, temperature=args.temperature, threshold=args.threshold)
    result = det.predict_file(args.audio)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
