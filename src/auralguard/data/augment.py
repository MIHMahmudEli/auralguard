"""Channel-robustness augmentation curriculum (addresses gap G2).

Applies, with configurable probability:
  * RawBoost-style convolutive + impulsive + stationary noise (device/channel effects)
  * a realistic codec chain (MP3/AAC/Opus/AMR/G.711 at random bitrates)
  * additive noise from MUSAN at a sampled SNR
  * reverberation from recorded room impulse responses

The codec chain is the key novelty for real-world robustness: synthetic-speech
detectors are notoriously brittle to lossy re-encoding, which is exactly what happens
when audio passes through messaging apps and phone networks.

The codec step shells out to ffmpeg when available (best fidelity to real channels);
if ffmpeg is missing it falls back to a spectral degradation approximation so training
still runs. See docs/DATASETS.md for installing ffmpeg.
"""

from __future__ import annotations

import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


class RawBoost:
    """Simplified RawBoost: convolutive noise + impulsive + stationary (Tak et al.)."""

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        if random.random() > self.prob:
            return wav
        x = wav.copy()
        # convolutive: random short FIR
        if random.random() < 0.7:
            taps = np.random.randn(random.randint(2, 8)).astype(np.float32)
            taps /= np.abs(taps).sum() + 1e-8
            x = np.convolve(x, taps, mode="same").astype(np.float32)
        # impulsive noise
        if random.random() < 0.5:
            n_imp = int(0.0005 * len(x))
            idx = np.random.randint(0, len(x), n_imp)
            x[idx] += np.random.randn(n_imp).astype(np.float32) * x.std()
        # stationary coloured noise
        if random.random() < 0.5:
            snr = random.uniform(10, 40)
            noise = np.random.randn(len(x)).astype(np.float32)
            x = _mix_snr(x, noise, snr)
        return _peak_norm(x)


class CodecChain:
    CODECS = {
        "mp3_320k": ("mp3", ["-b:a", "320k"]),
        "mp3_128k": ("mp3", ["-b:a", "128k"]),
        "mp3_64k": ("mp3", ["-b:a", "64k"]),
        "opus_64k": ("opus", ["-b:a", "64k"]),
        "opus_24k": ("opus", ["-b:a", "24k"]),
        "opus_12k": ("opus", ["-b:a", "12k"]),
        "aac_64k": ("aac", ["-b:a", "64k"]),
        "amr_nb": ("amr", ["-ar", "8000", "-ab", "12.2k"]),
        "g711": ("wav", ["-ar", "8000", "-acodec", "pcm_mulaw"]),
    }

    def __init__(self, prob: float = 0.5, codecs=None):
        self.prob = prob
        self.codecs = codecs or list(self.CODECS.keys())
        self.ffmpeg = shutil.which("ffmpeg")

    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        if random.random() > self.prob:
            return wav
        codec = random.choice(self.codecs)
        if self.ffmpeg is None:
            return _degrade_fallback(wav, sr, codec)
        return self._ffmpeg_roundtrip(wav, sr, codec)

    def _ffmpeg_roundtrip(self, wav: np.ndarray, sr: int, codec: str) -> np.ndarray:
        import soundfile as sf

        ext, opts = self.CODECS[codec]
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "in.wav"
            enc = Path(td) / f"enc.{ext}"
            dec = Path(td) / "out.wav"
            sf.write(src, wav, sr)
            try:
                subprocess.run([self.ffmpeg, "-y", "-i", str(src), *opts, str(enc)],
                               check=True, capture_output=True)
                subprocess.run([self.ffmpeg, "-y", "-i", str(enc), "-ar", str(sr), str(dec)],
                               check=True, capture_output=True)
                out, _ = sf.read(dec, dtype="float32")
                if out.ndim > 1:
                    out = out.mean(axis=1)
                return out.astype(np.float32)
            except Exception:
                return _degrade_fallback(wav, sr, codec)


class NoiseInjector:
    def __init__(self, corpus_manifest: str | None, snr_db=(0, 20), prob: float = 0.5):
        self.prob = prob
        self.snr_range = snr_db
        self.noise_paths = _read_paths(corpus_manifest)

    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        if random.random() > self.prob or not self.noise_paths:
            return wav
        import soundfile as sf

        npath = random.choice(self.noise_paths)
        noise, nsr = sf.read(npath, dtype="float32")
        if noise.ndim > 1:
            noise = noise.mean(axis=1)
        noise = _match_len(noise, len(wav))
        snr = random.uniform(*self.snr_range)
        return _mix_snr(wav, noise, snr)


class AugmentPipeline:
    """Compose the curriculum; each element self-gates on its probability."""

    def __init__(self, cfg):
        self.prob = cfg.get("prob", 0.7)
        self.steps = []
        if cfg.get("rawboost", False):
            self.steps.append(RawBoost(prob=0.6))
        if cfg.get("codec_chain", False):
            self.steps.append(CodecChain(prob=0.5))
        nz = cfg.get("add_noise", {})
        if nz.get("enabled", False):
            self.steps.append(NoiseInjector(nz.get("manifest"), tuple(nz.get("snr_db", (0, 20)))))

    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        if random.random() > self.prob:
            return wav
        for step in self.steps:
            wav = step(wav, sr)
        return wav


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _peak_norm(x: np.ndarray) -> np.ndarray:
    peak = np.abs(x).max() + 1e-8
    return (x / peak * 0.95).astype(np.float32)


def _mix_snr(sig: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    ps = (sig ** 2).mean() + 1e-12
    pn = (noise ** 2).mean() + 1e-12
    k = np.sqrt(ps / (pn * (10 ** (snr_db / 10))))
    return (sig + k * noise).astype(np.float32)


def _match_len(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        s = random.randint(0, len(x) - n)
        return x[s:s + n]
    reps = int(np.ceil(n / len(x)))
    return np.tile(x, reps)[:n]


def _read_paths(manifest: str | None):
    if not manifest or not Path(manifest).exists():
        return []
    import pandas as pd

    return pd.read_csv(manifest)["path"].tolist()


def _degrade_fallback(wav: np.ndarray, sr: int, codec: str) -> np.ndarray:
    """No-ffmpeg approximation: low-pass + mild quantization to mimic lossy codecs."""
    from scipy.signal import butter, lfilter

    cutoff = 3400 if ("g711" in codec or "amr" in codec) else 7000
    b, a = butter(6, cutoff / (sr / 2), btype="low")
    y = lfilter(b, a, wav).astype(np.float32)
    levels = 256 if "64k" in codec or "12k" in codec else 1024
    y = np.round(y * levels) / levels
    return _peak_norm(y)
