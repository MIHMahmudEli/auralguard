"""Manifest-driven audio dataset.

Reads the normalized CSV manifest schema documented in docs/DATASETS.md and yields
(waveform, label, meta). All corpora are adapted to this one schema so the training
code never special-cases a dataset.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    crop_seconds: float = 4.0
    random_crop: bool = True
    pad_mode: str = "repeat"  # repeat | zero

    @property
    def crop_len(self) -> int:
        return int(self.sample_rate * self.crop_seconds)


def _load_audio(path: str, sr: int) -> np.ndarray:
    import logging
    import warnings

    _log = logging.getLogger("auralguard.data")

    # Try soundfile first (fastest)
    try:
        import soundfile as sf
        wav, file_sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if file_sr != sr:
            import librosa
            wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
        return wav.astype(np.float32)
    except Exception as e:
        _log.debug("soundfile failed for %s: %s", path, e)

    # Fallback: pydub (requires ffmpeg for MP3/Opus/etc)
    try:
        from pydub import AudioSegment
        import io
        segment = AudioSegment.from_file(path)
        segment = segment.set_frame_rate(sr).set_channels(1)
        samples = segment.get_array_of_samples()
        wav = np.array(samples, dtype=np.float32)
        if segment.max_possible_amplitude > 0:
            wav /= segment.max_possible_amplitude
        return wav
    except Exception as e:
        _log.debug("pydub failed for %s: %s", path, e)

    # Fallback: librosa (uses audioread → needs ffmpeg for non-WAV)
    try:
        import librosa
        wav, file_sr = librosa.load(path, sr=None, mono=True)
        if file_sr != sr:
            wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
        return wav.astype(np.float32)
    except Exception as e:
        _log.debug("librosa failed for %s: %s", path, e)

    # Fallback: torchaudio
    try:
        import torchaudio
        wav_tensor, file_sr = torchaudio.load(path)
        if wav_tensor.ndim > 1:
            wav_tensor = wav_tensor.mean(dim=0)
        if file_sr != sr:
            wav_tensor = torchaudio.functional.resample(wav_tensor, file_sr, sr)
        return wav_tensor.numpy().astype(np.float32)
    except Exception as e:
        _log.debug("torchaudio failed for %s: %s", path, e)

    # Last resort: try raw bytes via ffmpeg subprocess
    try:
        import subprocess
        result = subprocess.run(
            ["ffmpeg", "-i", path, "-f", "wav", "-acodec", "pcm_s16le",
             "-ar", str(sr), "-ac", "1", "-"],
            capture_output=True, timeout=30,
        )
        if result.returncode == 0 and len(result.stdout) > 44:
            # Skip RIFF header (44 bytes), read raw PCM
            raw = result.stdout[44:]
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            return samples
        elif result.returncode != 0:
            _log.debug("ffmpeg failed for %s: %s", path,
                       result.stderr.decode(errors="replace")[:200])
    except FileNotFoundError:
        _log.debug("ffmpeg not found on PATH")
    except Exception as e:
        _log.debug("ffmpeg subprocess failed for %s: %s", path, e)

    # Detect file format for diagnostics
    try:
        with open(path, "rb") as f:
            header = f.read(16)
        magic = header[:4] if len(header) >= 4 else b""
        import os
        fsize = os.path.getsize(path)
        if magic == b"RIFF":
            fmt = header[8:12] if len(header) >= 12 else b"???"
            _log.warning("Failed to load %s: RIFF/%s file (%d bytes) — "
                         "likely MP3/Opus in .wav wrapper; install ffmpeg",
                         path, fmt.decode("ascii", errors="replace"), fsize)
        elif magic[:3] == b"ID3":
            _log.warning("Failed to load %s: MP3 file (%d bytes) — install ffmpeg",
                         path, fsize)
        elif magic[:4] == b"\x1aE\xdf\xa3":
            _log.warning("Failed to load %s: Matroska/WebM file (%d bytes) — install ffmpeg",
                         path, fsize)
        elif fsize == 0:
            _log.warning("Failed to load %s: file is empty (0 bytes) — broken symlink or mount",
                         path)
        else:
            _log.warning("Failed to load %s: unknown format %s (%d bytes)",
                         path, magic.hex(), fsize)
    except Exception:
        _log.warning("Failed to load %s (all backends failed, cannot inspect header)", path)

    return np.zeros(sr * 4, dtype=np.float32)


def _fix_length(wav: np.ndarray, target: int, random_crop: bool, pad_mode: str) -> np.ndarray:
    n = wav.shape[0]
    if n == target:
        return wav
    if n > target:
        start = random.randint(0, n - target) if random_crop else (n - target) // 2
        return wav[start:start + target]
    # pad
    if pad_mode == "repeat":
        reps = int(np.ceil(target / n))
        return np.tile(wav, reps)[:target]
    out = np.zeros(target, dtype=np.float32)
    out[:n] = wav
    return out


class AudioAntiSpoofDataset(Dataset):
    def __init__(self, manifest: str | Path, audio_cfg: AudioConfig,
                 augment=None, is_train: bool = True):
        self.df = pd.read_csv(manifest)
        required = {"utt_id", "path", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"manifest {manifest} missing columns {missing}")
        self.cfg = audio_cfg
        self.augment = augment
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        wav = _load_audio(row["path"], self.cfg.sample_rate)
        if self.augment is not None and self.is_train:
            wav = self.augment(wav, self.cfg.sample_rate)
        wav = _fix_length(wav, self.cfg.crop_len, self.cfg.random_crop and self.is_train,
                          self.cfg.pad_mode)
        label = int(row["label"])
        meta = {
            "utt_id": str(row["utt_id"]),
            "attack": str(row.get("attack", "unknown")),
            "dataset": str(row.get("dataset", "unknown")),
            "lang": str(row.get("lang", "und")),
        }
        return torch.from_numpy(wav), label, meta


def collate(batch):
    wavs, labels, metas = zip(*batch)
    return torch.stack(wavs), torch.tensor(labels, dtype=torch.long), list(metas)
