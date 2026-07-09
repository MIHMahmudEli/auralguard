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
    import soundfile as sf

    wav, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)  # to mono
    if file_sr != sr:
        import librosa

        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    return wav.astype(np.float32)


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
