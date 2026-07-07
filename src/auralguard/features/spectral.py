"""Spectral / phase feature extraction for the artifact front-end (View B).

All extractors operate on a batched waveform tensor (B, T) at 16 kHz and return a
stacked feature tensor (B, C, F, T) where C is the number of requested features.
Implemented with torch/torchaudio so they run on GPU and are differentiable where
possible.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _stft(wav: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    window = torch.hann_window(win, device=wav.device)
    return torch.stft(
        wav, n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, return_complex=True, center=True,
    )  # (B, F, T)


class ArtifactFeatureExtractor(nn.Module):
    def __init__(self, features=("lfcc", "mod_group_delay", "cqt_phase"),
                 sample_rate: int = 16000, lfcc_cfg: dict | None = None):
        super().__init__()
        self.features = list(features)
        self.sr = sample_rate
        cfg = {"n_lfcc": 60, "n_fft": 512, "hop": 160, "win": 400, **(lfcc_cfg or {})}
        self.n_fft = cfg["n_fft"]
        self.hop = cfg["hop"]
        self.win = cfg["win"]
        self.n_lfcc = cfg["n_lfcc"]

        try:
            import torchaudio  # noqa: F401
            self._lfcc = torchaudio.transforms.LFCC(
                sample_rate=sample_rate, n_lfcc=self.n_lfcc,
                speckwargs={"n_fft": self.n_fft, "hop_length": self.hop, "win_length": self.win},
            )
        except Exception:  # pragma: no cover
            self._lfcc = None

    # --- individual features ------------------------------------------------
    def lfcc(self, wav: torch.Tensor) -> torch.Tensor:
        if self._lfcc is None:
            raise RuntimeError("torchaudio required for LFCC")
        return self._lfcc(wav)  # (B, n_lfcc, T)

    def mod_group_delay(self, wav: torch.Tensor, rho: float = 0.4, gamma: float = 0.9) -> torch.Tensor:
        """Modified group-delay function — sensitive to phase artifacts."""
        spec = _stft(wav, self.n_fft, self.hop, self.win)  # (B, F, T)
        mag = spec.abs() + 1e-8
        # derivative along frequency via finite differences of unwrapped phase
        phase = torch.angle(spec)
        dphase = torch.diff(phase, dim=1, prepend=phase[:, :1])
        smooth = mag.pow(2 * gamma)
        tau = -dphase * mag.pow(2 * rho) / (smooth + 1e-8)
        return tau  # (B, F, T)

    def cqt_phase(self, wav: torch.Tensor) -> torch.Tensor:
        """Phase of a (approx.) constant-Q representation. Approximated with STFT
        phase here for a pure-torch, GPU-friendly path; swap for librosa CQT offline
        if you cache features."""
        spec = _stft(wav, self.n_fft, self.hop, self.win)
        return torch.angle(spec)

    # --- assembly -----------------------------------------------------------
    def _resize_to(self, x: torch.Tensor, F: int, T: int) -> torch.Tensor:
        # x: (B, f, t) -> (B, F, T) via interpolation so channels stack cleanly
        return torch.nn.functional.interpolate(
            x.unsqueeze(1), size=(F, T), mode="bilinear", align_corners=False
        ).squeeze(1)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        maps = []
        for name in self.features:
            m = getattr(self, name)(wav)  # (B, f, t)
            maps.append(m)
        F = max(m.shape[1] for m in maps)
        T = max(m.shape[2] for m in maps)
        maps = [self._resize_to(m, F, T) for m in maps]
        x = torch.stack(maps, dim=1)  # (B, C, F, T)
        # per-channel instance norm for scale invariance
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-5
        return (x - mean) / std
