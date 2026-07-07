"""Runnable baseline detectors (B1–B3 from docs/EXPERIMENTS.md).

Each is registered in the model registry and satisfies the same forward contract as
AuralGuard, so `python scripts/train.py experiment=b1_lcnn` etc. work with zero
pipeline changes. These are compact, literature-faithful-in-spirit implementations
meant for parity checks; cite the original papers in the manuscript.

  * lfcc_lcnn   — LFCC front-end + Light CNN (Max-Feature-Map) back-end + CE.
  * rawnet2     — raw-waveform SincConv-style front-end + residual GRU (RawNet2-lite) + CE.
  * aasist_raw  — raw waveform conv encoder + AASIST graph back-end + CE.

B0 (LFCC-GMM) is intentionally not a nn.Module — it lives in notebooks/02_baseline.ipynb
territory; B4/B5 are covered by the `auralguard` architecture via config (SSL-only view).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aasist_backend import AASISTBackend
from .registry import register


class _CEHead(nn.Module):
    """Binary cross-entropy head satisfying the repo score contract.

    Produces 2-class logits; the returned "score" is logit(spoof) - logit(bona),
    so higher = more spoof-like, matching OC-Softmax-based models.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 2)

    def forward(self, z: torch.Tensor, labels: torch.Tensor | None = None):
        logits = self.fc(z)                      # (B, 2): [bona, spoof]
        score = logits[:, 1] - logits[:, 0]
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return loss, score


class _MFM(nn.Module):
    """Max-Feature-Map activation (the LCNN building block)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return torch.max(a, b)


def _mfm_conv(cin: int, cout: int, k: int = 3, s: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, 2 * cout, k, s, k // 2), _MFM(), nn.BatchNorm2d(cout))


@register("lfcc_lcnn")
class LFCCLCNN(nn.Module):
    """B1: LFCC + Light CNN (Wu et al. MFM-style) + CE."""

    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or {}
        n_lfcc = cfg.get("n_lfcc", 60) if hasattr(cfg, "get") else 60
        import torchaudio

        self.lfcc = torchaudio.transforms.LFCC(
            sample_rate=16000, n_lfcc=n_lfcc,
            speckwargs={"n_fft": 512, "hop_length": 160, "win_length": 400},
        )
        self.net = nn.Sequential(
            _mfm_conv(1, 32), nn.MaxPool2d(2),
            _mfm_conv(32, 48), nn.MaxPool2d(2),
            _mfm_conv(48, 64), nn.MaxPool2d(2),
            _mfm_conv(64, 32), nn.AdaptiveAvgPool2d(1),
        )
        self.head = _CEHead(32)

    def forward(self, wav: torch.Tensor, labels: torch.Tensor | None = None) -> dict:
        feat = self.lfcc(wav).unsqueeze(1)       # (B, 1, n_lfcc, T)
        z = self.net(feat).flatten(1)            # (B, 32)
        loss, score = self.head(z, labels)
        out = {"score": score, "embedding": z}
        if loss is not None:
            out["loss"] = loss
        return out


class _ResBlock1d(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, padding=1)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(cin)
        self.bn2 = nn.BatchNorm1d(cout)
        self.short = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        y = self.conv1(F.leaky_relu(self.bn1(x), 0.3))
        y = self.conv2(F.leaky_relu(self.bn2(y), 0.3))
        return F.max_pool1d(y + self.short(x), 3)


@register("rawnet2")
class RawNet2Lite(nn.Module):
    """B2: raw-waveform residual encoder + GRU (RawNet2-lite, Tak et al.) + CE."""

    def __init__(self, cfg=None):
        super().__init__()
        self.front = nn.Sequential(
            nn.Conv1d(1, 64, 129, stride=1, padding=64),  # stand-in for SincConv
            nn.BatchNorm1d(64), nn.LeakyReLU(0.3), nn.MaxPool1d(3),
        )
        self.blocks = nn.Sequential(
            _ResBlock1d(64, 64), _ResBlock1d(64, 128),
            _ResBlock1d(128, 128), _ResBlock1d(128, 128),
        )
        self.gru = nn.GRU(128, 256, num_layers=2, batch_first=True)
        self.head = _CEHead(256)

    def forward(self, wav: torch.Tensor, labels: torch.Tensor | None = None) -> dict:
        x = self.front(wav.unsqueeze(1))
        x = self.blocks(x)                        # (B, 128, T')
        h, _ = self.gru(x.transpose(1, 2))        # (B, T', 256)
        z = h[:, -1]                              # last state
        loss, score = self.head(z, labels)
        out = {"score": score, "embedding": z}
        if loss is not None:
            out["loss"] = loss
        return out


@register("aasist_raw")
class AASISTRaw(nn.Module):
    """B3: raw waveform conv encoder + AASIST graph-attention back-end + CE."""

    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or {}
        dim = 128
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, 129, stride=10, padding=64), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, dim, 3, stride=2, padding=1), nn.BatchNorm1d(dim), nn.GELU(),
            nn.Conv1d(dim, dim, 3, stride=2, padding=1), nn.BatchNorm1d(dim), nn.GELU(),
        )
        bk = cfg.get("backend", {}) if hasattr(cfg, "get") else {}
        self.backend = AASISTBackend(
            in_dim=dim, gat_dims=tuple(bk.get("gat_dims", [64, 32])),
            embed_dim=bk.get("embed_dim", 160),
        )
        self.head = _CEHead(self.backend.embed_dim)

    def forward(self, wav: torch.Tensor, labels: torch.Tensor | None = None) -> dict:
        f = self.encoder(wav.unsqueeze(1))        # (B, C, T')
        z = self.backend(f)                       # (B, embed)
        loss, score = self.head(z, labels)
        out = {"score": score, "embedding": z}
        if loss is not None:
            out["loss"] = loss
        return out
