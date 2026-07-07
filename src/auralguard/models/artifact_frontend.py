"""Hand-crafted vocoder-artifact front-end (View B).

Computes low-level cues that SSL features tend to under-represent — LFCC, modified
group-delay, and CQT phase — and encodes them with a light 2-D CNN. These expose
phase/periodicity artifacts left by neural vocoders and codec-based TTS.

The feature extractors live in `auralguard.features`; here we only encode them.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..features.spectral import ArtifactFeatureExtractor


class _ConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=(2, 2)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
            nn.BatchNorm2d(cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class ArtifactFrontend(nn.Module):
    def __init__(self, features=("lfcc", "mod_group_delay", "cqt_phase"), proj_dim: int = 128,
                 sample_rate: int = 16000, lfcc_cfg: dict | None = None):
        super().__init__()
        self.extractor = ArtifactFeatureExtractor(
            features=features, sample_rate=sample_rate, lfcc_cfg=lfcc_cfg or {}
        )
        cin = len(features)  # one channel per feature map
        self.encoder = nn.Sequential(
            _ConvBlock(cin, 32),
            _ConvBlock(32, 64),
            _ConvBlock(64, proj_dim),
        )
        # collapse frequency axis, keep time -> (B, proj_dim, T'')
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.out_dim = proj_dim

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, T). Returns (B, proj_dim, T'')."""
        feats = self.extractor(wav)      # (B, C, F, T)
        x = self.encoder(feats)          # (B, proj_dim, F', T')
        x = self.freq_pool(x)            # (B, proj_dim, 1, T')
        return x.squeeze(2)              # (B, proj_dim, T')
