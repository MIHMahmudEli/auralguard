"""Gated cross-attention fusion of View A (SSL) and View B (artifact) sequences.

Both inputs are (B, C, T*) with possibly different T; we align by interpolation,
then let each view attend to the other and gate the contribution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _align(x: torch.Tensor, T: int) -> torch.Tensor:
    if x.shape[-1] == T:
        return x
    return F.interpolate(x, size=T, mode="linear", align_corners=False)


class GatedCrossAttentionFusion(nn.Module):
    def __init__(self, dim: int = 128, heads: int = 4):
        super().__init__()
        self.a2b = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.b2a = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(2 * dim, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)
        self.out_dim = dim

    def forward(self, fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
        # fa, fb: (B, C, T)
        T = max(fa.shape[-1], fb.shape[-1])
        fa = _align(fa, T).transpose(1, 2)  # (B, T, C)
        fb = _align(fb, T).transpose(1, 2)
        a_ctx, _ = self.a2b(fa, fb, fb)
        b_ctx, _ = self.b2a(fb, fa, fa)
        g = self.gate(torch.cat([a_ctx, b_ctx], dim=-1))
        fused = self.norm(g * a_ctx + (1 - g) * b_ctx + fa)
        return fused.transpose(1, 2)  # (B, C, T)


class ConcatFusion(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.proj = nn.Conv1d(2 * dim, dim, 1)
        self.out_dim = dim

    def forward(self, fa, fb):
        T = max(fa.shape[-1], fb.shape[-1])
        fa, fb = _align(fa, T), _align(fb, T)
        return self.proj(torch.cat([fa, fb], dim=1))


def build_fusion(cfg) -> nn.Module:
    t = cfg.get("type", "gated_cross_attention")
    dim = cfg.get("dim", 128)
    if t == "gated_cross_attention":
        return GatedCrossAttentionFusion(dim, cfg.get("heads", 4))
    if t == "concat":
        return ConcatFusion(dim)
    raise ValueError(f"unknown fusion type {t}")
