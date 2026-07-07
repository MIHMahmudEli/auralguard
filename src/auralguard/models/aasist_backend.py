"""Compact AASIST-style spectro-temporal graph-attention back-end.

A faithful-in-spirit, dependency-light re-implementation: it builds temporal and
spectral (channel) node sets from the fused feature map, applies graph attention
within and across the two domains, and pools to an utterance embedding. For exact
reproduction of the original AASIST numbers, swap this for the authors' reference
module — the interface (feature map in -> embedding out) is identical.

Reference: Jung et al., "AASIST: Audio Anti-Spoofing using Integrated
Spectro-Temporal Graph Attention Networks", ICASSP 2022.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.att = nn.Linear(2 * out_dim, 1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, in_dim)
        h = self.proj(x)                      # (B, N, out)
        B, N, D = h.shape
        hi = h.unsqueeze(2).expand(B, N, N, D)
        hj = h.unsqueeze(1).expand(B, N, N, D)
        e = self.att(torch.cat([hi, hj], dim=-1)).squeeze(-1)  # (B, N, N)
        a = F.softmax(F.leaky_relu(e, 0.2), dim=-1)
        out = torch.bmm(a, h)                 # (B, N, out)
        return self.norm(F.gelu(out) + h)


class AASISTBackend(nn.Module):
    def __init__(self, in_dim: int = 128, gat_dims=(64, 32), embed_dim: int = 160):
        super().__init__()
        # temporal nodes: pool over channels -> N_t time nodes (dim in_dim)
        # spectral nodes: pool over time    -> N_s channel nodes
        d1, d2 = gat_dims
        self.gat_t = GraphAttentionLayer(in_dim, d1)
        self.gat_s = GraphAttentionLayer(in_dim, d1)
        self.gat_st = GraphAttentionLayer(d1, d2)
        self.n_time_nodes = 64
        self.n_spec_nodes = 32
        self.head = nn.Sequential(
            nn.Linear(d2 * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T). Two complementary C-dim node sets over the temporal axis:
        #   average-pooled ("temporal") and max-pooled ("spectral"/salient) nodes.
        t_nodes = F.adaptive_avg_pool1d(x, self.n_time_nodes).transpose(1, 2)  # (B, N_t, C)
        s_nodes = F.adaptive_max_pool1d(x, self.n_spec_nodes).transpose(1, 2)  # (B, N_s, C)

        ht = self.gat_t(t_nodes)   # (B, N_t, d1)
        hs = self.gat_s(s_nodes)   # (B, N_s, d1)

        joint = torch.cat([ht, hs], dim=1)   # (B, N_t+N_s, d1)
        hst = self.gat_st(joint)             # (B, N, d2)

        # readout: max + mean pool
        emb = torch.cat([hst.max(dim=1).values, hst.mean(dim=1)], dim=-1)  # (B, 2*d2)
        return self.head(emb)  # (B, embed_dim)
