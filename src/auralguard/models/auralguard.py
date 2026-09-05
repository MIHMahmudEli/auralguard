"""AuralGuard — the assembled multi-view detector.

    waveform
      ├─ SSLFrontend        (View A)
      ├─ ArtifactFrontend   (View B, optional)
      ├─ Fusion             (gated cross-attention)
      ├─ AASISTBackend      -> utterance embedding z
      ├─ OCSoftmax head     -> spoof score  (inference)
      └─ SupCon projection  -> aux contrastive (train)

Config is the composed OmegaConf `model` node (see config/model/auralguard.yaml).
`forward` returns a dict so the trainer can access score, embedding, and losses.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .aasist_backend import AASISTBackend
from .artifact_frontend import ArtifactFrontend
from .fusion import build_fusion
from .losses import OCSoftmax, SupConLoss, layer_attention_entropy_reg
from .registry import register
from .ssl_frontend import SSLFrontend


def _get(cfg, key, default=None):
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@register("auralguard")
class AuralGuard(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        fs = cfg["frontend_ssl"]
        self.ssl = SSLFrontend(
            backbone=fs["backbone"], revision=_get(fs, "revision", "main"),
            finetune=_get(fs, "finetune", "last_k"),
            finetune_last_k=_get(fs, "finetune_last_k", 3),
            layer_attention=_get(fs, "layer_attention", True),
            proj_dim=_get(fs, "proj_dim", 128),
            grad_checkpointing=_get(cfg, "grad_checkpointing", False),
        )
        dim = self.ssl.out_dim

        fa = cfg["frontend_artifact"]
        self.use_artifact = _get(fa, "enabled", True)
        if self.use_artifact:
            self.artifact = ArtifactFrontend(
                features=tuple(_get(fa, "features", ["lfcc", "mod_group_delay", "cqt_phase"])),
                proj_dim=_get(fa, "proj_dim", dim),
                lfcc_cfg=_get(fa, "lfcc", {}),
            )
            self.fusion = build_fusion({**cfg["fusion"], "dim": dim})
        else:
            self.artifact = None
            self.fusion = None

        bk = cfg["backend"]
        self.backend = AASISTBackend(
            in_dim=dim, gat_dims=tuple(_get(bk, "gat_dims", [64, 32])),
            embed_dim=_get(bk, "embed_dim", 160),
        )
        embed_dim = self.backend.embed_dim

        hd = cfg["head"]["oc_softmax"]
        self.head = OCSoftmax(
            feat_dim=embed_dim, m_real=_get(hd, "m_real", 0.9),
            m_fake=_get(hd, "m_fake", 0.2), alpha=_get(hd, "alpha", 20.0),
        )

        loss_cfg = cfg["loss"]
        sc = _get(loss_cfg, "supcon", {})
        self.use_supcon = _get(sc, "enabled", False)
        if self.use_supcon:
            self.supcon = SupConLoss(temperature=_get(sc, "temperature", 0.1))
            self.supcon_weight = _get(sc, "weight", 0.3)
            self.proj_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, 128)
            )
        self.layer_reg = _get(loss_cfg, "layer_entropy_reg", 0.0)

    def embed(self, wav: torch.Tensor) -> torch.Tensor:
        fa = self.ssl(wav)                    # (B, C, T')
        if self.use_artifact:
            fb = self.artifact(wav)           # (B, C, T'')
            fused = self.fusion(fa, fb)       # (B, C, T*)
        else:
            fused = fa
        return self.backend(fused)            # (B, embed_dim)

    def forward(self, wav: torch.Tensor, labels: torch.Tensor | None = None) -> dict:
        z = self.embed(wav)
        loss_oc, score = self.head(z, labels)
        out = {"score": score, "embedding": z}
        if labels is None:
            return out

        loss = loss_oc
        out["loss_oc"] = loss_oc.detach()
        if self.use_supcon:
            l_sc = self.supcon(self.proj_head(z), labels)
            loss = loss + self.supcon_weight * l_sc
            out["loss_supcon"] = l_sc.detach()
        if self.layer_reg and self.ssl.layer_weight_logits is not None:
            l_reg = layer_attention_entropy_reg(self.ssl.layer_weight_logits)
            loss = loss + self.layer_reg * l_reg
        out["loss"] = loss
        return out


# NOTE: the repo-wide factory lives in .registry (model-agnostic). Import from there.
