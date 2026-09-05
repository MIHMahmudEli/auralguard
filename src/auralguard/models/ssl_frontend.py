"""Self-supervised speech front-end (View A).

Wraps a HuggingFace SSL encoder (WavLM / wav2vec2-XLS-R / HuBERT), exposes all
hidden layers, and combines them with a learnable layer-attention (weighted sum),
following the observation that spoofing artifacts concentrate in different
transformer depths depending on the attack.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSLFrontend(nn.Module):
    def __init__(
        self,
        backbone: str = "microsoft/wavlm-large",
        revision: str = "main",
        finetune: str = "last_k",        # frozen | last_k | full
        finetune_last_k: int = 3,
        layer_attention: bool = True,
        proj_dim: int = 128,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        from transformers import AutoModel  # local import: heavy dependency

        self.model = AutoModel.from_pretrained(backbone, revision=revision, output_hidden_states=True)
        # Disable spec augment to avoid CUDA indexing crash with PyTorch <= 2.3
        if hasattr(self.model.config, "apply_spec_augment"):
            self.model.config.apply_spec_augment = False
        if grad_checkpointing:
            self.model.gradient_checkpointing_enable()
        hidden = self.model.config.hidden_size
        n_layers = self.model.config.num_hidden_layers + 1  # + embedding output

        self.layer_attention = layer_attention
        if layer_attention:
            self.layer_weights = nn.Parameter(torch.zeros(n_layers))
        self.proj = nn.Sequential(
            nn.Conv1d(hidden, proj_dim, kernel_size=1),
            nn.BatchNorm1d(proj_dim),
            nn.GELU(),
        )
        self.out_dim = proj_dim
        self._configure_finetune(finetune, finetune_last_k)

    def _configure_finetune(self, mode: str, k: int):
        if mode == "full":
            for p in self.model.parameters():
                p.requires_grad = True
            return
        # freeze everything first
        for p in self.model.parameters():
            p.requires_grad = False
        if mode == "frozen":
            self.model.eval()
            return
        if mode == "last_k":
            # unfreeze the last-k transformer layers
            try:
                layers = self.model.encoder.layers
                for layer in layers[-k:]:
                    for p in layer.parameters():
                        p.requires_grad = True
            except AttributeError:
                # fall back: leave frozen if structure differs
                pass

    @property
    def layer_weight_logits(self) -> torch.Tensor | None:
        return self.layer_weights if self.layer_attention else None

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, T) float waveform at 16 kHz. Returns (B, proj_dim, T')."""
        out = self.model(wav)
        hs = torch.stack(out.hidden_states, dim=0)  # (L, B, T', H)
        if self.layer_attention:
            n_actual = hs.shape[0]
            w = F.softmax(self.layer_weights[:n_actual], dim=0).view(-1, 1, 1, 1)
            feat = (w * hs).sum(dim=0)              # (B, T', H)
        else:
            feat = hs[-1]                           # last layer
        feat = feat.transpose(1, 2)                 # (B, H, T')
        feat = self.proj(feat)                      # (B, proj_dim, T')
        return feat
