"""Model registry — the extension point that makes the pipeline model-agnostic.

Every detector (proposed model, baselines, future architectures) registers itself
under a name; the trainer / evaluator / inference / export code then builds whatever
`model.name` says in the config. Adding a new model requires ZERO changes to the
pipeline — just:

    from auralguard.models.registry import register

    @register("my_model")
    class MyModel(nn.Module):
        def __init__(self, cfg): ...                 # cfg = the composed `model` node
        def forward(self, wav, labels=None) -> dict:  # THE CONTRACT (see below)
            ...

The model contract
------------------
`forward(wav, labels=None)` where `wav` is (B, T) float32 @16 kHz must return a dict:
  * always:            "score"  (B,) tensor — HIGHER = more spoof/AI-like
  * when labels given: "loss"   scalar tensor to backprop
  * optional:          "embedding", any extra "loss_*" terms for logging

That is the ONLY interface the trainer, evaluator, Detector, and exporter rely on.
See docs/EXTENDING.md for a walkthrough.
"""

from __future__ import annotations

from typing import Callable, Type

import torch.nn as nn

_REGISTRY: dict[str, Type[nn.Module]] = {}


def register(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """Class decorator: register a detector under `name` (used as `model.name`)."""

    def wrap(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in _REGISTRY:
            raise KeyError(f"model '{name}' already registered by {_REGISTRY[name].__name__}")
        _REGISTRY[name] = cls
        return cls

    return wrap


def available() -> list[str]:
    return sorted(_REGISTRY)


def build_model(cfg) -> nn.Module:
    """Build whatever `cfg.name` (or cfg['name']) names. Single factory for the whole repo."""
    if isinstance(cfg, str):
        name = cfg
        arch = name
    else:
        name = cfg.name if hasattr(cfg, "name") else cfg["name"]
        arch = getattr(cfg, "arch", name) if hasattr(cfg, "arch") else cfg.get("arch", name)
    if arch not in _REGISTRY:
        raise KeyError(f"unknown model '{arch}'. Registered: {available()}")
    return _REGISTRY[arch](cfg)
