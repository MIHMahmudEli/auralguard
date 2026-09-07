"""`auralguard-eval` console entrypoint."""

from __future__ import annotations

import argparse

import torch, sys, importlib.abc
class _FakeSerializationMod(__import__('types').ModuleType):
    def __getattr__(self, name):
        return type(name, (), {})
class _FakeFinder(importlib.abc.MetaPathFinder):
    def find_module(self, fullname, path=None):
        return self if fullname == 'torch.utils.serialization' else None
    def load_module(self, fullname):
        if fullname not in sys.modules:
            sys.modules[fullname] = _FakeSerializationMod(fullname)
        return sys.modules[fullname]
if 'torch.utils.serialization' not in sys.modules:
    sys.meta_path.insert(0, _FakeFinder())

from ..models import build_model
from ..utils import get_logger
from .evaluate import evaluate_all

logger = get_logger(__name__)


def main():
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on the full protocol.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="experiments/eval")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = ckpt["cfg"]
    model = build_model(cfg["model"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    evaluate_all(model, cfg["data"], cfg["eval"], device=args.device, out_dir=args.out)


if __name__ == "__main__":
    main()
