"""`auralguard-eval` console entrypoint."""

from __future__ import annotations

import argparse

import torch, sys, types
class _FakeObj:
    def __getattr__(self, n): return _FakeObj()
    def __call__(self, *a, **kw): return _FakeObj()
    def __bool__(self): return False
class _FakeSerializationMod(types.ModuleType):
    def __getattr__(self, name): return _FakeObj()
sys.modules['torch.utils.serialization'] = _FakeSerializationMod('torch.utils.serialization')

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
