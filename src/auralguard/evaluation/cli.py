"""`auralguard-eval` console entrypoint."""

from __future__ import annotations

import argparse
import sys
import types

import torch

from ..models import build_model
from ..utils import get_logger
from .evaluate import evaluate_all

logger = get_logger(__name__)

_SHIM_KEY = "torch.utils.serialization"


def _install_shim():
    if _SHIM_KEY in sys.modules:
        return False

    class _FakeObj:
        def __getattr__(self, n): return _FakeObj()
        def __call__(self, *a, **kw): return _FakeObj()
        def __bool__(self): return False

    class _FakeSerializationMod(types.ModuleType):
        def __getattr__(self, name): return _FakeObj()

    sys.modules[_SHIM_KEY] = _FakeSerializationMod(_SHIM_KEY)
    return True


def _remove_shim():
    sys.modules.pop(_SHIM_KEY, None)


def main():
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on the full protocol.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="experiments/eval")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    added = _install_shim()
    try:
        ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    finally:
        if added:
            _remove_shim()

    cfg = ckpt["cfg"]
    model = build_model(cfg["model"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    evaluate_all(model, cfg["data"], cfg["eval"], device=args.device, out_dir=args.out)


if __name__ == "__main__":
    main()
