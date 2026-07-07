#!/usr/bin/env python
"""Export a trained checkpoint for serving (lean state_dict + optional ONNX/TorchScript).

    python scripts/export_model.py --ckpt .../best.ckpt --out deployment/artifacts \
        --formats torchscript onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from auralguard.models import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="deployment/artifacts")
    ap.add_argument("--formats", nargs="*", default=[], choices=["torchscript", "onnx"])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = build_model(ckpt["cfg"]["model"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    # slim checkpoint (weights + config only)
    torch.save({"model": model.state_dict(), "cfg": ckpt["cfg"]}, out / "best.ckpt")
    print(f"[ok] {out/'best.ckpt'}")

    dummy = torch.randn(1, 64000)
    if "torchscript" in args.formats:
        try:
            ts = torch.jit.trace(lambda x: model(x)["score"], dummy, strict=False)
            ts.save(str(out / "model.ts"))
            print(f"[ok] {out/'model.ts'}")
        except Exception as e:
            print(f"[warn] torchscript export failed (SSL backbone tracing): {e}")
    if "onnx" in args.formats:
        try:
            torch.onnx.export(
                model, dummy, str(out / "model.onnx"),
                input_names=["waveform"], output_names=["score"],
                dynamic_axes={"waveform": {1: "samples"}}, opset_version=17,
            )
            print(f"[ok] {out/'model.onnx'}")
        except Exception as e:
            print(f"[warn] onnx export failed: {e}")


if __name__ == "__main__":
    main()
