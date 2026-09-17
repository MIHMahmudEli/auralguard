#!/usr/bin/env python
"""Run zero-shot evaluation across all unseen datasets + summarise as a table.

Usage:
    python scripts/eval_all_zeroshot.py --ckpt <path>
                                        [--out experiments/zeroshot]
                                        [--device cuda]

Reads the checkpoint's embedded data config (asvspoof2019_la.yaml) which
already defines 'cross_eval' manifest paths.  Each dataset is scored with
the same audio decoding parameters and the full metric bundle.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import torch
from auralguard.evaluation.evaluate import evaluate_all, score_manifest, scores_to_probs
from auralguard.evaluation.metrics import summarize, bootstrap_eer_ci
from auralguard.models import build_model
from auralguard.data.datasets import AudioConfig
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Zero-shot evaluation on all cross-dataset benchmarks")
    ap.add_argument("--ckpt", required=True, help="path to .ckpt checkpoint")
    ap.add_argument("--out", default="experiments/zeroshot")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    _SHIM_KEY = "torch.utils.serialization"
    had_shim = _SHIM_KEY in sys.modules
    if not had_shim:
        class _FakeObj:
            def __getattr__(self, n): return _FakeObj()
            def __call__(self, *a, **kw): return _FakeObj()
            def __bool__(self): return False
        class _FakeSerializationMod(types.ModuleType):
            def __getattr__(self, name): return _FakeObj()
        sys.modules[_SHIM_KEY] = _FakeSerializationMod(_SHIM_KEY)
    try:
        ckpt = torch.load(args.ckpt, map_location=args.device)
    finally:
        if not had_shim:
            sys.modules.pop(_SHIM_KEY, None)
    cfg = ckpt["cfg"]
    model = build_model(cfg["model"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    data_cfg = cfg["data"]
    eval_cfg = cfg.get("eval", {})
    audio_cfg = AudioConfig(
        sample_rate=data_cfg["sample_rate"],
        crop_seconds=data_cfg.get("crop_seconds", 4.0),
        random_crop=False,
    )

    # Build manifest list: in-domain + all cross_eval
    manifests = {"in_domain_eval": data_cfg["manifests"]["eval"]}
    manifests.update(data_cfg.get("cross_eval", {}))

    results = {}
    for name, path in manifests.items():
        resolved = None
        if Path(path).exists():
            resolved = path
        else:
            name_only = Path(path).name
            cwd = Path.cwd()
            candidates = [
                cwd / path,
                cwd / "data" / "manifests" / name_only,
                Path("/kaggle/working/data/manifests") / name_only,
                Path("/kaggle/working/auralguard/data/manifests") / name_only,
                Path(path).parent / name_only,
            ]
            for c in candidates:
                if c.exists():
                    resolved = str(c)
                    break
        if resolved is None:
            print(f"  [skip] {name:20s}  manifest not found: {path}")
            continue

        scores, labels, _ = score_manifest(model, resolved, audio_cfg, args.device)
        if scores is None:
            print(f"  [skip] {name:20s}  empty or unreadable manifest")
            continue
        # Check for single-class dataset
        import numpy as _np
        unique_labels = _np.unique(labels)
        if len(unique_labels) < 2:
            print(f"  [skip] {name:20s}  single class only ({unique_labels}) — EER/tDCF undefined")
            continue
        probs = scores_to_probs(scores)
        m = summarize(scores, labels, probs)
        point, lo, hi = bootstrap_eer_ci(
            scores, labels,
            n_boot=eval_cfg.get("n_bootstrap", 1000),
        )
        m["eer_ci95"] = [lo, hi]
        results[name] = m

        print(f"  {name:20s}  EER={m.get('eer', float('nan')):.4f}  "
              f"[{m.get('eer_ci95', [float('nan'), float('nan')])[0]:.4f}, "
              f"{m.get('eer_ci95', [float('nan'), float('nan')])[1]:.4f}]  "
              f"AUROC={m.get('auroc', float('nan')):.4f}  "
              f"tDCF={m.get('min_tdcf', float('nan')):.4f}")

    # Write results
    result_path = out_dir / "results.json"
    result_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {result_path}")

    # Print markdown table
    print("\n## Zero-shot Evaluation Results\n")
    print(f"| {'Dataset':<20s} | {'EER':>8s} | {'EER CI95':>14s} | {'AUROC':>8s} | "
          f"{'tDCF':>8s} | {'F1':>8s} | {'Acc':>8s} |")
    sep_ds  = '-' * 20
    sep_eer = '-' * 8
    sep_ci  = '-' * 14
    print(f"| {sep_ds:>20s} | {sep_eer:>8s} | {sep_ci:>14s} | {sep_eer:>8s} | "
          f"{sep_eer:>8s} | {sep_eer:>8s} | {sep_eer:>8s} |")
    for name, m in results.items():
        ci = m.get('eer_ci95', [float('nan'), float('nan')])
        eer_val = m.get('eer', float('nan'))
        auroc_val = m.get('auroc', float('nan'))
        tdcf_val = m.get('min_tdcf', float('nan'))
        f1_val = m.get('f1', float('nan'))
        acc_val = m.get('balanced_accuracy', float('nan'))
        if isinstance(eer_val, float) and eer_val != eer_val:  # NaN check
            ci_str = "N/A"
            eer_str = "N/A"
        else:
            ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            eer_str = f"{eer_val:>8.4f}"
        print(f"| {name:<20s} | {eer_str:>8s} | {ci_str:>14s} | "
              f"{auroc_val:>8.4f} | {tdcf_val:>8.4f} | "
              f"{f1_val:>8.4f} | {acc_val:>8.4f} |")

    return results


if __name__ == "__main__":
    main()
