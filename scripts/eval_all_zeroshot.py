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
    ckpt = torch.load(args.ckpt, map_location=args.device)
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
        if not Path(path).exists():
            print(f"  [skip] {name:20s}  manifest not found: {path}")
            continue

        scores, labels, _ = score_manifest(model, path, audio_cfg, args.device)
        probs = scores_to_probs(scores)
        m = summarize(scores, labels, probs)
        point, lo, hi = bootstrap_eer_ci(
            scores, labels,
            n_boot=eval_cfg.get("n_bootstrap", 1000),
        )
        m["eer_ci95"] = [lo, hi]
        results[name] = m

        print(f"  {name:20s}  EER={m['eer']:.4f}  [{lo:.4f}, {hi:.4f}]  "
              f"AUROC={m['auroc']:.4f}  tDCF={m['min_tdcf']:.4f}")

    # Write results
    result_path = out_dir / "results.json"
    result_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {result_path}")

    # Print markdown table
    print("\n## Zero-shot Evaluation Results\n")
    print(f"| {'Dataset':<20s} | {'EER':>8s} | {'EER CI95':>14s} | {'AUROC':>8s} | "
          f"{'tDCF':>8s} | {'F1':>8s} | {'Acc':>8s} |")
    print(f"| {'-'*20s} | {'-'*8s} | {'-'*14s} | {'-'*8s} | "
          f"{'-'*8s} | {'-'*8s} | {'-'*8s} |")
    for name, m in results.items():
        ci = f"[{m['eer_ci95'][0]:.4f}, {m['eer_ci95'][1]:.4f}]"
        print(f"| {name:<20s} | {m['eer']:>8.4f} | {ci:>14s} | "
              f"{m['auroc']:>8.4f} | {m['min_tdcf']:>8.4f} | "
              f"{m.get('f1', 0):>8.4f} | {m.get('balanced_accuracy', 0):>8.4f} |")

    return results


if __name__ == "__main__":
    main()
