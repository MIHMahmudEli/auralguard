"""Evaluation runner: in-domain + cross-dataset zero-shot + calibration.

Given a trained checkpoint, scores one or more manifests and reports the full metric
bundle with bootstrap CIs. Robustness sweeps (E5) are driven separately by
scripts/run_robustness.py which re-encodes eval audio through the codec chain.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.datasets import AudioAntiSpoofDataset, AudioConfig, collate
from ..utils.logging import get_logger
from .metrics import bootstrap_eer_ci, summarize

logger = get_logger(__name__)


@torch.no_grad()
def score_manifest(model, manifest, audio_cfg, device="cuda", batch_size=16, num_workers=4):
    ds = AudioAntiSpoofDataset(manifest, audio_cfg, augment=None, is_train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate)
    model.eval()
    scores, labels, ids = [], [], []
    for wav, y, meta in loader:
        wav = wav.to(device)
        out = model(wav)
        scores.append(out["score"].cpu().numpy())
        labels.append(y.numpy())
        ids.extend(m["utt_id"] for m in meta)
    return np.concatenate(scores), np.concatenate(labels), ids


def scores_to_probs(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Map raw spoof scores to calibrated probabilities via a logistic + temperature."""
    return 1.0 / (1.0 + np.exp(-scores / max(temperature, 1e-6)))


def evaluate_all(model, data_cfg, eval_cfg, device="cuda", out_dir="experiments/eval"):
    audio_cfg = AudioConfig(
        sample_rate=data_cfg["sample_rate"],
        crop_seconds=data_cfg["crop_seconds"],
        random_crop=False,
    )
    results = {}

    # in-domain
    manifests = {"in_domain_eval": data_cfg["manifests"]["eval"]}
    if eval_cfg.get("cross_dataset", True):
        manifests.update(data_cfg.get("cross_eval", {}))

    for name, path in manifests.items():
        if not Path(path).exists():
            logger.warning("skip %s (manifest not found: %s)", name, path)
            continue
        scores, labels, _ = score_manifest(model, path, audio_cfg, device)
        probs = scores_to_probs(scores)
        m = summarize(scores, labels, probs)
        point, lo, hi = bootstrap_eer_ci(scores, labels,
                                         n_boot=eval_cfg.get("n_bootstrap", 1000))
        m["eer_ci95"] = [lo, hi]
        results[name] = m
        logger.info("%-18s EER=%.4f [%.4f, %.4f] tDCF=%.4f AUROC=%.4f",
                    name, m["eer"], lo, hi, m["min_tdcf"], m["auroc"])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", out / "results.json")
    return results
