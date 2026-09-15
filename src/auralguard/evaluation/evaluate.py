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


def _resolve_manifest(path: str) -> str | None:
    """Try the given path, then fall back to common Kaggle/repo locations."""
    p = Path(path)
    if p.exists():
        return path
    name = p.name
    cwd = Path.cwd()
    candidates = [
        cwd / path,
        cwd / "data" / "manifests" / name,
        Path("/kaggle/working/data/manifests") / name,
        Path("/kaggle/working/auralguard/data/manifests") / name,
        p.parent / name,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


@torch.no_grad()
def score_manifest(model, manifest, audio_cfg, device="cuda", batch_size=16, num_workers=0):
    ds = AudioAntiSpoofDataset(manifest, audio_cfg, augment=None, is_train=False)
    if len(ds) == 0:
        logger.warning("manifest is empty (%s), skipping", manifest)
        return None, None, []
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate)
    model.eval()
    scores, labels, ids = [], [], []
    n_failed = 0
    total = len(ds)
    n_batches = len(loader)
    import time as _time
    start_time = _time.time()
    last_log = start_time
    LOG_INTERVAL_SEC = 30
    LOG_INTERVAL_PCT = 5
    next_pct = LOG_INTERVAL_PCT
    ds_name = Path(manifest).stem
    for batch_idx, (wav, y, meta) in enumerate(loader):
        failed_mask = wav.sum(dim=1) == 0
        n_batch_failed = failed_mask.sum().item()
        n_failed += n_batch_failed
        if n_batch_failed > 0 and n_failed <= 5:
            for i in range(n_batch_failed):
                idx = failed_mask.nonzero(as_tuple=True)[0][i].item()
                logger.warning("  silence detected for %s (possible load failure)",
                               meta[idx]["utt_id"])
        wav = wav.to(device)
        out = model(wav)
        scores.append(out["score"].cpu().numpy())
        labels.append(y.numpy())
        ids.extend(m["utt_id"] for m in meta)
        processed = min((batch_idx + 1) * batch_size, total)
        pct = processed / total * 100
        now = _time.time()
        if pct >= next_pct or (now - last_log) >= LOG_INTERVAL_SEC or batch_idx == n_batches - 1:
            elapsed = now - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            eta_m, eta_s = divmod(int(eta), 60)
            if batch_idx == n_batches - 1:
                logger.info("scoring %s: 100%% (%d/%d) — done in %dm%02ds",
                            ds_name, total, total, int(elapsed) // 60, int(elapsed) % 60)
            else:
                logger.info("scoring %s: %d%% (%d/%d) — %.0f utt/s — ETA %dm%02ds",
                            ds_name, int(pct), processed, total, rate, eta_m, eta_s)
            next_pct += LOG_INTERVAL_PCT
            last_log = now
    if not scores:
        logger.warning("manifest yielded no batches (%s), skipping", manifest)
        return None, None, []
    fail_pct = n_failed / len(ds) * 100
    if fail_pct > 0:
        logger.warning("manifest %s: %d/%d files (%.1f%%) failed to load (returned silence)",
                       manifest, n_failed, len(ds), fail_pct)
    if fail_pct >= 90:
        logger.error("manifest %s: %.1f%% files failed — aborting (results would be meaningless). "
                     "Check: (1) ffmpeg installed? (2) files actual format (xxd header)? "
                     " (3) symlink/mount valid?",
                     manifest, fail_pct)
        return None, None, []
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

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest_items = list(manifests.items())
    n_manifests = len(manifest_items)
    for i, (name, path) in enumerate(manifest_items, 1):
        resolved = _resolve_manifest(path)
        if resolved is None:
            logger.warning("skip %s (manifest not found: %s)", name, path)
            continue
        logger.info("[%d/%d] scoring %-24s  manifest=%s", i, n_manifests, name, resolved)
        scores, labels, _ = score_manifest(model, resolved, audio_cfg, device)
        if scores is None:
            logger.warning("skip %s (empty or unreadable manifest)", name)
            continue
        probs = scores_to_probs(scores)
        m = summarize(scores, labels, probs)
        point, lo, hi = bootstrap_eer_ci(scores, labels,
                                         n_boot=eval_cfg.get("n_bootstrap", 1000))
        m["eer_ci95"] = [lo, hi]
        results[name] = m
        logger.info("[%d/%d] %-18s EER=%.4f [%.4f, %.4f] tDCF=%.4f AUROC=%.4f",
                    i, n_manifests, name, m["eer"], lo, hi, m["min_tdcf"], m["auroc"])
        # write incrementally so downstream failures don't erase earlier results
        (out / "results.json").write_text(json.dumps(results, indent=2))

    (out / "results.json").write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", out / "results.json")
    return results
