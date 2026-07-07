"""Detection metrics for anti-spoofing / synthetic-speech detection.

Implements the standard ASVspoof metrics:
  * Equal Error Rate (EER)
  * minimum tandem Detection Cost Function (min t-DCF)
plus general classification/calibration metrics (AUROC, ECE, Brier).

Score convention throughout this repo:
  a HIGHER score  ==>  MORE likely SPOOF (AI-generated, label 1).
Adjust the sign at call time if your model outputs the opposite.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# EER
# --------------------------------------------------------------------------- #
def compute_det_curve(target_scores: np.ndarray, nontarget_scores: np.ndarray):
    """Return (frr, far, thresholds) for a detection error trade-off curve.

    `target_scores`    : scores of the POSITIVE class (spoof, label 1).
    `nontarget_scores` : scores of the NEGATIVE class (bona-fide, label 0).
    Higher score = more spoof-like.
    """
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )

    # sort by score (ascending); threshold sweeps from low to high
    order = np.argsort(all_scores, kind="mergesort")
    labels = labels[order]

    # As the threshold rises through the sorted scores:
    #   tar_below      = # of target (spoof) trials at or below threshold  -> false rejects
    #   nontarget_above= # of non-target (bona-fide) trials above threshold -> false accepts
    tar_below = np.cumsum(labels)
    idx_below = np.arange(1, n_scores + 1)
    nontarget_above = nontarget_scores.size - (idx_below - tar_below)

    # false rejection: spoof (target) scored below threshold -> called bona-fide
    frr = np.concatenate((np.atleast_1d(0), tar_below / target_scores.size))
    # false acceptance: bona-fide (non-target) scored above threshold -> called spoof
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_above / nontarget_scores.size)
    )
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[order[0]] - 1e-6), all_scores[order])
    )
    return frr, far, thresholds


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate.

    `scores` : model scores, higher = more spoof-like.
    `labels` : 1 = spoof, 0 = bona-fide.
    Returns (eer, threshold_at_eer).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    target = scores[labels == 1]
    nontarget = scores[labels == 0]
    if target.size == 0 or nontarget.size == 0:
        raise ValueError("EER needs both classes present.")

    frr, far, thresholds = compute_det_curve(target, nontarget)
    abs_diff = np.abs(frr - far)
    idx = int(np.nanargmin(abs_diff))
    eer = float((frr[idx] + far[idx]) / 2.0)
    return eer, float(thresholds[idx])


# --------------------------------------------------------------------------- #
# min t-DCF (ASVspoof 2019 formulation)
# --------------------------------------------------------------------------- #
DEFAULT_TDCF_COST = {
    "Pspoof": 0.05,
    "Cmiss": 1.0,   # cost of ASV missing a target
    "Cfa": 10.0,    # cost of ASV false alarm
    "Cfa_spoof": 10.0,
}


def compute_min_tdcf(
    cm_scores: np.ndarray,
    cm_labels: np.ndarray,
    asv_frr: float | None = None,
    asv_far: float | None = None,
    asv_far_spoof: float | None = None,
    cost: dict | None = None,
) -> float:
    """Minimum tandem DCF (normalized), ASVspoof-2019 style.

    For a standalone CM (no ASV scores available), pass fixed ASV operating-point
    error rates (asv_frr/asv_far/asv_far_spoof). If left None, a common reference
    operating point is used so the number is comparable across runs of THIS repo,
    but for paper numbers use the official ASV scores + the official script.
    """
    cost = {**DEFAULT_TDCF_COST, **(cost or {})}
    if asv_frr is None:
        asv_frr, asv_far, asv_far_spoof = 0.05, 0.05, 0.05  # reference point

    cm_scores = np.asarray(cm_scores, dtype=np.float64)
    cm_labels = np.asarray(cm_labels, dtype=np.int64)
    bona = cm_scores[cm_labels == 0]
    spoof = cm_scores[cm_labels == 1]

    frr, far, _ = compute_det_curve(spoof, bona)
    p_spoof = cost["Pspoof"]
    p_tar = (1 - p_spoof) * (1 - asv_frr)
    p_non = (1 - p_spoof) * asv_far

    # weights on CM false-reject (of spoof->bona miss) and CM false-accept
    c1 = cost["Cmiss"] * p_tar + cost["Cfa"] * p_non
    c2 = cost["Cfa_spoof"] * p_spoof * (1 - asv_far_spoof)

    tdcf = c1 * frr + c2 * far
    tdcf_default = min(c1, c2)  # cost of a trivial detector
    if tdcf_default <= 0:
        return float("nan")
    return float(np.min(tdcf) / tdcf_default)


# --------------------------------------------------------------------------- #
# General metrics
# --------------------------------------------------------------------------- #
def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """ECE for the probability of the positive (spoof) class."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = probs.size
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs > lo) & (probs <= hi)
        if not mask.any():
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    return float(np.mean((probs - labels) ** 2))


def bootstrap_eer_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap (point, lo, hi) 95% CI for EER."""
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    n = scores.size
    point, _ = compute_eer(scores, labels)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            e, _ = compute_eer(scores[idx], labels[idx])
            boots.append(e)
        except ValueError:
            continue
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return point, lo, hi


def summarize(scores: np.ndarray, labels: np.ndarray, probs: np.ndarray | None = None) -> dict:
    """One-shot metric bundle for a score set."""
    eer, thr = compute_eer(scores, labels)
    out = {
        "eer": eer,
        "eer_threshold": thr,
        "min_tdcf": compute_min_tdcf(scores, labels),
        "auroc": auroc(scores, labels),
    }
    if probs is not None:
        out["ece"] = expected_calibration_error(probs, labels)
        out["brier"] = brier_score(probs, labels)
    return out
