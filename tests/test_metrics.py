import numpy as np

from auralguard.evaluation.metrics import (
    brier_score,
    compute_eer,
    expected_calibration_error,
    summarize,
)


def test_eer_perfect_separation():
    # spoof (label 1) high scores, bona-fide (0) low scores -> EER = 0
    scores = np.concatenate([np.zeros(100), np.ones(100)])
    labels = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)
    eer, _ = compute_eer(scores, labels)
    assert eer < 1e-6


def test_eer_random_is_near_half():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=2000)
    labels = rng.integers(0, 2, size=2000)
    eer, _ = compute_eer(scores, labels)
    assert 0.4 < eer < 0.6


def test_eer_symmetric_overlap():
    # two gaussians separated by 2 sigma -> EER ~ 0.16
    rng = np.random.default_rng(1)
    bona = rng.normal(0, 1, 5000)
    spoof = rng.normal(2, 1, 5000)
    scores = np.concatenate([bona, spoof])
    labels = np.concatenate([np.zeros(5000), np.ones(5000)]).astype(int)
    eer, _ = compute_eer(scores, labels)
    assert 0.13 < eer < 0.19


def test_ece_and_brier_bounds():
    probs = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    assert 0.0 <= expected_calibration_error(probs, labels) <= 1.0
    assert 0.0 <= brier_score(probs, labels) <= 1.0


def test_summarize_keys():
    rng = np.random.default_rng(2)
    scores = np.concatenate([rng.normal(0, 1, 500), rng.normal(2, 1, 500)])
    labels = np.concatenate([np.zeros(500), np.ones(500)]).astype(int)
    probs = 1 / (1 + np.exp(-scores))
    out = summarize(scores, labels, probs)
    for k in ["eer", "min_tdcf", "auroc", "ece", "brier"]:
        assert k in out
