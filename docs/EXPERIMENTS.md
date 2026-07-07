# Experiment matrix

Every row is a config + a seed sweep. Results land in `experiments/<name>/` and are
aggregated by `scripts/aggregate_results.py` into the paper's tables.

## E1 — Baselines (parity / reference)
| id | model | config | expected in-domain EER |
|----|-------|--------|------------------------|
| E1.0 | LFCC-GMM | `experiment=b0_lfcc_gmm` | ~9% |
| E1.1 | LFCC-LCNN | `experiment=b1_lcnn` | ~5% |
| E1.2 | RawNet2 | `experiment=b2_rawnet2` | ~4% |
| E1.3 | AASIST | `experiment=b3_aasist` | ~1% |
| E1.4 | wav2vec2-AASIST (SOTA) | `experiment=b4_w2v_aasist` | <1% |
| E1.5 | WavLM-AASIST + OC-Softmax | `experiment=b5_wavlm_ocs` | <1% |

## E2 — Proposed
| id | model | config |
|----|-------|--------|
| E2.0 | **AuralGuard (full)** | `experiment=auralguard` |

## E3 — Ablations (RQ1–RQ3)
| id | change | isolates |
|----|--------|----------|
| E3.1 | View A only | complementarity |
| E3.2 | View B only | complementarity |
| E3.3 | last-layer SSL (no layer-attn) | RQ layer-attention |
| E3.4 | CE loss instead of OC-Softmax | RQ2 |
| E3.5 | OC-Softmax, no SupCon | RQ2 |
| E3.6 | no augmentation | RQ3 |
| E3.7 | RawBoost only (no codec chain) | RQ3 |
| E3.8 | backbone = XLS-R-300M | backbone |
| E3.9 | backbone = HuBERT | backbone |
| E3.10 | frozen SSL vs. last-3 fine-tuned | compute/perf |

## E4 — Generalization (headline, RQ1)
Zero-shot EER of E1.4, E1.5, E2.0 on each of:
ASVspoof2021-LA, ASVspoof2021-DF, ASVspoof5, In-the-Wild, WaveFake, MLAAD, CodecFake.

## E5 — Robustness sweeps (RQ3 / G2)
EER vs. codec (MP3 320/128/64k, Opus 64/24/12k, AMR-NB, G.711) and vs. SNR (0–20 dB MUSAN)
and reverberation. Line plots per model.

## E6 — Calibration & explainability (RQ4 / G3)
ECE, Brier, reliability diagrams before/after temperature scaling; attention/Grad-CAM
attributions; qualitative case studies.

## E7 — Efficiency
Params, MACs, RTF (CPU + GPU), peak memory — for the deployment section.

## Reporting
- 3 seeds; mean ± std; 95% bootstrap CI on EER.
- Paired bootstrap significance vs. B4 and B5.
- One command reproduces each table: see `scripts/aggregate_results.py --table N`.
