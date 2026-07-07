# Research Plan — Precise, Generalizable Detection of AI-Generated Speech

**Working title:** *AuralGuard: Multi-View One-Class Learning for Generalizable
Detection of AI-Generated Speech in the Wild*

**Target venues (Q1):**
1. *IEEE/ACM Transactions on Audio, Speech, and Language Processing* (TASLP) — primary
2. *IEEE Transactions on Information Forensics and Security* (TIFS)
3. *Computer Speech & Language* (Elsevier)
4. *Pattern Recognition* / *Expert Systems with Applications* (fallbacks, faster turnaround)

---

## 1. Problem statement and motivation

Text-to-speech (TTS) and voice-conversion (VC) systems (VALL-E, XTTS, Tortoise,
StyleTTS2, diffusion vocoders, neural codec LMs) now synthesize speech that humans and
naïve classifiers cannot distinguish from real recordings. This enables fraud
(voice-cloning scams), disinformation, and bypass of voice biometrics.

The **anti-spoofing / synthetic-speech-detection (SSD)** community has strong in-domain
results (< 1% EER on ASVspoof 2019 LA). **But three failure modes block real deployment**,
and they are exactly the gaps a Q1 contribution must close:

| Gap | Symptom | Why it matters |
|-----|---------|----------------|
| **G1 — Unseen generators** | EER collapses (5–30%) on synthesizers absent from training | New TTS models appear monthly; a detector must generalize *without* retraining |
| **G2 — Channel/robustness** | Codec, MP3/Opus compression, telephony, noise, and re-encoding destroy accuracy | Real audio arrives via WhatsApp, phone lines, social platforms |
| **G3 — Calibration & explainability** | Scores are uncalibrated; decisions are opaque | Forensic/legal use needs calibrated confidence + evidence |

### Research questions

- **RQ1:** Can a *multi-view* representation (self-supervised + hand-crafted vocoder-artifact
  features) generalize to unseen generators better than either view alone?
- **RQ2:** Does a *one-class* objective (modelling only bona-fide speech tightly) outperform
  binary cross-entropy for open-set generalization (G1)?
- **RQ3:** Does *codec/compression-aware augmentation* recover robustness (G2) without
  sacrificing clean-condition accuracy?
- **RQ4:** Can we produce *calibrated* scores and *interpretable* evidence (time-frequency
  attributions) suitable for forensic use (G3)?

### Contributions (the "novelty" the reviewers will grade)

1. **AuralGuard**, a multi-view architecture fusing an SSL front-end (WavLM/XLS-R with
   *learnable layer attention*), a hand-crafted vocoder-artifact branch (LFCC + modified
   group-delay + CQT phase), and an AASIST graph-attention back-end.
2. A **one-class contrastive objective** (OC-Softmax + supervised contrastive auxiliary)
   that tightens the bona-fide manifold and pushes *any* spoof outward — improving unseen-attack
   generalization.
3. A **channel-robustness augmentation curriculum** (RawBoost + realistic codec chain:
   MP3/AAC/Opus/AMR/G.711 + reverberation + additive noise) and a study of its effect.
4. A **large cross-corpus generalization + robustness benchmark protocol** spanning
   ASVspoof 2019/2021, ASVspoof 5, In-the-Wild, MLAAD (multilingual), WaveFake, and
   CodecFake, with **calibration (ECE)** and **explainability** analysis.
5. Full **open-source release**: code, trained weights, a live HF inference API, and a
   demo UI — reproducibility is itself a selling point for Q1 review.

> **Positioning statement (for the intro):** Prior SOTA (SSL+AASIST, e.g. wav2vec2-AASIST)
> optimizes in-domain EER. We reframe the task as *open-set, channel-robust, calibrated*
> detection and show that multi-view one-class learning closes a large fraction of the
> cross-generator generalization gap.

---

## 2. Related work (to be expanded in `paper/sections/related.tex`)

- **Front-ends:** LFCC, CQCC, spectrogram, raw waveform (RawNet2), SSL features
  (wav2vec 2.0, XLS-R, WavLM, HuBERT), Whisper encoder features.
- **Back-ends:** LCNN, ResNet, RawNet2, **AASIST** (spectro-temporal graph attention),
  Conformer, SE-Rawformer.
- **SOTA line:** SSL front-end + AASIST back-end (Tak et al. 2022) — our primary baseline.
- **Objectives:** softmax/BCE, **OC-Softmax** (Zhang et al.), AM-Softmax, contrastive.
- **Generalization studies:** RawBoost augmentation, continual learning, domain
  adaptation, single-/one-class modelling.
- **Datasets:** ASVspoof 2019/2021/5, ADD 2022/2023, In-the-Wild, WaveFake, FakeAVCeleb,
  MLAAD, CodecFake.

*Gap we exploit:* few works jointly attack G1+G2+G3; almost none report **calibration**
and **cross-lingual** generalization together with **explainability**. That combination is
our defensible Q1 delta.

---

## 3. Proposed method — AuralGuard

See `docs/ARCHITECTURE.md` for tensor shapes and the module diagram. Summary:

```
          ┌─────────────────────── View A: SSL ───────────────────────┐
raw wave ─┤ WavLM-Large / XLS-R-300M (partially fine-tuned)            │
          │   → per-layer hidden states → learnable layer-attention    │──┐
          └────────────────────────────────────────────────────────────┘  │
          ┌────────────────── View B: artifact cues ──────────────────┐    │  fusion
raw wave ─┤ LFCC + modified group-delay + CQT-phase → light CNN        │──┤ (gated
          └────────────────────────────────────────────────────────────┘  │  cross-
                                                                            │  attention)
                         AASIST spectro-temporal graph-attention back-end ◄─┘
                                              │
                        embedding z ──► OC-Softmax head (score s)
                                     └─► SupCon auxiliary (train only)
```

**Why each choice (defensibility for reviewers):**
- *SSL front-end:* captures high-level generator inconsistencies; layer attention because
  spoof artifacts concentrate in specific transformer layers (we ablate this).
- *Artifact branch:* SSL features can miss low-level vocoder phase artifacts; group-delay /
  CQT-phase explicitly expose them. Complementarity is an ablation.
- *AASIST back-end:* SOTA at modelling spectro-temporal sub-band artifacts jointly.
- *OC-Softmax:* learns a compact bona-fide region → open-set spoofs fall outside → better G1.
- *Augmentation curriculum:* directly targets G2.

### Baselines we must beat / match
| Baseline | Front-end | Back-end | Loss |
|----------|-----------|----------|------|
| B0 | LFCC | GMM | — |
| B1 | LFCC | LCNN | CE |
| B2 | raw | RawNet2 | CE |
| B3 | raw | AASIST | CE |
| B4 (SOTA) | wav2vec2-XLS-R | AASIST | CE |
| B5 (strong SOTA) | WavLM | AASIST | OC-Softmax |
| **AuralGuard (ours)** | **WavLM + artifact (multi-view)** | **AASIST** | **OC-Softmax + SupCon** |

We claim novelty over B4/B5 via **multi-view fusion + curriculum + the calibration/XAI/
cross-lingual evaluation**, not by inventing a back-end from scratch (honest, reviewer-safe).

---

## 4. Datasets

Detailed licenses and download recipes in `docs/DATASETS.md`.

| Dataset | Role | Tests |
|---------|------|-------|
| **ASVspoof 2019 LA** | Train + in-domain dev/eval | sanity / SOTA parity |
| **ASVspoof 2021 LA & DF** | Cross-condition eval | codec/compression (G2) |
| **ASVspoof 5 (2024)** | Modern generators eval | unseen generators (G1) |
| **In-the-Wild** | Real-world eval | domain shift (G1+G2) |
| **MLAAD** (multi-lingual) | Cross-lingual eval | language shift |
| **WaveFake** | Cross-generator eval | unseen vocoders (G1) |
| **CodecFake** | Neural-codec spoof eval | codec-based TTS (emerging) |

**Golden rule:** *train on ASVspoof 2019 LA only (+ augmentation); evaluate zero-shot on
everything else.* This isolates generalization and is the protocol reviewers respect.

---

## 5. Experimental protocol

### Metrics
- **EER** (primary), **min t-DCF** (ASVspoof standard), **AUROC**, **balanced accuracy**, **F1**.
- **Calibration:** Expected Calibration Error (ECE), reliability diagrams, Brier score.
- **Robustness:** EER vs. codec/SNR sweep curves.
- **Efficiency:** params, MACs, real-time factor (RTF) on CPU + GPU (for deployment).

### Evaluation axes
1. **In-domain** (ASVspoof19 eval) — parity check with SOTA.
2. **Cross-dataset zero-shot** (all others) — the headline result (G1).
3. **Robustness sweeps** — MP3/Opus/AMR/G.711 bitrates, additive noise (MUSAN) SNR
   0–20 dB, reverberation (RIRs) (G2).
4. **Cross-lingual** (MLAAD by language family).
5. **Calibration + explainability** (G3): reliability diagrams; Grad-CAM / attention
   attributions over the spectro-temporal graph; listening-test spot checks.

### Statistical rigor
- ≥ 3 seeds; report mean ± std; **95% CIs via bootstrap** on EER.
- Significance: paired bootstrap / McNemar between ours and each baseline.
- Fix all splits; publish manifests + seeds. No test-set peeking for model selection.

### Ablations (each is a paper table)
- View A only / View B only / both (RQ1).
- CE vs. OC-Softmax vs. OC-Softmax+SupCon (RQ2).
- Layer-attention vs. last-layer vs. mean-pool of SSL layers.
- Augmentation off / RawBoost only / full curriculum (RQ3).
- Backbone: WavLM-Large vs. XLS-R-300M vs. HuBERT.
- Frozen vs. partial fine-tune of SSL front-end (compute/perf trade-off).

---

## 6. Reproducibility & compute

- **Compute:** single 24 GB GPU (RTX 3090/4090/A5000) is enough with frozen/partial SSL +
  gradient checkpointing; A100 speeds up full fine-tune. Budget: ~150–300 GPU-hours total.
- Deterministic seeds, pinned deps (`environment.yml`), config-as-code (Hydra), and
  released checkpoints. See `docs/REPRODUCIBILITY.md`.
- **Note:** use Python 3.10/3.11 for the ML env — several audio/ML wheels lag on 3.14.

---

## 7. Ethics, dual-use, and limitations

- **Dual-use:** a detector can be probed by adversaries. We report robustness to
  compression but **not** an anti-forensic attack recipe. We discuss adaptive-attacker
  limitations honestly (required by TIFS/TASLP reviewers).
- **Bias:** we audit performance across languages/accents/genders where labels allow.
- **Privacy:** all corpora are public research datasets; no scraping of private voices.
- **Deployment caveat:** the demo UI states clearly that detection is probabilistic and
  must not be the sole basis for high-stakes decisions.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| We don't beat B5 on in-domain EER | Frame contribution as generalization+calibration; in-domain parity is acceptable |
| SSL fine-tuning unstable / OOM | Freeze SSL, train layer-attention + back-end only; gradient checkpointing |
| Dataset access delays (ASVspoof 5) | Start with 2019/2021 + In-the-Wild + WaveFake (all readily available) |
| Reviewer: "just fusion of known parts" | Emphasize the *one-class multi-view* interaction + the new eval protocol + released artifacts |

---

## 9. Deliverables checklist (for submission)

- [ ] Reproducible codebase (this repo) + released weights
- [ ] Manuscript (`paper/`) with all tables/figures auto-generated from `experiments/`
- [ ] Live HF inference API + Next.js demo (strengthens the "impact/availability" score)
- [ ] Supplementary: full hyperparameters, per-language results, reliability diagrams
- [ ] Data/ethics statement, author checklist, CRediT roles

See `docs/ROADMAP.md` for the 24-week schedule and `docs/EXPERIMENTS.md` for the full run matrix.
