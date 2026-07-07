# AuralGuard — Precise Detection of AI-Generated Speech

> Generalizable, real-world-robust synthetic-speech (voice deepfake) detection.
> Research code, trained models, a Hugging Face inference API, and a Next.js demo UI.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Paper](https://img.shields.io/badge/paper-in%20progress-orange)]()

---

## What this is

A complete, reproducible research project targeting a **Q1 journal** (e.g. *IEEE/ACM
Transactions on Audio, Speech and Language Processing*, *Computer Speech & Language*,
*IEEE TIFS*, *Pattern Recognition*, *Expert Systems with Applications*).

The scientific goal is **not** to win a single in-domain benchmark — modern systems
already reach <1% EER on ASVspoof 2019 LA. The goal is the field's open problem:

> **Detect AI-generated speech from *unseen* synthesizers, in *real-world* channels
> (codecs, compression, noise, telephony), across *multiple languages* — reliably and
> with calibrated confidence.**

Our proposed model, **AuralGuard**, is a multi-view detector that fuses a
self-supervised speech front-end, hand-crafted vocoder-artifact features, and a
graph-attention back-end, trained with a one-class objective and aggressive channel
augmentation for out-of-distribution generalization.

## Repository map

```
.
├── docs/                  # Research plan, architecture, protocol, reproducibility
│   ├── RESEARCH_PLAN.md    # ← THE Q1 PLAN (read this first)
│   ├── ARCHITECTURE.md     # Model + system design
│   ├── DATASETS.md         # Corpora, licenses, splits, download recipes
│   ├── EXPERIMENTS.md      # Experiment matrix, ablations, baselines
│   ├── EXTENDING.md        # Plug in YOUR OWN model/dataset (registry contract)
│   └── ROADMAP.md          # 24-week timeline to submission
├── config/                # Hydra/YAML configs (data, model, training, eval)
├── src/                    # Library code (installable package: `auralguard`)
│   ├── data/               # Datasets, manifests, augmentation
│   ├── features/           # LFCC/CQT/phase front-ends
│   ├── models/             # SSL front-end, AASIST back-end, fusion, losses
│   ├── training/           # Trainer, schedulers, checkpointing
│   ├── evaluation/         # EER, min t-DCF, calibration, robustness sweeps
│   └── inference/          # Single-file / batch prediction
├── scripts/               # CLI entrypoints (train, evaluate, export)
├── notebooks/             # EDA, baseline reproduction, results analysis
├── tests/                 # Unit tests (metrics, shapes, augment)
├── deployment/            # Hugging Face Space (FastAPI + Gradio) + Docker
├── ui/                    # Next.js end-user demo (App Router + Tailwind)
└── paper/                 # LaTeX manuscript skeleton + bib
```

## Quickstart

```bash
# 1. Environment (use Python 3.10/3.11 for ML wheels — see docs/REPRODUCIBILITY.md)
conda env create -f environment.yml
conda activate auralguard
pip install -e .

# 2. Get data manifests (see docs/DATASETS.md for the download recipes)
python scripts/build_manifests.py --config config/data/asvspoof2019_la.yaml

# 3. Train the proposed model
python scripts/train.py --config config/config.yaml experiment=auralguard

# 4. Evaluate (in-domain + cross-dataset + robustness)
python scripts/evaluate.py --ckpt experiments/auralguard/best.ckpt \
    --protocol config/eval/full_protocol.yaml

# 5. Single-file inference
python -m auralguard.inference.predict --audio path/to/clip.wav --ckpt best.ckpt
```

## Using other models

The pipeline is **model-agnostic** — every detector registers in a model zoo and the
trainer/evaluator/API build whatever the config names:

```bash
python scripts/train.py experiment=b1_lcnn      # LFCC + LCNN baseline
python scripts/train.py experiment=b2_rawnet2   # RawNet2 baseline
python scripts/train.py experiment=b3_aasist    # raw AASIST baseline
python scripts/train.py experiment=auralguard   # proposed model
```

Adding your own architecture is a decorator + two small YAML files — see
[`docs/EXTENDING.md`](docs/EXTENDING.md). The HF Space serves any registered model
without code changes (the checkpoint carries its own config).

## Deployment

- **API / model host:** `deployment/hf_space/` — a Hugging Face Space exposing both a
  Gradio UI and a FastAPI JSON endpoint (`POST /api/detect`).
- **End-user UI:** `ui/` — a Next.js app that records/uploads audio and calls the HF API.

See `deployment/README.md` and `ui/README.md`.

## Citing

A `CITATION.cff` and BibTeX entry will be added on submission. Draft manuscript lives in
`paper/`.

## License

MIT for code. **Datasets and pretrained SSL backbones keep their own licenses** — check
`docs/DATASETS.md` before any redistribution or commercial deployment.
