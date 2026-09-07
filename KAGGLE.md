# AuralGuard — Kaggle Training Guide

Complete guide for training all models on Kaggle with checkpoint recovery, cross-account resume, and HuggingFace integration.

---

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Notebook Structure](#notebook-structure)
- [Progress Tracking](#progress-tracking)
- [Training](#training)
- [Checkpoint Recovery](#checkpoint-recovery)
- [Cross-Account Resume](#cross-account-resume)
- [Datasets](#datasets)
- [HuggingFace Uploads](#huggingface-uploads)
- [Model Deployment](#model-deployment)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Kaggle notebook trains 5 models sequentially:

| Model | Description | Est. Time |
|-------|-------------|-----------|
| B1 | LFCC + LCNN | ~1 hr |
| B2 | RawNet2 | ~2 hrs |
| B3 | AASIST | ~3 hrs |
| B5 | WavLM + AASIST + OCSoftmax | ~6 hrs |
| AuralGuard | Full multi-view model | ~8 hrs |

**Total: ~20 hours** (fits within Kaggle's 30hr/week limit)

### Key Features

- **Progress tracking** — status cell shows which cells are done after any crash
- **Checkpoint recovery** — resume from last epoch if session crashes
- **Live HuggingFace upload** — checkpoints uploaded after every epoch (background thread)
- **Cross-account resume** — pull checkpoints from HF in a new account
- **All datasets** — zero-shot eval + augmentation corpora auto-downloaded
- **Auto-upload** — figures, metrics, checkpoints pushed to HF Hub
- **Model deployment** — trained model uploaded to dedicated HF repo for serving

---

## Setup — Step by Step

### Step 1: Create a Kaggle Dataset with the Notebook

1. Go to **kaggle.com** → **Code** → **New Notebook**
2. In the notebook editor, click **File** → **Import Notebook**
3. Upload `notebooks/kaggle_training.ipynb`
4. **Rename** it to `AuralGuard Training`

### Step 2: Add Input Datasets (right sidebar → +Data)

Click **+Data** in the right sidebar and search + add each one:

| # | Search for | Kaggle path | Why |
|---|-----------|-------------|-----|
| 1 | `awsaf49/asvpoof-2019-dataset` | `datasets/awsaf49/asvpoof-2019-dataset` | Training (required) |
| 2 | `abdallamohamed312/in-the-wild-audio-deepfake` | `datasets/abdallamohamed312/in-the-wild-audio-deepfake` | Zero-shot eval |
| 3 | `walimuhammadahmad/fakeaudio` | `datasets/walimuhammadahmad/fakeaudio` | Zero-shot eval (WaveFake) |
| 4 | `kkijjaa/asvspoof2021-la` | `datasets/kkijjaa/asvspoof2021-la` | Zero-shot eval |
| 5 | `serjkalinovskiy/asvspoof2021-df` | `datasets/serjkalinovskiy/asvspoof2021-df` | Zero-shot eval |
| 6 | `riosgonzalo/musan-rirs` | `datasets/riosgonzalo/musan-rirs` | Augmentation (both) |
| 7 | `nhattruongdev/rirs-noises` | `datasets/nhattruongdev/rirs-noises` | Augmentation (RIRs) |

> **How to find them:** Click `+Data` → type the search term → click the dataset → click **Add**.
>
> **Can't find one?** No problem — the notebook auto-downloads missing datasets from HuggingFace/Zenodo. It's just slower.

### Step 3: Enable GPU + Internet

In the right sidebar:
1. **Accelerator** → select **GPU** (P100 or T4 x2)
2. **Internet** → toggle **ON**

### Step 4: Add HuggingFace Token (for checkpoint upload)

1. Right sidebar → **Settings** → **Secrets**
2. Click **Add new secret**
3. Name: `HF_TOKEN`
4. Value: `hf_xxxxxxxxxxxxx` (get from https://huggingface.co/settings/tokens)

### Step 5: Run Cells Sequentially

Run cells from top to bottom. The notebook tracks progress — if it crashes, re-run and it resumes from where it left off.

**First run order:**
```
Cell 1  → Helpers (logging, progress)
Cell 2  → Status display (shows what's done)
Cell 3  → HuggingFace setup
Cell 4  → Checkpoint recovery helpers
Cell 6  → Find ASVspoof 2019 dataset
Cell 7  → Download zero-shot datasets (or symlink from Kaggle input)
Cell 8  → Download augmentation datasets (or symlink from Kaggle input)
Cell 9  → Install AuralGuard
Cell 10 → HuggingFace login
Cell 11 → Build manifests
Cell 12 → Build augmentation manifests
```

**Then training (run one at a time, each takes 1-8 hrs):**
```
Cell 14 → Train B1 (LFCC-LCNN)       ~1 hr
Cell 15 → Train B2 (RawNet2)          ~2 hrs
Cell 16 → Train B3 (AASIST)           ~3 hrs
Cell 17 → Train B5 (WavLM+OCS)        ~6 hrs
Cell 18 → Train AuralGuard (full)     ~8 hrs
```

**After training:**
```
Cell 20 → Evaluate all models
Cell 22 → Zero-shot evaluation
Cell 24 → Generate paper figures
Cell 26 → Export results table
Cell 28 → Upload to HuggingFace
```

---

## Quick Setup Checklist

```
[ ] Create Kaggle notebook, upload .ipynb
[ ] Add 7 input datasets via +Data (search terms in Step 2 above)
[ ] Enable GPU accelerator
[ ] Enable Internet
[ ] Add HF_TOKEN to Kaggle Secrets
[ ] Run Cell 1 (helpers)
[ ] Run Cell 2 (status — shows what to do next)
[ ] Follow the "NEXT" pointer in status output
```

---

## Notebook Structure

```
Cell 0   — Title & overview
Cell 1   — Progress & logging helpers (mark_done, get_progress, run_cmd)
Cell 2   — Progress status (auto-shows which cells are done on every session start)
Cell 3   — HuggingFace Hub helpers (upload, download, login)
Cell 4   — Checkpoint recovery helpers
Cell 5   — [markdown] Dataset Setup
Cell 6   — Find ASVspoof 2019 LA dataset
Cell 7   — Download zero-shot eval datasets
Cell 8   — Download augmentation corpora (MUSAN + RIRs)
Cell 9   — Install AuralGuard + dependencies
Cell 10  — Login to HuggingFace + create repo
Cell 11  — Build manifests for ALL datasets
Cell 12  — Build augmentation manifests
Cell 13  — [markdown] Training
Cell 14  — Train B1 (LFCC-LCNN)
Cell 15  — Train B2 (RawNet2)
Cell 16  — Train B3 (AASIST)
Cell 17  — Train B5 (WavLM+OCS)
Cell 18  — Train AuralGuard (full)
Cell 19  — [markdown] In-Domain Evaluation
Cell 20  — Evaluate all models
Cell 21  — [markdown] Zero-Shot Evaluation
Cell 22  — Zero-shot eval on all unseen datasets
Cell 23  — [markdown] Generate Paper Figures
Cell 24  — Generate 5 paper figures
Cell 25  — [markdown] Export Results Table
Cell 26  — Export results as markdown + CSV
Cell 27  — [markdown] Upload to HuggingFace
Cell 28  — Upload all results + figures + checkpoints
Cell 29  — [markdown] Model Deployment
Cell 30  — Upload model to dedicated HF repo
Cell 31  — [markdown] Package for Download
Cell 32  — Package results as tar.gz
Cell 33  — [markdown] Recovery Guide
Cell 34  — Pull ALL checkpoints from HF
Cell 35  — Pull SINGLE model from HF (quick resume)
```

---

## Progress Tracking

The notebook tracks which cells have completed execution. This is critical for crash recovery.

### How It Works

- **Cell 1** defines `mark_done(cell_id)` and `get_progress()` functions
- **Cell 2** runs automatically on every session start and shows a status table
- **Every code cell** calls `mark_done("cell_id")` as its last line when it completes
- Progress is saved to `/kaggle/working/auralguard/progress.json`

### Status Cell Output (Cell 2)

```
============================================================
  AURALGUARD PROGRESS STATUS
============================================================
  ID                     │ Status
────────────────────────────────────────────────────────────
  1.  Helpers            │ ✅ DONE
  2.  HuggingFace setup  │ ✅ DONE
  ...
  8a. Train B1           │ ⏳ pending
  8b. Train B2           │ ⏳ pending
  ...
────────────────────────────────────────────────────────────
  ➜  NEXT: Run cell for "8a. Train B1 (LFCC-LCNN)"
============================================================
```

### After a Crash

1. Open a new Kaggle session
2. Run **Cell 1** (helpers — loads progress file)
3. Run **Cell 2** (status — shows exactly what's done and what to run next)
4. Follow the "NEXT" pointer to resume

### Cell IDs

| Cell | ID | Description |
|------|----|-------------|
| 1 | `helpers` | Helpers & logging |
| 2 | (auto) | Progress status display |
| 3 | `hf_setup` | HuggingFace Hub setup |
| 4 | `checkpoint_aids` | Checkpoint recovery helpers |
| 6 | `find_dataset` | Find ASVspoof dataset |
| 7 | `zero_shot_data` | Download zero-shot datasets |
| 8 | `augmentation_data` | Download augmentation corpora |
| 9 | `install` | Install AuralGuard |
| 10 | `hf_login` | HF Login + create repo |
| 11 | `build_manifests` | Build manifests |
| 12 | `aug_manifests` | Build augmentation manifests |
| 14 | `train_b1` | Train B1 (LFCC-LCNN) |
| 15 | `train_b2` | Train B2 (RawNet2) |
| 16 | `train_b3` | Train B3 (AASIST) |
| 17 | `train_b5` | Train B5 (WavLM+AASIST) |
| 18 | `train_auralguard` | Train AuralGuard |
| 20 | `eval_in_domain` | In-domain evaluation |
| 22 | `eval_zero_shot` | Zero-shot evaluation |
| 24 | `figures` | Generate paper figures |
| 26 | `results_table` | Export results table |
| 28 | `hf_upload_all` | Upload results to HF Hub |
| 30 | `model_deploy` | Upload model for deployment |
| 32 | `package` | Package for local download |

---

### Run All Models
Run cells 1-12 (setup), then cells 14-18 (training) sequentially.

### Run Single Model
Run cells 1-12, then the specific training cell:
- Cell 14: B1 (LFCC-LCNN)
- Cell 15: B2 (RawNet2)
- Cell 16: B3 (AASIST)
- Cell 17: B5 (WavLM+OCS)
- Cell 18: AuralGuard (full)

### What Happens During Training

After every epoch:
1. `best.ckpt` and `last.ckpt` saved locally
2. Background thread uploads both to HuggingFace Hub
3. Training continues immediately (upload doesn't block)

```
Epoch 5 complete → EER=0.032
  ├─ Save best.ckpt locally
  ├─ Spawn thread → upload best.ckpt to HF
  ├─ Save last.ckpt locally
  ├─ Spawn thread → upload last.ckpt to HF
  └─ Start epoch 6
```

---

## Checkpoint Recovery

### Scenario 1: Session Crashed (same account)

**Step 1: Check progress**
- Run Cell 1 (helpers)
- Run Cell 2 (status) — shows exactly what completed before crash

**Step 2: Resume**
- The status cell tells you which cell to run next
- Training cells auto-resume from `last.ckpt`

The trainer auto-detects `last.ckpt` and resumes:
```
resuming from epoch 12 (best_eer=0.0320)
```

### Scenario 2: Network Died Mid-Training

Checkpoints were already uploaded to HF (after each epoch). Options:

**Option A:** Re-run the same cell (if session still alive)
**Option B:** If session died, follow Scenario 3

### Scenario 3: New Session / New Account

#### Step 1: Setup (run these cells first)
- Cell 1 (helpers)
- Cell 3 (HF setup)
- Cell 10 (HF login — reads from Kaggle Secrets automatically)

#### Step 2: Pull Checkpoints
Run **Cell 34** (Pull ALL) or **Cell 35** (Pull SINGLE model).

**Cell 34 output:**
```
Files on HuggingFace (10 total):
  checkpoints/auralguard/best.ckpt
  checkpoints/auralguard/last.ckpt
  checkpoints/b1_lcnn/best.ckpt
  ...

Resume status:
  b1_lcnn               -> resume from epoch 35  (best EER=0.0210)
  b2_rawnet2            -> resume from epoch 28  (best EER=0.0340)
  b3_aasist             -> resume from epoch 22  (best EER=0.0290)
  b5_wavlm_ocs          -> resume from epoch 15  (best EER=0.0180)
  auralguard            -> resume from epoch 12  (best EER=0.0150)
```

#### Step 3: Install AuralGuard
Run Cell 9.

#### Step 4: Re-run Training Cell
Run the training cell for the model you want to continue. It auto-resumes.

---

## Cross-Account Resume

When your 30hr/week Kaggle limit is hit on one account:

1. **All checkpoints are on HuggingFace** (uploaded live during training)
2. **Create a new Kaggle account**
3. **Add HF_TOKEN to Kaggle Secrets** (same token as before)
4. **Create a new notebook** with the same notebook file
5. **Run Cell 1, 3, 10** (setup + HF login — auto-reads from Secrets)
6. **Run Cell 34** (pull all checkpoints)
7. **Run Cell 9** (install AuralGuard)
8. **Re-run training cells** — they resume from where they left off

### HuggingFace Repos Used

| Repo | Purpose |
|------|---------|
| `MIHMahmudEli/auralguard-checkpoints` | Training checkpoints + results + figures |
| `MIHMahmudEli/auralguard` | Deployment model (for serving) |

---

## Datasets

### Auto-Downloaded by Notebook (or Kaggle Input if added)

| Dataset | Kaggle path | Purpose | Source |
|---------|-------------|---------|--------|
| ASVspoof 2019 LA | `datasets/awsaf49/asvpoof-2019-dataset` | Training | Kaggle (you add) |
| In-the-Wild | `datasets/abdallamohamed312/in-the-wild-audio-deepfake` | Zero-shot eval | Kaggle (you add) |
| WaveFake | `datasets/walimuhammadahmad/fakeaudio` | Zero-shot eval | Kaggle (you add) |
| ASVspoof2021 LA | `datasets/kkijjaa/asvspoof2021-la` | Zero-shot eval | Kaggle (you add) |
| ASVspoof2021 DF | `datasets/serjkalinovskiy/asvspoof2021-df` | Zero-shot eval | Kaggle (you add) |
| MUSAN | `datasets/riosgonzalo/musan-rirs` | Noise augmentation | Kaggle (you add) |
| RIRs | `datasets/nhattruongdev/rirs-noises` | Reverb augmentation | Kaggle (you add) |

> **Tip:** Adding all datasets as Kaggle inputs makes setup instant — no download time, no corrupted zip issues.

### Manifests Built

The notebook builds CSV manifests for all datasets:
```
data/manifests/
├── asvspoof2019_la_train.csv
├── asvspoof2019_la_dev.csv
├── asvspoof2019_la_eval.csv
├── asvspoof2021_la_eval.csv
├── asvspoof2021_df_eval.csv
├── in_the_wild.csv
├── wavefake.csv
├── musan.csv
└── rirs.csv
```

---

## HuggingFace Uploads

### What Gets Uploaded

| When | What | Where |
|------|------|-------|
| After every epoch | `best.ckpt`, `last.ckpt` | `MIHMahmudEli/auralguard-checkpoints` |
| After training | All checkpoints | Same repo |
| After eval | `results.json` for each model | Same repo |
| After figures | PNG figures | Same repo |
| After model export | Slim `best.ckpt` + model card | `MIHMahmudEli/auralguard` |

### HF Repo Structure

```
MIHMahmudEli/auralguard-checkpoints/
├── checkpoints/
│   ├── b1_lcnn/best.ckpt, last.ckpt
│   ├── b2_rawnet2/best.ckpt, last.ckpt
│   ├── b3_aasist/best.ckpt, last.ckpt
│   ├── b5_wavlm_ocs/best.ckpt, last.ckpt
│   └── auralguard/best.ckpt, last.ckpt
├── results/
│   └── <model>/eval_results/results.json
│   └── <model>/zeroshot_results/results.json
└── figures/
    ├── training_curves.png
    ├── eer_comparison.png
    ├── det_curve.png
    ├── score_distribution.png
    └── robustness_heatmap.png

MIHMahmudEli/auralguard/
├── best.ckpt          (slim, deployment-ready)
├── model.onnx         (ONNX export)
└── README.md          (model card)
```

---

## Model Deployment

After training, Cell 30 uploads the AuralGuard model to `MIHMahmudEli/auralguard`.

### Deploy as HF Space

1. Create a new HF Space (Docker SDK)
2. Set environment variables:
   ```
   MODEL_REPO=MIHMahmudEli/auralguard
   MODEL_FILE=best.ckpt
   ```
3. The Space auto-downloads and serves the model

### Use via API

```python
from huggingface_hub import hf_hub_download
from auralguard.inference.predict import Detector

ckpt = hf_hub_download("MIHMahmudEli/auralguard", "best.ckpt")
detector = Detector(ckpt)
result = detector.predict_file("audio.wav")
print(result)
# {'verdict': 'ai_generated', 'p_ai_generated': 0.92, 'confidence': 'high', ...}
```

### Use via REST API

```bash
curl -X POST https://MIHMahmudEli-auralguard.hf.space/api/detect \
  -F "file=@audio.wav"
```

---

## Troubleshooting

### "I don't know which cell to run after a crash"
- Run Cell 1 (helpers) then Cell 2 (status)
- The status table shows ✅ DONE / ⏳ pending for every cell
- Follow the "NEXT" pointer at the bottom

### "progress.json is corrupted or missing"
- Delete `/kaggle/working/auralguard/progress.json`
- Run Cell 1 + Cell 2 — it will show everything as pending
- Resume manually from the first incomplete cell

### "Dataset not found"
- Ensure you added `awsaf49/asvpoof-2019-dataset` via +Data
- Check that Internet is enabled in notebook settings

### "CUDA out of memory"
- The notebook uses `data.num_workers=0` to avoid memory issues
- If still OOM, reduce batch size: add `train.batch_size=12` to the command

### "HF upload failed"
- Check your HF token is valid
- The training continues even if upload fails (background thread)
- Checkpoints are saved locally regardless

### "Training cell says resume from epoch X"
- This is normal — it found a previous checkpoint
- Training continues from that epoch

### Session killed before epoch completes
- Run Cell 1 + Cell 2 to see progress
- Checkpoints are uploaded after each epoch
- If killed mid-epoch, that epoch's checkpoint is lost
- Resume from the last completed epoch (status cell shows this)

### 30h/week limit hit
- Switch to a new Kaggle account
- Run Cell 34 to pull all checkpoints from HF
- Continue training where you left off

---

## Quick Reference

| Action | How |
|--------|-----|
| Start training | Run cells 1-18 |
| See progress after crash | Run Cell 1 (helpers) then Cell 2 (status) |
| Resume after crash | Re-run the training cell |
| Pull checkpoints | Cell 34 (all) or Cell 35 (single) |
| Check what's on HF | Cell 34 output shows epoch status |
| Upload model for deployment | Cell 30 |
| Download results | Cell 32 (tar.gz) or check HF repo |
| Set HF token | Kaggle Secrets → name: `HF_TOKEN`, value: `hf_xxx` |
