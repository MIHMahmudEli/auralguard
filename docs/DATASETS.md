# Datasets — corpora, licenses, splits, download recipes

> **Do not commit audio to git.** Only manifests (CSV/JSON lists) live in the repo, under
> `data/manifests/`. Raw audio goes in `data/raw/` (gitignored).

## Manifest schema (`data/manifests/*.csv`)

| column | type | notes |
|--------|------|-------|
| `utt_id` | str | unique id |
| `path` | str | absolute or `data/raw`-relative path |
| `label` | int | `0`=bona-fide (human), `1`=spoof (AI) |
| `attack` | str | generator/attack id (`bonafide` for real) |
| `dataset` | str | source corpus |
| `lang` | str | ISO code if known, else `und` |
| `split` | str | `train`/`dev`/`eval` |
| `codec` | str | `none`/`mp3`/`opus`/... (for robustness sets) |

`src/data/datasets.py` reads exactly this schema, so every corpus is normalized to it by
a small adapter in `scripts/build_manifests.py`.

## Corpora

| Dataset | Access | License (verify!) | Use |
|---------|--------|-------------------|-----|
| **ASVspoof 2019 LA** | ASVspoof/Edinburgh DataShare | ODC-BY-ish, research | **train + in-domain eval** |
| **ASVspoof 2021 LA/DF** | ASVspoof site | research | codec/compression eval |
| **ASVspoof 5 (2024)** | ASVspoof 5 site | research | modern generators eval |
| **In-the-Wild** | Fraunhofer AISEC (Zenodo) | research | real-world eval |
| **MLAAD** | multi-lingual TTS, Zenodo | check per-source | cross-lingual eval |
| **WaveFake** | Zenodo | research | unseen vocoders eval |
| **CodecFake** | project release | research | neural-codec spoof eval |
| **MUSAN** | OpenSLR | permissive | noise augmentation |
| **RIR (SLR28)** | OpenSLR | permissive | reverberation augmentation |

> **Licensing caution:** several corpora forbid redistribution and some restrict commercial
> use. The public HF demo runs the *model*, not the datasets — that's fine. Do **not** upload
> corpus audio to the Space. Confirm each license before any commercial deployment.

## Download recipes

Fill secrets/paths in `config/data/*.yaml`, then:

```bash
# ASVspoof 2019 LA (registration required — accept terms first)
bash scripts/download/asvspoof2019_la.sh   # → data/raw/ASVspoof2019_LA/

# In-the-Wild (Zenodo, direct)
bash scripts/download/in_the_wild.sh

# WaveFake (Zenodo)
bash scripts/download/wavefake.sh

# Augmentation corpora
bash scripts/download/musan.sh
bash scripts/download/rirs.sh

# Normalize everything to the manifest schema
python scripts/build_manifests.py --all
```

Each `download/*.sh` is a stub with the URL and expected checksum; you accept the dataset
terms manually (they require it) and drop the archive in `data/raw/`.

## Splits & the golden protocol

- **Train / model-selection:** ASVspoof 2019 LA `train` / `dev` **only** (+ augmentation).
- **In-domain test:** ASVspoof 2019 LA `eval`.
- **Zero-shot generalization tests:** everything else, untouched during training/selection.

This is the protocol reviewers expect for a *generalization* claim. Any deviation must be
justified in the paper.
