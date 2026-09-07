# notebooks/

Analysis and training notebooks.

## Kaggle Training Pipeline

**`kaggle_training.ipynb`** — Full training pipeline for Kaggle with:

- **Progress tracking** — status cell shows what's done after any crash
- Checkpoint recovery (resume from crashes)
- Live HuggingFace upload (every epoch)
- Cross-account resume (pull from HF)
- All datasets (zero-shot eval + augmentation)
- Model deployment upload

**See [KAGGLE.md](../KAGGLE.md) for complete guide.**

### Quick Start on Kaggle

1. Add ASVspoof 2019 LA dataset (`+Data` → search `awsaf49/asvpoof-2019-dataset`)
2. Enable Internet + GPU in notebook settings
3. Run cells 1-18 for full training

### Recovery

| Scenario | What to do |
|----------|------------|
| Session crashed | Run Cell 1 + Cell 2 (shows progress), then follow "NEXT" pointer |
| Network died | Re-run the same cell (checkpoints on HF) |
| New session/account | Run Cell 34 (pull from HF), then re-run training |
| 30h limit hit | Switch account, run Cell 34, continue training |

## Other Notebooks

- `01_eda.ipynb` — corpus stats, class balance, duration/lang distributions
- `02_baseline.ipynb` — reproduce B3/B4/B5, sanity-check in-domain EER
- `03_results.ipynb` — load `experiments/*/results.json`, build the paper figures
- `04_explainability.ipynb` — spectro-temporal attributions / case studies

Run `pip install -e .` first so `import auralguard` works from any notebook.
