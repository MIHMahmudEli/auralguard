# Reproducibility

## Environment
Use **Python 3.10 or 3.11** for the ML environment. (Your system Python is 3.14, which is
fine for the Next.js/tooling side, but several audio/ML wheels — `torchaudio`, `fairseq`-style
SSL loaders, `onnxruntime` — lag on 3.14. Use the conda env below for training.)

```bash
conda env create -f environment.yml      # creates `auralguard` on py3.11
conda activate auralguard
pip install -e .                          # installs the `auralguard` package
pre-commit install                        # black + ruff + isort
pytest -q                                 # metrics + shape tests must pass
```

## Determinism
- Global seed via `auralguard.utils.seed.seed_everything(seed)`.
- `torch.use_deterministic_algorithms(True)` where kernels allow; document exceptions.
- cuDNN benchmark off for final runs.
- SSL backbones pinned by HF revision hash in `config/model/*.yaml`.

## What we release on submission
- Trained weights (HF Hub), TorchScript + ONNX exports.
- All configs + seeds + manifests (not audio).
- `experiments/` result JSONs + the aggregation scripts that build every table/figure.
- `CITATION.cff`, model card, data statement.

## One-command reproduction (target)
```bash
make repro            # baselines + proposed + all evals (long; uses cached features)
make table3           # regenerates a specific paper table from experiments/
```
