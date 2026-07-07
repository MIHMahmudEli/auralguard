# Extending the pipeline — plug in your own model, dataset, or feature

The whole pipeline (training, evaluation, robustness sweeps, inference CLI, the HF API,
export) is **model-agnostic**: everything goes through `auralguard.models.build_model(cfg)`,
which dispatches on the config's `name`/`arch`. You never touch pipeline code to try a
new architecture.

## 1. Add a new model (≈10 minutes)

### The contract
One method. `forward(wav, labels=None)` with `wav: (B, T) float32 @ 16 kHz` returns a dict:

| key | when | meaning |
|-----|------|---------|
| `"score"` | always | `(B,)` tensor, **higher = more AI/spoof-like** |
| `"loss"` | `labels` given | scalar tensor to backprop |
| `"embedding"` | optional | utterance embedding (used for analysis plots) |
| `"loss_*"` | optional | extra terms, logged automatically |

### Steps
```python
# src/auralguard/models/my_model.py
import torch.nn as nn
from .registry import register

@register("my_model")
class MyModel(nn.Module):
    def __init__(self, cfg):          # cfg = the composed `model` config node
        super().__init__()
        ...
    def forward(self, wav, labels=None):
        score = ...
        out = {"score": score}
        if labels is not None:
            out["loss"] = ...
        return out
```

1. Import it in `src/auralguard/models/__init__.py` (one line: `from . import my_model`).
2. Create `config/model/my_model.yaml` with `name: my_model` + hyperparameters.
3. Create `config/experiment/my_model.yaml` (copy `b3_aasist.yaml`, change the names).
4. Run everything unchanged:
   ```bash
   python scripts/train.py experiment=my_model
   python scripts/evaluate.py --ckpt experiments/my_model/checkpoints/best.ckpt
   python -m auralguard.inference.predict --audio x.wav --ckpt .../best.ckpt
   ```
   The checkpoint stores the config, so eval/inference/export/HF-Space rebuild the right
   architecture automatically — the Space serves ANY registered model without edits.

### Reusing an architecture with different settings
Set `arch:` to the registered class and keep `name:` unique
(see `config/model/b5_wavlm_ocs.yaml`, which reuses `arch: auralguard`).

### Built-in zoo
`python -c "from auralguard.models import available; print(available())"`
→ `['aasist_raw', 'auralguard', 'lfcc_lcnn', 'rawnet2']` (+ whatever you add).

## 2. Add a new dataset (≈15 minutes)
1. Write an adapter in `scripts/build_manifests.py` mapping the corpus's protocol files to
   the manifest schema (`docs/DATASETS.md`). That's the only dataset-specific code.
2. Add the manifest path to `config/data/*.yaml` — under `cross_eval:` for zero-shot
   evaluation, or as `manifests:` for training.

## 3. Add a new hand-crafted feature (View B)
Add a method to `ArtifactFeatureExtractor` (`src/auralguard/features/spectral.py`)
returning `(B, f, t)` and list its name in `config/model/auralguard.yaml → frontend_artifact.features`.

## 4. Swap the SSL backbone
Config-only: `model.frontend_ssl.backbone=facebook/hubert-large-ll60k` (any HF encoder
with `output_hidden_states`). This is ablation E3.8/E3.9.

## 5. Deploy a different model
Upload its `best.ckpt` to your HF model repo and set `MODEL_REPO`/`MODEL_FILE` on the
Space — the API rebuilds the architecture from the checkpoint's embedded config.
