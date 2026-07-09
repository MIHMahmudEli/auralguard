---
title: AuralGuard
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# AuralGuard — AI-Generated Speech Detection

Multi-view one-class detection of AI-generated speech. Combines WavLM front-end,
AASIST spectro-temporal graph attention, and OC-Softmax one-class learning with
Gated Cross-Attention Fusion.

**Try the demo** — upload an audio file and get a verdict.

## Model Details

| Property | Value |
|----------|-------|
| **Architecture** | WavLM-Large + AASIST + Gated Cross-Attention Fusion + OC-Softmax |
| **Training Data** | ASVspoof 2019 Logical Access (LA) — 25,380 utterances |
| **Input** | 16 kHz mono audio, 4-second windows with 2-second hop |
| **Output** | Spoof score (higher = more likely AI-generated) + calibrated probability |

## Performance

| Dataset | EER (%) | AUROC |
|---------|---------|-------|
| ASVspoof 2019 LA (In-Domain) | TBD | TBD |
| ASVspoof 2021 LA (Zero-Shot) | TBD | TBD |
| ASVspoof 2021 DF (Zero-Shot) | TBD | TBD |
| In-the-Wild (Zero-Shot) | TBD | TBD |
| WaveFake (Zero-Shot) | TBD | TBD |
| MLAAD (Zero-Shot) | TBD | TBD |

*Results will be updated after training completes.*

## Usage

```python
from auralguard.inference.predict import Detector

det = Detector("path/to/best.ckpt")
result = det.predict_file("speech.wav")
print(result["verdict"])         # "human" or "ai_generated"
print(result["p_ai_generated"])  # calibrated probability
```

## Citation

```bibtex
@article{auralguard,
  title={AuralGuard: Multi-View One-Class Detection of AI-Generated Speech},
  author={Your Name and Others},
  journal={},
  year={2026}
}
```

## Links

- **GitHub**: https://github.com/MIHMahmudEli/auralguard
- **Paper**: (coming soon)
