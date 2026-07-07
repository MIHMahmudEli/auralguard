---
title: AuralGuard
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# AuralGuard — AI-Generated Speech Detector (HF Space)

This Space serves the trained AuralGuard model as **both** a Gradio demo (at `/`) and a
JSON API (`POST /api/detect`) consumed by the Next.js UI.

## Endpoints
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | liveness + whether real weights are loaded |
| POST | `/api/detect` | `multipart/form-data` with `file`; returns the verdict JSON |
| GET | `/` | Gradio UI |

## Configuration (Space → Settings → Variables / Secrets)
| var | meaning |
|-----|---------|
| `MODEL_REPO` | HF model repo holding the checkpoint, e.g. `youruser/auralguard` |
| `MODEL_FILE` | checkpoint filename (default `best.ckpt`) |
| `CKPT_PATH` | alternatively, a local path baked into the image |
| `THRESHOLD` | decision threshold (default `0.5`) |
| `CORS_ORIGINS` | comma-separated allowed origins for the Next.js app |
| `MAX_UPLOAD_MB` | reject larger uploads (default 25) |

Without a checkpoint the Space boots in **DEMO mode** (clearly labelled random scores) so
you can wire the UI before training finishes.

## Deploy
```bash
# 1. create the Space (Docker SDK) on huggingface.co, then:
git remote add space https://huggingface.co/spaces/MIHMahmudEli/auralguard
git push space main
# 2. upload trained weights to a model repo and set MODEL_REPO in Space settings
```

## Local run
```bash
pip install -r requirements.txt
CKPT_PATH=../../experiments/auralguard/checkpoints/best.ckpt uvicorn app:app --port 7860
```
