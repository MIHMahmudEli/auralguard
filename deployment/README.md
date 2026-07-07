# Deployment

Two artifacts turn the research model into a product:

1. **`hf_space/`** — a Hugging Face Space (Docker) exposing the model as a Gradio demo
   and a JSON API (`POST /api/detect`). This is the inference backend.
2. **`../ui/`** — the Next.js end-user app that calls the API.

## Data flow
```
browser (record/upload) → Next.js /api/detect (proxy) → HF Space /api/detect → JSON verdict
```
The Next.js route proxies the request so the HF URL and any token stay server-side and CORS
is simple. Set `AURALGUARD_API_URL` in the UI's environment to your Space URL.

## Export for serving
Before deploying, export a lean inference checkpoint (optionally TorchScript/ONNX):
```bash
python scripts/export_model.py --ckpt experiments/auralguard/checkpoints/best.ckpt \
    --out deployment/artifacts/ --formats torchscript onnx
```
Upload the checkpoint to an HF model repo and point the Space at it via `MODEL_REPO`.

## Notes
- CPU inference is fine for a demo (RTF ≈ 0.4). For scale, use a GPU Space or batch requests.
- Keep dataset audio OUT of the Space — only the model is served.
- The UI and Space both state that detection is probabilistic (see the ethics section of
  `docs/RESEARCH_PLAN.md`).
