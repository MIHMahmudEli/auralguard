# Architecture — AuralGuard (model + system)

## 1. Model architecture

Input: mono waveform, 16 kHz, cropped/padded to `T = 4 s` (64 000 samples) during training;
sliding windows at inference for longer clips.

```
 waveform x ∈ ℝ^{64000}
 │
 ├─────────────── View A (semantic / SSL) ───────────────────────────┐
 │  WavLM-Large (frozen or last-k layers fine-tuned)                  │
 │  hidden states H ∈ ℝ^{L×T'×D}   (L=25 layers, D=1024)             │
 │  learnable layer weights α ∈ Δ^{L}  →  Ĥ = Σ_l softmax(α)_l H_l    │
 │  Ĥ ∈ ℝ^{T'×1024}  →  Conv1d proj → F_A ∈ ℝ^{T'×C}                  │
 │                                                                    │
 ├─────────────── View B (low-level artifacts) ──────────────────────┤
 │  LFCC(60) ⊕ modified-group-delay ⊕ CQT-phase                       │
 │  → 2-D CNN (SincConv-free, light) → F_B ∈ ℝ^{T''×C}                │
 │                                                                    │
 │  Gated cross-attention fusion:  F = GatedXAttn(F_A, F_B) ∈ ℝ^{T*×C}│
 └────────────────────────────────────────────────────────────────────┘
 │
 AASIST back-end (spectro-temporal graph attention over F)
 │  → utterance embedding z ∈ ℝ^{d}   (d = 160)
 │
 ├─ OC-Softmax head → score s ∈ ℝ            (inference decision)
 └─ SupCon projection g(z) (train only, auxiliary contrastive loss)
```

### Losses (training)
```
L = L_oc-softmax(z, y)  +  λ_c · L_supcon(g(z), y)  +  λ_r · ||α||_entropy-reg
```
- `L_oc-softmax`: one-class softmax — compact target (bona-fide) region, margin m0/m1.
- `L_supcon`: supervised contrastive on the projection head, pulls same-class together.
- entropy reg on layer weights `α` keeps layer attention from collapsing to one layer.

### Why this beats B4/B5 (hypotheses, each = an ablation)
1. **Layer attention** > last-layer SSL: artifacts live in different depths per attack.
2. **View B** adds phase/group-delay cues SSL misses (neural-vocoder phase artifacts).
3. **OC-Softmax + SupCon** learns a tighter bona-fide manifold → unseen spoofs fall outside.
4. **Augmentation curriculum** matches deployment channels → robustness (G2).

### Parameter/compute budget
- WavLM-Large: 316M (frozen) — no grad; or last-3 layers fine-tuned (~40M trainable).
- Artifact CNN + fusion + AASIST + heads: ~2–5M trainable.
- Trains on a single 24 GB GPU with gradient checkpointing; inference RTF < 0.1 on GPU,
  ~0.4 on CPU (report exact numbers in paper).

## 2. Inference behavior
- Clips > 4 s: overlapping windows (50% hop) → mean/logsumexp pool of scores.
- Output: `p(spoof) ∈ [0,1]` (temperature-scaled → calibrated), decision threshold `τ`
  chosen on dev to hit a target FAR, plus a spectro-temporal attribution heatmap.

## 3. System architecture (research → product)

```
┌──────────────┐   train/eval    ┌───────────────┐   export     ┌────────────────────┐
│  Data +      │ ──────────────► │  AuralGuard    │ ───────────► │ artifacts:          │
│  manifests   │   (src/)        │  training      │  TorchScript │  best.ckpt / .pt    │
└──────────────┘                 └───────────────┘  / ONNX       │  + config + norm    │
                                                                 └─────────┬──────────┘
                                                                           │
                                            ┌──────────────────────────────▼───────────────┐
                                            │  Hugging Face Space (deployment/hf_space/)     │
                                            │  FastAPI  POST /api/detect  +  Gradio UI       │
                                            │  loads weights once; batching; rate limit      │
                                            └──────────────────────────────┬────────────────┘
                                                                           │ HTTPS JSON
                                            ┌──────────────────────────────▼────────────────┐
                                            │  Next.js UI (ui/)                              │
                                            │  record / upload → /api/detect proxy →         │
                                            │  verdict + confidence + spectrogram heatmap    │
                                            └───────────────────────────────────────────────┘
```

### API contract (`POST /api/detect`)
Request: `multipart/form-data` with `file` (wav/mp3/flac/ogg/m4a), optional `return_heatmap`.

Response:
```json
{
  "verdict": "ai_generated",          // or "human"
  "p_ai_generated": 0.984,            // calibrated probability
  "confidence": "high",              // low | medium | high (from p and margin)
  "score": 3.21,                      // raw model score
  "threshold": 0.5,
  "model_version": "auralguard-v1.0",
  "sample_rate": 16000,
  "duration_s": 6.4,
  "windows": 3,
  "heatmap_png_base64": "..."         // optional spectro-temporal attribution
}
```

Errors are JSON: `{"error": "message", "code": "UNSUPPORTED_FORMAT"}`.
