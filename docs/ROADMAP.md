# 24-week roadmap to Q1 submission

| Phase | Weeks | Milestones | Exit criteria |
|-------|-------|-----------|---------------|
| **0. Setup** | 1–2 | Env, data access requests, manifests, CI/tests | ASVspoof19 loads; metrics unit-tested |
| **1. Baselines** | 3–5 | Reproduce E1.3–E1.5 | In-domain EER within ±0.3% of literature |
| **2. Proposed v1** | 6–9 | AuralGuard trains; multi-view fusion works | Matches B5 in-domain |
| **3. Generalization** | 10–13 | E4 zero-shot on all corpora | Ours > B4/B5 avg cross-dataset EER (target: −20–40% rel.) |
| **4. Robustness** | 14–16 | E5 augmentation curriculum + sweeps | Recovers robustness with <0.5% clean cost |
| **5. Ablations** | 17–18 | E3.1–E3.10 | Each RQ answered with a table |
| **6. Calibration/XAI** | 19–20 | E6 + temperature scaling + attributions | ECE reported; qualitative figures |
| **7. Deploy** | 21 | HF Space API + Next.js UI live | End-to-end demo works |
| **8. Write-up** | 22–24 | Manuscript, figures, supplementary, internal review | Submission-ready PDF + released weights |

**Critical path:** dataset access (start week 1) → generalization results (phase 3) are the
paper's headline; protect that time. Deployment (phase 7) can proceed in parallel from
week 10 using an early checkpoint.

**Go/no-go at week 13:** if cross-dataset EER improvement over B5 is not clearly positive,
pivot the narrative to *calibration + robustness + multilingual* (still a full Q1 story)
rather than raw EER supremacy.
