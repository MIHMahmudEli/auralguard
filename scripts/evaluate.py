#!/usr/bin/env python
"""Evaluate a checkpoint on the full protocol (in-domain + zero-shot + calibration).

    python scripts/evaluate.py --ckpt experiments/auralguard/checkpoints/best.ckpt
"""
from auralguard.evaluation.cli import main

if __name__ == "__main__":
    main()
