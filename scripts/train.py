#!/usr/bin/env python
"""Train AuralGuard or a baseline.

Examples
--------
    python scripts/train.py experiment=auralguard
    python scripts/train.py experiment=b5_wavlm_ocs train.epochs=30
"""
import hydra
from omegaconf import DictConfig

from auralguard.training.cli import run


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
