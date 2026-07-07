"""`auralguard-train` console entrypoint (thin wrapper over scripts/train.py logic)."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from ..data import AudioAntiSpoofDataset, AudioConfig, AugmentPipeline
from ..models import build_model
from ..utils import get_logger, seed_everything
from .trainer import Trainer

logger = get_logger(__name__)


def run(cfg: DictConfig):
    seed_everything(cfg.seed)
    data = cfg.data
    audio_cfg = AudioConfig(
        sample_rate=data.sample_rate, crop_seconds=data.crop_seconds,
        random_crop=data.random_crop, pad_mode=data.pad_mode,
    )
    aug = AugmentPipeline(data.augment) if data.augment.get("enabled", False) else None
    train_ds = AudioAntiSpoofDataset(data.manifests.train, audio_cfg, augment=aug, is_train=True)
    dev_ds = AudioAntiSpoofDataset(data.manifests.dev, audio_cfg, augment=None, is_train=False)

    model = build_model(cfg.model)
    trainer = Trainer(model, train_ds, dev_ds, cfg, device=cfg.device)
    best = trainer.train()
    logger.info("training done. best dev EER = %.4f", best)
    return best


@hydra.main(version_base=None, config_path="../../../config", config_name="config")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
