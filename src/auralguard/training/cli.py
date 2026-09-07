"""`auralguard-train` console entrypoint (thin wrapper over scripts/train.py logic)."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

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

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg["grad_checkpointing"] = cfg.train.get("grad_checkpointing", False)
    model = build_model(model_cfg)

    # Inject HF token from env if not in config
    import os
    hf_cfg = cfg.train.get("hf_upload", {})
    if hf_cfg.get("enabled", False) and hf_cfg.get("token") is None:
        token = None
        # Try Kaggle Secrets
        try:
            from kaggle_secrets import UserSecretsClient
            secrets = UserSecretsClient()
            token = secrets.get_secret("HF_TOKEN")
            logger.info("Loaded HF token from Kaggle Secrets")
        except Exception:
            pass
        # Try env vars
        if not token:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HF")
        # Try .env file
        if not token:
            from pathlib import Path
            env_file = Path("auralguard/.env") if Path("auralguard/.env").exists() else Path(".env")
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    if line.startswith("HF="):
                        token = line.split("=", 1)[1].strip()
                        break
        OmegaConf.update(cfg, "train.hf_upload.token", token, force_add=True)

    trainer = Trainer(model, train_ds, dev_ds, cfg, device=cfg.device)

    # Resume from checkpoint if it exists
    import torch, sys, importlib.abc
    from pathlib import Path
    class _FakeSerializationMod(__import__('types').ModuleType):
        def __getattr__(self, name):
            return type(name, (), {})
    class _FakeFinder(importlib.abc.MetaPathFinder):
        def find_module(self, fullname, path=None):
            return self if fullname == 'torch.utils.serialization' else None
        def load_module(self, fullname):
            if fullname not in sys.modules:
                sys.modules[fullname] = _FakeSerializationMod(fullname)
            return sys.modules[fullname]
    if 'torch.utils.serialization' not in sys.modules:
        sys.meta_path.insert(0, _FakeFinder())
    out_dir = Path(cfg["output_dir"])
    last_ckpt = out_dir / "checkpoints" / "last.ckpt"
    best_ckpt = out_dir / "checkpoints" / "best.ckpt"
    start_epoch = 0
    resume_ckpt = last_ckpt if last_ckpt.exists() else best_ckpt if best_ckpt.exists() else None
    if resume_ckpt is not None:
        ckpt = torch.load(str(resume_ckpt), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", -1) + 1
        trainer.best_eer = ckpt.get("dev_eer", float("inf"))
        logger.info("resuming from epoch %d via %s (best_eer=%.4f)", start_epoch, resume_ckpt.name, trainer.best_eer)

    best = trainer.train(start_epoch=start_epoch)
    logger.info("training done. best dev EER = %.4f", best)
    return best


@hydra.main(version_base=None, config_path="../../../config", config_name="config")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
