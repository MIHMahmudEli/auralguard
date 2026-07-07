"""Training loop for AuralGuard and baselines.

Deliberately framework-light (plain PyTorch + AMP) so it is easy to audit for a paper
and easy to run on a single GPU. Handles: param-group LRs (small LR for fine-tuned SSL),
cosine schedule with warmup, grad clipping, AMP, checkpointing on best dev EER, and
early stopping.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..evaluation.metrics import compute_eer
from ..utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    def __init__(self, model, train_ds, dev_ds, cfg, device="cuda"):
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg
        tcfg = cfg["train"]
        self.epochs = tcfg["epochs"]
        self.amp = tcfg.get("amp", True)
        self.grad_clip = tcfg.get("grad_clip", 5.0)
        self.out_dir = Path(cfg["output_dir"])
        (self.out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        self.train_loader = DataLoader(
            train_ds, batch_size=tcfg["batch_size"], shuffle=True,
            num_workers=cfg["data"].get("num_workers", 4),
            pin_memory=cfg["data"].get("pin_memory", True),
            collate_fn=_collate, drop_last=True,
        )
        self.dev_loader = DataLoader(
            dev_ds, batch_size=tcfg["batch_size"], shuffle=False,
            num_workers=cfg["data"].get("num_workers", 4),
            collate_fn=_collate,
        )
        self.optimizer = self._build_optimizer(tcfg)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.best_eer = float("inf")
        self.patience = tcfg.get("early_stop", {}).get("patience", 8)
        self._since_improve = 0
        self._warmup = tcfg.get("scheduler", {}).get("warmup_epochs", 2)
        self._min_lr = tcfg.get("scheduler", {}).get("min_lr", 1e-7)
        self._base_lrs = [g["lr"] for g in self.optimizer.param_groups]

    def _build_optimizer(self, tcfg):
        opt = tcfg["optimizer"]
        ssl_params, other_params = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (ssl_params if n.startswith("ssl.model") else other_params).append(p)
        groups = [{"params": other_params, "lr": opt["lr"]}]
        if ssl_params:
            groups.append({"params": ssl_params, "lr": opt.get("ssl_lr", opt["lr"] * 0.01)})
        return torch.optim.AdamW(groups, weight_decay=opt.get("weight_decay", 1e-4))

    def _set_lr(self, epoch):
        for i, g in enumerate(self.optimizer.param_groups):
            base = self._base_lrs[i]
            if epoch < self._warmup:
                lr = base * (epoch + 1) / self._warmup
            else:
                t = (epoch - self._warmup) / max(1, self.epochs - self._warmup)
                lr = self._min_lr + 0.5 * (base - self._min_lr) * (1 + math.cos(math.pi * t))
            g["lr"] = lr

    def train(self):
        for epoch in range(self.epochs):
            self._set_lr(epoch)
            self._train_epoch(epoch)
            eer = self._validate(epoch)
            improved = eer < self.best_eer
            if improved:
                self.best_eer = eer
                self._since_improve = 0
                self._save("best.ckpt", epoch, eer)
            else:
                self._since_improve += 1
            logger.info("epoch %d dev_eer=%.4f best=%.4f", epoch, eer, self.best_eer)
            if self._since_improve >= self.patience:
                logger.info("early stopping at epoch %d", epoch)
                break
        return self.best_eer

    def _train_epoch(self, epoch):
        self.model.train()
        for step, (wav, labels, _) in enumerate(self.train_loader):
            wav, labels = wav.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.amp):
                out = self.model(wav, labels)
                loss = out["loss"]
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if step % self.cfg["train"].get("log_every_n_steps", 50) == 0:
                logger.info("e%d s%d loss=%.4f", epoch, step, loss.item())

    @torch.no_grad()
    def _validate(self, epoch):
        self.model.eval()
        scores, labels = [], []
        for wav, y, _ in self.dev_loader:
            wav = wav.to(self.device)
            out = self.model(wav)
            scores.append(out["score"].cpu().numpy())
            labels.append(y.numpy())
        scores = np.concatenate(scores)
        labels = np.concatenate(labels)
        eer, _ = compute_eer(scores, labels)
        return eer

    def _save(self, name, epoch, eer):
        path = self.out_dir / "checkpoints" / name
        torch.save(
            {"model": self.model.state_dict(), "cfg": _to_container(self.cfg),
             "epoch": epoch, "dev_eer": eer},
            path,
        )
        logger.info("saved %s (dev_eer=%.4f)", path, eer)


def _collate(batch):
    from ..data.datasets import collate

    return collate(batch)


def _to_container(cfg):
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        return dict(cfg)
