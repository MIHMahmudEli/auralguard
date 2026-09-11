"""Training loop for AuralGuard and baselines.

Deliberately framework-light (plain PyTorch + AMP) so it is easy to audit for a paper
and easy to run on a single GPU. Handles: param-group LRs (small LR for fine-tuned SSL),
cosine schedule with warmup, grad clipping, AMP, checkpointing on best dev EER, and
early stopping.
"""

from __future__ import annotations

import math
import threading
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..evaluation.metrics import compute_eer
from ..utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    def __init__(self, model, train_ds, dev_ds, cfg, device="cuda", on_epoch_end=None):
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg
        self.on_epoch_end = on_epoch_end  # callback: fn(epoch, eer, ckpt_path, is_best)
        tcfg = cfg["train"]
        self.epochs = tcfg["epochs"]
        self.amp = tcfg.get("amp", True)
        self.grad_clip = tcfg.get("grad_clip", 5.0)
        self.out_dir = Path(cfg["output_dir"])
        (self.out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        # Sanity-check thresholds (configurable via config/train/default.yaml)
        sc_cfg = tcfg.get("sanity_check", {})
        self._sc_eer_threshold = sc_cfg.get("dev_eer_threshold", 0.95)
        self._sc_enabled = sc_cfg.get("enabled", True)
        self._max_consecutive_suspect = sc_cfg.get("max_consecutive_suspect", 3)
        self._consecutive_suspect = 0

        # HuggingFace upload config (optional)
        hf_cfg = tcfg.get("hf_upload", {})
        self._hf_repo = hf_cfg.get("repo", None)
        self._hf_enabled = hf_cfg.get("enabled", False) and self._hf_repo is not None
        self._hf_token = hf_cfg.get("token", None)
        self._hf_upload_every = hf_cfg.get("upload_every_n_epochs", 1)
        if self._hf_enabled:
            logger.info("HF upload enabled: repo=%s, every=%d epochs", self._hf_repo, self._hf_upload_every)

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

    def train(self, start_epoch=0):
        for epoch in range(start_epoch, self.epochs):
            self._set_lr(epoch)
            self._train_epoch(epoch)
            eer = self._validate(epoch)
            improved = eer < self.best_eer
            if improved:
                self.best_eer = eer
                self._since_improve = 0
                self._save("best.ckpt", epoch, eer)
                self._callback(epoch, eer, "best.ckpt", is_best=True)
                self._hf_upload(epoch, eer, "best.ckpt")
            else:
                self._since_improve += 1

            # Sanity check: do NOT let a degenerate epoch corrupt the resume chain.
            suspect = self._is_suspect_eer(eer)
            if suspect:
                self._consecutive_suspect += 1
                logger.warning(
                    "EPOCH %d dev_eer=%.4f is suspect (threshold=%.4f) — "
                    "quarantining to last_suspect.ckpt, preserving previous last.ckpt "
                    "(consecutive suspect: %d/%d)",
                    epoch, eer, self._sc_eer_threshold,
                    self._consecutive_suspect, self._max_consecutive_suspect,
                )
                self._save("last_suspect.ckpt", epoch, eer)
                self._callback(epoch, eer, "last_suspect.ckpt", is_best=False)
                self._hf_upload(epoch, eer, "last_suspect.ckpt")

                # Self-heal: rollback model/optimizer/scaler to last known-good state
                rolled_back_epoch = self._rollback_to_good_checkpoint()
                if rolled_back_epoch >= 0:
                    logger.info(
                        "epoch %d rolled back to epoch %d state, will retry next epoch",
                        epoch, rolled_back_epoch,
                    )

                # Hard-stop if too many consecutive suspect epochs
                if self._consecutive_suspect >= self._max_consecutive_suspect:
                    logger.error(
                        "STOPPING: %d consecutive suspect epochs (max=%d) — "
                        "training cannot recover.  Last good checkpoint preserved.",
                        self._consecutive_suspect, self._max_consecutive_suspect,
                    )
                    break
            else:
                # Good epoch: reset consecutive suspect counter
                self._consecutive_suspect = 0
                self._save("last.ckpt", epoch, eer)
                self._callback(epoch, eer, "last.ckpt", is_best=False)
                self._hf_upload(epoch, eer, "last.ckpt")

            logger.info("epoch %d dev_eer=%.4f best=%.4f", epoch, eer, self.best_eer)
            if self._since_improve >= self.patience:
                logger.info("early stopping at epoch %d", epoch)
                break
        return self.best_eer

    def _is_suspect_eer(self, eer: float) -> bool:
        """Return True if dev_eer is NaN or outside a plausible range."""
        if not self._sc_enabled:
            return False
        if math.isnan(eer):
            return True
        if eer >= self._sc_eer_threshold:
            return True
        return False

    def _rollback_to_good_checkpoint(self):
        """Reload model, optimizer, and scaler from the last known-good checkpoint.

        Prefers last.ckpt (if it exists and is not itself suspect), falls back to
        best.ckpt.  Returns the epoch loaded from, or -1 if rollback failed.
        """
        ckpt_dir = self.out_dir / "checkpoints"
        last_path = ckpt_dir / "last.ckpt"
        best_path = ckpt_dir / "best.ckpt"

        # Prefer last.ckpt — it tracks the most recent non-suspect epoch
        source = None
        if last_path.exists():
            source = last_path
        elif best_path.exists():
            source = best_path

        if source is None:
            logger.error("rollback failed: no last.ckpt or best.ckpt found")
            return -1

        ckpt = torch.load(str(source), map_location=self.device, weights_only=False)
        epoch = ckpt.get("epoch", -1)

        # Restore model weights
        if "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
        # Restore optimizer state
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        # Restore scaler state
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        # Restore tracking fields
        if "best_eer" in ckpt:
            self.best_eer = ckpt["best_eer"]
        if "since_improve" in ckpt:
            self._since_improve = ckpt["since_improve"]

        logger.warning(
            "rollback: reloaded model/optimizer/scaler from %s (epoch=%d, dev_eer=%.4f)",
            source.name, epoch, ckpt.get("dev_eer", float("nan")),
        )
        return epoch

    def _callback(self, epoch, eer, ckpt_name, is_best):
        if self.on_epoch_end is None:
            return
        ckpt_path = self.out_dir / "checkpoints" / ckpt_name
        try:
            self.on_epoch_end(epoch, eer, str(ckpt_path), is_best)
        except Exception as e:
            logger.warning("on_epoch_end callback failed: %s", e)

    def _hf_upload(self, epoch, eer, ckpt_name):
        """Upload checkpoint to HuggingFace Hub in background thread."""
        if not self._hf_enabled:
            return
        if epoch % self._hf_upload_every != 0 and ckpt_name != "best.ckpt":
            return
        ckpt_path = self.out_dir / "checkpoints" / ckpt_name
        if not ckpt_path.exists():
            return

        def _upload():
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=self._hf_token)
                exp_name = self.out_dir.name
                repo_path = f"checkpoints/{exp_name}/{ckpt_name}"
                api.upload_file(
                    path_or_fileobj=str(ckpt_path),
                    path_in_repo=repo_path,
                    repo_id=self._hf_repo,
                    repo_type="model",
                )
                logger.info("HF uploaded: %s (epoch=%d, EER=%.4f)", repo_path, epoch, eer)
            except Exception as e:
                logger.warning("HF upload failed for %s: %s", ckpt_name, e)

        t = threading.Thread(target=_upload, daemon=True)
        t.start()

    def _train_epoch(self, epoch):
        self.model.train()
        nan_grad_steps = 0
        for step, (wav, labels, _) in enumerate(self.train_loader):
            wav, labels = wav.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.amp):
                out = self.model(wav, labels)
                loss = out["loss"]
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            # Check for NaN/Inf gradients before clipping and optimizer step.
            # NaN grads that slip through corrupt AdamW exp_avg/exp_avg_sq
            # permanently, surviving checkpoint save/load across restarts.
            has_nan_grad = False
            for p in self.model.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_nan_grad = True
                    break
            if has_nan_grad:
                nan_grad_steps += 1
                logger.warning(
                    "e%d s%d NaN/Inf gradient detected — skipping optimizer step "
                    "(total skipped this epoch: %d)", epoch, step, nan_grad_steps,
                )
                self.optimizer.zero_grad(set_to_none=True)
                continue

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if step % self.cfg["train"].get("log_every_n_steps", 50) == 0:
                logger.info("e%d s%d loss=%.4f", epoch, step, loss.item())
        if nan_grad_steps:
            logger.warning("e%d finished with %d NaN-gradient steps skipped", epoch, nan_grad_steps)

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
        n_nan = int(np.isnan(scores).sum())
        n_inf = int(np.isinf(scores).sum())
        if n_nan or n_inf:
            logger.warning(
                "e%d evaluation scores contain %d NaN + %d Inf out of %d — "
                "degenerate EER likely",
                epoch, n_nan, n_inf, scores.size,
            )
        eer, _ = compute_eer(scores, labels)
        return eer

    def _save(self, name, epoch, eer):
        path = self.out_dir / "checkpoints" / name
        torch.save(
            {
                "model": self.model.state_dict(),
                "cfg": _to_container(self.cfg),
                "epoch": epoch,
                "dev_eer": eer,
                "best_eer": self.best_eer,
                "since_improve": self._since_improve,
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
            },
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
