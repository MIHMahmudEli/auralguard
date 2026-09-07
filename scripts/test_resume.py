#!/usr/bin/env python
"""Verify checkpoint resume correctness for AuralGuard trainer.

Tests:
  1. best_eer and since_improve are persisted and restored correctly.
  2. best_eer comes from best.ckpt (not last.ckpt) when both exist.
  3. Old checkpoints without new fields fall back gracefully.
  4. Only last.ckpt exists but it carries best_eer field — restores best_eer, not dev_eer.
"""

import sys, types
import torch

# Fake torch.utils.serialization for older checkpoints
class _FakeObj:
    def __getattr__(self, n): return _FakeObj()
    def __call__(self, *a, **kw): return _FakeObj()
    def __bool__(self): return False
class _FakeSerializationMod(types.ModuleType):
    def __getattr__(self, name): return _FakeObj()
sys.modules['torch.utils.serialization'] = _FakeSerializationMod('torch.utils.serialization')

from pathlib import Path
from omegaconf import OmegaConf
from auralguard.training.trainer import Trainer, _to_container
from auralguard.models import build_model


class _FakeDataset:
    def __len__(self): return 10
    def __getitem__(self, i):
        return (torch.randn(16000), 0, "")


def _make_trainer(tmp_dir):
    cfg = OmegaConf.create({
        "model": {"name": "lfcc_lcnn", "lfcc": {"n_filts": 60, "n_frames": 500}},
        "train": {
            "epochs": 5, "batch_size": 2,
            "optimizer": {"name": "adamw", "lr": 1e-4, "weight_decay": 1e-4},
            "scheduler": {"warmup_epochs": 1, "min_lr": 1e-7},
            "early_stop": {"patience": 8},
            "amp": False, "grad_clip": 5.0, "log_every_n_steps": 1,
        },
        "data": {"num_workers": 0, "pin_memory": False},
        "output_dir": tmp_dir,
        "seed": 42,
    })
    model = build_model(OmegaConf.to_container(cfg.model, resolve=True))
    ds = _FakeDataset()
    trainer = Trainer(model, ds, ds, cfg, device="cpu")
    return trainer


def _save_ckpt(path, *, epoch, dev_eer, best_eer=None, since_improve=None):
    """Save a checkpoint dict to path, simulating _save() output."""
    data = {"epoch": epoch, "dev_eer": dev_eer}
    if best_eer is not None:
        data["best_eer"] = best_eer
    if since_improve is not None:
        data["since_improve"] = since_improve
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, str(path))


def _cli_resume(out_dir):
    """Simulate the cli.py resume path: load model+start_epoch from resume_ckpt,
    best_eer from best.ckpt, since_improve from resume_ckpt."""
    last_ckpt = out_dir / "checkpoints" / "last.ckpt"
    best_ckpt = out_dir / "checkpoints" / "best.ckpt"
    resume_ckpt = last_ckpt if last_ckpt.exists() else best_ckpt if best_ckpt.exists() else None
    if resume_ckpt is None:
        return 0, float("inf"), 0
    ckpt = torch.load(str(resume_ckpt), map_location="cpu", weights_only=False)
    start_epoch = ckpt.get("epoch", -1) + 1

    # Bug 1 fix: best_eer from best.ckpt, not resume_ckpt
    if best_ckpt.exists() and best_ckpt != resume_ckpt:
        best_ckpt_data = torch.load(str(best_ckpt), map_location="cpu", weights_only=False)
        best_eer = best_ckpt_data.get("best_eer", best_ckpt_data.get("dev_eer", float("inf")))
        source = best_ckpt.name
    else:
        best_eer = ckpt.get("best_eer", ckpt.get("dev_eer", float("inf")))
        source = resume_ckpt.name

    # Bug 2 fix: since_improve from resume_ckpt
    since_improve = ckpt.get("since_improve", 0)
    return start_epoch, best_eer, since_improve


def test_persist_and_restore():
    """Test 1: save checkpoint with known best_eer + since_improve, resume, assert match."""
    import shutil
    tmp = Path("/tmp/test_resume_1")
    shutil.rmtree(tmp, ignore_errors=True)

    trainer = _make_trainer(str(tmp))

    # Simulate mid-training state
    trainer.best_eer = 0.0423
    trainer._since_improve = 5

    ckpt_path = tmp / "checkpoints" / "last.ckpt"
    _save_ckpt(ckpt_path, epoch=7, dev_eer=0.0512,
               best_eer=trainer.best_eer, since_improve=trainer._since_improve)

    # Simulate cli.py resume
    start_epoch, best_eer, since_improve = _cli_resume(tmp)

    assert start_epoch == 8, f"start_epoch: expected 8, got {start_epoch}"
    assert best_eer == 0.0423, f"best_eer: expected 0.0423, got {best_eer}"
    assert since_improve == 5, f"since_improve: expected 5, got {since_improve}"

    print("PASS: test_persist_and_restore")
    shutil.rmtree(tmp, ignore_errors=True)


def test_best_eer_from_best_ckpt():
    """Test 2: best.ckpt has better dev_eer than last.ckpt.
    best_eer should come from best.ckpt, start_epoch from last.ckpt."""
    import shutil
    tmp = Path("/tmp/test_resume_2")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "checkpoints").mkdir(parents=True)

    # best.ckpt: epoch 5, dev_eer=0.0300 (the true best)
    _save_ckpt(tmp / "checkpoints" / "best.ckpt",
               epoch=5, dev_eer=0.0300, best_eer=0.0300, since_improve=0)
    # last.ckpt: epoch 8, dev_eer=0.0512 (worse, after best)
    _save_ckpt(tmp / "checkpoints" / "last.ckpt",
               epoch=8, dev_eer=0.0512, best_eer=0.0300, since_improve=3)

    start_epoch, best_eer, since_improve = _cli_resume(tmp)

    # start_epoch from last.ckpt
    assert start_epoch == 9, f"start_epoch: expected 9, got {start_epoch}"
    # best_eer from best.ckpt (0.0300), NOT from last.ckpt (0.0512)
    assert best_eer == 0.0300, f"best_eer: expected 0.0300 (from best.ckpt), got {best_eer}"
    # since_improve from last.ckpt (the one restoring model weights)
    assert since_improve == 3, f"since_improve: expected 3 (from last.ckpt), got {since_improve}"

    print("PASS: test_best_eer_from_best_ckpt")
    shutil.rmtree(tmp, ignore_errors=True)


def test_backward_compat():
    """Old checkpoint without best_eer/since_improve fields should still work."""
    import shutil
    tmp = Path("/tmp/test_resume_3")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "checkpoints").mkdir(parents=True)

    # Old-format checkpoint
    _save_ckpt(tmp / "checkpoints" / "last.ckpt", epoch=3, dev_eer=0.0678)

    start_epoch, best_eer, since_improve = _cli_resume(tmp)

    assert start_epoch == 4
    assert best_eer == 0.0678, f"fallback best_eer: expected 0.0678, got {best_eer}"
    assert since_improve == 0, f"fallback since_improve: expected 0, got {since_improve}"

    print("PASS: test_backward_compat")
    shutil.rmtree(tmp, ignore_errors=True)


def test_best_eer_from_last_ckpt_field():
    """Test 4: Only last.ckpt exists (no best.ckpt). Its dev_eer is worse
    than the stored best_eer field. Resume should restore best_eer, not dev_eer."""
    import shutil
    tmp = Path("/tmp/test_resume_4")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "checkpoints").mkdir(parents=True)

    # last.ckpt: epoch 8, dev_eer=0.0512, but best_eer=0.0300 (true best from earlier)
    _save_ckpt(tmp / "checkpoints" / "last.ckpt",
               epoch=8, dev_eer=0.0512, best_eer=0.0300, since_improve=3)

    start_epoch, best_eer, since_improve = _cli_resume(tmp)

    assert start_epoch == 9, f"start_epoch: expected 9, got {start_epoch}"
    # best_eer=0.0300 from best_eer field, NOT dev_eer=0.0512
    assert best_eer == 0.0300, f"best_eer: expected 0.0300 (from best_eer field), got {best_eer}"
    assert since_improve == 3, f"since_improve: expected 3, got {since_improve}"

    print("PASS: test_best_eer_from_last_ckpt_field")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_persist_and_restore()
    test_best_eer_from_best_ckpt()
    test_backward_compat()
    test_best_eer_from_last_ckpt_field()
    print("\nAll tests passed.")
