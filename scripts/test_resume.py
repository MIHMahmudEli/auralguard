#!/usr/bin/env python
"""Verify that Trainer.resume() correctly restores best_eer, since_improve, and optimizer state."""

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


def make_dummy_model():
    cfg = OmegaConf.create({
        "model": {"name": "lfcc_lcnn", "lfcc": {"n_filts": 60, "n_frames": 500}},
        "train": {
            "epochs": 5, "batch_size": 2, "lr": 1e-4,
            "optimizer": {"name": "adamw", "lr": 1e-4, "weight_decay": 1e-4},
            "scheduler": {"warmup_epochs": 1, "min_lr": 1e-7},
            "early_stop": {"patience": 3},
            "amp": False, "grad_clip": 5.0, "log_every_n_steps": 1,
        },
        "data": {"num_workers": 0, "pin_memory": False},
        "output_dir": "/tmp/test_resume_ckpt",
        "seed": 42,
    })
    model = build_model(OmegaConf.to_container(cfg.model, resolve=True))
    return model, cfg


def test_resume_persists_state():
    model, cfg = make_dummy_model()
    out_dir = Path(cfg.output_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Simulate a trainer mid-training with known state
    fake_ds = _FakeDataset()
    trainer = Trainer(model, fake_ds, fake_ds, cfg, device="cpu")

    # Set non-default state
    trainer.best_eer = 0.0423
    trainer._since_improve = 5

    # Save checkpoint
    ckpt_path = out_dir / "checkpoints" / "last.ckpt"
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": _to_container(cfg),
            "epoch": 7,
            "dev_eer": 0.0512,
            "best_eer": trainer.best_eer,
            "since_improve": trainer._since_improve,
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scaler_state_dict": trainer.scaler.state_dict(),
        },
        str(ckpt_path),
    )
    print(f"Saved checkpoint: best_eer={trainer.best_eer}, since_improve={trainer._since_improve}, epoch=7")

    # Create a fresh trainer and resume
    model2, cfg2 = make_dummy_model()
    fake_ds2 = _FakeDataset()
    trainer2 = Trainer(model2, fake_ds2, fake_ds2, cfg2, device="cpu")

    # Verify initial state is default
    assert trainer2.best_eer == float("inf"), f"Expected inf, got {trainer2.best_eer}"
    assert trainer2._since_improve == 0, f"Expected 0, got {trainer2._since_improve}"

    start_epoch = trainer2.resume(ckpt_path)

    # Verify restored state
    assert start_epoch == 8, f"Expected start_epoch=8, got {start_epoch}"
    assert trainer2.best_eer == 0.0423, f"Expected best_eer=0.0423, got {trainer2.best_eer}"
    assert trainer2._since_improve == 5, f"Expected since_improve=5, got {trainer2._since_improve}"

    # Verify optimizer state was loaded (param groups should match)
    orig_opt = trainer.optimizer
    new_opt = trainer2.optimizer
    for p1, p2 in zip(orig_opt.state.values(), new_opt.state.values()):
        for k in p1:
            if isinstance(p1[k], torch.Tensor):
                assert torch.equal(p1[k], p2[k]), f"Optimizer state mismatch for {k}"

    print("PASS: best_eer, since_improve, optimizer state all restored correctly")

    # Cleanup
    import shutil
    shutil.rmtree(out_dir, ignore_errors=True)


def test_backward_compat_no_state_fields():
    """Old checkpoints without best_eer/since_improve should still work."""
    model, cfg = make_dummy_model()
    out_dir = Path(cfg.output_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    fake_ds = _FakeDataset()
    trainer = Trainer(model, fake_ds, fake_ds, cfg, device="cpu")

    # Save in old format (no best_eer, since_improve, optimizer_state_dict)
    ckpt_path = out_dir / "checkpoints" / "last.ckpt"
    torch.save(
        {"model": model.state_dict(), "cfg": _to_container(cfg), "epoch": 3, "dev_eer": 0.0678},
        str(ckpt_path),
    )
    print(f"Saved old-format checkpoint: dev_eer=0.0678, epoch=3")

    # Resume with fresh trainer
    model2, cfg2 = make_dummy_model()
    fake_ds2 = _FakeDataset()
    trainer2 = Trainer(model2, fake_ds2, fake_ds2, cfg2, device="cpu")
    start_epoch = trainer2.resume(ckpt_path)

    assert start_epoch == 4
    assert trainer2.best_eer == 0.0678, f"Fallback: best_eer should be dev_eer, got {trainer2.best_eer}"
    assert trainer2._since_improve == 0, "Fallback: since_improve should be 0"

    print("PASS: backward compatibility works (old checkpoints)")

    import shutil
    shutil.rmtree(out_dir, ignore_errors=True)


class _FakeDataset:
    """Minimal dataset stub for Trainer construction."""
    def __len__(self): return 10
    def __getitem__(self, i):
        return (torch.randn(16000), 0, "")


if __name__ == "__main__":
    test_resume_persists_state()
    test_backward_compat_no_state_fields()
    print("\nAll tests passed.")
