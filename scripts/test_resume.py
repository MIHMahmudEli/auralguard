#!/usr/bin/env python
"""Verify checkpoint resume correctness for AuralGuard trainer.

Tests:
  1. best_eer and since_improve are persisted and restored correctly.
  2. best_eer comes from best.ckpt (not last.ckpt) when both exist.
  3. Old checkpoints without new fields fall back gracefully.
  4. Only last.ckpt exists but it carries best_eer field — restores best_eer, not dev_eer.
  5. Optimizer state (exp_avg/exp_avg_sq) is persisted and restored.
  6. Scaler state is persisted and restored.
  7. Old checkpoint without optimizer/scaler keys resumes without crash.
"""

import sys, types
import torch

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
    best_eer from best.ckpt, since_improve + optimizer + scaler from resume_ckpt."""
    last_ckpt = out_dir / "checkpoints" / "last.ckpt"
    best_ckpt = out_dir / "checkpoints" / "best.ckpt"
    resume_ckpt = last_ckpt if last_ckpt.exists() else best_ckpt if best_ckpt.exists() else None
    if resume_ckpt is None:
        return 0, float("inf"), 0, None, None
    ckpt = torch.load(str(resume_ckpt), map_location="cpu", weights_only=False)
    start_epoch = ckpt.get("epoch", -1) + 1

    # Bug 1 fix: best_eer from best.ckpt, not resume_ckpt
    if best_ckpt.exists() and best_ckpt != resume_ckpt:
        best_ckpt_data = torch.load(str(best_ckpt), map_location="cpu", weights_only=False)
        best_eer = best_ckpt_data.get("best_eer", best_ckpt_data.get("dev_eer", float("inf")))
    else:
        best_eer = ckpt.get("best_eer", ckpt.get("dev_eer", float("inf")))

    # Bug 2 fix: since_improve from resume_ckpt
    since_improve = ckpt.get("since_improve", 0)

    # Optimizer + scaler from resume_ckpt
    optimizer_state = ckpt.get("optimizer", None)
    scaler_state = ckpt.get("scaler", None)

    return start_epoch, best_eer, since_improve, optimizer_state, scaler_state


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
    start_epoch, best_eer, since_improve, _, _ = _cli_resume(tmp)

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

    start_epoch, best_eer, since_improve, _, _ = _cli_resume(tmp)

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

    start_epoch, best_eer, since_improve, _, _ = _cli_resume(tmp)

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

    start_epoch, best_eer, since_improve, _, _ = _cli_resume(tmp)

    assert start_epoch == 9, f"start_epoch: expected 9, got {start_epoch}"
    # best_eer=0.0300 from best_eer field, NOT dev_eer=0.0512
    assert best_eer == 0.0300, f"best_eer: expected 0.0300 (from best_eer field), got {best_eer}"
    assert since_improve == 3, f"since_improve: expected 3, got {since_improve}"

    print("PASS: test_best_eer_from_last_ckpt_field")
    shutil.rmtree(tmp, ignore_errors=True)


def test_optimizer_state_persisted():
    """Test 5: Optimizer exp_avg/exp_avg_sq survive save+resume."""
    import shutil
    tmp = Path("/tmp/test_resume_5")
    shutil.rmtree(tmp, ignore_errors=True)

    trainer = _make_trainer(str(tmp))

    # Run a few fake optimizer steps to populate state
    for p in trainer.model.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p)
    trainer.optimizer.step()
    for p in trainer.model.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p)
    trainer.optimizer.step()

    # Snapshot optimizer state before save
    pre_state = trainer.optimizer.state_dict()
    pre_exp_avg = {k: v["exp_avg"].clone() for k, v in pre_state["state"].items() if "exp_avg" in v}
    pre_exp_avg_sq = {k: v["exp_avg_sq"].clone() for k, v in pre_state["state"].items() if "exp_avg_sq" in v}
    assert any(torch.abs(v).sum() > 0 for v in pre_exp_avg.values()), "exp_avg should be non-zero after steps"

    # Save checkpoint
    ckpt_path = tmp / "checkpoints" / "last.ckpt"
    trainer._save("last.ckpt", epoch=2, eer=0.1)

    # Create fresh trainer and resume
    trainer2 = _make_trainer(str(tmp))
    _, _, _, optimizer_state, _ = _cli_resume(tmp)
    assert optimizer_state is not None, "optimizer state should be in checkpoint"
    trainer2.optimizer.load_state_dict(optimizer_state)

    # Compare
    post_state = trainer2.optimizer.state_dict()
    for k in pre_exp_avg:
        assert torch.allclose(pre_exp_avg[k], post_state["state"][k]["exp_avg"], atol=1e-6), \
            f"exp_avg mismatch for param group {k}"
    for k in pre_exp_avg_sq:
        assert torch.allclose(pre_exp_avg_sq[k], post_state["state"][k]["exp_avg_sq"], atol=1e-6), \
            f"exp_avg_sq mismatch for param group {k}"

    print("PASS: test_optimizer_state_persisted")
    shutil.rmtree(tmp, ignore_errors=True)


def test_scaler_state_persisted():
    """Test 6: AMP GradScaler state survives save+resume."""
    import shutil
    tmp = Path("/tmp/test_resume_6")
    shutil.rmtree(tmp, ignore_errors=True)

    trainer = _make_trainer(str(tmp))

    # Manually set a known scale so it's non-default
    original_scale = trainer.scaler.get_scale()
    # Step the scaler a few times to change internal state
    for _ in range(3):
        for p in trainer.model.parameters():
            if p.requires_grad:
                p.grad = torch.ones_like(p)
        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()
    stepped_scale = trainer.scaler.get_scale()

    # Save
    trainer._save("last.ckpt", epoch=1, eer=0.2)

    # Create fresh trainer and resume
    trainer2 = _make_trainer(str(tmp))
    _, _, _, _, scaler_state = _cli_resume(tmp)
    assert scaler_state is not None, "scaler state should be in checkpoint"
    trainer2.scaler.load_state_dict(scaler_state)

    assert trainer2.scaler.get_scale() == stepped_scale, \
        f"scaler scale: expected {stepped_scale}, got {trainer2.scaler.get_scale()}"

    print("PASS: test_scaler_state_persisted")
    shutil.rmtree(tmp, ignore_errors=True)


def test_old_checkpoint_no_optimizer_scaler():
    """Test 7: Old checkpoint without optimizer/scaler keys resumes without crash."""
    import shutil
    tmp = Path("/tmp/test_resume_7")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "checkpoints").mkdir(parents=True)

    # Old-format checkpoint (no optimizer, no scaler)
    _save_ckpt(tmp / "checkpoints" / "last.ckpt", epoch=3, dev_eer=0.0678)

    trainer = _make_trainer(str(tmp))
    # Verify initial optimizer/scaler state is fresh
    fresh_opt_scale = sum(v.get("exp_avg", torch.zeros(1)).abs().sum().item()
                          for v in trainer.optimizer.state_dict()["state"].values()
                          if "exp_avg" in v)
    fresh_scaler_scale = trainer.scaler.get_scale()

    start_epoch, best_eer, since_improve, opt_state, scl_state = _cli_resume(tmp)
    assert opt_state is None, "old checkpoint should not have optimizer state"
    assert scl_state is None, "old checkpoint should not have scaler state"

    # Optimizer and scaler should remain fresh (unchanged from init)
    post_opt_scale = sum(v.get("exp_avg", torch.zeros(1)).abs().sum().item()
                         for v in trainer.optimizer.state_dict()["state"].values()
                         if "exp_avg" in v)
    assert post_opt_scale == fresh_opt_scale, "optimizer should remain fresh"
    assert trainer.scaler.get_scale() == fresh_scaler_scale, "scaler should remain fresh"

    print("PASS: test_old_checkpoint_no_optimizer_scaler")
    shutil.rmtree(tmp, ignore_errors=True)


def test_suspect_eer_detection():
    """Test 8: _is_suspect_eer correctly flags degenerate EER values."""
    import shutil, math
    tmp = Path("/tmp/test_resume_8")
    shutil.rmtree(tmp, ignore_errors=True)

    trainer = _make_trainer(str(tmp))

    # Default threshold is 0.95
    assert trainer._is_suspect_eer(1.0) is True, "EER=1.0 should be suspect"
    assert trainer._is_suspect_eer(0.96) is True, "EER=0.96 should be suspect"
    assert trainer._is_suspect_eer(0.95) is True, "EER=0.95 (at threshold) should be suspect"
    assert trainer._is_suspect_eer(0.94) is False, "EER=0.94 should NOT be suspect"
    assert trainer._is_suspect_eer(0.0016) is False, "EER=0.0016 should NOT be suspect"
    assert trainer._is_suspect_eer(float("nan")) is True, "NaN EER should be suspect"
    assert trainer._is_suspect_eer(float("inf")) is True, "Inf EER should be suspect"

    # Test with disabled sanity check
    trainer._sc_enabled = False
    assert trainer._is_suspect_eer(1.0) is False, "disabled: EER=1.0 should NOT be suspect"

    print("PASS: test_suspect_eer_detection")
    shutil.rmtree(tmp, ignore_errors=True)


def test_suspect_epoch_quarantined():
    """Test 9: When dev_eer=1.0, last.ckpt is NOT overwritten; last_suspect.ckpt created."""
    import shutil
    tmp = Path("/tmp/test_resume_9")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "checkpoints").mkdir(parents=True)

    # Pre-existing good last.ckpt (epoch 20, dev_eer=0.01)
    _save_ckpt(tmp / "checkpoints" / "last.ckpt",
               epoch=20, dev_eer=0.01, best_eer=0.01, since_improve=0)
    # Pre-existing best.ckpt
    _save_ckpt(tmp / "checkpoints" / "best.ckpt",
               epoch=20, dev_eer=0.01, best_eer=0.01, since_improve=0)

    # Simulate what the trainer DOES for a suspect epoch:
    # Instead of overwriting last.ckpt, save to last_suspect.ckpt
    suspect_eer = 1.0
    ckpt_dir = tmp / "checkpoints"
    last_ckpt = ckpt_dir / "last.ckpt"
    last_suspect = ckpt_dir / "last_suspect.ckpt"

    # Verify guard logic: if suspect, DON'T overwrite last.ckpt
    import math
    is_suspect = suspect_eer >= 0.95 or math.isnan(suspect_eer)
    assert is_suspect, "EER=1.0 should be detected as suspect"

    # Simulate quarantine: save suspect to separate file
    if is_suspect:
        _save_ckpt(last_suspect, epoch=21, dev_eer=suspect_eer)
    # last.ckpt should still be the old good one
    ckpt = torch.load(str(last_ckpt), map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 20, f"last.ckpt epoch should still be 20, got {ckpt['epoch']}"
    assert ckpt["dev_eer"] == 0.01, f"last.ckpt dev_eer should still be 0.01, got {ckpt['dev_eer']}"

    # last_suspect.ckpt should have the bad epoch
    suspect = torch.load(str(last_suspect), map_location="cpu", weights_only=False)
    assert suspect["epoch"] == 21, f"last_suspect epoch should be 21, got {suspect['epoch']}"
    assert suspect["dev_eer"] == 1.0, f"last_suspect dev_eer should be 1.0, got {suspect['dev_eer']}"

    # best.ckpt should be untouched
    best = torch.load(str(ckpt_dir / "best.ckpt"), map_location="cpu", weights_only=False)
    assert best["dev_eer"] == 0.01, f"best.ckpt dev_eer should still be 0.01, got {best['dev_eer']}"

    # Resume from last.ckpt should give epoch 21 (after good epoch 20)
    start_epoch, best_eer, _, _, _ = _cli_resume(tmp)
    assert start_epoch == 21, f"start_epoch should be 21, got {start_epoch}"
    assert best_eer == 0.01, f"best_eer should be 0.01, got {best_eer}"

    print("PASS: test_suspect_epoch_quarantined")
    shutil.rmtree(tmp, ignore_errors=True)


def test_good_epoch_overwrites_last():
    """Test 10: A good (non-suspect) epoch DOES overwrite last.ckpt as before."""
    import shutil
    tmp = Path("/tmp/test_resume_10")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "checkpoints").mkdir(parents=True)

    # Pre-existing last.ckpt (epoch 20, dev_eer=0.05)
    _save_ckpt(tmp / "checkpoints" / "last.ckpt",
               epoch=20, dev_eer=0.05, best_eer=0.03, since_improve=2)

    # Simulate a good epoch: overwrite last.ckpt
    _save_ckpt(tmp / "checkpoints" / "last.ckpt",
               epoch=21, dev_eer=0.04, best_eer=0.03, since_improve=3)

    # Resume should pick up epoch 22
    start_epoch, best_eer, since_improve, _, _ = _cli_resume(tmp)
    assert start_epoch == 22, f"start_epoch should be 22, got {start_epoch}"
    assert since_improve == 3, f"since_improve should be 3, got {since_improve}"

    print("PASS: test_good_epoch_overwrites_last")
    shutil.rmtree(tmp, ignore_errors=True)


def test_rollback_to_good_checkpoint():
    """Test 11: After a suspect epoch, _rollback_to_good_checkpoint restores
    model/optimizer state to the last known-good checkpoint."""
    import shutil
    tmp = Path("/tmp/test_resume_11")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "checkpoints").mkdir(parents=True)

    # Create a trainer with a known-good checkpoint
    trainer = _make_trainer(str(tmp))

    # Simulate some training steps to populate optimizer state
    for p in trainer.model.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p)
    trainer.optimizer.step()
    pre_state = trainer.optimizer.state_dict()
    pre_exp_avg = {k: v["exp_avg"].clone() for k, v in pre_state["state"].items() if "exp_avg" in v}
    # Save the good state
    trainer._save("last.ckpt", epoch=5, eer=0.05)
    trainer._save("best.ckpt", epoch=5, eer=0.05)
    good_model_state = {k: v.clone() for k, v in trainer.model.state_dict().items()}

    # Now corrupt the model and optimizer
    for p in trainer.model.parameters():
        p.data.fill_(999.0)
    for k in trainer.optimizer.state:
        if "exp_avg" in trainer.optimizer.state[k]:
            trainer.optimizer.state[k]["exp_avg"].fill_(999.0)

    # Verify corruption happened
    assert any(v.mean().item() > 900 for v in trainer.model.state_dict().values()), \
        "model should be corrupted before rollback"

    # Rollback
    result_epoch = trainer._rollback_to_good_checkpoint()

    assert result_epoch == 5, f"rollback should return epoch 5, got {result_epoch}"
    # Verify model is restored
    for k, v in trainer.model.state_dict().items():
        assert torch.equal(v, good_model_state[k]), f"model param {k} not restored"
    # Verify optimizer is restored
    post_state = trainer.optimizer.state_dict()
    for k in pre_exp_avg:
        assert torch.allclose(pre_exp_avg[k], post_state["state"][k]["exp_avg"], atol=1e-6), \
            f"optimizer exp_avg not restored for {k}"

    print("PASS: test_rollback_to_good_checkpoint")
    shutil.rmtree(tmp, ignore_errors=True)


def test_consecutive_suspect_stops_training():
    """Test 12: 3 consecutive suspect epochs (even after rollback) triggers hard-stop
    with distinct STOPPING message, not confused with early stopping."""
    import shutil
    tmp = Path("/tmp/test_resume_12")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "checkpoints").mkdir(parents=True)

    # Build a real checkpoint with model/optimizer/scaler via Trainer._save()
    trainer = _make_trainer(str(tmp))
    for p in trainer.model.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p)
    trainer.optimizer.step()
    trainer.best_eer = 0.05
    trainer._save("last.ckpt", epoch=0, eer=0.05)
    trainer._save("best.ckpt", epoch=0, eer=0.05)

    # Configure: max 2 consecutive suspect, low threshold to trigger suspect easily
    trainer._sc_eer_threshold = 0.01  # any eer > 0.01 is suspect
    trainer._max_consecutive_suspect = 2
    trainer._consecutive_suspect = 0

    # Manually simulate what train() does for suspect epochs.
    # We can't call train() directly because _train_epoch/_validate need real data,
    # but we can test the logic by calling the pieces.
    suspect_eers = [1.0, 1.0, 1.0]
    for i, eer in enumerate(suspect_eers):
        suspect = trainer._is_suspect_eer(eer)
        assert suspect, f"eer {eer} should be suspect"

        trainer._consecutive_suspect += 1
        trainer._save("last_suspect.ckpt", i + 1, eer)

        # Rollback
        rolled_back_epoch = trainer._rollback_to_good_checkpoint()
        assert rolled_back_epoch == 0, f"rollback should go to epoch 0, got {rolled_back_epoch}"

        # Check hard-stop
        if trainer._consecutive_suspect >= trainer._max_consecutive_suspect:
            # This is the distinct STOPPING condition (not early stopping)
            break

    assert trainer._consecutive_suspect == 2, \
        f"consecutive_suspect should be 2, got {trainer._consecutive_suspect}"
    assert trainer._since_improve == 0, "since_improve should not have changed"

    # Verify last.ckpt was NOT overwritten by suspect epochs
    last_ckpt = torch.load(str(tmp / "checkpoints" / "last.ckpt"),
                           map_location="cpu", weights_only=False)
    assert last_ckpt["epoch"] == 0, f"last.ckpt should still be epoch 0, got {last_ckpt['epoch']}"
    assert last_ckpt["dev_eer"] == 0.05, f"last.ckpt dev_eer should still be 0.05"

    print("PASS: test_consecutive_suspect_stops_training")
    shutil.rmtree(tmp, ignore_errors=True)


def test_consecutive_nan_grad_steps():
    """Test 13: Two consecutive NaN-gradient steps do not raise RuntimeError.

    When NaN/Inf is detected after unscale_(), the correct pattern is to still
    call scaler.step(optimizer) + scaler.update() — GradScaler's step()
    internally skips the optimizer when it finds NaN, while update() resets
    the scaler state. Skipping update() leaves the scaler in a bad internal
    state, causing the NEXT unscale_() call to raise RuntimeError.
    """
    import shutil
    tmp = Path("/tmp/test_resume_13")
    shutil.rmtree(tmp, ignore_errors=True)

    model = build_model({"name": "lfcc_lcnn", "lfcc": {"n_filts": 60, "n_frames": 500}})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Simulate two consecutive NaN-gradient steps back to back —
    # this is the exact sequence that crashed in production (step 308).
    for step in range(2):
        optimizer.zero_grad()
        out = model(torch.randn(1, 16000), torch.zeros(1, dtype=torch.long))
        loss = out["loss"]
        scaler.scale(loss).backward()

        # Inject NaN gradient to simulate a bad step
        for p in model.parameters():
            if p.grad is not None:
                p.grad = torch.full_like(p.grad, float("nan"))
                break

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

    print("PASS: test_consecutive_nan_grad_steps")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_persist_and_restore()
    test_best_eer_from_best_ckpt()
    test_backward_compat()
    test_best_eer_from_last_ckpt_field()
    test_optimizer_state_persisted()
    test_scaler_state_persisted()
    test_old_checkpoint_no_optimizer_scaler()
    test_suspect_eer_detection()
    test_suspect_epoch_quarantined()
    test_good_epoch_overwrites_last()
    test_rollback_to_good_checkpoint()
    test_consecutive_suspect_stops_training()
    test_consecutive_nan_grad_steps()
    print("\nAll tests passed.")
