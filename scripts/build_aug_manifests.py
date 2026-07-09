"""Build manifest CSVs for augmentation corpora (MUSAN noise, RIRs).

Usage:
    python scripts/build_aug_manifests.py --rirs     # build RIRs manifest
    python scripts/build_aug_manifests.py --musan    # build MUSAN manifest
    python scripts/build_aug_manifests.py --all      # build both
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MANIFEST_DIR = Path("data/manifests")


def build_rirs(root: str = "data/raw/RIRS_NOISES") -> pd.DataFrame:
    root = Path(root)
    rows = []
    # Use real RIRs (measured) and simulated RIRs
    for rir_dir in [
        root / "real_rirs_isotropic_noises" / "real_rirs_isotropic_noises",
        root / "simulated_rirs" / "mediumroom",
        root / "simulated_rirs" / "largeroom",
        root / "simulated_rirs" / "smallroom",
    ]:
        if rir_dir.exists():
            for f in sorted(rir_dir.rglob("*.wav")):
                rows.append({"path": str(f.resolve())})
    return pd.DataFrame(rows)


def build_musan(root: str = "data/raw/musan") -> pd.DataFrame:
    root = Path(root)
    rows = []
    if not root.exists():
        print(f"WARNING: {root} not found. MUSAN manifest will be empty.")
        return pd.DataFrame(columns=["path"])
    for noise_file in sorted(root.rglob("*.wav")):
        rows.append({"path": str(noise_file.resolve())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rirs", action="store_true")
    ap.add_argument("--musan", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    if args.all or args.rirs:
        df = build_rirs()
        out = MANIFEST_DIR / "rirs.csv"
        df.to_csv(out, index=False)
        print(f"[ok] {out}  ({len(df)} RIR files)")

    if args.all or args.musan:
        df = build_musan()
        out = MANIFEST_DIR / "musan.csv"
        df.to_csv(out, index=False)
        print(f"[ok] {out}  ({len(df)} noise files)")


if __name__ == "__main__":
    main()
