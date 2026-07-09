#!/usr/bin/env python
"""Normalize each corpus's native protocol files into the repo manifest schema.

Add one adapter function per dataset. Each adapter reads the dataset's own protocol
and writes data/manifests/<name>.csv with columns:
    utt_id, path, label, attack, dataset, lang, split, codec

Only ASVspoof2019-LA is fully wired here as the reference adapter; the others are
stubs with the exact field mapping documented so you can fill them once the data is
downloaded (see docs/DATASETS.md).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MANIFEST_DIR = Path("data/manifests")
COLUMNS = ["utt_id", "path", "label", "attack", "dataset", "lang", "split", "codec"]


def asvspoof2019_la(root: str, split: str) -> pd.DataFrame:
    """root = data/raw/ASVspoof2019_LA. split in {train, dev, eval}."""
    root = Path(root)
    # train protocol is .trn.txt (not .trl.txt)
    suffix = "trn" if split == "train" else "trl"
    proto = root / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{split}.{suffix}.txt"
    audio_dir = root / f"ASVspoof2019_LA_{split}" / "flac"
    rows = []
    for line in proto.read_text().splitlines():
        # format: SPEAKER  UTT_ID  -  ATTACK_ID  KEY(bonafide/spoof)
        parts = line.split()
        utt, attack, key = parts[1], parts[3], parts[4]
        rows.append({
            "utt_id": utt,
            "path": str(audio_dir / f"{utt}.flac"),
            "label": 0 if key == "bonafide" else 1,
            "attack": "bonafide" if key == "bonafide" else attack,
            "dataset": "asvspoof2019_la",
            "lang": "en",
            "split": split,
            "codec": "none",
        })
    return pd.DataFrame(rows, columns=COLUMNS)


ADAPTERS = {
    "asvspoof2019_la_train": lambda root: asvspoof2019_la(root, "train"),
    "asvspoof2019_la_dev": lambda root: asvspoof2019_la(root, "dev"),
    "asvspoof2019_la_eval": lambda root: asvspoof2019_la(root, "eval"),
    # "in_the_wild": in_the_wild,   # TODO: label from meta.csv (0 bona-fide / 1 spoof)
    # "wavefake": wavefake,         # TODO: all spoof; pair with LJSpeech bona-fide
    # "mlaad": mlaad,               # TODO: keep `lang` per file for cross-lingual eval
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="single manifest to build", default=None)
    ap.add_argument("--root", default="data/raw/ASVspoof2019_LA")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    targets = ADAPTERS.keys() if args.all else [args.name]
    for name in targets:
        if name not in ADAPTERS:
            print(f"[skip] no adapter for {name}")
            continue
        df = ADAPTERS[name](args.root)
        out = MANIFEST_DIR / f"{name}.csv"
        df.to_csv(out, index=False)
        print(f"[ok] {out}  ({len(df)} rows, {df.label.sum()} spoof)")


if __name__ == "__main__":
    main()
