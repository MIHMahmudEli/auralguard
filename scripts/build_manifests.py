#!/usr/bin/env python
"""Normalize each corpus's native protocol files into the repo manifest schema.

Add one adapter function per dataset. Each adapter reads the dataset's own protocol
and writes data/manifests/<name>.csv with columns:
    utt_id, path, label, attack, dataset, lang, split, codec
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MANIFEST_DIR = Path("data/manifests")
COLUMNS = ["utt_id", "path", "label", "attack", "dataset", "lang", "split", "codec"]


# ── ASVspoof 2019 LA (train / dev / eval) ───────────────────────────────

def asvspoof2019_la(root: str, split: str) -> pd.DataFrame:
    root = Path(root)
    suffix = "trn" if split == "train" else "trl"
    proto = root / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{split}.{suffix}.txt"
    audio_dir = root / f"ASVspoof2019_LA_{split}" / "flac"
    rows = []
    for line in proto.read_text().splitlines():
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


# ── ASVspoof 2021 LA (eval only) ────────────────────────────────────────

def asvspoof2021_la(root: str) -> pd.DataFrame:
    root = Path(root)
    proto = root / "trial_metadata.txt"
    audio_dir = root / "flac"
    rows = []
    for line in proto.read_text().splitlines():
        parts = line.split()
        if len(parts) < 8:
            continue
        spk, utt, codec, _channel, attack, key, _trim, _subset = parts[:8]
        rows.append({
            "utt_id": utt,
            "path": str(audio_dir / f"{utt}.flac"),
            "label": 0 if key == "bonafide" else 1,
            "attack": "bonafide" if key == "bonafide" else attack,
            "dataset": "asvspoof2021_la",
            "lang": "en",
            "split": "eval",
            "codec": codec,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


# ── ASVspoof 2021 DF (eval only) ────────────────────────────────────────

def asvspoof2021_df(root: str) -> pd.DataFrame:
    root = Path(root)
    proto = root / "trial_metadata.txt"
    audio_dir = root / "flac"
    rows = []
    for line in proto.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        spk, utt, codec, _source, attack, key = parts[:6]
        rows.append({
            "utt_id": utt,
            "path": str(audio_dir / f"{utt}.flac"),
            "label": 0 if key == "bonafide" else 1,
            "attack": "bonafide" if key == "bonafide" else attack,
            "dataset": "asvspoof2021_df",
            "lang": "en",
            "split": "eval",
            "codec": codec,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


# ── In-the-Wild (eval only) ─────────────────────────────────────────────

def in_the_wild(root: str) -> pd.DataFrame:
    root = Path(root)
    meta = root / "meta.csv"
    rows = []
    for _, row in pd.read_csv(meta).iterrows():
        fname = row["file"]
        label = row["label"]
        utt = fname.replace(".wav", "")
        rows.append({
            "utt_id": utt,
            "path": str(root / fname),
            "label": 0 if label == "bonafide" else 1,
            "attack": "bonafide" if label == "bonafide" else "spoof",
            "dataset": "in_the_wild",
            "lang": "en",
            "split": "eval",
            "codec": "none",
        })
    return pd.DataFrame(rows, columns=COLUMNS)


# ── WaveFake (eval only — requires LJSpeech bonafide reference) ─────────

def wavefake(root: str) -> pd.DataFrame:
    root = Path(root)
    rows = []

    # All generated .wav files are spoof
    for wav in sorted(root.rglob("*_gen.wav")):
        utt = wav.stem
        attack = wav.parent.name
        rows.append({
            "utt_id": utt,
            "path": str(wav),
            "label": 1,
            "attack": attack,
            "dataset": "wavefake",
            "lang": "en",
            "split": "eval",
            "codec": "none",
        })

    # Try to pair with LJSpeech bona-fide (from data/raw/LJSpeech)
    ljspeech_dir = Path("data/raw/LJSpeech")
    if ljspeech_dir.exists():
        for spoof_row in rows:
            ref_name = spoof_row["utt_id"].replace("_gen", "") + ".wav"
            ref_path = ljspeech_dir / ref_name
            if ref_path.exists():
                rows.append({
                    "utt_id": ref_name.replace(".wav", ""),
                    "path": str(ref_path),
                    "label": 0,
                    "attack": "bonafide",
                    "dataset": "wavefake",
                    "lang": "en",
                    "split": "eval",
                    "codec": "none",
                })
    else:
        print("  [info] LJSpeech not found; WaveFake will lack bonafide references")

    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.drop_duplicates(subset=["utt_id"])
    return df


# ── MLAAD (eval only — all spoof; language-aware) ───────────────────────

def mlaad(root: str) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for meta_path in sorted(root.rglob("meta.csv")):
        for _, row in pd.read_csv(meta_path, delimiter="|").iterrows():
            audio_path = meta_path.parent / row["path"]
            if not audio_path.exists():
                continue
            lang = str(row.get("language", "und"))
            rows.append({
                "utt_id": audio_path.stem,
                "path": str(audio_path),
                "label": 1,
                "attack": row.get("model_name", "unknown"),
                "dataset": "mlaad",
                "lang": lang,
                "split": "eval",
                "codec": "none",
            })
    return pd.DataFrame(rows, columns=COLUMNS)


# ── Registry ────────────────────────────────────────────────────────────

ADAPTERS = {
    "asvspoof2019_la_train": lambda root: asvspoof2019_la(root, "train"),
    "asvspoof2019_la_dev": lambda root: asvspoof2019_la(root, "dev"),
    "asvspoof2019_la_eval": lambda root: asvspoof2019_la(root, "eval"),
    "asvspoof2021_la_eval": lambda root: asvspoof2021_la(root),
    "asvspoof2021_df_eval": lambda root: asvspoof2021_df(root),
    "in_the_wild": lambda root: in_the_wild(root),
    "wavefake": lambda root: wavefake(root),
    "mlaad": lambda root: mlaad(root),
}


# ── CLI ─────────────────────────────────────────────────────────────────

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

        # Determine default root for each dataset
        dataset_name = name.rsplit("_", 1)[0] if name.count("_") >= 2 else name
        default_roots = {
            "asvspoof2019_la": "data/raw/ASVspoof2019_LA",
            "asvspoof2021_la": "data/raw/ASVspoof2021_LA",
            "asvspoof2021_df": "data/raw/ASVspoof2021_DF",
            "in_the_wild": "data/raw/in_the_wild",
            "wavefake": "data/raw/wavefake",
            "mlaad": "data/raw/mlaad",
        }
        root = default_roots.get(dataset_name, args.root)

        df = ADAPTERS[name](root)
        out = MANIFEST_DIR / f"{name}.csv"
        df.to_csv(out, index=False)
        n_spoof = df["label"].sum()
        print(f"[ok] {out}  ({len(df)} rows, {n_spoof} spoof)")


if __name__ == "__main__":
    main()
