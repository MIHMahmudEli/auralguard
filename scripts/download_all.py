"""One-shot download + extract + manifest builder for all AuralGuard datasets.

Usage:
    python scripts/download_all.py                          # interactive prompts
    python scripts/download_all.py --asvspoof               # ASVspoof 2019 LA only
    python scripts/download_all.py --aug                    # MUSAN + RIRs only
    python scripts/download_all.py --zero-shot              # zero-shot eval datasets
    python scripts/download_all.py --all                    # everything
    python scripts/download_all.py --manifest-only          # build manifests from existing files

ASVspoof 2019 LA terms: https://datashare.ed.ac.uk/handle/10283/3336
In-the-Wild:           Hugging Face (no registration required)
WaveFake:              Zenodo CC BY 4.0 (no registration required)
MLAAD:                 Hugging Face — requires `huggingface-cli login` (free)
ASVspoof 2021 LA/DF:   Zenodo Open Data Commons Attribution Licence
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

RAW = Path("data/raw")
MANIFEST = Path("data/manifests")

# ── file info ──────────────────────────────────────────────────────────
ASVSPOOF_INFO = {
    "name": "ASVspoof2019_LA",
    "dir": RAW / "ASVspoof2019_LA",
    "expected_md5": "30c98f11d8b2bc21f2c257bfd78bb5c5",
    "url_hint": "https://datashare.ed.ac.uk/handle/10283/3336",
}

ZERO_SHOT_DATASETS = {
    "in_the_wild": {
        "name": "In-the-Wild",
        "url": "https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip",
        "dir": RAW / "in_the_wild",
        "extract_mode": "zip",
        "extract_subdir": "release_in_the_wild",
    },
    "wavefake": {
        "name": "WaveFake",
        "url": "https://zenodo.org/records/5642694/files/wavefake.zip",
        "dir": RAW / "wavefake",
        "extract_mode": "zip",
        "extract_subdir": "audio",
    },
    "asvspoof2021_la": {
        "name": "ASVspoof2021_LA",
        "url": "https://zenodo.org/records/4837263/files/ASVspoof2021_LA_eval.tar.gz",
        "dir": RAW / "ASVspoof2021_LA",
        "extract_mode": "tar",
        "extract_subdir": None,
    },
    "asvspoof2021_df": {
        "name": "ASVspoof2021_DF",
        "url": "https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval.tar.gz",
        "dir": RAW / "ASVspoof2021_DF",
        "extract_mode": "tar",
        "extract_subdir": None,
    },
}

AUG_DATASETS = {
    "musan": {
        "name": "MUSAN",
        "url": "https://www.openslr.org/resources/17/musan.tar.gz",
        "dir": RAW,
        "extract_mode": "tar",
        "extract_subdir": None,
    },
    "rirs": {
        "name": "RIRs (SLR28)",
        "url": "https://www.openslr.org/resources/28/rirs_noises.zip",
        "dir": RAW,
        "extract_mode": "zip",
        "extract_subdir": None,
    },
}


# ── helpers ─────────────────────────────────────────────────────────────
def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, label: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {label} already at {dest}")
        return dest
    print(f"  [dl] {label} ({url}) …", end=" ", flush=True)
    with urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            while chunk := resp.read(2**20):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  [dl] {label} … {pct:.0f}%", end="", flush=True)
        print(f"\r  [ok] {label} ({dest})")
    return dest


def extract_zip(path: Path, dest: Path) -> None:
    print(f"  [extract] {path.name} → {dest}")
    with zipfile.ZipFile(path) as zf:
        zf.extractall(dest)


def extract_tar(path: Path, dest: Path) -> None:
    print(f"  [extract] {path.name} → {dest}")
    with tarfile.open(path) as tf:
        tf.extractall(dest)


# ── per-dataset ─────────────────────────────────────────────────────────
def download_asvspoof(url: str | None) -> Path:
    target = RAW / "LA.zip"
    if url is None:
        print(
            "ASVspoof 2019 LA requires accepting terms at:\n"
            f"  {ASVSPOOF_INFO['url_hint']}\n"
            "Then re-run with --asvspoof-url <direct-download-link>"
        )
        raise SystemExit(1)
    return download(url, target, ASVSPOOF_INFO["name"])


def extract_asvspoof(zip_path: Path) -> Path:
    out = ASVSPOOF_INFO["dir"]
    if out.exists():
        print(f"  [skip] {ASVSPOOF_INFO['name']} already extracted at {out}")
        return out
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW)
    la_dir = RAW / "LA"
    if la_dir.exists():
        for item in la_dir.iterdir():
            shutil.move(str(item), str(out))
        la_dir.rmdir()
    return out


def download_zero_shot_dataset(key: str) -> None:
    info = ZERO_SHOT_DATASETS[key]
    archive = RAW / f"{key}.{('zip' if info['extract_mode'] == 'zip' else 'tar.gz')}"
    download(info["url"], archive, info["name"])

    target = info["dir"]
    if target.exists():
        print(f"  [skip] {info['name']} already extracted at {target}")
        return
    target.mkdir(parents=True, exist_ok=True)

    if info["extract_mode"] == "zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(RAW)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(RAW)

    # Move from subdirectory if needed
    if info["extract_subdir"]:
        sub = RAW / info["extract_subdir"]
        if sub.exists():
            for item in sub.iterdir():
                shutil.move(str(item), str(target))
            shutil.rmtree(sub)
    archive.unlink()
    print(f"  [ok] {info['name']} ready at {target}")


def download_aug_dataset(key: str) -> None:
    info = AUG_DATASETS[key]
    ext = "zip" if info["extract_mode"] == "zip" else "tar.gz"
    archive = RAW / f"{key}.{ext}"
    download(info["url"], archive, info["name"])
    target = RAW / key
    if not target.exists():
        if info["extract_mode"] == "zip":
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(RAW)
        else:
            with tarfile.open(archive) as tf:
                tf.extractall(RAW)
        # MUSAN extracts to musan/, RIRs to RIRS_NOISES/
        if key == "musan":
            (RAW / "musan").rename(target)
        elif key == "rirs":
            (RAW / "RIRS_NOISES").rename(target)
    else:
        print(f"  [skip] {info['name']} already extracted")


# ── main ────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Download and prepare AuralGuard datasets")
    ap.add_argument("--all", action="store_true", help="download everything")
    ap.add_argument("--asvspoof", action="store_true", help="download ASVspoof 2019 LA")
    ap.add_argument("--aug", action="store_true", help="download MUSAN + RIRs")
    ap.add_argument("--zero-shot", action="store_true", help="download zero-shot eval datasets")
    ap.add_argument("--asvspoof-url", help="direct download URL for ASVspoof 2019 LA")
    ap.add_argument("--manifest-only", action="store_true", help="build manifests from existing files")
    args = ap.parse_args()

    if args.manifest_only:
        subprocess.check_call([sys.executable, "scripts/build_manifests.py", "--all"])
        return

    RAW.mkdir(parents=True, exist_ok=True)

    if args.all or args.asvspoof:
        zip_path = download_asvspoof(args.asvspoof_url)
        extract_asvspoof(zip_path)

    if args.all or args.zero_shot:
        for key in ZERO_SHOT_DATASETS:
            download_zero_shot_dataset(key)

    if args.all or args.aug:
        for key in AUG_DATASETS:
            download_aug_dataset(key)

    # Build manifests if any ASVspoof 2019 data was downloaded
    if args.all or args.asvspoof or args.zero_shot:
        print("  Building manifests …")
        subprocess.check_call([sys.executable, "scripts/build_manifests.py", "--all"])

    print("\nAll done! Now you can train with:\n"
          "  python scripts/train.py experiment=auralguard\n"
          "  python scripts/evaluate.py experiment=auralguard")


if __name__ == "__main__":
    main()
