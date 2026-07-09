"""One-shot download + extract + manifest builder for all AuralGuard datasets.

Usage:
    python scripts/download_all.py                  # interactive prompts
    python scripts/download_all.py --asvspoof       # ASVspoof 2019 LA only
    python scripts/download_all.py --aug            # MUSAN + RIRs only
    python scripts/download_all.py --all            # everything
    python scripts/download_all.py --manifest-only  # build manifests from existing files

ASVspoof 2019 LA requires accepting terms at:
    https://datashare.ed.ac.uk/handle/10283/3336
After accepting, pass the direct download link via --asvspoof-url <URL>.
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

MUSAN_INFO = {
    "name": "MUSAN",
    "url": "https://www.openslr.org/resources/17/musan.tar.gz",
    "dir": RAW,
    "extract_mode": "tar",
}

RIR_INFO = {
    "name": "RIRs (SLR28)",
    "url": "https://www.openslr.org/resources/28/rirs_noises.zip",
    "dir": RAW,
    "extract_mode": "zip",
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
    print(f"  [extract] {zip_path.name} → {out}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW)
    # The zip creates LA/; move contents up
    la_dir = RAW / "LA"
    if la_dir.exists():
        for item in la_dir.iterdir():
            shutil.move(str(item), str(out))
        la_dir.rmdir()
    return out


def download_musan() -> None:
    archive = RAW / "musan.tar.gz"
    download(MUSAN_INFO["url"], archive, MUSAN_INFO["name"])
    target = RAW / "musan"
    if not target.exists():
        extract_tar(archive, RAW)


def download_rirs() -> None:
    archive = RAW / "rirs_noises.zip"
    download(RIR_INFO["url"], archive, RIR_INFO["name"])
    target = RAW / "RIRS_NOISES"
    if not target.exists():
        extract_zip(archive, RAW)


# ── main ────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Download and prepare AuralGuard datasets")
    ap.add_argument("--all", action="store_true", help="download everything")
    ap.add_argument("--asvspoof", action="store_true", help="download ASVspoof 2019 LA")
    ap.add_argument("--aug", action="store_true", help="download MUSAN + RIRs")
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
        print("  Building manifests …")
        subprocess.check_call([sys.executable, "scripts/build_manifests.py", "--all"])

    if args.all or args.aug:
        download_musan()
        download_rirs()

    print("\nAll done! Now you can train with:\n"
          "  python scripts/train.py experiment=auralguard\n"
          "  python scripts/evaluate.py experiment=auralguard")


if __name__ == "__main__":
    main()
