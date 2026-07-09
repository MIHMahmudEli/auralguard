#!/usr/bin/env bash
# Download zero-shot evaluation datasets + augmentation corpora.
# Usage: bash scripts/download/other_datasets.sh [data/raw]
set -euo pipefail

RAW="${1:-data/raw}"
mkdir -p "$RAW"

echo "================================================"
echo " In-the-Wild (Fraunhofer AISEC, Hugging Face)"
echo "================================================"
ITW_DIR="$RAW/in_the_wild"
if [ -d "$ITW_DIR" ] && [ -f "$ITW_DIR/meta.csv" ]; then
  echo "  [skip] already at $ITW_DIR"
else
  mkdir -p "$ITW_DIR"
  ZIP="$RAW/release_in_the_wild.zip"
  echo "  Downloading from Hugging Face …"
  curl -L "https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip" -o "$ZIP"
  echo "  Extracting …"
  unzip -q "$ZIP" -d "$RAW"
  # The zip creates a subdir; move contents
  if [ -d "$RAW/release_in_the_wild" ]; then
    mv "$RAW/release_in_the_wild/"* "$ITW_DIR/"
    rm -rf "$RAW/release_in_the_wild"
  fi
  rm -f "$ZIP"
  echo "  [ok] In-the-Wild ready at $ITW_DIR"
fi

echo "================================================"
echo " WaveFake (Zenodo)"
echo "================================================"
WF_DIR="$RAW/wavefake"
if [ -d "$WF_DIR" ] && ls "$WF_DIR"/*.wav >/dev/null 2>&1 || [ -d "$WF_DIR/generated_audio" ]; then
  echo "  [skip] already at $WF_DIR"
else
  mkdir -p "$WF_DIR"
  echo "  Downloading from Zenodo …"
  curl -L "https://zenodo.org/records/5642694/files/wavefake.zip" -o "$RAW/wavefake.zip"
  echo "  Extracting …"
  unzip -q "$RAW/wavefake.zip" -d "$RAW"
  if [ -d "$RAW/audio" ]; then
    mv "$RAW/audio/"* "$WF_DIR/"
    rm -rf "$RAW/audio"
  fi
  rm -f "$RAW/wavefake.zip"
  echo "  [ok] WaveFake ready at $WF_DIR"
fi

echo "================================================"
echo " MLAAD — download via Hugging Face datasets"
echo "================================================"
echo "  MLAAD requires Hugging Face authentication (free)."
echo "  Run the following Python snippet manually:"
echo ""
echo '    python -c "
# from datasets import load_dataset
# ds = load_dataset(\"mueller91/MLAAD\", split=\"train\")
# ds.save_to_disk(\"data/raw/mlaad\")
# '"
echo ""
echo "  Or clone with git-lfs:"
echo "    git lfs install"
echo "    git clone https://huggingface.co/datasets/mueller91/MLAAD data/raw/mlaad"
echo ""

echo "================================================"
echo " MUSAN (noise augmentation)"
echo "================================================"
bash scripts/download/musan.sh "$RAW"

echo "================================================"
echo " RIRs (reverb augmentation)"
echo "================================================"
bash scripts/download/rirs.sh "$RAW"

echo ""
echo "All done!"
echo "Now build manifests: python scripts/build_manifests.py --all"
