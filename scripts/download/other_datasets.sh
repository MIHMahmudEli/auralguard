#!/usr/bin/env bash
# Download stubs for the zero-shot evaluation + augmentation corpora.
# Each requires accepting the source's terms; fill the URLs / Zenodo record IDs.
set -euo pipefail

RAW="${1:-data/raw}"
mkdir -p "$RAW"

echo "== In-the-Wild (Fraunhofer AISEC, Zenodo) =="
# zenodo_get 10.5281/zenodo.XXXXXXX -o "$RAW/in_the_wild"

echo "== WaveFake (Zenodo) =="
# zenodo_get 10.5281/zenodo.5642694 -o "$RAW/wavefake"

echo "== MLAAD (multilingual, Zenodo) =="
# zenodo_get <record> -o "$RAW/mlaad"

echo "== MUSAN (noise) =="
bash scripts/download/musan.sh "$RAW"

echo "== RIRs (SLR28) =="
bash scripts/download/rirs.sh "$RAW"

echo "Fill the commented commands after accepting each dataset's terms (see docs/DATASETS.md)."
