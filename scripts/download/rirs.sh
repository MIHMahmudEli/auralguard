#!/usr/bin/env bash
# Download Room Impulse Response dataset (SLR28) from OpenSLR.
# Apache 2.0 — no registration required.
set -euo pipefail

DEST="${1:-data/raw}"
mkdir -p "$DEST"

echo "Downloading RIRs (≈1.3 GB) …"
curl -L https://www.openslr.org/resources/28/rirs_noises.zip -o "$DEST/rirs_noises.zip"
echo "Extracting …"
unzip -q "$DEST/rirs_noises.zip" -d "$DEST"
rm -f "$DEST/rirs_noises.zip"
echo "Done. RIRs are ready at $DEST/RIRS_NOISES"