#!/usr/bin/env bash
# Download MUSAN corpus (noise augmentation) from OpenSLR.
# CC BY 4.0 — no registration required.
set -euo pipefail

DEST="${1:-data/raw}"
mkdir -p "$DEST"

echo "Downloading MUSAN (≈11 GB) …"
curl -L https://www.openslr.org/resources/17/musan.tar.gz -o "$DEST/musan.tar.gz"
echo "Extracting …"
tar xzf "$DEST/musan.tar.gz" -C "$DEST"
rm -f "$DEST/musan.tar.gz"
echo "Done. MUSAN is ready at $DEST/musan"