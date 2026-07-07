#!/usr/bin/env bash
# Download ASVspoof 2019 LA. You must FIRST accept the dataset terms on the
# Edinburgh DataShare / ASVspoof site, then this script fetches + extracts it.
set -euo pipefail

DEST="${1:-data/raw/ASVspoof2019_LA}"
mkdir -p "$DEST"

# The LA partition is distributed as a tarball on Edinburgh DataShare (DOI 10.7488/ds/2555).
# Set the URL after accepting terms:
URL="${ASVSPOOF2019_LA_URL:?set ASVSPOOF2019_LA_URL to the accepted download link}"

echo "Downloading ASVspoof2019 LA -> $DEST"
curl -L "$URL" -o /tmp/asvspoof2019_la.zip
unzip -q /tmp/asvspoof2019_la.zip -d "$DEST"
echo "Done. Now run: python scripts/build_manifests.py --all --root $DEST"
