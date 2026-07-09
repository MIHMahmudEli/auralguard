#!/usr/bin/env bash
# Download ASVspoof 2019 LA.
#
# The dataset is available at Edinburgh DataShare (DOI 10.7488/ds/2555).
# While the DataShare page is open-access, ASVspoof requests you accept their terms.
#
# Usage:
#   export ASVSPOOF2019_LA_URL="<direct-download-link>"
#   bash scripts/download/asvspoof2019_la.sh
#
# If you already accept the terms, the direct download URL is:
#   https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content
# Pass it via ASVSPOOF2019_LA_URL or the first argument.
set -euo pipefail

DEST="${1:-data/raw/ASVspoof2019_LA}"
mkdir -p "$DEST"

URL="${ASVSPOOF2019_LA_URL:-}"
if [ -z "$URL" ]; then
  echo "ERROR: Set ASVSPOOF2019_LA_URL to the direct download link."
  echo "  If you accept the terms, use:"
  echo "    https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content"
  echo "  Or fetch it from https://datashare.ed.ac.uk/handle/10283/3336"
  exit 1
fi

TMP_ZIP=$(mktemp).zip
echo "Downloading ASVspoof2019 LA (≈7.1 GB) …"
curl -L "$URL" -o "$TMP_ZIP"
echo "Extracting …"
unzip -q "$TMP_ZIP" -d "$(dirname "$DEST")"
# The zip creates LA/; move into expected directory structure
if [ -d "$(dirname "$DEST")/LA" ]; then
  mv "$(dirname "$DEST")/LA/"* "$DEST/"
  rmdir "$(dirname "$DEST")/LA"
fi
rm -f "$TMP_ZIP"
echo "Done. ASVspoof 2019 LA is ready at $DEST"
echo "Now run: python scripts/build_manifests.py --all"