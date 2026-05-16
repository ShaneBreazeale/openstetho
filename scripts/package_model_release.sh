#!/usr/bin/env bash
#
# Package a Core ML `.mlpackage` directory as the GUI-downloadable release
# asset expected by stetho-ui.
#
# Usage:
#   scripts/package_model_release.sh model/runs/v1/MurmurCNN.mlpackage
#
# Output:
#   dist/MurmurCNN.mlpackage.zip

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 path/to/MurmurCNN.mlpackage" >&2
    exit 2
fi

SRC="$1"
if [ ! -d "$SRC" ]; then
    echo "error: '$SRC' is not a .mlpackage directory" >&2
    exit 1
fi

NAME="$(basename "$SRC")"
if [ "$NAME" != "MurmurCNN.mlpackage" ]; then
    echo "error: expected directory named MurmurCNN.mlpackage, got '$NAME'" >&2
    exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST="$ROOT/dist"
OUT="$DIST/MurmurCNN.mlpackage.zip"
mkdir -p "$DIST"
rm -f "$OUT"

(
    cd "$(dirname "$SRC")"
    zip -qry "$OUT" "$NAME"
)

du -h "$OUT"
echo "upload this file to a GitHub release as: MurmurCNN.mlpackage.zip"
