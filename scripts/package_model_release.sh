#!/usr/bin/env bash
#
# Package one or two Core ML `.mlpackage` directories as the GUI-downloadable
# release asset expected by stetho-ui.
#
# Usage:
#   scripts/package_model_release.sh \
#       model/runs/v1/MurmurCNN.mlpackage \
#       [model/runs/s3_circor_v10/S3CNN_v2.mlpackage]
#
# Output:
#   dist/MurmurCNN.mlpackage.zip
#       MurmurCNN.mlpackage/      (always present)
#       S3CNN_v2.mlpackage/       (only if the second arg is given)
#
# stetho-ui extracts the zip into its downloads directory and looks for
# both `MurmurCNN.mlpackage` and the optional `S3CNN_v2.mlpackage` sibling.

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 path/to/MurmurCNN.mlpackage [path/to/S3CNN_v2.mlpackage]" >&2
    exit 2
fi

MURMUR_SRC="$1"
S3_SRC="${2:-}"

if [ ! -d "$MURMUR_SRC" ]; then
    echo "error: '$MURMUR_SRC' is not a .mlpackage directory" >&2
    exit 1
fi
NAME="$(basename "$MURMUR_SRC")"
if [ "$NAME" != "MurmurCNN.mlpackage" ]; then
    echo "error: expected directory named MurmurCNN.mlpackage, got '$NAME'" >&2
    exit 1
fi

if [ -n "$S3_SRC" ]; then
    if [ ! -d "$S3_SRC" ]; then
        echo "error: S3 path '$S3_SRC' is not a .mlpackage directory" >&2
        exit 1
    fi
    S3_NAME="$(basename "$S3_SRC")"
    if [[ "$S3_NAME" != S3CNN*.mlpackage ]]; then
        echo "error: expected S3 dir named S3CNN*.mlpackage, got '$S3_NAME'" >&2
        exit 1
    fi
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST="$ROOT/dist"
OUT="$DIST/MurmurCNN.mlpackage.zip"
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT
mkdir -p "$DIST"
rm -f "$OUT"

cp -R "$MURMUR_SRC" "$STAGING/MurmurCNN.mlpackage"
if [ -n "$S3_SRC" ]; then
    cp -R "$S3_SRC" "$STAGING/$S3_NAME"
fi

(
    cd "$STAGING"
    if [ -n "$S3_SRC" ]; then
        zip -qry "$OUT" "MurmurCNN.mlpackage" "$S3_NAME"
    else
        zip -qry "$OUT" "MurmurCNN.mlpackage"
    fi
)

du -h "$OUT"
echo "upload this file to a GitHub release as: MurmurCNN.mlpackage.zip"
if [ -n "$S3_SRC" ]; then
    echo "(bundled both MurmurCNN.mlpackage and $S3_NAME)"
fi
