#!/usr/bin/env bash
#
# Download the PASCAL Heart Sound Challenge 2011 dataset (Bentley, Nouri,
# Mannor, Coimbra). Two sub-corpora, two recording devices:
#
#   Dataset A — recorded with the iStethoscope iPhone app.
#               Labels: artifact / extrahs / murmur / normal.
#   Dataset B — recorded with a DigiScope electronic stethoscope.
#               Labels: extrasystole / murmur / normal.
#
# Publicly downloadable, but the reuse terms are not clear enough for
# redistribution or commercial model releases. Treat as research-only unless
# you have separate written rights.
#
# Output: data/pascal_2011/{A,B}/<class>/*.wav   (gitignored)
#
# Usage:
#   bash scripts/download_pascal_2011.sh

set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/data/pascal_2011"
mkdir -p "$DEST"
BASE_URL="http://istethoscope.peterjbentley.com/heartchallenge/wav"

ZIPS=(
    "Atraining_artifact.zip"
    "Atraining_extrahs.zip"
    "Atraining_murmur.zip"
    "Atraining_normal.zip"
    "Aunlabelledtest.zip"
    "Btraining_extrasystole.zip"
    "Btraining_murmur.zip"
    "Btraining_normal.zip"
    "Bunlabelledtest.zip"
)

if command -v wget >/dev/null 2>&1; then
    FETCH="wget -c -q --show-progress -P"
elif command -v curl >/dev/null 2>&1; then
    FETCH="_curl_to_dir"
    _curl_to_dir() {
        local dir="$1"; local url="$2"
        local fname; fname="$(basename "$url")"
        local tgt="$dir/$fname"
        if [ -f "$tgt" ]; then
            curl -sSL -o "$tgt" --range "$(stat -f%z "$tgt" 2>/dev/null || stat -c%s "$tgt")-" --continue-at - "$url" || true
        else
            curl -sSL -o "$tgt" "$url"
        fi
    }
else
    echo "ERROR: need wget or curl" >&2; exit 1
fi

cd "$DEST"
echo "→ downloading PASCAL 2011 to $DEST"

for z in "${ZIPS[@]}"; do
    if [ -f "$z" ]; then
        echo "   cached $z"
    else
        echo "   pull $z"
        if command -v wget >/dev/null 2>&1; then
            wget -c -q --show-progress "$BASE_URL/$z"
        else
            curl -sSL -o "$z" "$BASE_URL/$z" || { echo "FAILED $z" >&2; continue; }
        fi
    fi
done

echo
echo "→ extracting"
for z in "${ZIPS[@]}"; do
    [ -f "$z" ] || continue
    # Strip the leading "Atraining_" / "Btraining_" / "Aunlabelledtest" prefix
    # to derive a class name, route into Dataset A or B directory.
    set_letter="${z:0:1}"
    base="${z%.zip}"
    case "$base" in
        Atraining_*) cls="${base#Atraining_}";;
        Btraining_*) cls="${base#Btraining_}";;
        Aunlabelledtest|Bunlabelledtest) cls="unlabelled";;
        *) cls="$base";;
    esac
    out="$DEST/$set_letter/$cls"
    if [ -d "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
        echo "   skip $z (already extracted)"
        continue
    fi
    mkdir -p "$out"
    unzip -q -o -j "$z" -d "$out" || { echo "FAILED $z" >&2; continue; }
done

echo
echo "✓ done."
du -sh "$DEST"
echo
echo "WAV counts:"
for sub in "$DEST"/A "$DEST"/B; do
    [ -d "$sub" ] || continue
    set_letter="$(basename "$sub")"
    for cls in "$sub"/*/; do
        [ -d "$cls" ] || continue
        cls_name="$(basename "$cls")"
        n="$(ls "$cls"/*.wav 2>/dev/null | wc -l | tr -d ' ')"
        printf "  %s / %-14s  %s WAVs\n" "$set_letter" "$cls_name" "$n"
    done
done
