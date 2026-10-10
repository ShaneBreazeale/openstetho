#!/usr/bin/env bash
#
# Download HLS-CMDS v2 (Torabi, Shirani, Reilly — IEEE Data Descriptions
# 2025, doi:10.1109/IEEEDATA.2025.3566012): heart and lung sounds recorded
# from a clinical manikin with a 3M Littmann CORE digital stethoscope
# (Eko Core 2 internals), sent over Bluetooth to the Eko app, exported from
# the Eko cloud as 15 s, 4 kHz, 16-bit WAV.
#
# The only public dataset we know of that went through real Eko hardware.
# Caveats: the "patient" is a manikin speaker, and the Eko app applied its
# own filter before export, so this is not the raw ADPCM path stetho-ui sees.
#
#   HS.zip          50 heart-only clips (9 Normal)          labels: HS.csv
#   Mix{1,2,3}.zip  145 mixed clips + their 145 source heart
#                   and 145 source lung recordings (13 Normal heart)
#                                                            labels: Mix.csv
#   LS.zip          50 lung-only clips                       labels: LS.csv
#
# License: MIT (GitHub mirror). Source: https://github.com/Torabiy/HLS-CMDS
#
# Output: data/hls_cmds/{HS,LS,Mix}/*.wav + *.csv   (gitignored)
#
# Usage:
#   bash scripts/download_hls_cmds.sh

set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/data/hls_cmds"
mkdir -p "$DEST"
BASE_URL="https://raw.githubusercontent.com/Torabiy/HLS-CMDS/main/Dataset.v2"

ZIPS=(HS.zip LS.zip Mix1.zip Mix2.zip Mix3.zip)
CSVS=(HS.csv LS.csv Mix.csv)

command -v curl >/dev/null 2>&1 || { echo "ERROR: need curl" >&2; exit 1; }

cd "$DEST"
echo "→ downloading HLS-CMDS v2 to $DEST"
for f in "${ZIPS[@]}" "${CSVS[@]}"; do
    if [ -f "$f" ]; then
        echo "   cached $f"
    else
        echo "   pull $f"
        curl -fsSL -o "$f" "$BASE_URL/$f" || { echo "FAILED $f" >&2; rm -f "$f"; continue; }
    fi
done

echo
echo "→ extracting"
for z in "${ZIPS[@]}"; do
    [ -f "$z" ] || continue
    base="${z%.zip}"
    out="$DEST/${base%[0-9]}"          # Mix1/Mix2/Mix3 all land in Mix/
    mkdir -p "$out"
    unzip -q -o -j "$z" '*.wav' -d "$out" || { echo "FAILED $z" >&2; continue; }
done

echo
echo "✓ done."
du -sh "$DEST"
echo
echo "WAV counts:"
for sub in HS LS Mix; do
    n="$(ls "$DEST/$sub"/*.wav 2>/dev/null | wc -l | tr -d ' ')"
    printf "  %-4s %s WAVs\n" "$sub" "$n"
done
