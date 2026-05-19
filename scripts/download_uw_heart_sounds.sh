#!/usr/bin/env bash
#
# Fetch the public University of Washington "Demonstrations: Heart Sounds &
# Murmurs" library and convert clips to mono 4 kHz WAV for the openstetho
# S3 detector validation pipeline.
#
# Source: https://depts.washington.edu/physdx/heart/demo.html
# Audio:  https://depts.washington.edu/physdx/audio/<name>.mp3
#
# Output:
#   data/s3_validation/uw/
#       *.wav
#       labels.csv     (filename, label_s3, kind)
#
# Usage:
#   scripts/download_uw_heart_sounds.sh [data/s3_validation/uw]

set -euo pipefail

OUT="${1:-data/s3_validation/uw}"
mkdir -p "$OUT"

BASE="https://depts.washington.edu/physdx/audio"
CLIPS=(
    "s31"        # S3 gallop
    "s41"        # S4 gallop (confounder)
    "splits21"   # split S2 (confounder)
    "normal"     # clean baseline
    "innocent"   # innocent murmur
    "ar" "as-early" "asd" "lateas" "mr" "ms" "pda" "ps" "vsd"
    "rub" "rub2" # pericardial rub
)

for c in "${CLIPS[@]}"; do
    mp3="$OUT/$c.mp3"
    wav="$OUT/$c.wav"
    if [ ! -f "$wav" ]; then
        echo "fetch $c.mp3"
        curl -fsSL "$BASE/$c.mp3" -o "$mp3"
        # 4 kHz mono PCM_16, peak-normalize so quiet teaching clips reach the
        # detector's expected level range.
        ffmpeg -nostdin -y -loglevel error -i "$mp3" -ac 1 -ar 4000 -filter:a "loudnorm=I=-20:LRA=11:TP=-1.5" -sample_fmt s16 "$wav"
        rm -f "$mp3"
    fi
done

# Label CSV — S3 positive only for s31. S4 / split-S2 are labeled 0 but
# tagged so we can compute confounder-specific false-positive rates later.
{
    echo "filename,label_s3,kind"
    echo "s31.wav,1,s3_gallop"
    echo "s41.wav,0,s4_gallop"
    echo "splits21.wav,0,split_s2"
    echo "normal.wav,0,normal"
    echo "innocent.wav,0,innocent_murmur"
    echo "ar.wav,0,aortic_regurgitation"
    echo "as-early.wav,0,aortic_stenosis_early"
    echo "asd.wav,0,asd"
    echo "lateas.wav,0,aortic_stenosis_late"
    echo "mr.wav,0,mitral_regurgitation"
    echo "ms.wav,0,mitral_stenosis"
    echo "pda.wav,0,patent_ductus_arteriosus"
    echo "ps.wav,0,pulmonic_stenosis"
    echo "vsd.wav,0,vsd"
    echo "rub.wav,0,pericardial_rub"
    echo "rub2.wav,0,pericardial_rub"
} > "$OUT/labels.csv"

echo
echo "wrote $OUT (16 clips + labels.csv)"
echo "next: uv run python -m openstetho_model.validate_clips \\"
echo "          --checkpoint runs/s3_circor_v10/best.pt \\"
echo "          --backbone s3cnn_v2 --crop-anchor s2 \\"
echo "          --labels $OUT/labels.csv"
