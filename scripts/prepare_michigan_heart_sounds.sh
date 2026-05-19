#!/usr/bin/env bash
#
# Extract the University of Michigan Heart Sound & Murmur Library
# (Deep Blue item d9d03331-12f5-414e-9c19-3c5985bedb49, 23 labeled MP3s)
# and convert to mono 4 kHz WAV for the openstetho S3 detector validation
# pipeline.
#
# The clip filenames are self-labeling (e.g. `05_apex_s3_lld_bell.mp3`),
# so this script also builds `labels.csv` mapping each WAV to:
#   * `label_s3`   - 1 if the filename mentions s3, else 0
#   * `kind`       - short tag describing the clip (s3_alone, s4, click,
#                    opening_snap, split_s2, normal, murmur)
#
# Usage:
#   scripts/prepare_michigan_heart_sounds.sh \
#       ~/Downloads/medical_resources-heart_sound_and_murmur_library-April15.zip \
#       data/s3_validation/umich
set -euo pipefail

ZIP="${1:?path to Michigan zip}"
OUT="${2:-data/s3_validation/umich}"
mkdir -p "$OUT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

unzip -q -o "$ZIP" -d "$TMP"

for mp3 in "$TMP"/*.mp3; do
    [ -f "$mp3" ] || continue
    name="$(basename "$mp3" .mp3)"
    wav="$OUT/$name.wav"
    if [ ! -f "$wav" ]; then
        ffmpeg -nostdin -y -loglevel error -i "$mp3" \
            -ac 1 -ar 4000 \
            -filter:a "loudnorm=I=-20:LRA=11:TP=-1.5" \
            -sample_fmt s16 "$wav"
    fi
done

# Build labels.csv by parsing filenames.
{
    echo "filename,label_s3,kind"
    for wav in "$OUT"/*.wav; do
        name="$(basename "$wav")"
        # S3 positive if filename contains _s3 (but not _s3_ as part of "vsd" etc).
        if [[ "$name" =~ _s3[_\.] ]] || [[ "$name" =~ _s3$ ]]; then
            label=1
        else
            label=0
        fi
        # Tag the kind based on filename clues.
        if   [[ "$name" =~ _s3[^a-z] ]];                 then kind="s3"
        elif [[ "$name" =~ _s4[^a-z] ]];                 then kind="s4"
        elif [[ "$name" =~ split_s1 ]];                  then kind="split_s1"
        elif [[ "$name" =~ s?pilt_s2 || "$name" =~ split_s2 ]]; then kind="split_s2"
        elif [[ "$name" =~ click ]];                     then kind="click"
        elif [[ "$name" =~ os_ || "$name" =~ _os_ ]];    then kind="opening_snap"
        elif [[ "$name" =~ normal ]];                    then kind="normal"
        elif [[ "$name" =~ dias_mur ]];                  then kind="diastolic_murmur"
        elif [[ "$name" =~ sys_mur ]];                   then kind="systolic_murmur"
        elif [[ "$name" =~ single_s2 ]];                 then kind="single_s2"
        else                                                  kind="other"
        fi
        echo "$name,$label,$kind"
    done
} > "$OUT/labels.csv"

echo "wrote $OUT (clips + labels.csv)"
cat "$OUT/labels.csv"
