#!/usr/bin/env bash
#
# Fetch the public teaching-library clips used as held-out validation for the
# S3 detector. The actual hosting URLs change occasionally and some require a
# brief consent click, so this script does *not* download silently — it
# prints the canonical sources and the expected on-disk layout, then performs
# whatever direct downloads are stable.
#
# Usage:
#   scripts/download_s3_validation.sh [data/s3_validation]
#
# Expected layout after running this + filling in manual steps:
#
#   data/s3_validation/
#       michigan/
#           s3_gallop_<id>.wav
#           normal_<id>.wav
#           ...
#           labels.csv         # columns: filename,label_s3 (0|1)
#       texas/
#           <same>
#       README.md              # produced by this script
#
# All recordings must be mono, any sample rate (we resample to 4 kHz). Keep
# 16-bit PCM where possible to avoid float quantisation surprises.

set -euo pipefail

ROOT="${1:-data/s3_validation}"
MICH="$ROOT/michigan"
TEX="$ROOT/texas"

mkdir -p "$MICH" "$TEX"

cat > "$ROOT/README.md" <<'EOF'
# S3 detector — held-out teaching-library validation

Two open auscultation libraries provide a small but clinically-curated
held-out set of "S3 gallop" exemplars and matched negatives. Both ship as
short MP3 / WAV clips with cardiologist-confirmed labels.

## Michigan Heart Sound & Murmur Library

Source: <https://www.med.umich.edu/lrc/psb_open/repo/primer_heartsounds/primer_heartsounds.html>

The relevant categories for S3 detection are:

* `S3 Gallop` — positive (label_s3 = 1)
* `Normal` — negative (label_s3 = 0)
* `S4 Gallop` — *exclude* from S3 evaluation (different sound, but easy to
  confuse; revisit when we add an S4 head).

Download each MP3 from the site and save it under `michigan/` with a
descriptive filename. Convert to mono WAV with:

    ffmpeg -i input.mp3 -ac 1 -ar 4000 output.wav

Populate `michigan/labels.csv` with the filename and the binary S3 label.

## Texas Heart Institute auscultation tutorial

Source: <https://www.texasheart.org/heart-health/heart-information-center/>

Similar workflow. Use the same `labels.csv` schema in `texas/`.

## Labels schema

`labels.csv` columns:

    filename,label_s3
    s3_gallop_apex.wav,1
    normal_lub_dub.wav,0

If a clip clearly contains both S3 and an unrelated murmur, label it `1` —
the detector targets S3 presence, not exclusivity.

## Counting on these clips for what?

* They are *teaching* recordings: low-noise, mic-positioned by an expert.
  Performance here is an upper bound on what the detector achieves on
  real-world handheld recordings.
* Total sample size is small (tens of clips). Treat as a sanity gate, not
  the primary metric — the primary metric is the cardiologist-annotated
  CirCor subset (see docs/s3_annotation_protocol.md).
EOF

echo "wrote $ROOT/README.md"
echo
echo "Manual steps to complete:"
echo "  1. Open each library URL listed in $ROOT/README.md"
echo "  2. Save S3/normal/(optionally) S4 clips under michigan/ and texas/"
echo "  3. ffmpeg-convert to mono 4 kHz WAV"
echo "  4. Fill michigan/labels.csv and texas/labels.csv"
echo "  5. Run: uv run python -m openstetho_model.validate_clips \\"
echo "         --checkpoint runs/s3_v1/best.pt \\"
echo "         --labels $MICH/labels.csv $TEX/labels.csv"
