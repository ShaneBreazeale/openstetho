"""Cut a 2-second mono WAV + mel-spectrogram PNG per selected cycle.

Outputs are written to a flat directory the annotation viewer can serve:

    <out_dir>/
        manifest.csv          # one row per clip: filename, score, stratum, ...
        clips/
            cycle_000123.wav  # 2-second audio centered on S2
            cycle_000123.png  # mel-spectrogram visualization

Cardiologists then load `manifest.csv` into the viewer, label each row, and
return the same CSV with a `label_s3` column appended (0 = no S3, 1 = S3,
9 = unannotatable).
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from .preprocess import SAMPLE_RATE, apply_s3_preset, load_audio, log_mel

log = logging.getLogger(__name__)

CLIP_PRE_S2_S = 0.7
CLIP_POST_S2_S = 1.3
CLIP_LEN_S = CLIP_PRE_S2_S + CLIP_POST_S2_S  # 2.0 s
CLIP_LEN_SAMPLES = int(CLIP_LEN_S * SAMPLE_RATE)


def export_clip(audio: np.ndarray, s2_idx: int) -> np.ndarray:
    pre = int(CLIP_PRE_S2_S * SAMPLE_RATE)
    post = int(CLIP_POST_S2_S * SAMPLE_RATE)
    start = s2_idx - pre
    end = s2_idx + post
    out = np.zeros(pre + post, dtype=np.float32)
    valid_start = max(0, start)
    valid_end = min(len(audio), end)
    if valid_end > valid_start:
        out[valid_start - start : valid_end - start] = audio[valid_start:valid_end]
    return out


def _try_save_mel_png(mel: np.ndarray, png_path: Path) -> None:
    """Optional mel-spec image — only writes if matplotlib is importable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 2.4), dpi=100)
    ax.imshow(mel.T, aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("frame")
    ax.set_ylabel("mel bin")
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def run(selection_csv: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    with selection_csv.open() as f:
        rows = list(csv.DictReader(f))

    manifest_rows: list[dict] = []
    audio_cache: dict[str, np.ndarray] = {}
    for i, row in enumerate(rows):
        wav = row["wav"]
        if wav not in audio_cache:
            try:
                audio_cache[wav] = load_audio(wav)
            except Exception as e:  # noqa: BLE001
                log.warning("skip %s: %s", wav, e)
                continue
            # Bounded cache to avoid OOM on large selections.
            if len(audio_cache) > 64:
                # Evict any one entry — Python 3.7+ dicts preserve order, drop oldest.
                oldest_key = next(iter(audio_cache))
                if oldest_key != wav:
                    audio_cache.pop(oldest_key)

        audio = audio_cache[wav]
        s2 = int(row["s2_idx"])
        clip = export_clip(audio, s2)

        slug = f"cycle_{i:06d}"
        wav_out = clips_dir / f"{slug}.wav"
        png_out = clips_dir / f"{slug}.png"
        sf.write(str(wav_out), clip, SAMPLE_RATE, subtype="PCM_16")

        mel = log_mel(apply_s3_preset(clip))
        _try_save_mel_png(mel, png_out)

        manifest_rows.append(
            {
                "slug": slug,
                "wav_file": f"clips/{slug}.wav",
                "png_file": f"clips/{slug}.png",
                "source_wav": wav,
                "cycle_no": row["cycle_no"],
                "s1_idx": row["s1_idx"],
                "s2_idx": row["s2_idx"],
                "next_s1_idx": row["next_s1_idx"],
                "score": row["score"],
                "stratum": row.get("stratum", ""),
            }
        )
        if (i + 1) % 50 == 0:
            log.info("exported %d/%d clips", i + 1, len(rows))

    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "slug",
                "wav_file",
                "png_file",
                "source_wav",
                "cycle_no",
                "s1_idx",
                "s2_idx",
                "next_s1_idx",
                "score",
                "stratum",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    log.info("wrote manifest %s (%d clips)", manifest, len(manifest_rows))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--selection", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    run(args.selection, args.out)


if __name__ == "__main__":
    main()
