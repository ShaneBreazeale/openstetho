# model/

Python training pipeline for the murmur-detector. Outputs a Core ML
`.mlpackage` consumed by `stetho-ui` / future iOS app via the Apple Neural
Engine.

The supported default training path uses CirCor 2022 only. Additional
dataset loaders are research tooling; do not publish weights trained on
PASCAL 2011 data without a separate license review.

Current exported models are experimental. Public training datasets were
recorded with different microphones, acoustic paths, gain/filter settings,
patient populations, and labeling protocols than compatible device audio
captured by this toolkit. Treat the pipeline as infrastructure for
developing and testing future matched-data models, not as a validated
classifier.

The `MurmurCNN` architecture has 74,129 trainable parameters. A PyTorch
`state_dict` checkpoint is about 300 KB; exported Core ML packages should
generally stay in the low single-digit MB range. Keep generated `.pt`,
`.mlpackage`, `.mlmodelc`, and `.onnx` files under `runs/` or release
assets, not in git.

## Setup

```bash
# from repo root
bash scripts/download_circor.sh         # ~3 GB
uv sync --project model                 # creates model/.venv, installs deps
```

Pins: `torch>=2.7,<2.8` (most recent coremltools-tested version), `coremltools>=8`, Python 3.12.

## Layout

```
model/
├── openstetho_model/
│   ├── __init__.py
│   ├── preprocess.py    # 4 kHz resample, cardiac biquads, mel-spec
│   ├── dataset.py       # CirCor 2022 PyTorch Dataset
│   └── bench_murmur.py  # CoreML/PyTorch/ONNX murmur benchmark
├── tests/
│   └── test_preprocess.py
├── pyproject.toml
└── README.md            # this file
```

## Pipeline shape

Matches `stetho-core::dsp` so the model sees the same features at
train time and inference time:

```
WAV (any sr, any channels)
  └─ load_audio()         → mono float32 @ 4 kHz
     └─ apply_cardiac()   → HP35 → HP55 → LP100 Butterworth (causal sosfilt)
        └─ split_windows()  → 4 s windows, 50 % overlap
           └─ log_mel()    → 62 frames × 32 mels, log10×10, z-score, -80 dB clip
```

`MEL_FFT_BINS = 65` and `f_max ≈ 1000 Hz` are intentional — they restrict
the model to the lower half of the FFT (heart-sound band) and match
`stetho-core::dsp::mel::MEL_FFT_BINS`.

## Tests

```bash
uv run --project model python -m pytest model/tests -q
```

## Murmur training

The default path trains per 4-second window. For the current deployment
shape, prefer recording-level multiple-instance training so the loss
matches recording/session-level aggregation:

```bash
uv run --project model python -m openstetho_model.train \
    --data data/circor \
    --level recording \
    --aggregation mean \
    --no-cardiac \
    --epochs 30 \
    --batch-size 16 \
    --out model/runs/murmur_recording_mean_v1
```

Use `--aggregation topk_mean --topk 3` when a murmur should be learned
from only the strongest few windows in each recording.

Experimental temporal-head training is available with `--architecture
cnn_bigru`. It is a PyTorch research path, not an exported ANE/Core ML
deployment path yet:

```bash
uv run --project model python -m openstetho_model.train \
    --data $CIRCOR_ROOT \
    --level recording \
    --architecture cnn_bigru \
    --aggregation mean \
    --no-cardiac \
    --epochs 5 \
    --batch-size 16 \
    --workers 0 \
    --device cpu \
    --out model/runs/murmur_cnn_bigru_mean_full_v1
```

The research path supports alternate feature and window experiments:

- `--feature-mode mfcc`: single-channel MFCC-only input
- `--feature-mode logmel128`: HearHeart-style research log-mel input
  with 128 mel bins, 25-2000 Hz, 50 ms Hamming windows, and 25 ms hop
- `--feature-mode multi`: stacked log-mel, MFCC, and log-STFT energy maps
- `--feature-mode scattering`: wavelet-scattering coefficients for the
  PyTorch-only `scattering_cnn1d` architecture
- `--architecture scattering_cnn1d`: small 1D-CNN for scattering features
- `--architecture scattering_transformer`: contextualized scattering
  baseline with fixed sinusoidal positional encoding and lightweight
  self-attention; use with `--feature-mode scattering`
- `--wide-features`: add a recording-level side branch with train-fold
  normalized patient metadata and recording statistics; currently requires
  `--level recording --architecture cnn_bigru`
- `--window-seconds 5`: 5-second windows with 50 percent overlap by default
- `--loss focal_bce`: focal BCE for sensitivity-weighted experiments
- `--loss asymmetric_focal`: asymmetric focal BCE for separate positive /
  negative focusing
- `--pos-weight-multiplier N`: multiply the class-balanced positive loss
  weight
- `--positive-sample-weight N`: replacement-sample positives more often
- `--audio-noise-snr-db 15 --audio-noise-prob 0.5`: add train-fold-only
  Gaussian audio noise
- `--train-random-crop`: train on one random window per recording while
  keeping validation/test aggregation on deterministic overlapped windows
- `--window-jitter-seconds N`: train-fold-only random crop/window-start
  jitter
- `--time-shift-seconds N --time-shift-prob P`: train-fold-only circular
  audio shift jitter
- `--freq-mask-max-width N --time-mask-max-width N`: train-fold-only
  SpecAugment-style masking after feature extraction
- `--select-metric {auc,f1,youden_j}`: choose the best CV checkpoint by a
  validation metric
- `--teacher-predictions-csv PATH`: optional recording-level soft targets
  for local distillation experiments; current docs label the local source as
  CirCor2022
- `--teacher-distill-weight N`: extra BCE weight applied where finite
  soft targets exist
- `--lr-scheduler plateau`: validation-AUC `ReduceLROnPlateau`
- `--early-stopping-patience N`: stop after N non-improving validation epochs

This is intentionally limited to PyTorch training/benchmarking for now.
The first full-CirCor 5-second MFCC-only CNN+BiGRU run improved best-F1
over the released 4-second log-mel checkpoint; the stacked multi-channel
run trailed both. Follow-up 5-fold patient-level CV is available with
`openstetho_model.cv_murmur`; it showed MFCC had stronger mean fold AUROC
but weaker pooled out-of-fold calibration than 5-second log-mel. The
initial wavelet-scattering + 1D-CNN run was a negative result in this
pipeline, with pooled out-of-fold AUROC near chance, so keep it as a
research branch rather than a model candidate.

The scattering follow-up should use the contextualized
`scattering_transformer` architecture, not the old plain scattering 1D-CNN:

```bash
uv run --project model python -m openstetho_model.cv_murmur \
    --data $CIRCOR_ROOT \
    --architecture scattering_transformer \
    --feature-mode scattering \
    --window-seconds 5 \
    --aggregation mean \
    --folds 5 \
    --epochs 8 \
    --batch-size 16 \
    --workers 0 \
    --device mps \
    --no-cardiac \
    --loss bce \
    --lr 3e-4 \
    --grad-clip-norm 5 \
    --select-metric f1 \
    --feature-cache-dir model/runs/feature_cache \
    --out model/runs/murmur_cv_scattering_transformer_5s_v1
```

Use `--feature-cache-dir` for scattering runs. Kymatio feature extraction is
the slow part; the cache stores deterministic per-recording feature tensors
and avoids recomputing them across folds or reruns. For quick integration
checks, add `--max-patients N` to run a stratified patient subset. A
10-patient / 2-fold / 1-epoch smoke completed successfully. The full
5-fold contextual scattering benchmark also completed, but it is not a lead
candidate: fold AUROC mean `0.711`, pooled OoF AUROC `0.699`, Platt AUROC
`0.691`, and Platt transferred best-F1 `0.412`.

`cv_murmur` also writes a `cross_fold_calibration` report. For each held-out
fold it fits Platt/isotonic probability calibration and best-F1 / Youden-J
thresholds on the other folds' out-of-fold predictions, then applies those
choices to the held-out fold. Use that section for calibrated Brier/ECE and
threshold-transfer metrics; the pooled `best_f1` threshold is optimistic.
The first combined focal / positive-weight / positive-sampling run improved
some raw sensitivity operating points but reduced pooled AUROC and
transferred-threshold F1. The clean baseline is the 5-second log-mel
CNN+BiGRU trained with standard BCE and selected by validation F1 or
Youden-J; both selection policies picked the same epochs in the seed-0
5-fold run and improved Platt-calibrated transferred best-F1 to `0.597`.
The current local research lead adds CirCor2022 positive-only soft targets
at `--teacher-distill-weight 0.2`: pooled OoF AUROC `0.829`, Platt AUROC
`0.823`, and Platt-calibrated transferred best-F1 `0.626` with sensitivity
`0.573` / specificity `0.934`. It is not a clean public recipe because the
CirCor2022 soft-target CSV covers only present-label recordings.

A full positive+negative CirCor2022 soft-target score file was also
generated under
`model/runs/murmur_bench/full_all_tflite_teacher_predictions.csv`, covering
all `2,964` present/absent recordings. The CirCor2022 soft-target source is
only a moderate CirCor ranker by itself, and a full-target soft-target +
positive-weight CV run hurt calibrated transfer badly. Do not promote
full-target distillation from that source.

HearHeart-style wide features are available for the current 5-second
CNN+BiGRU path. The vector includes age bucket, sex, pregnancy status,
height/weight with missing indicators, auscultation location, duration,
RMS, zero-crossing rate, spectral centroid, and spectral bandwidth. Scaling
is fit on each train fold only before validation:

```bash
uv run --project model python -m openstetho_model.cv_murmur \
    --data $CIRCOR_ROOT \
    --architecture cnn_bigru \
    --wide-features \
    --feature-mode logmel \
    --window-seconds 5 \
    --aggregation mean \
    --folds 5 \
    --epochs 8 \
    --batch-size 16 \
    --workers 0 \
    --device mps \
    --no-cardiac \
    --loss bce \
    --lr 3e-4 \
    --grad-clip-norm 5 \
    --select-metric f1 \
    --out model/runs/murmur_cv_logmel_5s_wide_v2
```

The full 5-fold run completed. It improved mean fold AUROC versus the
5-second log-mel lead (`0.844` vs `0.825`), but hurt pooled and
cross-fold-calibrated metrics: pooled OoF AUROC `0.776`, Platt AUROC
`0.741`, and Platt transferred best-F1 `0.499`. Do not promote the
wide-feature branch yet; treat it as evidence that the side features can
help within-fold ranking while making cross-fold score calibration worse.

HearHeart-style 15-second log-mel CV is now expressible without changing
the deployed mel path. Training uses one random 15-second crop per
recording; validation/test folds stay deterministic with overlapped
recording-level mean aggregation. Use MPS when available. The full 5-fold
run completed, but it was a negative result and should not replace the
5-second log-mel lead.

```bash
uv run --project model python -m openstetho_model.cv_murmur \
    --data $CIRCOR_ROOT \
    --architecture cnn_bigru \
    --feature-mode logmel128 \
    --window-seconds 15 \
    --hop-seconds 7.5 \
    --aggregation mean \
    --folds 5 \
    --epochs 8 \
    --batch-size 16 \
    --workers 0 \
    --device mps \
    --no-cardiac \
    --loss bce \
    --lr 1e-4 \
    --grad-clip-norm 5 \
    --select-metric f1 \
    --train-random-crop \
    --audio-noise-snr-db 15 \
    --audio-noise-prob 0.5 \
    --window-jitter-seconds 0.5 \
    --time-shift-seconds 0.5 \
    --time-shift-prob 0.5 \
    --freq-mask-max-width 16 \
    --time-mask-max-width 40 \
    --out model/runs/murmur_cv_logmel128_15s_aug_v3
```

Result summary for `murmur_cv_logmel128_15s_aug_v3`: fold AUROC mean
`0.520`, pooled OoF AUROC `0.500`, Platt AUROC `0.496`, and Platt
transferred best-F1 `0.325`. Treat this as a negative benchmark result.

```bash
uv run --project model python -m openstetho_model.cv_murmur \
    --data $CIRCOR_ROOT \
    --feature-mode mfcc \
    --window-seconds 5 \
    --folds 5 \
    --epochs 8 \
    --batch-size 16 \
    --workers 0 \
    --device cpu \
    --no-cardiac \
    --out model/runs/murmur_cv_mfcc_5s_v1
```

## Murmur benchmark

Use `bench_murmur` to compare the current exported murmur detector against
a PyTorch checkpoint and/or an ONNX baseline on the same CirCor patient
split. The ONNX model must accept the same log-mel tensor contract as
`MurmurCNN`: `(1, 1, 62, 32)` float32, one logit or probability output.
By default the benchmark skips the legacy cardiac filter to match the
current `stetho-ui` inference path; pass `--apply-cardiac` only when
benchmarking an older model trained with that filter.

```bash
uv run --project model python -m openstetho_model.bench_murmur \
    --data data/circor \
    --coreml model/runs/release-circor-v1/MurmurCNN.mlpackage \
    --onnx path/to/baseline.onnx \
    --split val \
    --out model/runs/murmur_bench/predictions.csv \
    --sweep-out model/runs/murmur_bench/threshold_sweep.csv \
    --json model/runs/murmur_bench/report.json
```

The JSON report includes threshold summaries for window-level scores and
recording-level aggregations (`max`, `mean`, `top3_mean` by default). The
optional sweep CSV writes every threshold point for plotting sensitivity /
specificity tradeoffs.

The benchmark also supports the 2025 Scientific Reports-style vote
experiment: overlapping 2.5-second windows, 1.25-second hop, and recording
positive if at least `k` windows exceed a window threshold. For the current
fixed-shape Core ML package, pass `--pad-to-model-window` because the
exported model expects 4-second `(1, 1, 62, 32)` features:

```bash
uv run --project model python -m openstetho_model.bench_murmur \
    --data $CIRCOR_ROOT \
    --coreml model/runs/release-circor-v1/MurmurCNN.mlpackage \
    --split all \
    --bench-window-seconds 2.5 \
    --bench-hop-seconds 1.25 \
    --max-recording-seconds 10 \
    --pad-to-model-window \
    --bandpass 25 700 \
    --vote-thresholds 0.3,0.4,0.49331352,0.5,0.6,0.7,0.8 \
    --vote-counts 1,2,3,4,5,6,7 \
    --json model/runs/murmur_bench/full_all_coreml_vote_2p5_bandpass25_700_report.json
```

See `docs/murmur_detector_benchmark.md` for the latest Core ML, vote-rule,
bandpass, CNN+BiGRU, multi-channel feature, MFCC CV, and scattering CV
results.

Current app-facing post-processing tune for the exported `MurmurCNN`: use
recording-level mean aggregation and threshold at `0.49331352`. This is the
running-session threshold currently wired into `stetho-ui`; the full
benchmark note above also records stricter best-F1 operating points for
offline comparison.

## Current roadmap — S3 validation + annotation

The old Phase 3 CreateML/Core ML/UI checklist is complete or superseded:
the repository now has a PyTorch S3 detector, Core ML export, and parallel
Murmur/S3 inference in `stetho-ui`. The remaining bottleneck is real
cycle-level S3 ground truth.

- [x] Train/export `S3CNN_v2` from synthetic S3 augmentation.
- [x] Package `MurmurCNN.mlpackage` and `S3CNN_v2.mlpackage` together for
      `stetho-ui`.
- [x] Wire `stetho-ui` to load the S3 model opportunistically next to the
      murmur model and display both probabilities.
- [x] Validate `runs/s3_circor_v10/best.pt` on small cardiologist-labeled
      public teaching libraries; see `docs/real_validation_results.md`.
- [x] Generate an initial v10 annotation export:
      `model/runs/s3_circor_v10/annotation_export`.
- [ ] Build the clinician annotation viewer described in
      `docs/s3_annotation_protocol.md`: audio playback, S1/S2 markers,
      mel image, and `label_s3 ∈ {0, 1, 9}` capture.
- [ ] Run a 60-cycle pilot with two independent raters, then compute kappa
      with `openstetho_model.compute_kappa`.
- [ ] Scale the stratified annotation export to the target batch size once
      the viewer/protocol are usable.
- [ ] Adjudicate disagreements and write `data/s3_circor_labels.csv`.
- [ ] Add a real-label S3 cycle dataset and fine-tune from the synthetic
      checkpoint.
