# Murmur Detector Benchmark Notes

This note tracks the current murmur-detector benchmark direction after
adding recording-level training/eval, 2.5-second vote aggregation, optional
25-700 Hz bandpass preprocessing, and a CNN+BiGRU experimental backbone.

## Baseline

Current exported model:

```bash
model/runs/release-circor-v1/MurmurCNN.mlpackage
```

Full CirCor benchmark data:

```bash
$CIRCOR_ROOT
```

The current app-facing model still uses the fixed Core ML input contract
`(1, 1, 62, 32)`, corresponding to 4-second log-mel windows. Direct 2.5 s
spectrogram input fails against the exported package because it produces
`(1, 1, 39, 32)`. The 2.5 s vote benchmark therefore pads each 2.5 s audio
window back to the 4 s model input length before scoring. Treat that as a
post-processing experiment for the current model, not a faithful 2.5 s
model architecture.

## Current Core ML Result

Command:

```bash
uv run --project model python -m openstetho_model.bench_murmur \
    --data $CIRCOR_ROOT \
    --coreml model/runs/release-circor-v1/MurmurCNN.mlpackage \
    --split all \
    --label-filter all \
    --threshold 0.49331352 \
    --recording-aggregates mean top3_mean max \
    --out model/runs/murmur_bench/full_all_coreml_predictions.csv \
    --sweep-out model/runs/murmur_bench/full_all_coreml_sweep.csv \
    --json model/runs/murmur_bench/full_all_coreml_report.json
```

Best recording-level mean operating point:

| model | aggregation | threshold | AUROC | sensitivity | specificity | F1 |
|---|---|---:|---:|---:|---:|---:|
| CoreML MurmurCNN | mean | 0.586 | 0.831 | 0.536 | 0.965 | 0.641 |

The app currently uses recording/session mean aggregation with threshold
`0.49331352`, which is more sensitivity-oriented than the best-F1 point.

## 2.5 s Vote Aggregation

Inspired by the Scientific Reports 2025 murmur-classification paper's
2.5 s / 1.25 s overlapping-window voting setup, the benchmark now supports:

- `--bench-window-seconds 2.5`
- `--bench-hop-seconds 1.25`
- `--max-recording-seconds 10`
- `--vote-thresholds ...`
- `--vote-counts ...`
- `--bandpass 25 700`

Unfiltered padded 2.5 s vote command:

```bash
uv run --project model python -m openstetho_model.bench_murmur \
    --data $CIRCOR_ROOT \
    --coreml model/runs/release-circor-v1/MurmurCNN.mlpackage \
    --split all \
    --label-filter all \
    --threshold 0.49331352 \
    --bench-window-seconds 2.5 \
    --bench-hop-seconds 1.25 \
    --max-recording-seconds 10 \
    --pad-to-model-window \
    --recording-aggregates mean top3_mean max \
    --vote-thresholds 0.3,0.4,0.49331352,0.5,0.6,0.7,0.8 \
    --vote-counts 1,2,3,4,5,6,7 \
    --out model/runs/murmur_bench/full_all_coreml_vote_2p5_predictions.csv \
    --sweep-out model/runs/murmur_bench/full_all_coreml_vote_2p5_sweep.csv \
    --json model/runs/murmur_bench/full_all_coreml_vote_2p5_report.json
```

Best vote-count result:

| model | window threshold | min positive windows | AUROC | sensitivity | specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|
| CoreML MurmurCNN | 0.600 | 5 | 0.802 | 0.563 | 0.916 | 0.596 |

With offline zero-phase 25-700 Hz bandpass:

```bash
uv run --project model python -m openstetho_model.bench_murmur \
    --data $CIRCOR_ROOT \
    --coreml model/runs/release-circor-v1/MurmurCNN.mlpackage \
    --split all \
    --label-filter all \
    --threshold 0.49331352 \
    --bench-window-seconds 2.5 \
    --bench-hop-seconds 1.25 \
    --max-recording-seconds 10 \
    --pad-to-model-window \
    --bandpass 25 700 \
    --recording-aggregates mean top3_mean max \
    --vote-thresholds 0.3,0.4,0.49331352,0.5,0.6,0.7,0.8 \
    --vote-counts 1,2,3,4,5,6,7 \
    --out model/runs/murmur_bench/full_all_coreml_vote_2p5_bandpass25_700_predictions.csv \
    --sweep-out model/runs/murmur_bench/full_all_coreml_vote_2p5_bandpass25_700_sweep.csv \
    --json model/runs/murmur_bench/full_all_coreml_vote_2p5_bandpass25_700_report.json
```

Best vote-count result:

| model | preprocessing | window threshold | min positive windows | AUROC | sensitivity | specificity | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| CoreML MurmurCNN | 25-700 Hz bandpass | 0.700 | 4 | 0.817 | 0.543 | 0.937 | 0.607 |

Interpretation: the bandpass helped the padded 2.5 s path slightly, but
neither vote-count setup beat the existing 4 s recording-mean Core ML
benchmark. Vote aggregation is still useful as a benchmark dimension, but
the current exported model was not trained for 2.5 s inputs.

## CNN+BiGRU Experiment

The new `MurmurCNNBiGRU` backbone keeps a CNN frontend but preserves the
time axis for a small bidirectional GRU head. This is an offline PyTorch
experiment, not an ANE/Core ML deployment architecture yet.

Training command:

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

Best checkpoint:

```bash
model/runs/murmur_cnn_bigru_mean_full_v1/best.pt
```

`best_meta.json`:

```json
{"epoch": 3, "val_auc": 0.8600583090379008, "level": "recording", "architecture": "cnn_bigru", "aggregation": "mean", "topk": 3}
```

Benchmark command:

```bash
uv run --project model python -m openstetho_model.bench_murmur \
    --data $CIRCOR_ROOT \
    --checkpoint model/runs/murmur_cnn_bigru_mean_full_v1/best.pt \
    --architecture cnn_bigru \
    --split all \
    --label-filter all \
    --threshold 0.5 \
    --recording-aggregates mean top3_mean max \
    --out model/runs/murmur_bench/full_all_cnn_bigru_mean_full_v1_predictions.csv \
    --sweep-out model/runs/murmur_bench/full_all_cnn_bigru_mean_full_v1_sweep.csv \
    --json model/runs/murmur_bench/full_all_cnn_bigru_mean_full_v1_report.json
```

Best recording-level mean operating point:

| model | aggregation | threshold | AUROC | sensitivity | specificity | F1 |
|---|---|---:|---:|---:|---:|---:|
| CNN+BiGRU | mean | 0.682 | 0.868 | 0.634 | 0.956 | 0.702 |

This is the first run in this thread that materially improves on the
current Core ML benchmark. It should still be treated as an experimental
checkpoint until it is evaluated with patient-level cross-validation and an
export/deployment plan.

## Multi-Channel Feature Experiment

The FLAIRS PCG representation paper motivated a PyTorch-only research path
that stacks three feature maps as model input:

- log-mel, matching the current baseline channel
- MFCC, derived from the same 32-band mel frames
- log-STFT energy, pair-averaged from the lower 64 linear-frequency bins

Training used the same patient-level split, recording-level mean
aggregation, CNN+BiGRU architecture, and 5-epoch budget as the released
log-mel-only checkpoint:

```bash
uv run --project model python -m openstetho_model.train \
    --data $CIRCOR_ROOT \
    --level recording \
    --architecture cnn_bigru \
    --feature-mode multi \
    --aggregation mean \
    --no-cardiac \
    --epochs 5 \
    --batch-size 16 \
    --workers 0 \
    --device cpu \
    --out model/runs/murmur_cnn_bigru_multi_full_v1
```

`best_meta.json`:

```json
{"epoch": 3, "val_auc": 0.7886464930799906, "level": "recording", "architecture": "cnn_bigru", "feature_mode": "multi", "input_channels": 3, "aggregation": "mean", "topk": 3}
```

Benchmark command:

```bash
uv run --project model python -m openstetho_model.bench_murmur \
    --data $CIRCOR_ROOT \
    --checkpoint model/runs/murmur_cnn_bigru_multi_full_v1/best.pt \
    --architecture cnn_bigru \
    --feature-mode multi \
    --split all \
    --label-filter all \
    --threshold 0.5 \
    --recording-aggregates mean top3_mean max \
    --out model/runs/murmur_bench/full_all_cnn_bigru_multi_full_v1_predictions.csv \
    --sweep-out model/runs/murmur_bench/full_all_cnn_bigru_multi_full_v1_sweep.csv \
    --json model/runs/murmur_bench/full_all_cnn_bigru_multi_full_v1_report.json
```

Result against the released log-mel CNN+BiGRU:

| model | feature mode | aggregation | AUROC | best-F1 threshold | sensitivity | specificity | F1 |
|---|---|---|---:|---:|---:|---:|---:|
| CNN+BiGRU | logmel | mean | 0.868 | 0.682 | 0.634 | 0.956 | 0.702 |
| CNN+BiGRU | multi | mean | 0.777 | 0.336 | 0.583 | 0.858 | 0.545 |
| CNN+BiGRU | multi | top3_mean | 0.766 | 0.415 | 0.546 | 0.873 | 0.535 |
| CNN+BiGRU | multi | max | 0.763 | 0.492 | 0.515 | 0.896 | 0.536 |

Interpretation: simply stacking MFCC and linear-STFT energy did not improve
this architecture. The added channels likely need either different
normalization, channel dropout/regularization, a wider frontend, or
cross-validation before revisiting. Keep the released log-mel CNN+BiGRU as
the stronger benchmark.

## 5-Second MFCC Experiment

The Tsai et al. CapsNet paper is not directly comparable to this CirCor
murmur-present benchmark: it uses MFCC spectrum images, 5-second segments,
and a normal/abnormal framing. The useful transfer is the representation
and window-length choice, not the full CapsNet architecture. The PyTorch
research path now supports both:

- `--window-seconds 5` with 50 percent overlap by default
- `--feature-mode mfcc` for a single-channel MFCC-only model
- `--lr-scheduler plateau` for validation-AUC `ReduceLROnPlateau`
- `--early-stopping-patience N`

Training commands:

```bash
uv run --project model python -m openstetho_model.train \
    --data $CIRCOR_ROOT \
    --level recording \
    --architecture cnn_bigru \
    --feature-mode logmel \
    --window-seconds 5 \
    --aggregation mean \
    --no-cardiac \
    --epochs 8 \
    --batch-size 16 \
    --workers 0 \
    --device cpu \
    --lr-scheduler plateau \
    --plateau-patience 1 \
    --plateau-factor 0.5 \
    --early-stopping-patience 2 \
    --out model/runs/murmur_cnn_bigru_logmel_5s_v1

uv run --project model python -m openstetho_model.train \
    --data $CIRCOR_ROOT \
    --level recording \
    --architecture cnn_bigru \
    --feature-mode mfcc \
    --window-seconds 5 \
    --aggregation mean \
    --no-cardiac \
    --epochs 8 \
    --batch-size 16 \
    --workers 0 \
    --device cpu \
    --lr-scheduler plateau \
    --plateau-patience 1 \
    --plateau-factor 0.5 \
    --early-stopping-patience 2 \
    --out model/runs/murmur_cnn_bigru_mfcc_5s_v1
```

Best checkpoints:

```json
{"epoch": 7, "val_auc": 0.832512315270936, "level": "recording", "architecture": "cnn_bigru", "feature_mode": "logmel", "input_channels": 1, "window_seconds": 5.0, "hop_seconds": 2.5, "lr_scheduler": "plateau", "early_stopping_patience": 2, "aggregation": "mean", "topk": 3}
{"epoch": 6, "val_auc": 0.8624040749304648, "level": "recording", "architecture": "cnn_bigru", "feature_mode": "mfcc", "input_channels": 1, "window_seconds": 5.0, "hop_seconds": 2.5, "lr_scheduler": "plateau", "early_stopping_patience": 2, "aggregation": "mean", "topk": 3}
```

Benchmark command shape:

```bash
uv run --project model python -m openstetho_model.bench_murmur \
    --data $CIRCOR_ROOT \
    --checkpoint model/runs/murmur_cnn_bigru_mfcc_5s_v1/best.pt \
    --architecture cnn_bigru \
    --feature-mode mfcc \
    --split all \
    --label-filter all \
    --threshold 0.5 \
    --bench-window-seconds 5 \
    --bench-hop-seconds 2.5 \
    --recording-aggregates mean top3_mean max \
    --out model/runs/murmur_bench/full_all_cnn_bigru_mfcc_5s_v1_predictions.csv \
    --sweep-out model/runs/murmur_bench/full_all_cnn_bigru_mfcc_5s_v1_sweep.csv \
    --json model/runs/murmur_bench/full_all_cnn_bigru_mfcc_5s_v1_report.json
```

Results:

| model | feature | window | aggregation | AUROC | best-F1 threshold | sensitivity | specificity | F1 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| CNN+BiGRU released | logmel | 4s | mean | 0.868 | 0.682 | 0.634 | 0.956 | 0.702 |
| CNN+BiGRU | logmel | 5s | mean | 0.902 | 0.535 | 0.761 | 0.898 | 0.705 |
| CNN+BiGRU | logmel | 5s | top3_mean | 0.895 | 0.925 | 0.658 | 0.947 | 0.706 |
| CNN+BiGRU | mfcc | 5s | mean | 0.882 | 0.758 | 0.663 | 0.969 | 0.744 |
| CNN+BiGRU | mfcc | 5s | top3_mean | 0.879 | 0.955 | 0.677 | 0.958 | 0.735 |
| CNN+BiGRU | mfcc | 5s | max | 0.875 | 0.987 | 0.650 | 0.955 | 0.713 |

Interpretation from this single split: the 5-second window is useful. The
5s log-mel model has the best ranking metric in this set (`AUROC=0.902`),
while the 5s MFCC model gives the strongest calibrated best-F1 operating
point (`F1=0.744`, specificity `0.969`). This needed cross-validation
before promoting either checkpoint.

## 5-Fold Patient-Level CV

The patient-level CV runner trains a separate model per fold, keeps every
patient wholly in train or validation, and writes both per-fold checkpoints
and pooled out-of-fold recording predictions.

Command:

```bash
uv run --project model python -m openstetho_model.cv_murmur \
    --data $CIRCOR_ROOT \
    --feature-mode mfcc \
    --window-seconds 5 \
    --folds 5 \
    --epochs 8 \
    --batch-size 16 \
    --workers 0 \
    --device mps \
    --no-cardiac \
    --lr-scheduler plateau \
    --plateau-patience 1 \
    --plateau-factor 0.5 \
    --early-stopping-patience 2 \
    --out model/runs/murmur_cv_mfcc_5s_v1
```

Wavelet scattering / 1D-CNN command shape, motivated by the KAUST
WST+1D-CNN paper:

```bash
uv run --project model python -m openstetho_model.cv_murmur \
    --data $CIRCOR_ROOT \
    --architecture scattering_cnn1d \
    --feature-mode scattering \
    --window-seconds 5 \
    --folds 5 \
    --epochs 8 \
    --batch-size 16 \
    --workers 0 \
    --device cpu \
    --no-cardiac \
    --lr-scheduler plateau \
    --plateau-patience 1 \
    --plateau-factor 0.5 \
    --early-stopping-patience 2 \
    --out model/runs/murmur_cv_scattering_5s_v1
```

CV artifacts:

- `model/runs/murmur_cv_logmel_5s_v1/cv_report.json`
- `model/runs/murmur_cv_logmel_5s_v1/oof_predictions.csv`
- `model/runs/murmur_cv_mfcc_5s_v1/cv_report.json`
- `model/runs/murmur_cv_mfcc_5s_v1/oof_predictions.csv`
- `model/runs/murmur_cv_scattering_5s_v1/cv_report.json`
- `model/runs/murmur_cv_scattering_5s_v1/oof_predictions.csv`
- `model/runs/murmur_cv_logmel_5s_focal_sens_v1/cv_report.json`
- `model/runs/murmur_cv_logmel_5s_focal_sens_v1/oof_predictions.csv`
- `model/runs/murmur_cv_logmel_5s_bce_select_f1_v1/cv_report.json`
- `model/runs/murmur_cv_logmel_5s_bce_select_f1_v1/oof_predictions.csv`
- `model/runs/murmur_cv_logmel_5s_bce_select_youden_v1/cv_report.json`
- `model/runs/murmur_cv_logmel_5s_bce_select_youden_v1/oof_predictions.csv`

Results:

| feature | mean fold AUROC | fold AUROC std | pooled OoF AUROC | OoF best-F1 threshold | sensitivity | specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| logmel 5s | 0.825 | 0.027 | 0.809 | 0.471 | 0.630 | 0.872 | 0.592 |
| mfcc 5s | 0.854 | 0.022 | 0.800 | 0.557 | 0.573 | 0.898 | 0.582 |
| scattering 5s + 1D-CNN | 0.522 | 0.069 | 0.517 | 0.298 | 0.998 | 0.002 | 0.340 |
| logmel 5s + focal sensitivity tune | 0.783 | 0.023 | 0.748 | 0.790 | 0.408 | 0.938 | 0.494 |
| logmel 5s + BCE F1/Youden selection | 0.825 | 0.016 | 0.817 | 0.822 | 0.538 | 0.940 | 0.607 |

Fold AUROCs:

| feature | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 |
|---|---:|---:|---:|---:|---:|
| logmel 5s | 0.825 | 0.874 | 0.805 | 0.824 | 0.796 |
| mfcc 5s | 0.881 | 0.866 | 0.849 | 0.862 | 0.815 |
| scattering 5s + 1D-CNN | 0.565 | 0.469 | 0.462 | 0.637 | 0.476 |
| logmel 5s + focal sensitivity tune | 0.744 | 0.812 | 0.774 | 0.800 | 0.785 |
| logmel 5s + BCE F1/Youden selection | 0.831 | 0.849 | 0.819 | 0.822 | 0.802 |

Interpretation from CV: MFCC is more consistent fold-by-fold and has the
better mean validation AUROC, but pooled out-of-fold AUROC/F1 does not beat
5s log-mel. That means the single-split MFCC operating point was optimistic
and fold-specific score calibration is a problem. The next useful step is
not another single split; it is per-fold threshold calibration, probability
calibration, or a training objective that gives better cross-fold score
alignment.

The CV runner now also reports deployable cross-fold calibration metrics.
For each held-out fold, it fits threshold and probability calibration only
on the other folds' out-of-fold predictions, then applies those choices to
the held-out fold. This is stricter than the pooled `best_f1` rows above,
which pick a single threshold after seeing all out-of-fold labels.

Leave-one-fold-out probability calibration and threshold transfer:

| feature | probability view | AUROC | Brier | ECE-10 | transferred best-F1 F1 | sensitivity | specificity | mean threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| logmel 5s | raw | 0.809 | 0.134 | 0.121 | 0.569 | 0.568 | 0.890 | 0.543 |
| logmel 5s | Platt calibrated | 0.803 | 0.114 | 0.036 | 0.569 | 0.568 | 0.890 | 0.308 |
| mfcc 5s | raw | 0.800 | 0.145 | 0.143 | 0.566 | 0.574 | 0.883 | 0.549 |
| mfcc 5s | Platt calibrated | 0.780 | 0.127 | 0.029 | 0.566 | 0.574 | 0.883 | 0.275 |
| scattering 5s + 1D-CNN | raw | 0.517 | 0.254 | 0.296 | 0.325 | 0.800 | 0.196 | 0.323 |
| scattering 5s + 1D-CNN | Platt calibrated | 0.473 | 0.164 | 0.015 | 0.319 | 0.782 | 0.199 | 0.187 |
| logmel 5s + BCE F1/Youden selection | Platt calibrated | 0.814 | 0.110 | 0.040 | 0.597 | 0.525 | 0.940 | 0.437 |
| logmel 5s + BCE F1/Youden selection | isotonic calibrated | 0.805 | 0.109 | 0.021 | 0.597 | 0.525 | 0.940 | 0.368 |

Platt calibration improves Brier score and ECE for the 5s log-mel and MFCC
models, but it does not recover sensitivity/specificity tradeoff by itself.
Because the threshold is selected per fold on the other folds, monotonic
probability calibration keeps the best-F1 decisions effectively unchanged
for log-mel and MFCC. The useful calibrated headline is therefore:
log-mel remains slightly ahead (`F1=0.569`, sensitivity `0.568`,
specificity `0.890`) and has the better pooled ranking after calibration.

Checkpoint selection by validation F1 or Youden-J with the standard BCE
loss is a positive result. Both selection metrics picked the same best epoch
per fold in the seed-0 run, so the resulting reports are identical:

```bash
uv run --project model python -m openstetho_model.cv_murmur \
    --data $CIRCOR_ROOT \
    --architecture cnn_bigru \
    --feature-mode logmel \
    --window-seconds 5 \
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
    --lr-scheduler plateau \
    --plateau-patience 1 \
    --plateau-factor 0.5 \
    --early-stopping-patience 2 \
    --out model/runs/murmur_cv_logmel_5s_bce_select_f1_v1
```

Compared with the earlier AUROC-selected 5s log-mel baseline, this improves
pooled OoF AUROC (`0.817` vs `0.809`) and deployable Platt-calibrated
best-F1 transfer (`0.597` vs `0.569`). The specificity-oriented operating
point is also better with isotonic calibration: F1 `0.601`, sensitivity
`0.545`, specificity `0.931`. The high-sensitivity operating point is still
not good enough for screening: Platt transfer gets sensitivity `0.794`,
specificity `0.642`, F1 `0.498`.

The sensitivity-weighted run used focal BCE, 1.5x positive loss weight,
1.5x positive replacement sampling, and F1-based checkpoint selection:

```bash
uv run --project model python -m openstetho_model.cv_murmur \
    --data $CIRCOR_ROOT \
    --architecture cnn_bigru \
    --feature-mode logmel \
    --window-seconds 5 \
    --folds 5 \
    --epochs 8 \
    --batch-size 16 \
    --workers 0 \
    --device cpu \
    --no-cardiac \
    --loss focal_bce \
    --pos-weight-multiplier 1.5 \
    --positive-sample-weight 1.5 \
    --select-metric f1 \
    --lr-scheduler plateau \
    --plateau-patience 1 \
    --plateau-factor 0.5 \
    --early-stopping-patience 2 \
    --out model/runs/murmur_cv_logmel_5s_focal_sens_v1
```

This was a negative tuning result. It increased raw threshold-0.5
sensitivity on several folds, but transferred-threshold performance got
worse than the baseline 5s log-mel model:

| model | probability view | AUROC | Brier | ECE-10 | policy | F1 | sensitivity | specificity |
|---|---|---:|---:|---:|---|---:|---:|---:|
| logmel 5s baseline | Platt calibrated | 0.803 | 0.114 | 0.036 | best-F1 transfer | 0.569 | 0.568 | 0.890 |
| logmel 5s focal tune | Platt calibrated | 0.739 | 0.136 | 0.038 | best-F1 transfer | 0.410 | 0.498 | 0.760 |
| logmel 5s baseline | Platt calibrated | 0.803 | 0.114 | 0.036 | sensitivity >= 0.80 transfer | 0.451 | 0.800 | 0.551 |
| logmel 5s focal tune | Platt calibrated | 0.739 | 0.136 | 0.038 | sensitivity >= 0.80 transfer | 0.431 | 0.807 | 0.503 |
| logmel 5s baseline | Platt calibrated | 0.803 | 0.114 | 0.036 | specificity >= 0.90 transfer | 0.578 | 0.568 | 0.898 |
| logmel 5s focal tune | Platt calibrated | 0.739 | 0.136 | 0.038 | specificity >= 0.90 transfer | 0.476 | 0.454 | 0.883 |

The focal run can force the sensitivity target, but only by giving up too
much specificity and ranking. Do not promote it over the baseline. If
revisiting sensitivity weighting, try one knob at a time: checkpoint
selection by F1/Youden without focal loss, or a smaller positive-sampling
weight, before combining both.

## CirCor2022 Soft-Target Sensitivity Sweep

The training path now accepts recording-level soft targets via the generic
`--teacher-predictions-csv` distillation interface. The local artifact is
labeled CirCor2022: the first pass used the existing CirCor2022 soft-target
CSV for murmur-present recordings only, so it should be treated as a
positive-only recall-bias experiment rather than clean public distillation.
CirCor2022 soft-target coverage for that file is `606` present recordings
and `2358` recordings without soft targets.

Command shape for the best sweep point:

```bash
uv run --project model python -m openstetho_model.cv_murmur \
    --data $CIRCOR_ROOT \
    --architecture cnn_bigru \
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
    --teacher-predictions-csv model/runs/murmur_bench/full_present_coreml_vs_teacher_predictions.csv \
    --teacher-prob-column onnx_prob \
    --teacher-aggregation max \
    --teacher-distill-weight 0.2 \
    --feature-cache-dir model/runs/feature_cache \
    --out model/runs/murmur_cv_logmel_5s_teacher_posdistill_v1
```

Soft-target-weight sweep, all 5-fold patient-level CV with 5-second
log-mel CNN+BiGRU, BCE, recording-level mean aggregation, and fold-held-out
calibrated threshold transfer:

| run | fold AUROC mean | pooled OoF AUROC | Platt AUROC | Platt best-F1 | sensitivity | specificity | isotonic spec>=0.90 sensitivity | specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no soft-target BCE F1/Youden lead | 0.825 | 0.817 | 0.814 | 0.597 | 0.525 | 0.940 | 0.545 | 0.931 |
| CirCor2022 positive-only w=0.05 | 0.845 | 0.809 | 0.786 | 0.477 | 0.523 | 0.828 | 0.533 | 0.824 |
| CirCor2022 positive-only w=0.10 | 0.836 | 0.821 | 0.815 | 0.557 | 0.584 | 0.868 | 0.561 | 0.914 |
| CirCor2022 positive-only w=0.20 | 0.841 | 0.829 | 0.823 | 0.626 | 0.573 | 0.934 | 0.612 | 0.906 |
| CirCor2022 positive-only w=0.30 | 0.844 | 0.815 | 0.805 | 0.566 | 0.558 | 0.894 | 0.571 | 0.885 |
| CirCor2022 positive-only w=0.50 | 0.837 | 0.807 | 0.797 | 0.617 | 0.573 | 0.927 | 0.586 | 0.916 |

The only CirCor2022 soft-target weight worth keeping is `0.2`. It improves
pooled OoF AUROC (`0.829` vs `0.817`), Platt AUROC (`0.823` vs `0.814`),
Platt transferred best-F1 (`0.626` vs `0.597`), and the isotonic
specificity-target sensitivity (`0.612` vs `0.545`). It does not materially
improve the high-sensitivity target: Platt sensitivity>=0.80 transfer is
`0.792` sensitivity / `0.656` specificity, essentially the same recall as
the no-soft-target lead with slightly better specificity.

Two sensitivity-weighted CirCor2022 soft-target follow-ups were negative:

| run | Platt AUROC | Platt best-F1 | sensitivity | specificity | Platt spec>=0.90 sensitivity | specificity |
|---|---:|---:|---:|---:|---:|---:|
| CirCor2022 positive-only w=0.20 + pos_weight 1.25 + Youden selection | 0.805 | 0.534 | 0.540 | 0.876 | 0.594 | 0.831 |
| CirCor2022 full soft-target w=0.10 + pos_weight 1.25 + Youden selection | 0.707 | 0.439 | 0.391 | 0.900 | 0.452 | 0.871 |

The full positive+negative CirCor2022 soft-target score file now exists at:

```text
model/runs/murmur_bench/full_all_tflite_teacher_predictions.csv
```

It covers `29,474` 4-second windows across all `2,964` present/absent
recordings, but the CirCor2022 soft-target source is only a moderate CirCor
ranker by itself: recording-level max aggregation over `teacher_max` gives
AUROC `0.692`, best-F1 `0.451`, sensitivity `0.517`, and specificity
`0.801`. Full-target distillation therefore pulled the student in the wrong
direction. Do not promote full-target distillation from this source. Keep
the positive-only `w=0.20` run as the current research lead, with the caveat
that it is a local CirCor2022-assisted experiment and not a clean public
training recipe.

The simple wavelet-scattering implementation is a negative result in this
pipeline. It uses Kymatio WST coefficients (`J=8`, `Q=4`) from 5-second
windows and a small 1D-CNN, but pooled OoF AUROC is near chance. Do not
pursue this branch as standard scattering. The arXiv Scattering Transformer
paper reports that its own plain WSN baseline was weak (`W.acc=0.481`,
`UAR=0.46`) and that the gain came from parameter-free positional encoding
and self-attention over scattering features (`W.acc=0.786`, `UAR=0.697`).
The codebase now has a separate `scattering_transformer` architecture for
that contextualized path: scattering coefficients are projected into a small
token space, receive fixed sinusoidal positional encoding, pass through
lightweight self-attention, and are mean-pooled before classification.
Treat it as a new lightweight comparator, not a retune of the current
1D-CNN run.

Command shape:

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

Kymatio feature extraction is the bottleneck, so `cv_murmur` now supports
`--feature-cache-dir` for deterministic per-recording features. The same
runner also supports `--max-patients N` for stratified smoke runs before a
full CV benchmark.

Smoke command:

```bash
uv run --project model python -m openstetho_model.cv_murmur \
    --data $CIRCOR_ROOT \
    --architecture scattering_transformer \
    --feature-mode scattering \
    --window-seconds 5 \
    --aggregation mean \
    --folds 2 \
    --max-patients 10 \
    --epochs 1 \
    --batch-size 4 \
    --workers 0 \
    --device cpu \
    --no-cardiac \
    --loss bce \
    --lr 3e-4 \
    --grad-clip-norm 5 \
    --select-metric f1 \
    --feature-cache-dir model/runs/feature_cache \
    --out model/runs/murmur_cv_scattering_transformer_5s_smoke
```

The smoke completed and wrote `cv_report.json`. Its metrics are not a model
result because the subset has only 10 patients / 31 recordings, but it
proves that the contextualized scattering architecture, feature extraction,
recording aggregation, calibration reporting, and disk feature cache run
end-to-end.

The full contextual scattering benchmark also completed:

| run | fold AUROC mean | fold AUROC std | pooled OoF AUROC | pooled best-F1 | Platt AUROC | Platt best-F1 transfer |
|---|---:|---:|---:|---:|---:|---:|
| scattering transformer 5s | 0.711 | 0.058 | 0.699 | 0.448 | 0.691 | 0.412 |

This is much better than the old plain scattering + 1D-CNN branch, but it
is still well below the current 5-second log-mel CNN+BiGRU lead. Keep it as
a documented lightweight comparator, not a model candidate.

## HearHeart-Style Augmentation Path

The HearHeart/CinC 2022 review and preprint are a better fit for the
current CNN+BiGRU direction than the ViT/MiniROCKET paper. The transferable
pieces are 15-second windows, 128-bin mel spectrograms over 25-2000 Hz,
50 ms Hamming windows with 25 ms hop, 15 dB Gaussian noise,
SpecAugment-style time/frequency masking, and a side branch for wide
features. The research path now supports the 15-second mel/augmentation
pieces as `--feature-mode logmel128` plus `--train-random-crop` and
train-fold-only augmentation knobs; validation folds remain deterministic
and patient-disjoint with overlapped segment aggregation.

Command shape:

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

A 2-fold / 1-epoch MPS smoke run completed first as a runtime check. The
full 5-fold run then completed and wrote
`model/runs/murmur_cv_logmel128_15s_aug_v3/cv_report.json`.

| run | fold AUROC mean | fold AUROC std | pooled OoF AUROC | pooled best-F1 | Platt AUROC | Platt best-F1 transfer |
|---|---:|---:|---:|---:|---:|---:|
| logmel128 15s aug, 5 folds | 0.520 | 0.058 | 0.500 | 0.343 | 0.496 | 0.325 |

This is a negative result. The 15-second `logmel128` recipe runs, but it
does not improve ranking or transferred threshold metrics over the current
5-second log-mel CNN+BiGRU lead. The run required a manual stable BCE loss,
`--lr 1e-4`, and `--grad-clip-norm 5` for MPS stability; validation metrics
are usable, but the training-loss scalar can still be noisy on MPS and
should not be used for model selection.

### Wide-Feature Branch

The remaining useful HearHeart component is now wired as an optional
recording-level side branch. `--wide-features` adds a small MLP fed by
patient metadata and recording statistics, then concatenates that embedding
with the CNN+BiGRU pooled representation before classification. The feature
vector currently contains age bucket, sex, pregnancy status, height/weight
with missing indicators, auscultation location, duration, RMS,
zero-crossing rate, spectral centroid, and spectral bandwidth. Normalization
is fit on each training fold only and then reused for that fold's validation
set.

Command shape for the real benchmark:

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

The recording-level train/eval loop now scores all windows in a loader batch
with one model call, then splits logits back by recording before
mean/max/top-k aggregation. This keeps the decision rule unchanged while
making wide-feature CV practical.

A 2-fold / 1-epoch smoke run completed successfully after that optimization:

| run | fold AUROC mean | fold AUROC std | pooled OoF AUROC | pooled best-F1 | Platt AUROC | Platt best-F1 transfer |
|---|---:|---:|---:|---:|---:|---:|
| logmel 5s wide vectorized smoke, 2 folds x 1 epoch | 0.695 | 0.008 | 0.637 | 0.376 | 0.601 | 0.300 |

Treat this only as an integration check, not a model result.

The full 5-fold wide-feature benchmark then completed:

| run | fold AUROC mean | fold AUROC std | pooled OoF AUROC | pooled best-F1 | Platt AUROC | Platt best-F1 transfer |
|---|---:|---:|---:|---:|---:|---:|
| logmel 5s BCE F1/Youden lead | 0.825 | 0.016 | 0.817 | 0.607 | 0.814 | 0.597 |
| logmel 5s wide v2 | 0.844 | 0.015 | 0.776 | 0.559 | 0.741 | 0.499 |

Wide features improved the average per-fold AUROC, but they worsened pooled
out-of-fold ranking and fold-held-out calibration/threshold transfer. This
looks like fold-specific score-scale instability or overfitting from the
metadata/stat branch. Do not promote it over the current 5-second log-mel
BCE F1/Youden lead. If revisited, try stronger regularization on the wide
branch, a late-fusion/calibration-only use of the wide features, or freezing
the acoustic head before fitting the side branch.

## Regression Gate (proving improvement, not just running experiments)

Every result above was produced by hand and pasted into this note. There was
no committed, machine-checkable definition of "this model is better than the
last one." The regression gate closes that loop.

Two small modules and a committed baseline turn each `cv_report.json` into a
provable claim:

- `openstetho_model.scorecard` — projects a (large, gitignored, schema-drifting)
  `cv_report.json` into a flat, committable **scorecard** of canonical metrics
  plus provenance. Missing sections (e.g. older runs without
  `cross_fold_calibration`) extract as `null` instead of raising.
- `openstetho_model.regression_gate` — compares a candidate scorecard against
  the frozen baseline under a gate policy and returns `IMPROVED` / `PASS` /
  `REGRESSED`. The CLI exits non-zero on `REGRESSED`.
- `model/benchmarks/murmur_baseline.json` — the frozen baseline scorecard
  **bundled with** the gate policy, so the rule travels with the number.
- `model/benchmarks/scorecards/*.json` — committed candidate scorecards.
- `model/tests/test_regression_gate.py` — offline pytest (no data/GPU) that
  fails CI on a regression.

Gate policy (decided, see commit): primary metric is the calibrated,
fold-held-out **transferred best-F1** (`platt_tt_bestf1_f1`) — the
least-optimistic deployable operating point. Improvement margin `0.005`. Guard
metrics `oof_auroc`, `platt_auroc`, `platt_tt_spec90_sensitivity` may not drop
more than `0.01`, so a candidate cannot trade ranking or screening recall for
F1.

Frozen baseline (v2): the clean, public-data-only **3-seed bagged ensemble** of
the 5s log-mel CNN+BiGRU BCE/F1 recipe (`logmel_5s_ensemble3`). See the next
section for how it was produced.

Refresh a scorecard and run the gate:

```bash
uv run --project model python -m openstetho_model.scorecard \
    model/runs/<run>/cv_report.json --name <id> \
    --out model/benchmarks/scorecards/<id>.json

uv run --project model python -m openstetho_model.regression_gate \
    --baseline model/benchmarks/murmur_baseline.json \
    --candidate model/benchmarks/scorecards/<id>.json
```

**Promotion is reserved for clean public-data recipes** — the baseline file
enforces `teacher_distillation == false` via the test suite. The teacher-distilled
CirCor2022 w=0.20 run (transferred F1 `0.626`) was gate-IMPROVED over the v1
baseline but is *not* promotable; it stays a documented research lead, and it
no longer clears the raised v2 bar (`0.644`).

## Clean Win: Seed-Ensemble Promotion (v1 -> v2 baseline)

An augmentation candidate (train-fold SpecAugment + noise + time-shift on the
proven 5s log-mel recipe) was trained and gated first. It **regressed**: fold
val AUROC rose to `0.839` but pooled calibrated transferred F1 fell to `0.560`
(`platt_auroc` and `spec90` sensitivity also down). This is the same
fold-vs-pooled optimism seen with the wide-feature and 15s `logmel128` runs;
the gate caught it automatically and it was not promoted.

The clean win came from **seed bagging**. `--seed` controls the fold split,
model init, and sampler, so three runs at seeds 0/1/2 of the identical clean
recipe are well-decorrelated. `openstetho_model.ensemble_oof` averages the
per-recording out-of-fold probabilities (each prediction is still out-of-fold,
so averaging is valid) and rescoring with the same `summarize` /
`cross_fold_calibration_report` path yields a drop-in `cv_report.json`.

```bash
# seeds 1 and 2 (seed 0 is the existing logmel_5s_bce_f1 run)
for s in 1 2; do
  uv run --project model python -m openstetho_model.cv_murmur \
      --data $CIRCOR_ROOT --architecture cnn_bigru --feature-mode logmel \
      --window-seconds 5 --aggregation mean --folds 5 --epochs 8 \
      --batch-size 16 --workers 0 --device mps --no-cardiac \
      --loss bce --lr 3e-4 --grad-clip-norm 5 --select-metric f1 \
      --lr-scheduler plateau --plateau-patience 1 --plateau-factor 0.5 \
      --early-stopping-patience 2 --seed $s \
      --out model/runs/murmur_cv_logmel_5s_seed$s
done

uv run --project model python -m openstetho_model.ensemble_oof \
    --runs model/runs/murmur_cv_logmel_5s_bce_select_f1_v1 \
           model/runs/murmur_cv_logmel_5s_seed1 \
           model/runs/murmur_cv_logmel_5s_seed2 \
    --out model/runs/murmur_cv_logmel_5s_ensemble_v1
```

Gate verdict, frozen v1 baseline vs the clean 3-seed ensemble:

```text
verdict: IMPROVED
  metric                      role     baseline candidate    delta  status
  platt_tt_bestf1_f1          primary   0.5972   0.6439   +0.0467  improved
  oof_auroc                   guard     0.8172   0.8415   +0.0243  improved
  platt_auroc                 guard     0.8137   0.8406   +0.0269  improved
  platt_tt_spec90_sensitivity guard     0.5809   0.6452   +0.0644  improved
```

Every gate metric improved with no teacher signal, and calibration error
roughly halved (Platt ECE-10 `0.040 -> 0.019`). The ensemble also beats the
teacher-distilled w=0.20 lead (`0.6439 > 0.6264`) on clean public data, so it
is now the strongest result in this thread and the frozen v2 baseline.

Deployment note: the v2 baseline is a research/benchmark lead. Shipping it as a
single Core ML artifact needs either multi-model export or distilling the
ensemble into one student — that is the next deployment step. The released
app-facing checkpoint is governed separately and is unchanged.

## Distillation to a Single Deployable Model (negative, gate-rejected)

The v2 ensemble is a research/benchmark lead but not a single Core ML artifact.
The clean route to a deployable single model is distillation: train one student
to mimic the ensemble's out-of-fold soft targets (teacher = our own ensemble on
public CirCor, not a vendor model). `ensemble_oof.py` now also emits
`patient_id`/`location`, so its OoF CSV is a drop-in teacher for
`cv_murmur --teacher-predictions-csv ... --teacher-prob-column prob`. The OoF
targets are leak-free (each came from models that did not train on that
recording), and coverage is full (2964/2964).

A distill-weight sweep was gated against the **prior single-model** baseline
(`logmel_5s_bce_f1`, transferred F1 `0.5972`):

| distill weight | verdict | transferred F1 | OoF AUROC | Platt AUROC | spec>=0.90 sens | Platt ECE |
|---:|---|---:|---:|---:|---:|---:|
| 0.1 | REGRESSED | 0.3795 | 0.7723 | 0.7498 | 0.4950 | 0.0715 |
| 0.2 | REGRESSED | 0.5564 | 0.8245 | 0.8123 | 0.5809 | 0.0234 |
| 0.5 | REGRESSED | 0.5532 | 0.8064 | 0.7840 | 0.5627 | 0.0252 |
| 1.0 | REGRESSED | 0.5618 | 0.7933 | 0.7715 | 0.5561 | 0.0529 |

All four regressed on the gated primary. High weights (0.5/1.0) inflate the loss
scale and destabilize pooled ranking (the recurring fold-vs-pooled optimism:
fold val AUROC reached ~0.85 while pooled OoF fell). The best point, `w=0.2`, is
a **lateral move, not a win**: OoF AUROC nudges up (`0.8245` vs `0.8172`) and
calibration improves markedly (ECE `0.0234` vs `0.0396`), but the calibrated
transferred best-F1 drops to `0.5564` because that operating point shifts toward
sensitivity (transferred specificity `0.94 -> 0.885`). It does help the
high-recall screening point (`sens>=0.80` specificity `0.642 -> 0.665`).

Conclusion: a single distilled student does not beat the prior single model on
the gated primary metric at any tested weight. The clean shipped win remains the
3-seed ensemble. To deploy it, prefer multi-model Core ML export (run the three
checkpoints, average logits) over distillation. If distillation is revisited,
try specificity-oriented checkpoint selection (the gap is operating-point, not
ranking), distillation temperature, or distilling from per-window rather than
recording-level teacher targets.

## Deployment: Fused Ensemble Export (Option B, built)

The clean ensemble ships as a single Core ML artifact with the averaging baked
in -- no multi-file app change, just a fused mlprogram.

Two pieces:

1. Deployment checkpoints. The CV ensemble is out-of-fold (15 fold-models); it
   is a generalization *estimate*, not deployable weights. For deployment, train
   one full-data model per seed with the identical clean recipe
   (`train.py --level recording`, seeds 0/1/2 -> `runs/murmur_deploy_seed{0,1,2}/best.pt`).
2. Fused export. `openstetho_model.export_ensemble` loads the N members, wraps
   them in `ProbMeanEnsemble`, and converts to one `.mlpackage`. It averages
   member **probabilities** and re-encodes the mean as a logit
   (`logit(mean(sigmoid(member)))`), so the app's existing `sigmoid(output)`
   step recovers the mean member probability -- exactly what `ensemble_oof`
   scored. The app contract is unchanged: one `log_mel` input, one
   `murmur_logit` output.

```bash
uv run --project model python -m openstetho_model.export_ensemble \
    --checkpoints runs/murmur_deploy_seed0/best.pt \
                  runs/murmur_deploy_seed1/best.pt \
                  runs/murmur_deploy_seed2/best.pt \
    --window-seconds 5 \
    --out runs/release-ensemble-v2/MurmurCNN.mlpackage --verify
```

Parity check (PyTorch fused ensemble vs exported Core ML, random input):

```text
torch_logit  -1.9128   coreml_logit  -1.9121
max_abs_diff  0.0007    coreml_latency_ms  ~19 (3 members fused)
```

The exported package discriminates end-to-end through the 5 s / 78-frame Core ML
path (in-sample sanity bench; the unbiased performance claim remains the CV
estimate, transferred F1 `0.644`).

### Release is gated on a stetho-ui window-length change (NOT yet shipped)

The released contract is 4 s / 62 frames `(1,1,62,32)`. This ensemble is
**5 s / 78 frames** `(1,1,78,32)`. The GitHub release asset name is hardcoded
(`MurmurCNN.mlpackage.zip`) and installs auto-upgrade via
`/releases/latest/download/`, so publishing the 5 s package **before** the UI is
updated would feed 62-frame input to a 78-frame model and break every install.

Before packaging/publishing this artifact:

- update `MurmurEngine` in `stetho-ui` to a 5 s / 78-frame window (mel framing +
  rolling buffer length), keeping the z-scored-frame-to-inference contract;
- set the app's default operating threshold from the ensemble's calibrated
  transfer numbers (it is better calibrated, ECE `0.019`), not the old 4 s value;
- re-run Rust/Python mel parity (`dump_mel_parity.rs` + `test_parity.py`);
- only then `scripts/package_model_release.sh runs/release-ensemble-v2/MurmurCNN.mlpackage`
  and publish.

The export tooling, deployment recipe, and parity check are committed; the
binary `.mlpackage` is gitignored and the release step is intentionally left for
the coordinated UI change.

## Current Recommendation

- The clean, gate-promoted lead is now the **3-seed bagged ensemble** of the 5s
  log-mel CNN+BiGRU BCE/F1 recipe (frozen v2 baseline, `logmel_5s_ensemble3`):
  pooled OoF AUROC `0.842`, calibrated transferred F1 `0.644`, Platt ECE `0.019`.
  It is public-data-only and beats both the prior single-model baseline and the
  teacher-distilled w=0.20 run.
- Do not replace the app model with the earlier `murmur_recording_top3_v1`
  checkpoint; its full benchmark AUROC was below 0.5.
- The CirCor2022 positive-only soft-target w=0.20 run remains a documented but
  non-promotable research lead (teacher-distilled, and now below the v2 bar).
- The single-model no-soft-target 5s log-mel BCE F1/Youden model is the v1
  baseline, superseded by the ensemble but kept as the committed prior-baseline
  scorecard for the regression proof.
- Seed bagging was the clean lever that worked; train-fold augmentation on the
  same recipe regressed pooled transferred metrics and was rejected by the gate.
- Keep the released app-facing checkpoint unchanged until the 5s log-mel
  CNN+BiGRU path has an export/deployment plan.
- The 5s MFCC single-split result did not hold up as a pooled OoF
  improvement under patient-level CV.
- Treat 5s MFCC-only as a useful representation candidate because its fold
  AUROC is consistently strong, but do not promote it without calibration.
- Do not pursue the stacked multi-channel path unless changing the
  architecture or regularization.
- Do not pursue the current wavelet-scattering + 1D-CNN path as-is; its
  5-fold patient-level pooled OoF AUROC was near chance. Use
  `scattering_transformer` only as a separate contextualized comparator;
  its full 5-fold pooled OoF AUROC was `0.699`, below the log-mel lead.
- Do not promote the HearHeart-style 15s `logmel128` augmented run; its
  5-fold pooled OoF AUROC is near chance.
- Do not promote the HearHeart-style wide-feature branch yet. It improved
  fold AUROC but hurt pooled OoF AUROC, Platt AUROC, and transferred F1.
- Use recording-level mean aggregation as the primary decision rule.
- Keep vote-count sweeps in the benchmark for sensitivity/specificity
  exploration, especially once a model is trained natively on 2.5 s windows.
- Use the new `cross_fold_calibration` section in `cv_report.json` for model
  selection. It reports Brier/ECE plus fold-held-out threshold transfer, so
  it is less optimistic than pooled best-F1 threshold selection.
- Do not promote the first focal sensitivity-weighted log-mel run. It hurts
  AUROC and transferred-threshold F1 despite improving some raw recall
  operating points.
- Do not promote the positive-weighted CirCor2022 follow-ups or full-target
  CirCor2022 distillation. Both hurt calibrated threshold transfer.
- Next CNN+BiGRU work should focus on export/deployment feasibility for the
  CirCor2022-assisted `w=0.20` model if local-only use is acceptable, or the
  clean BCE F1/Youden-selected 5s log-mel model otherwise.
