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

Command shape:

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
    --device cpu \
    --no-cardiac \
    --loss bce \
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

The simple wavelet-scattering implementation is a negative result in this
pipeline. It uses Kymatio WST coefficients (`J=8`, `Q=4`) from 5-second
windows and a small 1D-CNN, but pooled OoF AUROC is near chance. Do not
pursue this branch unless reproducing the paper's fuller preprocessing
stack, including denoising, segmentation, noise-only segment relabeling, and
normalization details, or unless retuning the scattering/frontend
architecture from scratch.

## Current Recommendation

- Do not replace the app model with the earlier `murmur_recording_top3_v1`
  checkpoint; its full benchmark AUROC was below 0.5.
- Treat the 5s log-mel CNN+BiGRU with BCE loss and F1/Youden checkpoint
  selection as the current research lead. It has the best pooled OoF AUROC
  and best transferred F1 so far.
- Keep the released app-facing checkpoint unchanged until the 5s log-mel
  CNN+BiGRU path has an export/deployment plan.
- The 5s MFCC single-split result did not hold up as a pooled OoF
  improvement under patient-level CV.
- Treat 5s MFCC-only as a useful representation candidate because its fold
  AUROC is consistently strong, but do not promote it without calibration.
- Do not pursue the stacked multi-channel path unless changing the
  architecture or regularization.
- Do not pursue the current wavelet-scattering + 1D-CNN path as-is; its
  5-fold patient-level pooled OoF AUROC was near chance.
- Use recording-level mean aggregation as the primary decision rule.
- Keep vote-count sweeps in the benchmark for sensitivity/specificity
  exploration, especially once a model is trained natively on 2.5 s windows.
- Use the new `cross_fold_calibration` section in `cv_report.json` for model
  selection. It reports Brier/ECE plus fold-held-out threshold transfer, so
  it is less optimistic than pooled best-F1 threshold selection.
- Do not promote the first focal sensitivity-weighted log-mel run. It hurts
  AUROC and transferred-threshold F1 despite improving some raw recall
  operating points.
- Next CNN+BiGRU work should focus on export/deployment feasibility for the
  BCE F1/Youden-selected 5s log-mel model, or on a smaller sensitivity
  nudge that does not damage ranking.
