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

## Current Recommendation

- Do not replace the app model with the earlier `murmur_recording_top3_v1`
  checkpoint; its full benchmark AUROC was below 0.5.
- Keep current Core ML for the app until the CNN+BiGRU result is validated
  across patient-level folds.
- Use recording-level mean aggregation as the primary decision rule.
- Keep vote-count sweeps in the benchmark for sensitivity/specificity
  exploration, especially once a model is trained natively on 2.5 s windows.
- Run the next CNN+BiGRU experiment with 5-fold patient-level CV, early
  stopping, and threshold calibration per fold.
