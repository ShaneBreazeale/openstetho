# openstetho

Pure-Rust toolkit for BLE digital stethoscopes — capture, DSP,
log-mel spectrograms, and on-device murmur classification on the
Apple Neural Engine.

Independent project. Trademarks (Eko, Eko Core, Littmann CORE) used
descriptively to identify the hardware this toolkit interoperates
with. Not a medical device — see [`DISCLAIMER.md`](DISCLAIMER.md).

![stetho-ui live waveform + log-mel spectrogram](docs/stetho-ui-live.png)

## Model status

Current murmur models are experimental and should not be treated as
accurate. They are trained on public phonocardiogram datasets collected
with different microphones, acoustic paths, gain/filter settings, patient
populations, and labeling protocols than compatible Eko/Littmann devices.
That domain mismatch can dominate model output even when the DSP pipeline
is working correctly.

The useful role of this toolkit today is to capture device audio, make the
BLE/codec/DSP path reproducible, and provide a Core ML export/inference
loop that can support future model development on properly matched data.

Model artifacts are intentionally not committed to git. Two Core ML
packages are produced by the training pipeline:

* `MurmurCNN.mlpackage` (~450 KB mlpackage in the current release) —
  experimental CNN+BiGRU murmur classifier trained on CirCor 2022.
* `S3CNN_v2.mlpackage` (~1.2 MB mlpackage) — third-heart-sound detector
  with S2-anchored cycle-level inference. See
  [`docs/real_validation_results.md`](docs/real_validation_results.md)
  for the cardiologist-labeled validation summary.

Latest model bundle:
[`v0.3.0-murmur-bigru`](https://github.com/ShaneBreazeale/openstetho/releases/tag/v0.3.0-murmur-bigru)
publishes `MurmurCNN.mlpackage.zip`, containing the CNN+BiGRU murmur model
and the `S3CNN_v2` sibling. On the full CirCor benchmark, the CNN+BiGRU
recording-mean murmur model reached AUROC 0.868 and best-F1 0.702
(sensitivity 0.634, specificity 0.956). See
[`docs/murmur_detector_benchmark.md`](docs/murmur_detector_benchmark.md)
for commands, thresholds, and caveats.

`stetho-ui` runs both engines in parallel — it loads
`MurmurCNN.mlpackage` from the configured download dir, then
opportunistically looks for an `S3CNN_v2.mlpackage` sibling next to it
and pumps the same z-scored mel frames into both models. The murmur and
S3 probabilities appear side by side in the top status bar.

Set `OPENSTETHO_MODEL_DOWNLOAD_URL` to override the default release URL
and `OPENSTETHO_MODEL_DOWNLOAD_DIR` to override the local destination.
The default button URL works after a GitHub release asset named
`MurmurCNN.mlpackage.zip` exists; the current latest release already
provides that asset. Package one or both models with:

```bash
# murmur only
scripts/package_model_release.sh model/runs/v1/MurmurCNN.mlpackage

# murmur + S3 (recommended for current releases)
scripts/package_model_release.sh \
    model/runs/release-circor-v2/MurmurCNN.mlpackage \
    model/runs/s3_circor_v10/S3CNN_v2.mlpackage
```

## Crates

- **`stetho-core`** — BLE GATT, IMA-ADPCM decoder, biquad DSP,
  Slaney log-mel spectrogram. Pure DSP from public formulas.
- **`stetho-cli`** — `stetho` binary: `scan`, `connect`, `stream`,
  `capture`, `decode-hex`.
- **`stetho-ui`** — egui live dev viewer: waveform + mel-spec + on-
  device Core ML inference.
- **`model/`** — Python pipeline. Trains a small Conv2D classifier
  on CirCor 2022 by default and exports a Core ML
  `.mlpackage` for the Apple Neural Engine.

## Quick start

Requires macOS, Rust, and Python 3.12 with `uv`.

```bash
# Scan for compatible devices
cargo run -p stetho-cli --release -- scan --seconds 10

# Capture 30 s to WAV + raw hex
cargo run -p stetho-cli --release -- capture "eko core" --seconds 30 \
    --out /tmp/capture

# Live dev viewer
cargo run -p stetho-ui --release
```

Train from public data:

```bash
bash scripts/download_circor.sh
cd model && uv sync
uv run python -m openstetho_model.train --data ../data/circor \
    --epochs 30 --out runs/v1
uv run python -m openstetho_model.export --checkpoint runs/v1/best.pt \
    --out runs/v1/MurmurCNN.mlpackage --target macOS13 --verify
```

Experimental CNN+BiGRU murmur checkpoints can be exported with
`--architecture cnn_bigru`; keep the package name `MurmurCNN.mlpackage`
when building GUI-downloadable release assets.

## Provenance + license

DSP, protocol observations, and dataset attributions are documented
in [`PROVENANCE.md`](PROVENANCE.md). Code is [Apache 2.0](LICENSE).
Packaged binaries should include applicable dependency notices; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
