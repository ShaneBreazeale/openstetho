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
│   └── dataset.py       # CirCor 2022 PyTorch Dataset
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

## Next steps (Phase 3 task list)

- [x] step 2 — Python env via uv (this folder)
- [x] step 3 — preprocess pipeline
- [ ] step 4 — Apple `MLSoundClassifier` baseline (CreateML, ~10 min)
- [ ] step 5 — custom PyTorch model (MobileNetV3-tiny / AST-mini) on MPS
- [ ] step 6 — `coremltools.convert` → `.mlpackage`, ANE verify
- [ ] step 7 — wire into `stetho-ui` via `objc2-core-ml`
