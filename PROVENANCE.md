# Provenance

## Source code

Original work under Apache 2.0 (see [`LICENSE`](LICENSE)). DSP and
codec routines are independently implemented from public formulas and
owned-device interoperability observations:

- IMA-ADPCM: ITU-T Rec. G.726 / Annex G
- Butterworth biquad: Bristow-Johnson, "Cookbook formulae for audio
  equalizer biquad filter coefficients", 2005
- Slaney mel filterbank: Malcolm Slaney, "Auditory Toolbox", Apple
  Technical Report #45, 1998
- Core ML deployment: Apple developer documentation

## Training data

The model pipeline under `model/` uses public phonocardiogram
datasets only:

| Dataset | Source | Licence |
|---|---|---|
| CirCor DigiScope Phonocardiogram 2022 | PhysioNet | ODC-By 1.0 |
| PhysioNet / CinC 2016 Heart Sound Challenge | PhysioNet | ODC-By |
| PASCAL Heart Sound Challenge 2011 | Bentley et al. | unclear; use research-only unless you have separate rights |

Cite when using:

- Reyna MA, Kiarashi Y, Elola A, et al. *Heart murmur detection from
  phonocardiogram recordings: The George B. Moody PhysioNet
  Challenge 2022.* PLoS Digit Health 2(9):e0000324, 2023.
- Liu C, Springer D, Li Q, et al. *An open access database for the
  evaluation of heart sound algorithms.* Physiol Meas 37(12), 2016.
- Bentley P, Nouri G, et al. *PASCAL Heart Sound Challenge 2011.*

The default training path uses CirCor only. PhysioNet/CinC 2016 and
PASCAL loaders/scripts are optional research tooling and are not used by
the default trainer. Do not publish model weights trained on PASCAL data
without an independent license review.

The scripts under `scripts/download_*.sh` mirror the official
distribution URLs into a git-ignored `data/` directory. Dataset audio is
not redistributed in this repo.

Evaluation-only data (never used for training or model selection):

| Dataset | Source | Licence |
|---|---|---|
| HLS-CMDS v2 (manikin heart/lung sounds, Littmann CORE) | Torabi, Shirani, Reilly, IEEE Data Descriptions 2025, doi:10.1109/IEEEDATA.2025.3566012; GitHub `Torabiy/HLS-CMDS` | MIT (GitHub mirror) |

## Synthetic training signals

`model/openstetho_model/murmur_synth.py`, `channel.py` and
`recording_mix.py` generate train-only augmentation from public formulas
and CirCor's own segmentation annotations: band-limited noise murmurs
inserted into annotated systole/diastole of train-split Murmur-Absent
recordings, no-heart signals (coloured noise, mains hum, rubbing bursts),
random smooth EQ curves, and an IMA-ADPCM round trip using the public
IMA/DVI step and index tables. No vendor data, coefficients or model
outputs are involved. Synthetic items never enter validation.

## Test fixtures

Rust codec tests synthesize packet payloads in memory.
`model/tests/fixtures/adpcm_rust.txt` is the Rust decoder's output for a
synthetic nibble sequence (`stetho-core/examples/adpcm_vectors.rs`). Committed audio,
raw BLE captures, vendor binaries, model weights, and downloaded dataset
files are intentionally excluded from the repository.

## Documentation media

`docs/stetho-ui-live.png` is a maintainer-created screenshot using the
maintainer's own heartbeat recording. Do not add third-party screenshots,
logos, dataset audio visualizations, or patient recordings unless their
source, consent, and license are documented here.

## Protocol observations

BLE service / characteristic UUIDs and ADPCM framing constants in
`stetho-core/src/ble/uuids.rs` are functional facts about wire-level
behaviour, observed on hardware the maintainer owns.

## Trademarks

"Eko", "Eko Core", "Eko Core 2", "Eko Core 500" are trademarks of
Eko Health, Inc. "Littmann" and "Littmann CORE" are trademarks of
3M Company. Used here descriptively to identify the hardware this
toolkit interoperates with. Not endorsed by or affiliated with either
company.
