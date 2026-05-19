"""Cycle-level dataset for S3 detection.

Wraps any list of mono PCG WAV files. Each `__getitem__` returns one cardiac
cycle as a log-mel-spectrogram tensor plus a binary S3-presence label.

Positive labels come from `s3_inject`: we synthetically add an S3 to a random
subset of cycles per recording, at calibrated SNR. Negative labels are cycles
left untouched. Real-S3 contamination of negatives is a known label-noise
source — see [[s3-annotation-protocol]] for the eventual ground-truth path.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .pcg_augment import random_pcg_augment
from .preprocess import N_MELS, SAMPLE_RATE, apply_s3_preset, load_audio, log_mel
from .s3_inject import (
    ejection_click_inject,
    opening_snap_inject,
    s3_inject,
    s4_inject,
    split_s2_inject,
)
from .segment import Segmentation, segment_unified


def _load_and_segment(wav_path: str, method: str = "heuristic") -> tuple[np.ndarray, Segmentation]:
    """Load a WAV at 4 kHz mono float32 + segment with the chosen method."""
    audio = load_audio(wav_path)
    audio.setflags(write=False)
    return audio, segment_unified(audio, method=method)


log = logging.getLogger(__name__)

CYCLE_WINDOW_S = 1.5
CYCLE_WINDOW_SAMPLES = int(CYCLE_WINDOW_S * SAMPLE_RATE)  # 6000
CYCLE_WINDOW_FRAMES = CYCLE_WINDOW_SAMPLES // 256  # 23 STFT frames @ hop=256

# S2-anchored crop: gives S3 (early diastolic) and S4 (late diastolic) fixed
# frame positions independent of cycle period. S3 lives at S2+100-200 ms;
# S4 at next_S1-100 ms which varies with HR but stays inside [S2+0.3 s,
# S2+1.0 s] for HR 50-150 bpm.
CYCLE_PRE_S2_S = 0.3
CYCLE_POST_S2_S = 1.2
assert CYCLE_PRE_S2_S + CYCLE_POST_S2_S == CYCLE_WINDOW_S


@dataclass(frozen=True)
class CycleIndex:
    wav: Path
    cycle_no: int  # 0-based index into recording's cycle list
    s1_idx: int    # cached so we can re-crop without re-segmenting
    s2_idx: int    # anchor for S2-centered crops (introduced in v8)
    is_seed_positive: bool  # decided at index build time for label stability


def _cycle_crop(audio: np.ndarray, s1_idx: int) -> np.ndarray:
    """S1-anchored crop (legacy, used by older checkpoints v4-v7).

    Crop a fixed-length window starting at S1 onset. Zero-pad if the
    recording ends before the window does.
    """
    end = s1_idx + CYCLE_WINDOW_SAMPLES
    if end <= len(audio):
        return audio[s1_idx:end].astype(np.float32, copy=False)
    out = np.zeros(CYCLE_WINDOW_SAMPLES, dtype=np.float32)
    tail = audio[s1_idx:]
    out[: len(tail)] = tail
    return out


def _cycle_crop_s2(audio: np.ndarray, s2_idx: int) -> np.ndarray:
    """S2-anchored crop (v8+). Places S2 at a fixed position in the window
    so S3 (early diastolic) and S4 (late diastolic) land in HR-invariant
    frame ranges. Zero-pads on either side when the recording boundary
    clips the window.
    """
    pre = int(CYCLE_PRE_S2_S * SAMPLE_RATE)
    post = int(CYCLE_POST_S2_S * SAMPLE_RATE)
    start = s2_idx - pre
    end = s2_idx + post
    out = np.zeros(pre + post, dtype=np.float32)
    valid_start = max(0, start)
    valid_end = min(len(audio), end)
    if valid_end > valid_start:
        out[valid_start - start : valid_end - start] = audio[valid_start:valid_end]
    return out


class S3CycleDataset(Dataset[tuple[torch.Tensor, int]]):
    """One cardiac cycle per item.

    Args
    ----
    wavs:           list of WAV paths (any sample rate; resampled to 4 kHz).
    positive_rate:  fraction of cycles to inject S3 into (controls class
                    balance; recommended 0.5 for training).
    snr_db_range:   amplitude calibration vs diastolic noise.
    min_segment_confidence:
                    recordings with segmenter confidence below this are
                    dropped at index build time.
    seed:           controls which cycles get synthetic S3 (label stability
                    across epochs). Augmentation params draw from a fresh
                    rng per `__getitem__` to keep mel-spec diversity.
    """

    def __init__(
        self,
        wavs: Sequence[str | Path],
        positive_rate: float = 0.5,
        snr_db_range: tuple[float, float] = (0.0, 12.0),
        min_segment_confidence: float = 0.3,
        seed: int = 0,
        prob_multi: float = 0.0,
        prob_s4: float = 0.0,
        freq_mask_max_width: int = 0,
        time_mask_max_width: int = 0,
        apply_spec_masks: bool = True,
        crop_anchor: str = "s2",
        emit_multiclass: bool = False,
        segmenter: str = "heuristic",
        apply_pcg_augment: bool = False,
        prob_split_s2: float = 0.0,
        prob_opening_snap: float = 0.0,
        prob_ejection_click: float = 0.0,
    ):
        self.positive_rate = float(positive_rate)
        self.snr_db_range = (float(snr_db_range[0]), float(snr_db_range[1]))
        self._epoch_rng_base = int(seed)
        self.prob_multi = float(prob_multi)
        self.prob_s4 = float(prob_s4)
        self.freq_mask_max_width = int(freq_mask_max_width)
        self.time_mask_max_width = int(time_mask_max_width)
        self.apply_spec_masks = bool(apply_spec_masks)
        if crop_anchor not in {"s1", "s2"}:
            raise ValueError(f"crop_anchor must be 's1' or 's2', got {crop_anchor!r}")
        self.crop_anchor = crop_anchor
        # Multiclass labels: 0=clean, 1=S3 positive, 2=S4 confounder. Same
        # cycle index drives label assignment so seed-time decisions are stable
        # across epochs.
        self.emit_multiclass = bool(emit_multiclass)
        if segmenter not in {"heuristic", "hsmm"}:
            raise ValueError(f"segmenter must be 'heuristic' or 'hsmm', got {segmenter!r}")
        self.segmenter = segmenter
        self.apply_pcg_augment = bool(apply_pcg_augment)
        self.prob_split_s2 = float(prob_split_s2)
        self.prob_opening_snap = float(prob_opening_snap)
        self.prob_ejection_click = float(prob_ejection_click)

        index_rng = np.random.default_rng(seed)
        self._index: list[CycleIndex] = []
        # Pre-load + segment every WAV once, store in an instance dict. Keeps
        # the cache scoped to this dataset (no surprise process-global LRU)
        # and survives pickling to DataLoader workers — at the cost of
        # duplicating the audio across worker processes. Run with
        # num_workers=0 to keep memory single-copy.
        self._cache: dict[str, tuple[np.ndarray, Segmentation]] = {}

        for wav_path in wavs:
            wav_path = Path(wav_path)
            key = str(wav_path)
            try:
                audio, segmentation = _load_and_segment(key, method=self.segmenter)
            except Exception as e:  # noqa: BLE001
                log.warning("skip %s: %s", wav_path, e)
                continue
            if segmentation.confidence < min_segment_confidence:
                continue
            self._cache[key] = (audio, segmentation)
            for k, cycle in enumerate(segmentation.cycles):
                positive_seed = bool(index_rng.random() < self.positive_rate)
                self._index.append(
                    CycleIndex(
                        wav=wav_path,
                        cycle_no=k,
                        s1_idx=cycle.s1_idx,
                        s2_idx=cycle.s2_idx,
                        is_seed_positive=positive_seed,
                    )
                )

    @staticmethod
    def from_wav_root(root: str | Path, pattern: str = "*.wav", **kwargs) -> "S3CycleDataset":
        return S3CycleDataset(sorted(Path(root).rglob(pattern)), **kwargs)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        entry = self._index[idx]
        cached_audio, segmentation = self._cache[str(entry.wav)]
        # Copy on every access — s3_inject writes into the buffer and we must
        # not mutate the cached, read-only array shared across getitem calls.
        audio: np.ndarray = cached_audio
        if entry.cycle_no >= len(segmentation.cycles):
            # Segmentation differs from index time (rare; rounding / boundary
            # peaks). Fall back to negative label with the recorded indices.
            cropped = self._crop_cycle(audio, entry)
            cropped = apply_s3_preset(cropped)
            mel = log_mel(cropped)
            return _to_tensor(mel), 0

        s4_injected = False
        # Negative-cycle S4 confounder injection — forces the model to use
        # timing (early vs late diastole) rather than spectral content alone.
        if (not entry.is_seed_positive) and self.prob_s4 > 0.0:
            single = type(segmentation)(
                cycles=[segmentation.cycles[entry.cycle_no]],
                cycle_period_s=segmentation.cycle_period_s,
                confidence=segmentation.confidence,
            )
            audio_copy = cached_audio.copy()
            audio_copy, flags = s4_inject(  # type: ignore[misc]
                audio_copy,
                single,
                np.random.default_rng(),
                prob_per_cycle=self.prob_s4,
                snr_db_range=self.snr_db_range,
                return_flags=True,
            )
            s4_injected = bool(flags[0]) if flags else False
            if s4_injected:
                audio = audio_copy

        if entry.is_seed_positive:
            # Force injection at this cycle only; per-call rng keeps mel
            # diversity epoch over epoch.
            single = type(segmentation)(
                cycles=[segmentation.cycles[entry.cycle_no]],
                cycle_period_s=segmentation.cycle_period_s,
                confidence=segmentation.confidence,
            )
            aug_rng = np.random.default_rng()
            audio_copy = cached_audio.copy() if audio is cached_audio else audio
            audio_copy, records = s3_inject(
                audio_copy,
                single,
                aug_rng,
                prob_per_cycle=1.0,
                snr_db_range=self.snr_db_range,
                prob_multi=self.prob_multi,
            )
            audio = audio_copy
            s3_landed = bool(records and records[0].positive)
        else:
            s3_landed = False

        # Cycle-level confounder injections — keep cycle label 0/non-S3 but
        # plant hard negatives in the audio so the encoder learns timing /
        # morphology rather than "any low-freq diastolic event = positive".
        rng_aug = np.random.default_rng()
        if (
            self.prob_split_s2 > 0.0
            or self.prob_opening_snap > 0.0
            or self.prob_ejection_click > 0.0
        ):
            single = type(segmentation)(
                cycles=[segmentation.cycles[entry.cycle_no]],
                cycle_period_s=segmentation.cycle_period_s,
                confidence=segmentation.confidence,
            )
            if audio is cached_audio:
                audio = cached_audio.copy()
            if self.prob_split_s2 > 0.0:
                audio = split_s2_inject(audio, single, rng_aug,
                                        prob_per_cycle=self.prob_split_s2,
                                        snr_db_range=self.snr_db_range)
            if self.prob_opening_snap > 0.0:
                audio = opening_snap_inject(audio, single, rng_aug,
                                            prob_per_cycle=self.prob_opening_snap,
                                            snr_db_range=self.snr_db_range)
            if self.prob_ejection_click > 0.0:
                audio = ejection_click_inject(audio, single, rng_aug,
                                              prob_per_cycle=self.prob_ejection_click,
                                              snr_db_range=self.snr_db_range)

        # Physio-level augmentation applied to the *whole* audio buffer
        # before cropping. Respiration noise / baseline drift / EQ /
        # attenuation. Two consecutive calls see fresh random parameters
        # so contrastive pairs differ at the physiology level.
        if self.apply_pcg_augment:
            if audio is cached_audio:
                audio = cached_audio.copy()
            audio = random_pcg_augment(audio, rng_aug)

        if self.emit_multiclass:
            if s3_landed:
                label: int = 1  # S3
            elif s4_injected:
                label = 2       # S4 confounder
            else:
                label = 0       # clean
        else:
            label = 1 if s3_landed else 0

        cropped = self._crop_cycle(audio, entry)
        cropped = apply_s3_preset(cropped)
        mel = log_mel(cropped)
        if self.apply_spec_masks and (self.freq_mask_max_width > 0 or self.time_mask_max_width > 0):
            mel = _apply_spec_masks(mel, self.freq_mask_max_width, self.time_mask_max_width)
        return _to_tensor(mel), label

    def _crop_cycle(self, audio: np.ndarray, entry: "CycleIndex") -> np.ndarray:
        if self.crop_anchor == "s2":
            return _cycle_crop_s2(audio, entry.s2_idx)
        return _cycle_crop(audio, entry.s1_idx)

    def class_balance(self) -> dict[str, int]:
        pos = sum(1 for e in self._index if e.is_seed_positive)
        return {"negative": len(self._index) - pos, "positive": pos}


def _apply_spec_masks(
    mel: np.ndarray,
    freq_mask_max_width: int,
    time_mask_max_width: int,
) -> np.ndarray:
    """SpecAugment-style frequency + time masking.

    Zeros a contiguous random-width band along the mel axis and / or a
    contiguous random-width band along the time axis. Fill value is the mel
    array's minimum so it looks like "silence" after our z-score + clip
    normalization. Operates on a copy to keep the cached mel untouched.
    """
    out = mel.copy()
    rng = np.random.default_rng()
    fill = float(out.min())

    if freq_mask_max_width > 0 and out.shape[1] > 0:
        w = int(rng.integers(0, freq_mask_max_width + 1))
        if w > 0 and w < out.shape[1]:
            start = int(rng.integers(0, out.shape[1] - w + 1))
            out[:, start : start + w] = fill

    if time_mask_max_width > 0 and out.shape[0] > 0:
        w = int(rng.integers(0, time_mask_max_width + 1))
        if w > 0 and w < out.shape[0]:
            start = int(rng.integers(0, out.shape[0] - w + 1))
            out[start : start + w, :] = fill

    return out


def _to_tensor(mel: np.ndarray) -> torch.Tensor:
    """Pad / truncate to `CYCLE_WINDOW_FRAMES` so DataLoader can batch."""
    t = mel.shape[0]
    if t == CYCLE_WINDOW_FRAMES:
        return torch.from_numpy(mel)
    out = np.zeros((CYCLE_WINDOW_FRAMES, N_MELS), dtype=np.float32)
    keep = min(t, CYCLE_WINDOW_FRAMES)
    out[:keep] = mel[:keep]
    return torch.from_numpy(out)


def write_synthetic_pcg_wav(
    path: str | Path,
    n_cycles: int = 8,
    cycle_period_s: float = 0.857,
    s3_in_cycles: Sequence[int] | None = None,
    sample_rate: int = SAMPLE_RATE,
    seed: int = 0,
) -> None:
    """Test helper: synthesize a multi-cycle PCG WAV with optional planted S3
    events. Used by tests and quick smoke checks of the cycle dataset."""
    from .s3_synth import synth_s3

    rng = np.random.default_rng(seed)
    n = int(n_cycles * cycle_period_s * sample_rate) + sample_rate
    x = rng.normal(0.0, 0.005, size=n).astype(np.float32)

    def thump(f0: float, dur_s: float, amp: float, tau_s: float = 0.025) -> np.ndarray:
        m = int(dur_s * sample_rate)
        t = np.arange(m, dtype=np.float32) / sample_rate
        return (amp * np.exp(-t / tau_s) * np.sin(2 * np.pi * f0 * t)).astype(np.float32)

    s1 = thump(60.0, 0.10, 1.0)
    s2 = thump(80.0, 0.08, 0.6)
    for k in range(n_cycles):
        s1_idx = int((k * cycle_period_s + 0.20) * sample_rate)
        s2_idx = int((k * cycle_period_s + 0.20 + 0.35 * cycle_period_s) * sample_rate)
        x[s1_idx : s1_idx + len(s1)] += s1
        x[s2_idx : s2_idx + len(s2)] += s2
        if s3_in_cycles is not None and k in s3_in_cycles:
            s3_wave = synth_s3(sample_rate=sample_rate, amp=0.3)
            onset = s2_idx + int(0.150 * sample_rate)
            x[onset : onset + len(s3_wave)] += s3_wave

    sf.write(str(path), x, sample_rate, subtype="PCM_16")


__all__ = [
    "S3CycleDataset",
    "CycleIndex",
    "CYCLE_WINDOW_SAMPLES",
    "CYCLE_WINDOW_FRAMES",
    "write_synthetic_pcg_wav",
]
