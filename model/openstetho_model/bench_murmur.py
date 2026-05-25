"""Benchmark murmur classifiers on the same CirCor window split.

The intended use is comparing the current exported `MurmurCNN.mlpackage`
against an ONNX baseline that accepts the same log-mel input:

    uv run --project model python -m openstetho_model.bench_murmur \\
        --data data/circor \\
        --coreml model/runs/release-circor-v1/MurmurCNN.mlpackage \\
        --onnx path/to/baseline.onnx \\
        --split val --seed 0 --max-windows 500

Inputs are 4-second windows preprocessed by `CirCorMurmurDataset`, shape
`(1, 1, 62, 32)`. If the ONNX model uses a different input contract, wrap
or export it to this feature shape before using this benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from .dataset import CirCorMurmurDataset, _patient_recordings
from .model import MurmurCNN, MurmurCNNBiGRU, MurmurScatteringCNN1D
from .preprocess import (
    FEATURE_MODES,
    FEATURE_MODE_LOGMEL,
    SAMPLE_RATE,
    WINDOW_HOP,
    WINDOW_SAMPLES,
    apply_bandpass,
    apply_cardiac,
    feature_channels,
    load_audio,
    split_windows,
    window_features,
)
from .train import patient_split

log = logging.getLogger("bench_murmur")

COREML_INPUT_NAME = "log_mel"
COREML_OUTPUT_NAME = "murmur_logit"


class Scorer(Protocol):
    name: str

    def score(self, x: np.ndarray) -> tuple[float, float]:
        """Return `(probability, latency_ms)` for one model input."""


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (scores >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    return {
        "n": int(labels.size),
        "positives": int(labels.sum()),
        "threshold": float(threshold),
        "auroc": auroc(labels, scores),
        "accuracy": (tp + tn) / max(labels.size, 1),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "precision": tp / max(tp + fp, 1),
        "f1": 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn),
        "youden_j": (tp / max(tp + fn, 1)) + (tn / max(tn + fp, 1)) - 1.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    rank_sum_pos = float(ranks[labels == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def sweep_thresholds(labels: np.ndarray, scores: np.ndarray) -> list[dict[str, float]]:
    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
    return [binary_metrics(labels, scores, float(t)) for t in thresholds]


def best_by(rows: list[dict[str, float]], key: str) -> dict[str, float]:
    if not rows:
        return {}
    # Prefer higher specificity then higher threshold for ties; this avoids
    # choosing noisier low thresholds when the headline metric is identical.
    return max(rows, key=lambda r: (r[key], r["specificity"], r["threshold"]))


def sweep_summary(labels: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    rows = sweep_thresholds(labels, scores)
    return {
        "threshold_0_5": binary_metrics(labels, scores, 0.5),
        "best_f1": best_by(rows, "f1"),
        "best_youden_j": best_by(rows, "youden_j"),
    }


def aggregate_values(values: list[float], mode: str) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if mode == "mean":
        return float(arr.mean())
    if mode == "median":
        return float(np.median(arr))
    if mode == "top3_mean":
        return float(np.sort(arr)[-3:].mean())
    return float(arr.max())


def recording_level(
    example_meta: list[dict[str, object]],
    labels: np.ndarray,
    scores: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, dict[str, object]] = {}
    for meta, label, score in zip(example_meta, labels.tolist(), scores.tolist()):
        key = str(meta["recording"])
        item = grouped.setdefault(key, {"label": label, "scores": []})
        if int(item["label"]) != int(label):
            raise ValueError(f"recording {key} has mixed labels")
        item["scores"].append(float(score))  # type: ignore[union-attr]
    out_labels: list[int] = []
    out_scores: list[float] = []
    for item in grouped.values():
        out_labels.append(int(item["label"]))
        out_scores.append(aggregate_values(item["scores"], mode))  # type: ignore[arg-type]
    return np.asarray(out_labels, dtype=np.int64), np.asarray(out_scores, dtype=np.float64)


def grouped_recordings(
    example_meta: list[dict[str, object]],
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for meta, label, score in zip(example_meta, labels.tolist(), scores.tolist()):
        key = str(meta["recording"])
        item = grouped.setdefault(key, {"label": label, "scores": []})
        if int(item["label"]) != int(label):
            raise ValueError(f"recording {key} has mixed labels")
        item["scores"].append(float(score))  # type: ignore[union-attr]
    return grouped


def vote_level(
    example_meta: list[dict[str, object]],
    labels: np.ndarray,
    scores: np.ndarray,
    window_threshold: float,
    min_positive_windows: int,
) -> dict[str, float]:
    grouped = grouped_recordings(example_meta, labels, scores)
    out_labels: list[int] = []
    out_scores: list[float] = []
    for item in grouped.values():
        rec_scores = np.asarray(item["scores"], dtype=np.float64)  # type: ignore[arg-type]
        out_labels.append(int(item["label"]))
        out_scores.append(float((rec_scores >= window_threshold).sum()))
    metrics = binary_metrics(
        np.asarray(out_labels, dtype=np.int64),
        np.asarray(out_scores, dtype=np.float64),
        float(min_positive_windows),
    )
    metrics["window_threshold"] = float(window_threshold)
    metrics["min_positive_windows"] = int(min_positive_windows)
    return metrics


def vote_sweep(
    example_meta: list[dict[str, object]],
    labels: np.ndarray,
    scores: np.ndarray,
    window_thresholds: list[float],
    min_positive_windows: list[int],
) -> list[dict[str, float]]:
    return [
        vote_level(example_meta, labels, scores, threshold, count)
        for threshold in window_thresholds
        for count in min_positive_windows
    ]


def vote_summary(rows: list[dict[str, float]]) -> dict[str, object]:
    return {
        "best_f1": best_by(rows, "f1"),
        "best_youden_j": best_by(rows, "youden_j"),
    }


@dataclass
class TorchScorer:
    path: Path
    architecture: str = "cnn"
    in_channels: int = 1
    name: str = "torch"

    def __post_init__(self) -> None:
        if self.architecture == "cnn":
            self.model = MurmurCNN(in_channels=self.in_channels)
        elif self.architecture == "cnn_bigru":
            self.model = MurmurCNNBiGRU(in_channels=self.in_channels)
        elif self.architecture == "scattering_cnn1d":
            self.model = MurmurScatteringCNN1D()
        else:
            raise ValueError(f"unknown architecture {self.architecture}")
        state = torch.load(self.path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def score(self, x: np.ndarray) -> tuple[float, float]:
        t0 = time.perf_counter()
        with torch.no_grad():
            logit = float(self.model(torch.from_numpy(x)).item())
        return sigmoid(logit), (time.perf_counter() - t0) * 1000


@dataclass
class CoreMLScorer:
    path: Path
    name: str = "coreml"

    def __post_init__(self) -> None:
        try:
            import coremltools as ct
        except ImportError as e:
            raise SystemExit("coremltools is required for --coreml") from e
        self.model = ct.models.MLModel(str(self.path))

    def score(self, x: np.ndarray) -> tuple[float, float]:
        t0 = time.perf_counter()
        out = self.model.predict({COREML_INPUT_NAME: x})
        latency_ms = (time.perf_counter() - t0) * 1000
        logit = float(np.asarray(out[COREML_OUTPUT_NAME]).reshape(-1)[0])
        return sigmoid(logit), latency_ms


@dataclass
class OnnxScorer:
    path: Path
    output_kind: str = "logit"
    output_index: int = 0
    name: str = "onnx"

    def __post_init__(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise SystemExit(
                "onnxruntime is required for --onnx; install it in the model env first"
            ) from e
        self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def score(self, x: np.ndarray) -> tuple[float, float]:
        t0 = time.perf_counter()
        out = self.session.run([self.output_name], {self.input_name: x})[0]
        latency_ms = (time.perf_counter() - t0) * 1000
        flat = np.asarray(out).reshape(-1)
        if self.output_index >= flat.size:
            raise ValueError(f"ONNX output has {flat.size} values; index {self.output_index} is invalid")
        value = float(flat[self.output_index])
        prob = value if self.output_kind == "prob" else sigmoid(value)
        return prob, latency_ms

    def score_raw_4s(self, audio: np.ndarray, aggregate: str) -> tuple[float, float]:
        probs: list[float] = []
        latencies: list[float] = []
        hop = SAMPLE_RATE // 2
        for start in range(0, max(1, len(audio) - SAMPLE_RATE + 1), hop):
            chunk = audio[start : start + SAMPLE_RATE]
            if len(chunk) < SAMPLE_RATE:
                padded = np.zeros(SAMPLE_RATE, dtype=np.float32)
                padded[: len(chunk)] = chunk
                chunk = padded
            x = chunk.reshape(1, SAMPLE_RATE, 1).astype(np.float32)
            prob, latency_ms = self.score(x)
            probs.append(prob)
            latencies.append(latency_ms)
        arr = np.asarray(probs, dtype=np.float64)
        if aggregate == "mean":
            prob = float(arr.mean())
        elif aggregate == "center":
            prob = float(arr[len(arr) // 2])
        else:
            prob = float(arr.max())
        return prob, float(np.sum(latencies))


def choose_split(ds: CirCorMurmurDataset, split: str, val_fraction: float, seed: int):
    if split == "all":
        return ds
    train_ds, val_ds = patient_split(ds, val_fraction=val_fraction, seed=seed)
    return train_ds if split == "train" else val_ds


def dataset_entry(dataset, i: int):
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        return dataset_entry(dataset.dataset, dataset.indices[i])
    return dataset._index[i]


def filter_by_label(dataset, label_filter: str):
    if label_filter == "all":
        return dataset
    wanted = 1 if label_filter == "present" else 0
    indices = [i for i in range(len(dataset)) if int(dataset_entry(dataset, i)[3]) == wanted]
    return torch.utils.data.Subset(dataset, indices)


def patient_ids_for_dataset(dataset) -> set[int] | None:
    if isinstance(dataset, CirCorMurmurDataset):
        return {int(row["Patient ID"]) for _, row in dataset.metadata.iterrows()}
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        return {int(dataset_entry(dataset, i)[0]) for i in range(len(dataset))}
    return None


def raw_window(dataset, i: int, apply_filter: bool) -> np.ndarray:
    _patient_id, _loc, wav, _label, w_idx = dataset_entry(dataset, i)
    audio = load_audio(str(wav))
    if apply_filter:
        audio = apply_cardiac(audio)
    start = w_idx * WINDOW_HOP
    chunk = audio[start : start + WINDOW_SAMPLES]
    if len(chunk) < WINDOW_SAMPLES:
        padded = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        padded[: len(chunk)] = chunk
        chunk = padded
    return chunk.astype(np.float32, copy=False)


def split_audio_windows(audio: np.ndarray, window_samples: int, hop_samples: int) -> np.ndarray:
    if window_samples == WINDOW_SAMPLES and hop_samples == WINDOW_HOP:
        return split_windows(audio)
    if len(audio) < window_samples:
        return np.zeros((0, window_samples), dtype=audio.dtype)
    n_windows = 1 + (len(audio) - window_samples) // hop_samples
    return np.stack(
        [audio[i * hop_samples : i * hop_samples + window_samples] for i in range(n_windows)],
        axis=0,
    )


def prepare_audio(
    wav: Path,
    apply_filter: bool,
    bandpass: tuple[float, float] | None,
    max_seconds: float | None,
) -> np.ndarray:
    audio = load_audio(str(wav))
    if max_seconds is not None:
        audio = audio[: int(round(max_seconds * SAMPLE_RATE))]
    if apply_filter:
        audio = apply_cardiac(audio)
    if bandpass is not None:
        audio = apply_bandpass(audio, bandpass[0], bandpass[1])
    return audio


def model_input_from_features(features: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = features.detach().cpu().numpy() if isinstance(features, torch.Tensor) else features
    if arr.ndim == 2:
        return arr[None, None, :, :].astype(np.float32)
    if arr.ndim == 3:
        return arr[None, :, :, :].astype(np.float32)
    raise ValueError(f"expected feature shape (T,F) or (C,T,F), got {arr.shape}")


def model_input_from_window(
    window: np.ndarray,
    pad_to_samples: int | None,
    feature_mode: str,
) -> np.ndarray:
    if pad_to_samples is not None and len(window) < pad_to_samples:
        padded = np.zeros(pad_to_samples, dtype=np.float32)
        padded[: len(window)] = window
        window = padded
    return model_input_from_features(window_features(window, feature_mode))


def iter_examples(dataset, max_windows: int | None, include_raw: bool, apply_filter: bool):
    n = len(dataset) if max_windows is None else min(len(dataset), max_windows)
    for i in range(n):
        patient_id, loc, wav, _entry_label, w_idx = dataset_entry(dataset, i)
        mel, label = dataset[i]
        x = model_input_from_features(mel)
        raw = raw_window(dataset, i, apply_filter) if include_raw else None
        meta = {
            "idx": i,
            "patient_id": int(patient_id),
            "location": str(loc),
            "recording": str(wav),
            "window_idx": int(w_idx),
        }
        yield meta, x, raw, int(label)


def iter_custom_examples(
    ds: CirCorMurmurDataset,
    selected_patient_ids: set[int] | None,
    label_filter: str,
    max_windows: int | None,
    include_raw: bool,
    apply_filter: bool,
    bandpass: tuple[float, float] | None,
    window_samples: int,
    hop_samples: int,
    pad_to_samples: int | None,
    max_recording_seconds: float | None,
    feature_mode: str,
):
    yielded = 0
    for _, row in ds.metadata.iterrows():
        patient_id = int(row["Patient ID"])
        if selected_patient_ids is not None and patient_id not in selected_patient_ids:
            continue
        label = 1 if row["Murmur"] == "Present" else 0
        if label_filter == "present" and label != 1:
            continue
        if label_filter == "absent" and label != 0:
            continue
        for loc, wav in _patient_recordings(ds.root, patient_id, row["Recording locations:"]):
            audio = prepare_audio(wav, apply_filter, bandpass, max_recording_seconds)
            windows = split_audio_windows(audio, window_samples, hop_samples)
            if len(windows) == 0:
                continue
            for w_idx, window in enumerate(windows):
                if max_windows is not None and yielded >= max_windows:
                    return
                x = model_input_from_window(window, pad_to_samples, feature_mode)
                raw = window.astype(np.float32, copy=False) if include_raw else None
                meta = {
                    "idx": yielded,
                    "patient_id": patient_id,
                    "location": str(loc),
                    "recording": str(wav),
                    "window_idx": int(w_idx),
                }
                yielded += 1
                yield meta, x, raw, label


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def run(args: argparse.Namespace) -> dict:
    if args.feature_mode != FEATURE_MODE_LOGMEL and (args.coreml or args.onnx):
        raise SystemExit("--feature-mode mfcc/multi is only supported for --checkpoint benchmarking")

    ds = CirCorMurmurDataset(
        args.data,
        apply_cardiac=args.apply_cardiac,
        feature_mode=args.feature_mode,
    )
    bench_ds = choose_split(ds, args.split, args.val_fraction, args.seed)
    bench_ds = filter_by_label(bench_ds, args.label_filter)
    custom_windowing = (
        args.bench_window_seconds != WINDOW_SAMPLES / SAMPLE_RATE
        or args.bench_hop_seconds != WINDOW_HOP / SAMPLE_RATE
        or args.bandpass is not None
        or args.max_recording_seconds is not None
    )
    selected_patient_ids = patient_ids_for_dataset(bench_ds) if custom_windowing else None
    if custom_windowing:
        log.info(
            "dataset: %s label=%s custom windows %.3fs hop %.3fs",
            args.split,
            args.label_filter,
            args.bench_window_seconds,
            args.bench_hop_seconds,
        )
    else:
        log.info("dataset: %s label=%s windows=%d", args.split, args.label_filter, len(bench_ds))

    scorers: list[Scorer] = []
    if args.checkpoint:
        scorers.append(
            TorchScorer(
                args.checkpoint,
                architecture=args.architecture,
                in_channels=feature_channels(args.feature_mode),
            )
        )
    if args.coreml:
        scorers.append(CoreMLScorer(args.coreml))
    if args.onnx:
        scorers.append(
            OnnxScorer(
                args.onnx,
                output_kind=args.onnx_output_kind,
                output_index=args.onnx_output_index,
            )
        )
    if not scorers:
        raise SystemExit("provide at least one of --checkpoint, --coreml, or --onnx")

    labels: list[int] = []
    example_meta: list[dict[str, object]] = []
    scores: dict[str, list[float]] = {s.name: [] for s in scorers}
    latencies: dict[str, list[float]] = {s.name: [] for s in scorers}
    rows: list[dict[str, float | int | str]] = []

    bandpass = tuple(args.bandpass) if args.bandpass is not None else None
    window_samples = int(round(args.bench_window_seconds * SAMPLE_RATE))
    hop_samples = int(round(args.bench_hop_seconds * SAMPLE_RATE))
    pad_to_samples = WINDOW_SAMPLES if args.pad_to_model_window else None
    examples = (
        iter_custom_examples(
            ds,
            selected_patient_ids,
            args.label_filter,
            args.max_windows,
            include_raw=bool(args.onnx and args.onnx_input == "raw4000"),
            apply_filter=args.apply_cardiac,
            bandpass=bandpass,
            window_samples=window_samples,
            hop_samples=hop_samples,
            pad_to_samples=pad_to_samples,
            max_recording_seconds=args.max_recording_seconds,
            feature_mode=args.feature_mode,
        )
        if custom_windowing
        else iter_examples(
            bench_ds,
            args.max_windows,
            include_raw=bool(args.onnx and args.onnx_input == "raw4000"),
            apply_filter=args.apply_cardiac,
        )
    )

    for meta, x, raw, label in examples:
        labels.append(label)
        example_meta.append(meta)
        row: dict[str, float | int | str] = {
            "idx": int(meta["idx"]),
            "patient_id": int(meta["patient_id"]),
            "location": str(meta["location"]),
            "recording": str(meta["recording"]),
            "window_idx": int(meta["window_idx"]),
            "label": label,
        }
        for scorer in scorers:
            if isinstance(scorer, OnnxScorer) and args.onnx_input == "raw4000":
                if raw is None:
                    raise RuntimeError("raw audio was not loaded for ONNX raw input")
                prob, latency_ms = scorer.score_raw_4s(raw, args.onnx_aggregate)
            else:
                prob, latency_ms = scorer.score(x)
            scores[scorer.name].append(prob)
            latencies[scorer.name].append(latency_ms)
            row[f"{scorer.name}_prob"] = prob
            row[f"{scorer.name}_latency_ms"] = latency_ms
        rows.append(row)

    labels_arr = np.asarray(labels, dtype=np.int64)
    report: dict[str, object] = {
        "data": str(args.data),
        "split": args.split,
        "label_filter": args.label_filter,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "apply_cardiac": args.apply_cardiac,
        "feature_mode": args.feature_mode,
        "input_channels": feature_channels(args.feature_mode),
        "bandpass": list(args.bandpass) if args.bandpass is not None else None,
        "bench_window_seconds": args.bench_window_seconds,
        "bench_hop_seconds": args.bench_hop_seconds,
        "pad_to_model_window": args.pad_to_model_window,
        "max_recording_seconds": args.max_recording_seconds,
        "onnx_input": args.onnx_input if args.onnx else None,
        "onnx_aggregate": args.onnx_aggregate if args.onnx and args.onnx_input == "raw4000" else None,
        "n_windows": int(labels_arr.size),
        "n_recordings": len({str(m["recording"]) for m in example_meta}),
        "models": {},
        "threshold_summaries": {},
        "recording_level": {},
        "recording_vote": {},
        "pairwise": {},
    }

    vote_thresholds = parse_float_list(args.vote_thresholds)
    vote_counts = parse_int_list(args.vote_counts)
    for name, vals in scores.items():
        arr = np.asarray(vals, dtype=np.float64)
        latency = np.asarray(latencies[name], dtype=np.float64)
        model_report = binary_metrics(labels_arr, arr, args.threshold)
        model_report["mean_latency_ms"] = float(latency.mean()) if latency.size else float("nan")
        model_report["p95_latency_ms"] = float(np.percentile(latency, 95)) if latency.size else float("nan")
        report["models"][name] = model_report  # type: ignore[index]
        report["threshold_summaries"][name] = sweep_summary(labels_arr, arr)  # type: ignore[index]

        rec_modes: dict[str, object] = {}
        for mode in args.recording_aggregates:
            rec_labels, rec_scores = recording_level(example_meta, labels_arr, arr, mode)
            rec_modes[mode] = {
                "threshold_0_5": binary_metrics(rec_labels, rec_scores, args.threshold),
                "best_f1": best_by(sweep_thresholds(rec_labels, rec_scores), "f1"),
                "best_youden_j": best_by(sweep_thresholds(rec_labels, rec_scores), "youden_j"),
            }
        report["recording_level"][name] = rec_modes  # type: ignore[index]
        vote_rows = vote_sweep(example_meta, labels_arr, arr, vote_thresholds, vote_counts)
        report["recording_vote"][name] = vote_summary(vote_rows)  # type: ignore[index]

    names = list(scores)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            a = np.asarray(scores[left], dtype=np.float64)
            b = np.asarray(scores[right], dtype=np.float64)
            key = f"{left}_vs_{right}"
            report["pairwise"][key] = {  # type: ignore[index]
                "mean_abs_prob_diff": float(np.mean(np.abs(a - b))),
                "max_abs_prob_diff": float(np.max(np.abs(a - b))),
                "corrcoef": float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else float("nan"),
                "same_decision_at_threshold": float(
                    np.mean((a >= args.threshold) == (b >= args.threshold))
                ),
            }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            fieldnames = list(rows[0].keys()) if rows else ["idx", "label"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        report["predictions_csv"] = str(args.out)

    if args.sweep_out:
        args.sweep_out.parent.mkdir(parents=True, exist_ok=True)
        with args.sweep_out.open("w", newline="") as f:
            fieldnames = [
                "model",
                "level",
                "aggregation",
                "threshold",
                "n",
                "positives",
                "auroc",
                "accuracy",
                "sensitivity",
                "specificity",
                "precision",
                "f1",
                "youden_j",
                "tp",
                "fp",
                "tn",
                "fn",
                "window_threshold",
                "min_positive_windows",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name, vals in scores.items():
                arr = np.asarray(vals, dtype=np.float64)
                for metrics in sweep_thresholds(labels_arr, arr):
                    writer.writerow({"model": name, "level": "window", "aggregation": "none", **metrics})
                for mode in args.recording_aggregates:
                    rec_labels, rec_scores = recording_level(example_meta, labels_arr, arr, mode)
                    for metrics in sweep_thresholds(rec_labels, rec_scores):
                        writer.writerow({
                            "model": name,
                            "level": "recording",
                            "aggregation": mode,
                            **metrics,
                        })
                for metrics in vote_sweep(example_meta, labels_arr, arr, vote_thresholds, vote_counts):
                    writer.writerow({
                        "model": name,
                        "level": "recording",
                        "aggregation": "vote_count",
                        **metrics,
                    })
        report["threshold_sweep_csv"] = str(args.sweep_out)

    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True, help="CirCor root")
    p.add_argument("--checkpoint", type=Path, default=None, help="PyTorch MurmurCNN checkpoint")
    p.add_argument("--architecture", choices=["cnn", "cnn_bigru", "scattering_cnn1d"], default="cnn")
    p.add_argument(
        "--feature-mode",
        choices=FEATURE_MODES,
        default=FEATURE_MODE_LOGMEL,
        help="PyTorch checkpoint feature representation; Core ML/ONNX benchmark inputs remain logmel",
    )
    p.add_argument("--coreml", type=Path, default=None, help="Core ML MurmurCNN.mlpackage")
    p.add_argument("--onnx", type=Path, default=None, help="ONNX baseline model")
    p.add_argument("--onnx-output-kind", choices=["logit", "prob"], default="logit")
    p.add_argument("--onnx-output-index", type=int, default=0)
    p.add_argument(
        "--onnx-input",
        choices=["logmel", "raw4000"],
        default="logmel",
        help="ONNX input contract: logmel=(1,1,62,32), raw4000=(1,4000,1)",
    )
    p.add_argument("--onnx-aggregate", choices=["max", "mean", "center"], default="max")
    p.add_argument("--split", choices=["train", "val", "all"], default="val")
    p.add_argument("--label-filter", choices=["all", "present", "absent"], default="all")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--bench-window-seconds", type=float, default=WINDOW_SAMPLES / SAMPLE_RATE)
    p.add_argument("--bench-hop-seconds", type=float, default=WINDOW_HOP / SAMPLE_RATE)
    p.add_argument(
        "--pad-to-model-window",
        action="store_true",
        help="pad shorter benchmark windows back to the exported model's 4-second input length",
    )
    p.add_argument(
        "--max-recording-seconds",
        type=float,
        default=None,
        help="optional leading duration to score from each recording, e.g. 10 for a 7-window 2.5s/1.25s vote rule",
    )
    p.add_argument(
        "--bandpass",
        nargs=2,
        type=float,
        metavar=("LOW_HZ", "HIGH_HZ"),
        default=None,
        help="optional offline zero-phase bandpass before mel features, e.g. --bandpass 25 700",
    )
    p.add_argument(
        "--vote-thresholds",
        default="0.3,0.4,0.49331352,0.5,0.6,0.7",
        help="comma-separated window probability thresholds for recording vote sweeps",
    )
    p.add_argument(
        "--vote-counts",
        default="1,2,3,4,5,6,7",
        help="comma-separated minimum positive-window counts for recording vote sweeps",
    )
    p.add_argument(
        "--recording-aggregates",
        nargs="+",
        choices=["max", "mean", "median", "top3_mean"],
        default=["max", "mean", "top3_mean"],
    )
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument(
        "--apply-cardiac",
        action="store_true",
        help="apply the legacy cardiac filter before mel features; default off matches stetho-ui",
    )
    p.add_argument("--out", type=Path, default=None, help="optional per-window predictions CSV")
    p.add_argument("--sweep-out", type=Path, default=None, help="optional threshold sweep CSV")
    p.add_argument("--json", type=Path, default=None, help="optional JSON report path")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = json_safe(run(args))
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
