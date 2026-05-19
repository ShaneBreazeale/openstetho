"""Linear-probe make-or-break test for SSL pretraining.

Per [[feedback-ssl-must-teach-physiology]]: after pretraining, freeze the
encoder and train tiny linear classifiers on synthetically-labeled cycle
sets. If any probe scores near chance, the SSL run is "vibes" — it has
learned to memorize recording fingerprints, not physiology — and the
augmentation suite must be revised before any downstream fine-tune.

Tasks (all binary unless stated):
    s3_vs_clean         — class 0 untouched cycle, class 1 cycle with synth S3
    s3_vs_s4            — class 0 cycle with synth S4, class 1 cycle with synth S3
    s3_vs_artifact      — class 0 cycle with split-S2 / opening-snap /
                          ejection-click confounder, class 1 cycle with S3
    s3_vs_others        — three-way: clean / S3 / S4

Each probe uses ~`--n-per-class` cycles drawn from the supplied WAV roots,
encodes them through the frozen backbone, and fits a tiny MLP head
(1 hidden layer, default 32 units) for ~50 epochs.

Output is a per-task accuracy report. We call the SSL run "physiologic" if
accuracy is >= 0.75 on all three single-binary probes — that gate is
adjustable via `--pass-threshold`.

CLI:
    uv run python -m openstetho_model.ssl_probe \\
        --encoder runs/ssl_v1/encoder.pt \\
        --wavs /path/to/circor \\
        --task s3_vs_s4 \\
        --n-per-class 300
"""
from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model import MurmurCNN, S3CNN, S3CNN_v2, S3CNN_v3, count_parameters
from .preprocess import apply_s3_preset, load_audio, log_mel
from .s3_dataset import (
    CYCLE_WINDOW_FRAMES,
    N_MELS,
    _cycle_crop_s2,
)
from .s3_inject import (
    ejection_click_inject,
    opening_snap_inject,
    s3_inject,
    s4_inject,
    split_s2_inject,
)
from .segment import segment_unified

log = logging.getLogger(__name__)

_BACKBONES = {
    "murmur": MurmurCNN,
    "s3cnn": S3CNN,
    "s3cnn_v2": S3CNN_v2,
    "s3cnn_v3": S3CNN_v3,
}


# ─── synthetic-event cycle generators ────────────────────────────────────────

def _make_single_cycle(audio: np.ndarray, segmentation, cycle_idx: int):
    return type(segmentation)(
        cycles=[segmentation.cycles[cycle_idx]],
        cycle_period_s=segmentation.cycle_period_s,
        confidence=segmentation.confidence,
    )


def _cycle_to_mel(audio: np.ndarray, cycle, sample_rate: int = 4000) -> np.ndarray:
    cropped = _cycle_crop_s2(audio, cycle.s2_idx)
    cropped = apply_s3_preset(cropped)
    mel = log_mel(cropped)
    if mel.shape[0] != CYCLE_WINDOW_FRAMES:
        padded = np.zeros((CYCLE_WINDOW_FRAMES, N_MELS), dtype=np.float32)
        keep = min(mel.shape[0], CYCLE_WINDOW_FRAMES)
        padded[:keep] = mel[:keep]
        mel = padded
    return mel


def _gather_cycles(
    wavs: list[Path],
    n_per_class: int,
    class_specs: list[tuple[str, Callable[[np.ndarray, object, np.random.Generator], np.ndarray]]],
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterate WAVs, segment, draw cycles, apply the per-class event
    injection, and return (mels, labels).

    `class_specs[i] = (name, inject_fn)` where `inject_fn(audio, single_seg,
    rng)` returns the audio with the appropriate event planted (or the
    audio unchanged for the "clean" class).
    """
    rng = np.random.default_rng(seed)
    mels: list[np.ndarray] = []
    labels: list[int] = []
    needed = {cls: n_per_class for cls in range(len(class_specs))}
    wav_iter = list(wavs)
    rng.shuffle(wav_iter)

    for wav in wav_iter:
        if all(v <= 0 for v in needed.values()):
            break
        try:
            audio = load_audio(str(wav))
        except Exception as e:  # noqa: BLE001
            log.debug("skip %s: %s", wav, e)
            continue
        seg = segment_unified(audio, method="heuristic")
        if seg.confidence < 0.3 or not seg.cycles:
            continue
        cycle_order = list(range(len(seg.cycles)))
        rng.shuffle(cycle_order)
        for k in cycle_order:
            if all(v <= 0 for v in needed.values()):
                break
            cls_id = int(rng.integers(0, len(class_specs)))
            if needed[cls_id] <= 0:
                continue
            single = _make_single_cycle(audio, seg, k)
            _, fn = class_specs[cls_id]
            mutated = fn(audio.copy(), single, rng)
            mel = _cycle_to_mel(mutated, single.cycles[0])
            mels.append(mel)
            labels.append(cls_id)
            needed[cls_id] -= 1

    if not mels:
        raise RuntimeError("no probe cycles gathered — check WAV roots and segmenter confidence floor")
    return np.stack(mels, axis=0), np.asarray(labels, dtype=np.int64)


# ─── injection wrappers matching the class spec interface ───────────────────

def _inject_clean(audio, _seg, _rng):
    return audio


def _inject_s3(audio, seg, rng):
    out, _ = s3_inject(audio, seg, rng, prob_per_cycle=1.0, snr_db_range=(8.0, 15.0))
    return out


def _inject_s4(audio, seg, rng):
    out = s4_inject(audio, seg, rng, prob_per_cycle=1.0, snr_db_range=(8.0, 15.0))
    # return_flags default is False so out is the array.
    return out


def _inject_split_s2(audio, seg, rng):
    return split_s2_inject(audio, seg, rng, prob_per_cycle=1.0, snr_db_range=(8.0, 15.0))


def _inject_opening_snap(audio, seg, rng):
    return opening_snap_inject(audio, seg, rng, prob_per_cycle=1.0, snr_db_range=(8.0, 15.0))


def _inject_ejection_click(audio, seg, rng):
    return ejection_click_inject(audio, seg, rng, prob_per_cycle=1.0, snr_db_range=(8.0, 15.0))


def _inject_random_artifact(audio, seg, rng):
    fn = rng.choice([_inject_split_s2, _inject_opening_snap, _inject_ejection_click])
    return fn(audio, seg, rng)


TASK_SPECS: dict[str, list[tuple[str, Callable]]] = {
    "s3_vs_clean":    [("clean", _inject_clean), ("s3", _inject_s3)],
    "s3_vs_s4":       [("s4", _inject_s4),       ("s3", _inject_s3)],
    "s3_vs_artifact": [("artifact", _inject_random_artifact), ("s3", _inject_s3)],
    "s3_vs_others":   [("clean", _inject_clean), ("s3", _inject_s3), ("s4", _inject_s4)],
}


# ─── probe model: frozen encoder + small MLP head ───────────────────────────

def _encoder_only(backbone_cls, ckpt_path: Path, device: torch.device) -> tuple[nn.Module, int]:
    """Load a backbone, strip its classification head to expose the encoder
    output, and load encoder weights from `ckpt_path`. Returns the encoder
    module and the encoder output dimensionality.
    """
    base = backbone_cls(n_classes=1).to(device)
    head: nn.Sequential = base.head  # type: ignore[assignment]
    encoder_dim = None
    new_head: list[nn.Module] = []
    for module in head:
        if isinstance(module, nn.Linear):
            encoder_dim = module.in_features
            break
        new_head.append(module)
    if encoder_dim is None:
        raise RuntimeError("backbone.head has no Linear layer")
    base.head = nn.Sequential(*new_head)

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "encoder_state_dict" in ckpt:
        # SSL checkpoint format.
        missing, unexpected = base.load_state_dict(ckpt["encoder_state_dict"], strict=False)
    else:
        # Plain torch state_dict — supervised checkpoint fallback. The head's
        # Linear weights won't match (we stripped them) so use strict=False.
        missing, unexpected = base.load_state_dict(ckpt, strict=False)
    log.info("encoder load: missing=%d unexpected=%d (head Linears expected to be missing)",
             len(missing), len(unexpected))

    for p in base.parameters():
        p.requires_grad = False
    base.train(False)
    return base, encoder_dim


class ProbeHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def _encode_all(encoder: nn.Module, mels: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(mels), batch_size):
            batch = torch.from_numpy(mels[i : i + batch_size]).to(device)
            feat = encoder(batch).cpu().numpy()
            out.append(feat)
    return np.concatenate(out, axis=0)


def _train_probe(
    features: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    hidden: int = 32,
    epochs: int = 50,
    lr: float = 3e-3,
    val_frac: float = 0.2,
    device: torch.device | None = None,
) -> dict:
    device = device or torch.device("cpu")
    n = len(features)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    features = features[perm]
    labels = labels[perm]
    n_val = max(8, int(n * val_frac))
    x_train = torch.from_numpy(features[n_val:]).to(device)
    y_train = torch.from_numpy(labels[n_val:]).to(device)
    x_val = torch.from_numpy(features[:n_val]).to(device)
    y_val = torch.from_numpy(labels[:n_val]).to(device)

    model = ProbeHead(features.shape[1], hidden, n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    train_ds = TensorDataset(x_train, y_train)
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
        model.train(False)
        with torch.no_grad():
            preds = model(x_val).argmax(dim=1)
            acc = float((preds == y_val).float().mean().item())
        best_val_acc = max(best_val_acc, acc)

    # Per-class precision / recall on the final epoch.
    model.train(False)
    with torch.no_grad():
        preds = model(x_val).argmax(dim=1).cpu().numpy()
    y_val_np = y_val.cpu().numpy()
    per_class: dict[int, dict] = {}
    for cls in range(n_classes):
        tp = int(((preds == cls) & (y_val_np == cls)).sum())
        fp = int(((preds == cls) & (y_val_np != cls)).sum())
        fn = int(((preds != cls) & (y_val_np == cls)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        per_class[cls] = {"precision": prec, "recall": rec, "support": int((y_val_np == cls).sum())}
    return {"best_val_acc": best_val_acc, "n_train": int(n - n_val), "n_val": int(n_val), "per_class": per_class}


def run(
    encoder_ckpt: Path,
    wavs: list[Path],
    task: str,
    n_per_class: int,
    backbone: str,
    pass_threshold: float = 0.75,
    seed: int = 0,
) -> bool:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    encoder, encoder_dim = _encoder_only(_BACKBONES[backbone], encoder_ckpt, device)
    log.info("frozen encoder loaded (dim=%d, params=%d)", encoder_dim, count_parameters(encoder))

    if task not in TASK_SPECS:
        raise ValueError(f"unknown task {task}; choose from {sorted(TASK_SPECS)}")
    specs = TASK_SPECS[task]
    log.info("probe task=%s | classes=%s | n_per_class=%d", task, [s[0] for s in specs], n_per_class)

    mels, labels = _gather_cycles(wavs, n_per_class, specs, seed=seed)
    log.info("gathered %d labeled cycles", len(mels))
    features = _encode_all(encoder, mels, device)
    log.info("encoded features shape: %s", features.shape)

    metrics = _train_probe(features, labels, n_classes=len(specs), device=device)
    passed = metrics["best_val_acc"] >= pass_threshold
    print(f"task           : {task}")
    print(f"n_train        : {metrics['n_train']}")
    print(f"n_val          : {metrics['n_val']}")
    print(f"best_val_acc   : {metrics['best_val_acc']:.3f}")
    for cls, info in metrics["per_class"].items():
        cls_name = specs[cls][0]
        print(f"  class {cls} ({cls_name:10s}) | prec={info['precision']:.3f} rec={info['recall']:.3f} n={info['support']}")
    print(f"pass threshold : {pass_threshold:.2f}")
    print(f"verdict        : {'PASS' if passed else 'FAIL — encoder may be teaching vibes'}")
    return passed


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", required=True, type=Path)
    p.add_argument("--wavs", required=True, nargs="+")
    p.add_argument("--pattern", default="*.wav")
    p.add_argument("--task", choices=sorted(TASK_SPECS), required=True)
    p.add_argument("--n-per-class", type=int, default=300)
    p.add_argument("--backbone", choices=list(_BACKBONES), default="s3cnn_v3")
    p.add_argument("--pass-threshold", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    wavs: list[Path] = []
    for root in args.wavs:
        rp = Path(root)
        if rp.is_file():
            wavs.append(rp)
        else:
            wavs.extend(sorted(rp.rglob(args.pattern)))
    if not wavs:
        raise SystemExit("no WAVs matched")
    log.info("found %d wavs for probe sampling", len(wavs))

    run(
        encoder_ckpt=args.encoder,
        wavs=wavs,
        task=args.task,
        n_per_class=args.n_per_class,
        backbone=args.backbone,
        pass_threshold=args.pass_threshold,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
