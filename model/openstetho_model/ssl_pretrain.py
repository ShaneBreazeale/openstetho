"""SimCLR-style self-supervised pretraining for the S3 detector encoder.

Trains an S3CNN_v3 (or any other backbone in `_BACKBONES`) without any S3
labels, using NT-Xent contrastive loss over augmented view pairs. After
pretraining, the encoder weights are saved as `encoder.pt`; downstream
supervised fine-tuning loads those weights via `train_s3 --init-from`.

The intent is to break past the ~AUPRC 0.89 synth-label ceiling by giving
the encoder richer features grounded in the *real* acoustic distribution
of all 7235 PCG recordings, not just the patterns our synthetic injection
can produce.

CLI:
    uv run python -m openstetho_model.ssl_pretrain \\
        --wavs <roots ...> \\
        --epochs 30 --batch-size 256 \\
        --backbone s3cnn_v3 \\
        --out runs/ssl_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import MurmurCNN, S3CNN, S3CNN_v2, S3CNN_v3, count_parameters
from .ssl_dataset import ContrastivePCGDataset

log = logging.getLogger(__name__)

_BACKBONES = {
    "murmur": MurmurCNN,
    "s3cnn": S3CNN,
    "s3cnn_v2": S3CNN_v2,
    "s3cnn_v3": S3CNN_v3,
}


# ─── encoder wrapper: strip the classifier head ──────────────────────────────

class EncoderProjector(nn.Module):
    """Wraps a backbone (without its head) + a 2-layer MLP projector.

    We rely on `nn.Module.children` to take everything except the last
    `head` attribute. This keeps the contrastive loss simple — no need
    to surgery each backbone individually.
    """

    def __init__(self, backbone: nn.Module, projection_dim: int = 64):
        super().__init__()
        self.backbone = backbone
        # Determine the encoder output size by running a dummy forward up
        # to the head input. Every backbone in this project funnels through
        # a layer immediately before `head` whose output is `head[0]` input
        # size — for `Sequential` heads with a `Linear` at index 2 (after
        # Flatten + Dropout) we read `in_features`.
        head: nn.Sequential = backbone.head  # type: ignore[assignment]
        # The first `Linear` inside the head reveals the encoder output dim.
        for module in head:
            if isinstance(module, nn.Linear):
                self.encoder_dim = module.in_features
                break
        else:
            raise RuntimeError("backbone.head has no Linear layer")
        # Disable the original head so forward(...) returns the pooled
        # encoder output. We replicate the head's pre-Linear ops (Flatten +
        # Dropout-or-not) by running everything up to the first Linear.
        new_head: list[nn.Module] = []
        for module in head:
            if isinstance(module, nn.Linear):
                break
            new_head.append(module)
        backbone.head = nn.Sequential(*new_head)
        self.projector = nn.Sequential(
            nn.Linear(self.encoder_dim, self.encoder_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_dim, projection_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode(x)
        z = self.projector(h)
        # L2-normalise for cosine similarity in NT-Xent.
        return nn.functional.normalize(z, dim=1)


# ─── NT-Xent loss (Chen et al. 2020) ────────────────────────────────────────

def nt_xent_loss(z: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Inputs: `z` of shape `(2N, D)` where rows `[2i, 2i+1]` are positive
    pairs. Returns scalar NT-Xent loss.
    """
    n2 = z.size(0)
    sim = z @ z.t() / temperature  # (2N, 2N)
    # Mask self-similarities.
    mask = torch.eye(n2, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -1e9)
    # Positive index for row 2i is 2i+1; for row 2i+1 is 2i.
    targets = torch.arange(n2, device=z.device)
    targets = targets ^ 1  # XOR with 1 flips the last bit -> pairs row<->row±1
    return nn.functional.cross_entropy(sim, targets)


# ─── training loop ──────────────────────────────────────────────────────────

def _interleave_views(view_a: torch.Tensor, view_b: torch.Tensor) -> torch.Tensor:
    """Returns (2N, ...) tensor with views interleaved so row 2i and 2i+1
    are a positive pair."""
    n = view_a.size(0)
    out = torch.empty(
        (2 * n, *view_a.shape[1:]),
        dtype=view_a.dtype,
        device=view_a.device,
    )
    out[0::2] = view_a
    out[1::2] = view_b
    return out


def pretrain(
    wavs: list[Path],
    out_dir: Path,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature: float = 0.1,
    backbone: str = "s3cnn_v3",
    seed: int = 0,
    segmenter: str = "heuristic",
    crop_anchor: str = "s2",
    snr_db_range: tuple[float, float] = (-5.0, 15.0),
    freq_mask_max_width: int = 12,
    time_mask_max_width: int = 4,
    prob_multi: float = 0.5,
    prob_s4: float = 0.5,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    # Note: prob_multi / prob_s4 / snr_db_range are intentionally NOT passed.
    # Cardiac content must be identical across the two contrastive views
    # of the same cycle, otherwise the encoder learns to ignore S3/S4. The
    # only aug applied during SSL is recording-condition jitter + spec masks.
    dataset = ContrastivePCGDataset(
        wavs,
        freq_mask_max_width=freq_mask_max_width,
        time_mask_max_width=time_mask_max_width,
        seed=seed,
        segmenter=segmenter,
        crop_anchor=crop_anchor,
    )
    if len(dataset) == 0:
        raise RuntimeError("SSL dataset is empty")
    log.info("SSL dataset: %d cycles across %d wavs", len(dataset), len(wavs))

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    base = _BACKBONES[backbone](n_classes=1).to(device)
    model = EncoderProjector(base, projection_dim=64).to(device)
    log.info("backbone=%s | total params=%d | encoder dim=%d",
             backbone, count_parameters(model), model.encoder_dim)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    metrics_csv = out_dir / "metrics.csv"
    with metrics_csv.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "items_per_sec", "wall_seconds"])

    live_path = out_dir / "live.json"

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        running = 0.0
        seen = 0
        for batch_no, (view_a, view_b) in enumerate(loader, start=1):
            view_a = view_a.to(device)
            view_b = view_b.to(device)
            x = _interleave_views(view_a, view_b)
            optimizer.zero_grad()
            z = model(x)
            loss = nt_xent_loss(z, temperature=temperature)
            loss.backward()
            optimizer.step()
            n = view_a.size(0)
            running += float(loss.item()) * n
            seen += n

            if batch_no % 5 == 0:
                elapsed = time.perf_counter() - epoch_start
                _write_live(
                    live_path,
                    epoch=epoch,
                    epochs=epochs,
                    batch_no=batch_no,
                    total_batches=len(loader),
                    seen=seen,
                    running_loss=running / max(seen, 1),
                    items_per_sec=seen / max(elapsed, 1e-6),
                    phase="ssl_pretrain",
                )

        epoch_loss = running / max(seen, 1)
        elapsed = time.perf_counter() - epoch_start
        log.info(
            "epoch %d: ssl_loss=%.4f items/sec=%.1f wall=%.1fs",
            epoch, epoch_loss, seen / max(elapsed, 1e-6), elapsed,
        )
        with metrics_csv.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, epoch_loss, seen / max(elapsed, 1e-6), elapsed])

        # Save encoder + projector each epoch so a crash doesn't lose progress.
        torch.save(
            {
                "backbone": backbone,
                "encoder_state_dict": model.backbone.state_dict(),
                "projector_state_dict": model.projector.state_dict(),
                "encoder_dim": model.encoder_dim,
                "epoch": epoch,
            },
            out_dir / "encoder.pt",
        )
        scheduler.step()

    log.info("SSL pretraining complete -> %s", out_dir / "encoder.pt")


def _write_live(path: Path, **fields) -> None:
    now = time.perf_counter()
    last = getattr(_write_live, "_last_t", 0.0)
    if now - last < 0.25:
        return
    _write_live._last_t = now  # type: ignore[attr-defined]
    fields["timestamp"] = time.time()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(fields))
    tmp.replace(path)


# ─── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--wavs", required=True, nargs="+")
    p.add_argument("--pattern", default="*.wav")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--backbone", choices=list(_BACKBONES), default="s3cnn_v3")
    p.add_argument("--segmenter", choices=["heuristic", "hsmm"], default="heuristic")
    p.add_argument("--crop-anchor", choices=["s1", "s2"], default="s2")
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

    pretrain(
        wavs=wavs,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        backbone=args.backbone,
        seed=args.seed,
        segmenter=args.segmenter,
        crop_anchor=args.crop_anchor,
    )


if __name__ == "__main__":
    main()
