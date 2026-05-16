"""Train MurmurCNN on CirCor 2022.

Usage (from `model/`):

    uv run python -m openstetho_model.train \
        --data ../data/circor \
        --epochs 30 \
        --batch-size 64 \
        --out runs/circor_v1
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from .dataset import CirCorMurmurDataset
from .model import MurmurCNN, count_parameters

log = logging.getLogger("train")


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def patient_split(ds: CirCorMurmurDataset, val_fraction: float = 0.15, seed: int = 0):
    """Group-aware split: every window of a given patient lands in one side
    of the split. Prevents leakage at the patient level."""
    patient_ids = sorted({pid for pid, *_ in ds._index})
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)
    cut = int(len(patient_ids) * (1.0 - val_fraction))
    train_ids = set(patient_ids[:cut])
    val_ids = set(patient_ids[cut:])
    train_idx = [i for i, (pid, *_) in enumerate(ds._index) if pid in train_ids]
    val_idx = [i for i, (pid, *_) in enumerate(ds._index) if pid in val_ids]
    return Subset(ds, train_idx), Subset(ds, val_idx)


def run_epoch(
    model: nn.Module,
    loader: Iterable,
    loss_fn: nn.Module,
    opt: torch.optim.Optimizer | None,
    dev: torch.device,
) -> tuple[float, float]:
    model.train(opt is not None)
    losses: list[float] = []
    probs: list[float] = []
    labels: list[float] = []
    for mel, label in loader:
        mel = mel.to(dev, non_blocking=True)
        label = label.to(dev, non_blocking=True).float()
        with torch.set_grad_enabled(opt is not None):
            logit = model(mel)
            loss = loss_fn(logit, label)
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()
        losses.append(loss.item())
        probs.extend(torch.sigmoid(logit).detach().cpu().numpy().tolist())
        labels.extend(label.detach().cpu().numpy().tolist())
    auc = float("nan")
    if len(set(labels)) == 2:
        auc = roc_auc_score(labels, probs)
    return float(np.mean(losses)), auc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True, help="CirCor root (training_data.csv lives here)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("runs/last"))
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    dev = device()
    torch.manual_seed(args.seed)
    log.info("device: %s", dev)

    ds = CirCorMurmurDataset(args.data)
    log.info("dataset: %d windows | balance: %s", len(ds), ds.class_balance())
    train_ds, val_ds = patient_split(ds, args.val_fraction, args.seed)
    log.info("split:  train=%d  val=%d", len(train_ds), len(val_ds))

    pin = dev.type in ("cuda",)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin,
    )

    model = MurmurCNN().to(dev)
    log.info("params: %d", count_parameters(model))
    # Class-balanced loss: pos_weight = N_neg / N_pos so the minority
    # (murmur-present) class gets a proportional gradient pull.
    bal = ds.class_balance()
    pos_weight_val = max(1.0, bal["absent"] / max(bal["present"], 1))
    log.info("pos_weight: %.3f", pos_weight_val)
    pos_weight = torch.tensor([pos_weight_val], device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_auc = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_auc = run_epoch(model, train_loader, loss_fn, opt, dev)
        va_loss, va_auc = run_epoch(model, val_loader, loss_fn, None, dev)
        history.append({"epoch": epoch, "tr_loss": tr_loss, "tr_auc": tr_auc, "va_loss": va_loss, "va_auc": va_auc})
        log.info("ep %02d | tr loss %.4f auc %.3f | va loss %.4f auc %.3f", epoch, tr_loss, tr_auc, va_loss, va_auc)
        if va_auc > best_auc:
            best_auc = va_auc
            torch.save(model.state_dict(), args.out / "best.pt")
            (args.out / "best_meta.json").write_text(json.dumps({"epoch": epoch, "val_auc": va_auc}))

    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    log.info("best val AUC: %.3f → %s/best.pt", best_auc, args.out)


if __name__ == "__main__":
    main()
