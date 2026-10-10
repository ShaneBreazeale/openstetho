from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from openstetho_model.train import run_epoch


class _RecordingLoss(nn.Module):
    """BCE that keeps a CPU copy of every label batch it is handed."""

    def __init__(self):
        super().__init__()
        self.seen: list[torch.Tensor] = []

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self.seen.append(labels.detach().cpu().clone())
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)


class _Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an MPS device")
def test_run_epoch_labels_survive_transfer_to_mps():
    # Many small batches give the allocator plenty of chances to reuse a
    # freed CPU buffer while an async host-to-device copy is in flight.
    torch.manual_seed(0)
    x = torch.randn(4096, 8)
    y = (torch.rand(4096) > 0.5).float()
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
    dev = torch.device("mps")
    model = _Linear().to(dev)
    loss_fn = _RecordingLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    loss, _auc = run_epoch(model, loader, loss_fn, opt, dev)

    seen = torch.cat(loss_fn.seen)
    assert seen.numel() == 4096
    assert set(seen.unique().tolist()) <= {0.0, 1.0}
    assert loss < 10.0
