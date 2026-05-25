from __future__ import annotations

import torch

from openstetho_model.train import aggregate_logits, build_model, recording_collate


def test_aggregate_logits_mean():
    logits = torch.tensor([-1.0, 0.0, 2.0])
    assert aggregate_logits(logits, "mean", topk=2).item() == torch.tensor(1.0 / 3.0).item()


def test_aggregate_logits_topk_mean_clamps_k():
    logits = torch.tensor([-1.0, 0.0, 2.0])
    assert aggregate_logits(logits, "topk_mean", topk=2).item() == 1.0
    assert aggregate_logits(logits, "topk_mean", topk=99).item() == torch.tensor(1.0 / 3.0).item()


def test_recording_collate_keeps_variable_window_counts():
    rec_a = torch.zeros(2, 62, 32)
    rec_b = torch.zeros(5, 62, 32)
    recordings, labels = recording_collate([(rec_a, 0), (rec_b, 1)])
    assert [r.shape[0] for r in recordings] == [2, 5]
    assert labels.tolist() == [0.0, 1.0]


def test_cnn_bigru_forward_accepts_window_batch():
    model = build_model("cnn_bigru")
    x = torch.zeros(2, 62, 32)
    y = model(x)
    assert y.shape == (2,)
