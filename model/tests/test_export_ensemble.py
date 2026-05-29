"""Unit test for the fused-ensemble export math.

The deployment-critical invariant is that the fused model averages member
*probabilities* (matching how the ensemble was benchmarked) and re-encodes the
mean as a logit, so the app's `sigmoid(output)` recovers the mean member
probability. Core ML conversion itself is covered by the export script's
`--verify` parity check; here we lock the numerics in a fast, offline test.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
# export_ensemble imports coremltools transitively; skip cleanly if absent.
pytest.importorskip("coremltools")

import torch.nn as nn

from openstetho_model.export_ensemble import ProbMeanEnsemble


class _ConstLogit(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0],), self.value, dtype=torch.float32)


def test_output_sigmoid_equals_mean_member_probability():
    members = [_ConstLogit(0.0), _ConstLogit(2.0), _ConstLogit(-1.0)]
    ensemble = ProbMeanEnsemble(members)
    x = torch.zeros(1, 1, 8, 4)

    out = ensemble(x)
    recovered_prob = torch.sigmoid(out)

    expected = torch.stack([torch.sigmoid(torch.tensor(v)) for v in (0.0, 2.0, -1.0)]).mean()
    assert torch.allclose(recovered_prob.squeeze(), expected, atol=1e-6)


def test_single_member_is_identity_in_prob_space():
    ensemble = ProbMeanEnsemble([_ConstLogit(1.3)])
    out = ensemble(torch.zeros(2, 1, 8, 4))
    assert torch.allclose(torch.sigmoid(out), torch.sigmoid(torch.tensor(1.3)), atol=1e-6)
