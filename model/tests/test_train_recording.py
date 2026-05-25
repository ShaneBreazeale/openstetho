from __future__ import annotations

import torch
import numpy as np

from openstetho_model.cv_murmur import (
    calibrate_probs,
    cross_fold_calibration_report,
    expected_calibration_error,
    stratified_patient_folds,
)
from openstetho_model.dataset import window_hop_samples
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


def test_cnn_bigru_forward_accepts_multi_channel_batch():
    model = build_model("cnn_bigru", in_channels=3)
    x = torch.zeros(2, 3, 62, 32)
    y = model(x)
    assert y.shape == (2,)


def test_cnn_bigru_forward_accepts_five_second_frames():
    model = build_model("cnn_bigru")
    x = torch.zeros(2, 78, 32)
    y = model(x)
    assert y.shape == (2,)


def test_scattering_cnn1d_forward_accepts_scattering_frames():
    model = build_model("scattering_cnn1d")
    x = torch.zeros(2, 79, 121)
    y = model(x)
    assert y.shape == (2,)


def test_window_hop_samples_defaults_to_half_overlap():
    assert window_hop_samples(5.0) == (20000, 10000)
    assert window_hop_samples(5.0, 1.25) == (20000, 5000)


def test_stratified_patient_folds_keep_each_patient_once():
    labels = {i: i % 2 for i in range(20)}
    folds = stratified_patient_folds(labels, n_folds=5, seed=0)
    assert sorted(pid for fold in folds for pid in fold) == list(range(20))
    assert all({labels[pid] for pid in fold} == {0, 1} for fold in folds)


def test_expected_calibration_error_is_low_for_matched_bins():
    labels = np.asarray([0, 0, 1, 1])
    probs = np.asarray([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(labels, probs, n_bins=2) < 1e-5


def test_calibrate_probs_single_class_train_falls_back_to_identity():
    train_labels = np.asarray([0, 0, 0])
    train_probs = np.asarray([0.1, 0.2, 0.3])
    val_probs = np.asarray([0.4, 0.6])
    np.testing.assert_allclose(
        calibrate_probs(train_labels, train_probs, val_probs, "platt"),
        val_probs,
    )


def test_cross_fold_calibration_reports_transfer_metrics():
    folds = np.asarray([1, 1, 2, 2, 3, 3, 4, 4])
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    probs = np.asarray([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    report = cross_fold_calibration_report(folds, labels, probs)

    assert set(report) == {"none", "platt", "isotonic"}
    none_report = report["none"]
    assert none_report["probability"]["auroc"] == 1.0
    assert set(none_report["threshold_transfer"]) == {"best_f1", "best_youden_j"}
    assert none_report["threshold_transfer"]["best_f1"]["f1"] > 0.8
    assert "threshold_mean" in none_report["threshold_transfer"]["best_f1"]
    assert len(none_report["folds"]) == 4
