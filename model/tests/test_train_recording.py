from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from openstetho_model.cv_murmur import (
    calibrate_probs,
    cross_fold_calibration_report,
    expected_calibration_error,
    limit_patient_labels,
    stratified_patient_folds,
    threshold_policy_row,
)
from openstetho_model.dataset import (
    MurmurAugmentationConfig,
    WIDE_FEATURE_NAMES,
    _augment_audio,
    _augment_features,
    _recording_windows,
    _window_at_index,
    load_recording_teacher_targets,
    wide_feature_vector,
)
from openstetho_model.dataset import window_hop_samples
from openstetho_model.train import (
    aggregate_logits,
    build_loss_fn,
    build_model,
    recording_batch_logits,
    recording_collate,
)


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


def test_recording_collate_keeps_wide_features():
    rec_a = torch.zeros(2, 62, 32)
    rec_b = torch.zeros(5, 62, 32)
    wide_a = torch.ones(3)
    wide_b = torch.zeros(3)
    recordings, wides, labels = recording_collate([(rec_a, wide_a, 0), (rec_b, wide_b, 1)])
    assert [r.shape[0] for r in recordings] == [2, 5]
    assert wides.shape == (2, 3)
    assert labels.tolist() == [0.0, 1.0]


def test_recording_collate_keeps_teacher_targets():
    rec_a = torch.zeros(2, 62, 32)
    rec_b = torch.zeros(5, 62, 32)
    recordings, teachers, labels = recording_collate([
        (rec_a, torch.tensor(0.8), 0),
        (rec_b, torch.tensor(float("nan")), 1),
    ])
    assert [r.shape[0] for r in recordings] == [2, 5]
    assert teachers[0].item() == torch.tensor(0.8).item()
    assert torch.isnan(teachers[1])
    assert labels.tolist() == [0.0, 1.0]


def test_recording_collate_keeps_wide_features_and_teacher_targets():
    rec_a = torch.zeros(2, 62, 32)
    rec_b = torch.zeros(5, 62, 32)
    wide_a = torch.ones(3)
    wide_b = torch.zeros(3)
    recordings, wides, teachers, labels = recording_collate([
        (rec_a, wide_a, torch.tensor(0.7), 0),
        (rec_b, wide_b, torch.tensor(0.2), 1),
    ])
    assert [r.shape[0] for r in recordings] == [2, 5]
    assert wides.shape == (2, 3)
    assert teachers.tolist() == [torch.tensor(0.7).item(), torch.tensor(0.2).item()]
    assert labels.tolist() == [0.0, 1.0]


def test_load_recording_teacher_targets_topk_mean(tmp_path):
    csv_path = tmp_path / "teacher.csv"
    csv_path.write_text(
        "patient_id,location,onnx_prob\n"
        "1,AV,0.2\n"
        "1,AV,0.8\n"
        "1,AV,0.6\n"
        "2,MV,0.4\n"
    )
    targets = load_recording_teacher_targets(csv_path, aggregation="topk_mean", topk=2)
    assert np.isclose(targets[(1, "AV")], 0.7)
    assert np.isclose(targets[(2, "MV")], 0.4)


def test_recording_batch_logits_matches_per_recording_calls():
    torch.manual_seed(0)
    model = build_model("cnn_bigru")
    model.eval()
    recordings = [torch.randn(2, 78, 32), torch.randn(3, 78, 32)]
    batched = recording_batch_logits(model, recordings, torch.device("cpu"), "mean", topk=2)
    expected = torch.stack([model(recording).mean() for recording in recordings])
    torch.testing.assert_close(batched, expected)


def test_recording_batch_logits_repeats_wide_features_per_window():
    torch.manual_seed(0)
    model = build_model("cnn_bigru", wide_feature_dim=4)
    model.eval()
    recordings = [torch.randn(2, 78, 32), torch.randn(3, 78, 32)]
    wides = torch.randn(2, 4)
    batched = recording_batch_logits(
        model,
        recordings,
        torch.device("cpu"),
        "topk_mean",
        topk=2,
        wides=wides,
    )
    expected = []
    for recording, wide in zip(recordings, wides):
        repeated = wide.unsqueeze(0).expand(recording.shape[0], -1)
        expected.append(aggregate_logits(model(recording, repeated), "topk_mean", topk=2))
    torch.testing.assert_close(batched, torch.stack(expected))


def test_wide_feature_vector_contains_metadata_and_stats():
    row = pd.Series({
        "Age": "Child",
        "Sex": "Female",
        "Pregnancy status": False,
        "Height": 100.0,
        "Weight": 15.0,
    })
    audio = np.sin(np.linspace(0, 20 * np.pi, 4000, dtype=np.float32))
    features = wide_feature_vector(row, "AV", audio)
    assert features.shape == (len(WIDE_FEATURE_NAMES),)
    assert features[WIDE_FEATURE_NAMES.index("age_child")] == 1.0
    assert features[WIDE_FEATURE_NAMES.index("sex_female")] == 1.0
    assert features[WIDE_FEATURE_NAMES.index("location_av")] == 1.0
    assert features[WIDE_FEATURE_NAMES.index("duration_s")] == 1.0
    assert features[WIDE_FEATURE_NAMES.index("spectral_centroid_hz")] > 0.0


def test_murmur_audio_augmentation_adds_noise_without_shape_change():
    x = np.ones(4000, dtype=np.float32)
    cfg = MurmurAugmentationConfig(audio_noise_snr_db=15.0, audio_noise_prob=1.0)
    y = _augment_audio(x, cfg, np.random.default_rng(0))
    assert y.shape == x.shape
    assert not np.allclose(y, x)


def test_murmur_spec_augmentation_masks_feature_bins():
    x = np.ones((20, 16), dtype=np.float32)
    x[0, 0] = -1.0
    cfg = MurmurAugmentationConfig(freq_mask_max_width=4, time_mask_max_width=4)
    y = _augment_features(x, cfg, np.random.default_rng(1))
    assert y.shape == x.shape
    assert np.count_nonzero(y == -1.0) > 1


def test_murmur_window_jitter_changes_crop_start():
    x = np.arange(100, dtype=np.float32)
    cfg = MurmurAugmentationConfig(window_jitter_seconds=0.001)
    y = _window_at_index(x, 1, window_samples=20, hop_samples=20, config=cfg, rng=np.random.default_rng(0))
    assert y.shape == (20,)
    assert not np.array_equal(y, x[20:40])


def test_recording_random_crop_returns_one_window():
    x = np.arange(100, dtype=np.float32)
    cfg = MurmurAugmentationConfig(random_crop=True)
    y = _recording_windows(x, window_samples=20, hop_samples=10, config=cfg, rng=np.random.default_rng(0))
    assert y.shape == (1, 20)
    assert not np.array_equal(y[0], x[:20])


def test_focal_loss_downweights_easy_examples():
    logits = torch.tensor([-4.0, 4.0])
    labels = torch.tensor([0.0, 1.0])
    pos_weight = torch.tensor([1.0])
    bce = build_loss_fn("bce", pos_weight)(logits, labels)
    focal = build_loss_fn("focal_bce", pos_weight, focal_gamma=2.0)(logits, labels)
    assert focal.item() < bce.item()


def test_bce_loss_is_finite_for_extreme_logits():
    logits = torch.tensor([-1e6, 1e6])
    labels = torch.tensor([0.0, 1.0])
    pos_weight = torch.tensor([4.0])
    loss = build_loss_fn("bce", pos_weight)(logits, labels)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


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


def test_cnn_bigru_forward_accepts_logmel128_fifteen_second_frames():
    model = build_model("cnn_bigru")
    x = torch.zeros(2, 599, 128)
    y = model(x)
    assert y.shape == (2,)


def test_cnn_bigru_forward_accepts_wide_features():
    model = build_model("cnn_bigru", wide_feature_dim=4)
    x = torch.zeros(2, 78, 32)
    wide = torch.zeros(2, 4)
    y = model(x, wide)
    assert y.shape == (2,)


def test_scattering_cnn1d_forward_accepts_scattering_frames():
    model = build_model("scattering_cnn1d")
    x = torch.zeros(2, 79, 121)
    y = model(x)
    assert y.shape == (2,)


def test_scattering_transformer_forward_accepts_scattering_frames():
    model = build_model("scattering_transformer")
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


def test_limit_patient_labels_keeps_stratified_smoke_subset():
    labels = {i: i % 2 for i in range(100)}
    limited = limit_patient_labels(labels, max_patients=20, n_folds=5, seed=0)
    assert len(limited) == 20
    assert sum(1 for label in limited.values() if label == 0) == 10
    assert sum(1 for label in limited.values() if label == 1) == 10


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
    # Core policies must always be present; the set may grow with added
    # specificity targets (e.g. specificity_ge_0_93/94/95), so assert a
    # superset rather than exact equality to avoid brittleness.
    assert set(none_report["threshold_transfer"]) >= {
        "best_f1",
        "best_youden_j",
        "sensitivity_ge_0_80",
        "specificity_ge_0_90",
    }
    assert none_report["threshold_transfer"]["best_f1"]["f1"] > 0.8
    assert "threshold_mean" in none_report["threshold_transfer"]["best_f1"]
    assert len(none_report["folds"]) == 4


def test_threshold_policy_reports_clinical_targets():
    rows = [
        {"threshold": 0.2, "sensitivity": 1.0, "specificity": 0.2, "f1": 0.5, "youden_j": 0.2},
        {"threshold": 0.5, "sensitivity": 0.8, "specificity": 0.9, "f1": 0.7, "youden_j": 0.7},
        {"threshold": 0.8, "sensitivity": 0.4, "specificity": 1.0, "f1": 0.6, "youden_j": 0.4},
    ]
    sens = threshold_policy_row(rows, "sensitivity_ge_0_80")
    spec = threshold_policy_row(rows, "specificity_ge_0_90")
    assert sens["threshold"] == 0.5
    assert spec["threshold"] == 0.5
    assert sens["constraint_met"] == 1.0
    assert spec["constraint_met"] == 1.0
