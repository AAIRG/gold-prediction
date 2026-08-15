"""Data-layer tests (roadmap M0 / M3).

The bugs these lock down were all silent: a scale computed over the wrong axis, a
lag channel that was really a zero pad, calendar features that never changed, a
window sampler that happily walked into the test range, and -- worst of the lot --
training covariates that did not match the ones the sampling loop builds.
"""

import pytest
import torch

from flowforecast.data.covariates import (build_covariates, covariate_dim,
                                          lag_features, lag_features_from_history,
                                          lag_vector_from_history, lags_for_freq,
                                          make_timestamps, normalize_freq,
                                          num_time_features, time_features)
from flowforecast.data.datasets import DATASET_SPECS, train_test_split
from flowforecast.data.scaling import (apply_scale, compute_scale, dequantize,
                                       invert_scale)
from flowforecast.data.windowing import RollingWindowDataset, WindowedDataset
from flowforecast.models.inputs import prepare_model_inputs


def _series(d=3, t=60, seed=0):
    torch.manual_seed(seed)
    return torch.rand(d, t) * 5.0 + 1.0


def _time_feat(t=60, freq="D"):
    return time_features(make_timestamps("2020-01-01", t, freq), freq)


# ------------------------------------------------------------------- scaling M3
def test_scaling_round_trip_raw_layout():
    x = _series()
    scale = compute_scale(x, time_dim=-1)
    assert scale.shape == (x.shape[0],)
    assert torch.allclose(invert_scale(apply_scale(x, scale), scale), x, atol=1e-5)


def test_scaling_round_trip_batch_layout():
    x = torch.randn(4, 12, 3)                       # [B, T, D]
    scale = compute_scale(x, time_dim=1)
    assert scale.shape == (4, 3)
    round_trip = invert_scale(apply_scale(x, scale, time_dim=1), scale, time_dim=1)
    assert torch.allclose(round_trip, x, atol=1e-5)


def test_scale_broadcasts_over_sample_sets():
    samples = torch.randn(7, 3, 10)                 # [S, D, P]
    scale = compute_scale(_series(), time_dim=-1)   # [D]
    scaled = invert_scale(samples, scale, time_dim=-1)
    assert scaled.shape == samples.shape
    assert torch.allclose(scaled[0, 0], samples[0, 0] * scale[0], atol=1e-5)


def test_all_zero_series_gets_official_minimum_scale():
    x = torch.zeros(2, 20)
    assert torch.allclose(compute_scale(x), torch.full((2,), 1e-10))


def test_batch_mean_scaler_uses_batch_fallback_for_observed_zero_series():
    x = torch.tensor([[[0.0, 2.0], [0.0, 4.0]],
                      [[2.0, 0.0], [4.0, 0.0]]])
    observed = torch.ones_like(x)
    scale = compute_scale(x, time_dim=1, observed=observed)
    assert torch.allclose(
        scale, torch.tensor([[1.5, 3.0], [3.0, 1.5]])
    )


def test_batch_mean_scaler_uses_global_fallback_without_observations():
    x = torch.tensor([[[0.0, 2.0], [0.0, 4.0]],
                      [[2.0, 0.0], [4.0, 0.0]]])
    observed = torch.tensor([[[0.0, 1.0], [0.0, 1.0]],
                             [[1.0, 0.0], [1.0, 0.0]]])
    scale = compute_scale(x, time_dim=1, observed=observed)
    assert torch.allclose(scale, torch.tensor([[3.0, 3.0], [3.0, 3.0]]))


def test_dequantize_stays_in_the_unit_interval_above_the_count():
    counts = torch.randint(0, 5, (4, 30)).float()
    noisy = dequantize(counts)
    assert torch.all(noisy >= counts)
    assert torch.all(noisy < counts + 1.0)


# ---------------------------------------------------------------- covariates M3
def test_normalize_freq_and_lookups():
    assert normalize_freq("h") == "H"
    assert normalize_freq("30T") == "30min"
    assert normalize_freq("1B") == "B"
    assert lags_for_freq("D") == [1, 7, 14]
    assert num_time_features("H") == 4
    assert covariate_dim(8, "D", [1, 7, 14]) == 2 + 8 * 3
    with pytest.raises(Exception):
        normalize_freq("W")


def test_time_features_actually_vary_with_real_timestamps():
    tf = _time_feat(t=40, freq="D")
    assert tf.shape == (2, 40)
    assert tf.std() > 0.1
    assert tf.min() >= -1.0 - 1e-6 and tf.max() <= 1.0 + 1e-6


def test_time_features_reject_integer_ranges():
    """range(T) is read as nanoseconds: every feature becomes a constant."""
    with pytest.raises(Exception):
        time_features(range(40), "D")


def test_lag_features_padding_and_history_versions_agree():
    series = _series(d=2, t=40)
    lags = [1, 7]
    padded = lag_features(series, lags)
    assert padded.shape == (2 * 2, 40)
    assert torch.allclose(padded[:2, :1], torch.zeros(2, 1))       # zero pad
    assert torch.allclose(padded[:2, 1:], series[:, :-1])          # lag 1

    max_lag = max(lags)
    window_len = 10
    start = 12
    extended = series[:, start - max_lag:start + window_len]
    windowed = lag_features_from_history(extended, lags, window_len)
    assert windowed.shape == (2 * 2, window_len)
    # no zero padding anywhere, and the lag-1 channel is the previous value
    assert torch.allclose(windowed[:2, 0], series[:, start - 1])
    assert torch.allclose(windowed[2:, 0], series[:, start - 7])


def test_lag_vector_matches_the_windowed_matrix():
    series = _series(d=3, t=50)
    lags = [1, 4]
    max_lag = max(lags)
    window_len = 8
    start = 20
    extended = series[:, start - max_lag:start + window_len]
    matrix = lag_features_from_history(extended, lags, window_len)

    history = extended.t().unsqueeze(0)                 # [1, max_lag + T, D]
    for i in range(window_len):
        vector = lag_vector_from_history(history[:, :max_lag + i, :], lags)[0]
        assert torch.allclose(vector, matrix[:, i], atol=1e-6)


def test_build_covariates_layout():
    series = _series(d=4, t=30)
    covariates = build_covariates(make_timestamps("2020-01-01", 30, "D"),
                                 series, "D", [1, 7])
    assert covariates.shape == (num_time_features("D") + 4 * 2, 30)


# ----------------------------------------------------------------- windowing M3
def test_windowed_dataset_shapes_and_scaling():
    series = _series(d=3, t=60)
    tf = _time_feat(60)
    dataset = WindowedDataset(series, tf, context_length=10, prediction_length=5,
                              lags=[1, 7], num_windows_per_epoch=4)
    item = dataset[0]
    assert item["x"].shape == (15, 3)
    assert item["c"].shape == (15, tf.shape[0] + 3 * 2)
    assert item["scale"].shape == (3,)

    # the scale is the mean absolute value of the CONTEXT part only
    start = int(item["start"])
    context = series[:, start:start + 10]
    assert torch.allclose(item["scale"], context.abs().mean(dim=1), atol=1e-5)
    # Dataset keeps raw targets; batch-aware scaling happens inside the model.
    assert torch.allclose(item["x"][:10].t(), context, atol=1e-5)


def test_windowed_dataset_never_pads_a_lag():
    series = _series(d=2, t=40)
    tf = _time_feat(40)
    lags = [1, 7]
    dataset = WindowedDataset(series, tf, context_length=6, prediction_length=6,
                              lags=lags, num_windows_per_epoch=50)
    for i in range(50):
        item = dataset[i]
        assert int(item["start"]) >= 7
        lag_part = item["c"][:, :series.shape[0] * len(lags)]
        assert torch.all(lag_part.abs() > 0)


def test_windowed_dataset_respects_the_training_boundary():
    series = _series(d=2, t=100)
    tf = _time_feat(100)
    train_end = 70
    dataset = WindowedDataset(series, tf, context_length=10, prediction_length=10,
                              lags=[1, 7], num_windows_per_epoch=200,
                              end_index=train_end)
    for i in range(200):
        item = dataset[i]
        assert int(item["start"]) + 20 <= train_end


def test_training_covariates_match_the_sampling_loop():
    """The single most important consistency check in the data layer.

    data/windowing.py builds the lag channels for a whole window at once, while
    evaluation/sampling.py builds them one step at a time from the buffer of
    already sampled values. If the two disagree by even one index, the model is
    trained on one covariate layout and evaluated on another.
    """
    series = _series(d=3, t=60)
    tf = _time_feat(60)
    lags = [1, 4]
    context_length, prediction_length = 8, 6
    dataset = WindowedDataset(series, tf, context_length=context_length,
                              prediction_length=prediction_length, lags=lags,
                              num_windows_per_epoch=1)
    start = 20
    item = dataset.make_window(start)

    max_lag = max(lags)
    window = context_length + prediction_length
    extended = series[:, start - max_lag:start + window]
    history = extended.t().unsqueeze(0)               # [1, max_lag + T, D]

    for i in range(window):
        expected = lag_vector_from_history(history[:, :max_lag + i, :], lags)[0]
        assert torch.allclose(item["c"][i, :series.shape[0] * len(lags)],
                              expected, atol=1e-5)
        assert torch.allclose(item["c"][i, series.shape[0] * len(lags):],
                              tf[:, start + i], atol=1e-6)


def test_model_input_scaling_preserves_lag_then_time_layout():
    x = torch.tensor([[[2.0, 4.0], [4.0, 8.0]]])
    observed = torch.ones_like(x)
    lag_part = torch.tensor([[[6.0, 12.0, 10.0, 20.0],
                              [8.0, 16.0, 12.0, 24.0]]])
    time_part = torch.tensor([[[0.25, -0.5], [0.75, 1.0]]])
    covariates = torch.cat([lag_part, time_part], dim=-1)

    _, scaled_covariates, _, scale = prepare_model_inputs(
        x, covariates, observed, context_length=2, num_series=2,
        time_feature_size=2, dequantize_target=False,
    )

    expected_lags = lag_part / scale.repeat(1, 2).unsqueeze(1)
    assert torch.allclose(scaled_covariates[:, :, :4], expected_lags)
    assert torch.equal(scaled_covariates[:, :, 4:], time_part)


def test_rolling_window_dataset_tiles_the_range():
    series = _series(d=2, t=100)
    tf = _time_feat(100)
    dataset = RollingWindowDataset(series, tf, context_length=10,
                                   prediction_length=10, lags=[1, 7],
                                   num_windows=3, end_index=100)
    assert dataset.starts == [60, 70, 80]
    assert len(dataset) == 3
    assert dataset[0]["x"].shape == (20, 2)
    # the prediction parts tile the last num_windows * prediction_length steps
    prediction_starts = [s + 10 for s in dataset.starts]
    assert prediction_starts == [70, 80, 90]


def test_window_dataset_rejects_impossible_ranges():
    series = _series(d=2, t=20)
    tf = _time_feat(20)
    with pytest.raises(Exception):
        WindowedDataset(series, tf, context_length=10, prediction_length=10,
                        lags=[1, 7], num_windows_per_epoch=1)


# ------------------------------------------------------------------ datasets M0
def test_train_test_split_boundaries():
    series = _series(d=2, t=100)
    train, test = train_test_split(series, prediction_length=10,
                                  num_test_windows=5)
    assert train.shape == (2, 50)
    assert test.shape == (2, 50)
    assert torch.equal(train[:, -1], series[:, 49])
    assert torch.equal(test[:, 0], series[:, 50])


def test_dataset_specs_match_table_2():
    assert DATASET_SPECS["exchange"][:4] == (8, "B", 30, 6071)
    assert DATASET_SPECS["solar"][:4] == (137, "H", 24, 7009)
    assert DATASET_SPECS["electricity"][:4] == (370, "H", 24, 5790)
    assert DATASET_SPECS["traffic"][:4] == (963, "H", 24, 10413)
    assert DATASET_SPECS["taxi"][:4] == (1214, "30min", 24, 1488)
    assert DATASET_SPECS["wikipedia"][:4] == (2000, "D", 30, 792)
    # every dataset of Table 2 is reachable through gluonts
    assert all(spec.gluonts_name for spec in DATASET_SPECS.values())
