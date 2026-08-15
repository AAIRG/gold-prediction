"""Reference-parity tests for TempFlow backbones and sampling."""

import torch

from flowforecast.data.covariates import make_timestamps, time_features
from flowforecast.data.windowing import WindowedDataset
from flowforecast.evaluation.sampling import sample_trajectories
from flowforecast.models.backbone_lstm import LSTMBackbone
from flowforecast.models.backbone_transformer import TransformerBackbone
from flowforecast.utils.build import build_model

D, C, T, B = 3, 4, 12, 2


def _inputs(seed=0):
    torch.manual_seed(seed)
    return torch.randn(B, T, D), torch.randn(B, T, C)


def test_lstm_consumes_aligned_current_covariates():
    x, c = _inputs()
    backbone = LSTMBackbone(D, C, hidden_size=8, num_layers=1)
    backbone.eval()
    with torch.no_grad():
        h = backbone(x, c)
        changed = c.clone()
        changed[:, 5] += 10.0
        h_changed = backbone(x, changed)
    assert torch.allclose(h[:, :5], h_changed[:, :5], atol=1e-6)
    assert (h[:, 5] - h_changed[:, 5]).abs().max() > 1e-4


def test_lstm_does_not_read_raw_target_directly():
    x, c = _inputs()
    backbone = LSTMBackbone(D, C, hidden_size=8, num_layers=1)
    backbone.eval()
    with torch.no_grad():
        h = backbone(x, c)
        h_changed = backbone(x + 100.0, c)
    assert torch.allclose(h, h_changed, atol=1e-6)


def test_lstm_has_one_embedding_per_target_dimension():
    backbone = LSTMBackbone(D, C, hidden_size=8, num_layers=1)
    assert backbone.target_embedding.weight.shape == (D, 1)
    assert backbone.lstm.input_size == C + D


def test_lstm_step_matches_forward():
    x, c = _inputs()
    backbone = LSTMBackbone(D, C, hidden_size=8, num_layers=2)
    backbone.eval()
    with torch.no_grad():
        full = backbone(x, c)
        state = None
        rows = []
        for t in range(T):
            row, state = backbone.step(x[:, t:t + 1], c[:, t:t + 1], state)
            rows.append(row[:, 0])
        stepwise = torch.stack(rows, dim=1)
    assert torch.allclose(full, stepwise, atol=1e-5)


def test_transformer_returns_future_decoder_states_only():
    x, c = _inputs()
    backbone = TransformerBackbone(
        D, C, d_model=16, n_heads=4, num_encoder_layers=1,
        num_decoder_layers=1, dropout=0.0, context_length=6,
    )
    output = backbone(x, c)
    assert output.shape == (B, T - 6, 16)
    assert backbone.target_embedding.weight.shape == (D, 1)


def test_builder_uses_official_transformer_d_model_not_lstm_cells():
    config = {
        "backbone": {
            "type": "transformer", "hidden_size": 40, "d_model": 32,
            "n_heads": 8, "num_encoder_layers": 1,
            "num_decoder_layers": 1, "dropout": 0.0,
        },
        "flow": {"type": "maf", "num_blocks": 1, "hidden": 8},
    }
    model = build_model(config, num_series=D, cond_size=C, context_length=6)

    assert model.backbone.d_model == 32
    assert model.condition_proj.in_features == 32


def test_transformer_decoder_is_causal_and_encoder_is_unmasked():
    x, c = _inputs()
    backbone = TransformerBackbone(
        D, C, d_model=16, n_heads=4, num_encoder_layers=1,
        num_decoder_layers=1, dropout=0.0, context_length=6,
    )
    backbone.eval()
    with torch.no_grad():
        output = backbone(x, c)
        future_changed = c.clone()
        future_changed[:, 8] += 10.0
        output_future = backbone(x, future_changed)
        context_changed = c.clone()
        context_changed[:, 5] += 10.0
        output_context = backbone(x, context_changed)
    assert torch.allclose(output[:, :2], output_future[:, :2], atol=1e-5)
    assert (output[:, 2] - output_future[:, 2]).abs().max() > 1e-5
    assert (output - output_context).abs().max() > 1e-5


def _tiny_setup(num_series=3, total=80, context=6, prediction=6, lags=(1, 4)):
    torch.manual_seed(0)
    series = torch.rand(num_series, total) * 3.0 + 1.0
    tf = time_features(make_timestamps("2020-01-01", total, "D"), "D")
    dataset = WindowedDataset(
        series, tf, context_length=context, prediction_length=prediction,
        lags=list(lags), num_windows_per_epoch=8, end_index=total,
    )
    return series, tf, dataset


def test_sampling_shapes_and_finiteness_for_both_backbones():
    series, tf, dataset = _tiny_setup()
    lags = dataset.lags
    max_lag = dataset.max_lag
    context, prediction = dataset.context_length, dataset.prediction_length
    cond_size = tf.shape[0] + series.shape[0] * len(lags)

    for kind in ("lstm", "transformer"):
        config = {
            "model": {"conditioning_length": 12},
            "backbone": {
                "type": kind, "hidden_size": 16, "num_layers": 1,
                "n_heads": 4, "num_encoder_layers": 1,
                "num_decoder_layers": 1, "dropout": 0.0,
            },
            "flow": {
                "type": "realnvp", "num_blocks": 2, "hidden": 16,
                "bn_min_var": 0.0,
            },
        }
        model = build_model(
            config, series.shape[0], cond_size, context_length=context
        )
        item = dataset.make_window(max_lag)
        x = item["x"].unsqueeze(0)
        c = item["c"].unsqueeze(0)
        observed = item["observed"].unsqueeze(0)
        model.train()
        loss = model.loss(x, c, observed=observed)
        loss.backward()
        assert torch.isfinite(loss)

        forecast_start = 40
        hist_start = forecast_start - context - max_lag
        history = series[:, hist_start:forecast_start]
        scale = history[:, max_lag:].abs().mean(dim=1)
        history_input = history.t().unsqueeze(0)
        future_tf = tf[:, hist_start:forecast_start + prediction].t().unsqueeze(0)
        model.eval()
        with torch.no_grad():
            samples = sample_trajectories(
                model, history_input, future_tf, lags, prediction,
                num_samples=7, context_length=context,
                scale=scale,
            )
        assert samples.shape == (7, series.shape[0], prediction)
        assert torch.isfinite(samples).all()


def test_maf_conditioning_matches_official_linear_projection():
    config = {
        "model": {"conditioning_length": 12},
        "backbone": {
            "type": "lstm", "hidden_size": 8, "num_layers": 1,
            "dropout": 0.0,
        },
        "flow": {"type": "maf", "num_blocks": 1, "hidden": 8},
    }
    model = build_model(config, num_series=D, cond_size=C, context_length=6)
    hidden = torch.randn(B, T, 8)
    condition = model.project_condition(hidden)

    assert torch.equal(condition, model.condition_proj(hidden))
    assert not hasattr(model, "condition_norm")


def test_transformer_loss_uses_future_observations_only():
    series, tf, dataset = _tiny_setup()
    cond_size = tf.shape[0] + series.shape[0] * len(dataset.lags)
    config = {
        "model": {"conditioning_length": 12},
        "backbone": {
            "type": "transformer", "hidden_size": 16, "n_heads": 4,
            "num_encoder_layers": 1, "num_decoder_layers": 1, "dropout": 0.0,
        },
        "flow": {"type": "realnvp", "num_blocks": 1, "hidden": 8},
    }
    model = build_model(
        config, series.shape[0], cond_size,
        context_length=dataset.context_length,
    )
    item = dataset.make_window(dataset.max_lag)
    x = item["x"].unsqueeze(0)
    c = item["c"].unsqueeze(0)
    observed = item["observed"].unsqueeze(0)
    observed[:, :dataset.context_length] = 0.0
    loss = model.loss(x, c, observed=observed)
    assert torch.isfinite(loss)
