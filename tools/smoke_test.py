"""End-to-end CPU smoke test for the goldflow layer.

Runs the full pipeline on a short synthetic price series: four variants are
built and briefly trained, rolling windows are sampled and scored, both boosting
stages are fitted, and the horizon reconstruction is checked against a hand
computed reference. No network access and no Colab.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CHECKPOINTS = Path(tempfile.gettempdir()) / "goldflow_smoke_checkpoints"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "project"))

import goldflow as gf                                          # noqa: E402
from flowforecast.data.windowing import (RollingWindowDataset,   # noqa: E402
                                         WindowedDataset)
from flowforecast.training.trainer import Trainer               # noqa: E402
from flowforecast.utils.build import count_parameters           # noqa: E402
from goldflow import boosting, experiments, features, variants  # noqa: E402

HORIZONS = (1, 3, 7)
CONTEXT = 30
PREDICTION = 10
LAGS = [1, 2]
TEST_WINDOWS = 4
VALIDATION_WINDOWS = 3
SAMPLES = 24
BOOST_WINDOWS = 30


def synthetic_prices(rows: int = 900, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=rows)
    shocks = rng.normal(0.0002, 0.009, rows)
    close = 1500.0 * np.exp(np.cumsum(shocks))
    spread = np.abs(rng.normal(0.004, 0.002, rows)) * close
    return pd.DataFrame(
        {
            "Open": close * np.exp(rng.normal(0, 0.002, rows)),
            "High": close + spread,
            "Low": close - spread,
            "Close": close,
        },
        index=dates,
    )


def check_horizon_reconstruction(prices: pd.DataFrame) -> None:
    close = prices["Close"].to_numpy()
    rng = np.random.default_rng(3)
    daily = rng.normal(0.0, 0.008, (16, PREDICTION))
    paths = gf.forecast.reconstruct_price_paths(float(close[-1]), daily)
    derived = gf.derive_horizon_returns(close, paths, HORIZONS)

    extended = np.concatenate([np.tile(close, (16, 1)), paths], axis=1)
    anchor = len(close)
    for horizon in HORIZONS:
        expected = np.log(
            extended[:, anchor:anchor + PREDICTION]
            / extended[:, anchor - horizon:anchor - horizon + PREDICTION]
        )
        error = np.abs(derived[horizon] - expected).max()
        if error > 1e-12:
            raise AssertionError(f"horizon {horizon} reconstruction off by {error}")

    # Day 1 of the 3-day horizon must equal the sampled daily return plus the
    # two already observed daily returns, exactly.
    observed_two_day = float(np.log(close[-1] / close[-3]))
    expected_day1 = observed_two_day + daily[:, 0]
    error = np.abs(derived[3][:, 0] - expected_day1).max()
    if error > 1e-12:
        raise AssertionError(f"3-day day-1 splice off by {error}")

    # The 3-day return at forecast day 3 must be the sum of the first three
    # sampled daily returns, with no observed component left.
    error = np.abs(derived[3][:, 2] - daily[:, :3].sum(axis=1)).max()
    if error > 1e-12:
        raise AssertionError(f"3-day day-3 compounding off by {error}")

    fractions = gf.forecast.observed_fractions(HORIZONS, PREDICTION)
    if fractions[3][0] != 2 / 3 or fractions[3][2] != 0.0 or fractions[1][0] != 0.0:
        raise AssertionError(f"observed fractions are wrong: {fractions}")
    print("[ok] horizon reconstruction, splice and observed shares")


def check_candles(prices: pd.DataFrame) -> None:
    for window in (3, 7):
        candles = gf.aggregate_candles(prices, window)
        row = candles.iloc[-1]
        block = prices.iloc[-window:]
        if abs(row["Open"] - block["Open"].iloc[0]) > 1e-9:
            raise AssertionError(f"{window}-day open is not the window open")
        if abs(row["High"] - block["High"].max()) > 1e-9:
            raise AssertionError(f"{window}-day high is not the window max")
        if abs(row["Low"] - block["Low"].min()) > 1e-9:
            raise AssertionError(f"{window}-day low is not the window min")
        if abs(row["Close"] - block["Close"].iloc[-1]) > 1e-9:
            raise AssertionError(f"{window}-day close is not the day close")

        returns = gf.build_return_targets(prices["Close"], HORIZONS)
        candle_return = float(np.log(
            prices["Close"].iloc[-1] / prices["Close"].iloc[-1 - window]
        ))
        target = float(returns[f"return_{window}d"].iloc[-1])
        if abs(candle_return - target) > 1e-6:
            raise AssertionError(
                f"{window}-day candle return {candle_return} does not match "
                f"the training target {target}"
            )
    print("[ok] aggregate candles match the training targets")


def check_stable_features() -> None:
    long_index = pd.bdate_range("2015-01-01", periods=1500)
    short_index = long_index[-60:]
    full = features.stable_time_features(long_index)[:, -60:]
    short = features.stable_time_features(short_index)
    error = float((full - short).abs().max().item())
    if error > 1e-12:
        raise AssertionError(f"time features depend on window length: {error}")
    if features.time_feature_count() != 4:
        raise AssertionError("expected four calendar channels")
    print("[ok] calendar features are window independent")


def train_all(returns: pd.DataFrame, time_feat: torch.Tensor,
              config: dict, cond_size: int, splits: dict) -> dict:
    series = torch.from_numpy(returns.to_numpy()).t().contiguous()
    train_dataset = WindowedDataset(
        series=series, time_feat=time_feat, context_length=CONTEXT,
        prediction_length=PREDICTION, lags=LAGS,
        num_windows_per_epoch=16 * 4, end_index=splits["train_end"],
    )
    val_dataset = RollingWindowDataset(
        series=series, time_feat=time_feat, context_length=CONTEXT,
        prediction_length=PREDICTION, lags=LAGS,
        num_windows=VALIDATION_WINDOWS, end_index=splits["validation_end"],
    )
    train_loader = DataLoader(train_dataset, batch_size=16, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))

    trained = {}
    for variant in variants.MODEL_VARIANTS:
        model = variants.build_variant(
            variant.key, config, num_series=series.shape[0],
            cond_size=cond_size, context_length=CONTEXT, device="cpu",
        )
        trainer = Trainer(
            model, train_loader, val_loader=val_loader, device="cpu", lr=1e-3,
            grad_clip=5.0, batches_per_epoch=4, select_metric="last",
            validation_num_samples=4, validation_interval=99,
            lr_schedule="onecycle", max_lr=3e-3, verbose=False,
        )
        trainer.fit(
            num_epochs=2,
            checkpoint_dir=str(CHECKPOINTS),
            run_name=f"smoke_{variant.key}",
            extra_state={
                "config": variants.variant_config(variant.key, config),
                "num_series": int(series.shape[0]), "cond_size": cond_size,
                "context_length": CONTEXT, "prediction_length": PREDICTION,
                "lags": LAGS, "freq": "B", "baseline": variant.baseline,
                "model_contract": variants.MODEL_CONTRACT,
            },
        )
        model.eval()
        trained[variant.key] = model
        print(f"[ok] trained {variant.name} ({count_parameters(model):,} parameters)")
    return trained, series


def main() -> None:
    prices = synthetic_prices()
    check_stable_features()
    check_horizon_reconstruction(prices)
    check_candles(prices)

    returns = gf.build_return_targets(prices["Close"], HORIZONS)
    time_feat = features.stable_time_features(pd.DatetimeIndex(returns.index))
    cond_size = int(time_feat.shape[0] + len(HORIZONS) * len(LAGS))
    if cond_size != 10:
        raise AssertionError(f"cond_size drifted to {cond_size}")

    splits = gf.data.split_indices(
        len(returns), PREDICTION, TEST_WINDOWS, VALIDATION_WINDOWS
    )
    config = variants.base_config(
        num_series=len(HORIZONS), context_length=CONTEXT,
        prediction_length=PREDICTION, lags=LAGS, epochs=2, batch_size=16,
        batches_per_epoch=4, validation_windows=VALIDATION_WINDOWS,
        test_windows=TEST_WINDOWS, evaluation_samples=SAMPLES,
        validation_samples=4, seed=1,
    )
    trained, series = train_all(returns, time_feat, config, cond_size, splits)

    summaries = []
    test_samples = {}
    for variant in variants.MODEL_VARIANTS:
        rolled = experiments.rolling_window_samples(
            trained[variant.key], series, time_feat, LAGS, CONTEXT, PREDICTION,
            num_windows=TEST_WINDOWS, end_index=splits["test_end"],
            num_samples=SAMPLES, device="cpu", seed=1,
        )
        per_window = experiments.evaluate_samples(rolled["samples"], rolled["targets"])
        summary = experiments.summarize_windows(per_window)
        summaries.append(experiments.SeedSummary(
            model_key=variant.key, model_name=variant.name, seed=1,
            metrics=summary,
            per_horizon=experiments.per_dimension_metrics(
                rolled["samples"], rolled["targets"], HORIZONS
            ),
            per_window=per_window,
        ))
        test_samples[variant.key] = rolled["samples"]
        print(f"[ok] evaluated {variant.name}: "
              f"CRPS_sum={summary['crps_sum']['mean']:.5f} "
              f"CRPS={summary['crps']['mean']:.5f} "
              f"MSE={summary['mse']['mean']:.3g}")

    targets = rolled["targets"]
    starts = rolled["forecast_starts"]

    labels = {v.key: v.name for v in variants.MODEL_VARIANTS}
    order = [v.key for v in variants.MODEL_VARIANTS]
    tables = experiments.build_paper_tables(
        summaries, HORIZONS, "Gold (GC=F)", order, labels
    )
    for name, payload in tables.items():
        frame = payload["frame"]
        expected_rows = 1 + (len(HORIZONS) if name != "table1_crps_sum" else 0)
        if frame.shape != (expected_rows, len(order)):
            raise AssertionError(f"{name} has shape {frame.shape}")
        if (frame.to_numpy() == "-").any():
            raise AssertionError(f"{name} has an empty cell")
        print(f"\n[{name}] {payload['caption']}")
        print(frame.to_string())

    board = experiments.leaderboard(summaries)
    print("\n[leaderboard]")
    print(board[["Rank", "Model", "crps_sum_mean", "crps_mean", "mse_mean"]].to_string(index=False))

    coverage = experiments.coverage_report(
        test_samples[order[0]], targets, HORIZONS
    )
    if len(coverage) != len(HORIZONS):
        raise AssertionError("coverage report has the wrong number of rows")

    ensemble = boosting.fit_greedy_crps_ensemble(test_samples, targets, rounds=8)
    print(f"\n[ok] ensemble weights: {ensemble.weights}")
    mixture = boosting.mix_samples(test_samples, ensemble.weights, ensemble.pool_size)
    if mixture.dim() != 4 or mixture.shape[0] != TEST_WINDOWS:
        raise AssertionError(f"mixture has shape {tuple(mixture.shape)}")
    mixture_summary = experiments.summarize_windows(
        experiments.evaluate_samples(mixture, targets)
    )
    print(f"[ok] mixture CRPS_sum={mixture_summary['crps_sum']['mean']:.5f}")

    # Closed-form selection loss must agree with the project's own metric.
    energy = ensemble.rounds[-1]["crps_sum"]
    reference = float(mixture_summary["crps_sum_energy"]["mean"])
    if abs(energy - reference) > 5e-3:
        raise AssertionError(
            f"closed-form mixture CRPS {energy} disagrees with {reference}"
        )
    print("[ok] closed-form mixture CRPS matches compute_metrics")

    boost_starts = experiments.window_starts(
        splits["validation_end"], BOOST_WINDOWS, 1
    )
    boost_starts = [s for s in boost_starts if s - CONTEXT - max(LAGS) >= 0]
    boost_windows = len(boost_starts)
    rolled_boost = {}
    for variant in variants.MODEL_VARIANTS:
        rolled_boost[variant.key] = experiments.rolling_window_samples(
            trained[variant.key], series, time_feat, LAGS, CONTEXT, PREDICTION,
            num_windows=boost_windows, end_index=splits["validation_end"],
            num_samples=12, device="cpu", seed=2, stride=1,
        )
    boost_targets = rolled_boost[order[0]]["targets"]
    boost_start_list = rolled_boost[order[0]]["forecast_starts"]
    boost_mixture = boosting.mix_samples(
        {k: v["samples"] for k, v in rolled_boost.items()},
        ensemble.weights, ensemble.pool_size,
    )
    timestamps = pd.DatetimeIndex(returns.index)
    matrix, residuals = boosting.build_residual_dataset(
        boost_mixture, boost_targets, series, timestamps, boost_start_list,
        list(HORIZONS),
    )
    print(f"[ok] residual dataset: {matrix.shape[0]} rows, {matrix.shape[1]} features")
    booster = boosting.GradientResidualBooster(max_iter=60, seed=1).fit(matrix, residuals)
    gate = boosting.gate_residual_booster(
        booster, mixture, targets, series, timestamps, starts, list(HORIZONS)
    )
    print(f"[ok] booster gate: enabled={gate['enabled']} "
          f"gain={gate['gain_pct']:.2f}% "
          f"({gate['crps_sum_before']:.5f} -> {gate['crps_sum_after']:.5f})")

    single = {key: block[0] for key, block in test_samples.items()}
    flat = boosting.mix_samples(single, ensemble.weights, ensemble.pool_size)
    if flat.dim() != 3:
        raise AssertionError(f"flat mixture has shape {tuple(flat.shape)}")

    result = gf.forecast_prices(
        trained["transformer_maf"], "transformer_maf", "Transformer-MAF",
        prices.iloc[:splits["train_end"] + 7], returns.iloc[:splits["train_end"]],
        HORIZONS, LAGS, CONTEXT, PREDICTION, num_samples=32, device="cpu", seed=5,
    )
    probabilities = gf.forecast.probability_frame(result, 0.75)
    consistency = gf.forecast.consistency_report(result)
    if len(probabilities) != len(HORIZONS) * PREDICTION:
        raise AssertionError("probability table has the wrong number of rows")
    if len(consistency) != len(HORIZONS) * PREDICTION:
        raise AssertionError("consistency report has the wrong number of rows")
    print(f"[ok] forecast for {result.model_name}: last close {result.last_close:,.2f}, "
          f"day {PREDICTION} median "
          f"{np.median(result.price_paths[:, -1]):,.2f}")
    print("\n[consistency, 3-day horizon, first three forecast days]")
    print(consistency[consistency["Return horizon"] == "3-day"].head(3).to_string(index=False))

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
