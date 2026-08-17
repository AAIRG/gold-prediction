"""Headless check of the Streamlit app's logic against the exported artifacts.

Streamlit's caching decorators are bypassed with `__wrapped__` so the real
loading, forecasting and table code runs outside a browser session. This proves
the app can rebuild every exported variant, reconstruct the horizons, pool the
boosted mixture, and read the paper tables, without needing a server or network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "project"))

from run_pipeline_local import synthetic_prices  # noqa: E402


def unwrap(function):
    return getattr(function, "__wrapped__", function)


def main() -> None:
    import app as dashboard

    metadata = unwrap(dashboard.load_metadata)()
    print(f"[ok] metadata: {len(metadata['models'])} models, "
          f"contract {metadata['model_contract']}")

    models = unwrap(dashboard.load_models)("test")
    if set(models) != {entry["key"] for entry in metadata["models"]}:
        raise AssertionError("the app did not rebuild every exported model")
    print(f"[ok] rebuilt models on CPU: {sorted(models)}")

    booster = unwrap(dashboard.load_booster)("test")
    print(f"[ok] stage 2 booster loaded: {booster is not None} "
          f"(gate said {metadata['boosting']['booster_enabled']})")
    if booster is not None and not metadata["boosting"]["booster_enabled"]:
        raise AssertionError("a gated-off booster was loaded anyway")

    prices = synthetic_prices(rows=320, seed=99)
    bundle = unwrap(dashboard.run_forecasts)("test", prices, 64)
    results = bundle["results"]
    if set(results) != set(models):
        raise AssertionError("not every model produced a forecast")

    horizons = tuple(int(v) for v in metadata["return_horizons"])
    for key, result in results.items():
        if result.price_paths.shape != (64, metadata["prediction_length"]):
            raise AssertionError(f"{key} price paths have shape {result.price_paths.shape}")
        for horizon in horizons:
            block = result.horizon_returns[horizon]
            if block.shape != result.price_paths.shape:
                raise AssertionError(f"{key} horizon {horizon} has shape {block.shape}")
            if not np.isfinite(block).all():
                raise AssertionError(f"{key} horizon {horizon} has non-finite values")
        print(f"[ok] {result.model_name}: day-10 median "
              f"${np.median(result.price_paths[:, -1]):,.2f}")

    # The reported 1-day return must equal the daily dimension of the path, and
    # the 3-day return at day 3 must equal the sum of the first three 1-day
    # reported returns. If either fails the reconstruction is not self-consistent.
    reference = results[next(iter(results))]
    daily = reference.horizon_returns[1]
    error = float(np.abs(daily[:, :3].sum(axis=1) - reference.horizon_returns[3][:, 2]).max())
    if error > 1e-10:
        raise AssertionError(f"reported horizons are not self-consistent: {error}")
    print("[ok] reported 3-day day-3 return equals the sum of three 1-day returns")

    ensemble = bundle["ensemble"]
    if ensemble is None:
        raise AssertionError("no boosted ensemble was produced")
    weights = metadata["boosting"]["weights"]
    active = sum(1 for value in weights.values() if value > 0)
    expected = sum(
        max(1, round(value / sum(v for v in weights.values() if v > 0)
                     * metadata["boosting"]["pool_size"]))
        for value in weights.values() if value > 0
    )
    if ensemble.model_returns.shape[0] != 64 * expected:
        raise AssertionError(
            f"mixture has {ensemble.model_returns.shape[0]} draws, expected "
            f"{64 * expected} from {active} active models"
        )
    print(f"[ok] boosted mixture: {ensemble.model_returns.shape[0]} pooled draws "
          f"from {active} active models")

    for key, result in list(results.items()) + [("boosted", ensemble)]:
        table = dashboard.forecast.probability_frame(result, 1.0)
        if len(table) != len(horizons) * metadata["prediction_length"]:
            raise AssertionError(f"{key} probability table is the wrong size")
        probabilities = table["P(return > 0)"] + table["P(return < 0)"]
        if not np.allclose(probabilities, 100.0, atol=1e-6):
            raise AssertionError(f"{key} up/down probabilities do not sum to 100")
    print("[ok] probability tables are complete and normalized")

    consistency = dashboard.forecast.consistency_report(ensemble)
    day1 = consistency[
        (consistency["Return horizon"] == "3-day")
        & (consistency["Forecast day"] == 1)
    ].iloc[0]
    if abs(day1["Observed share of window"] - 2 / 3) > 1e-9:
        raise AssertionError("the 3-day day-1 observed share is not 2/3")
    print(f"[ok] consistency report: 3-day day-1 observed share "
          f"{day1['Observed share of window']:.3f}, median gap "
          f"{day1['Median gap (pp)']:+.3f}pp")

    tables = unwrap(dashboard.load_experiment_tables)()
    required = {"table1_crps_sum", "table4_crps", "table5_mse", "leaderboard",
                "coverage", "per_window_metrics"}
    missing = required - set(tables)
    if missing:
        raise AssertionError(f"the app cannot see these tables: {sorted(missing)}")
    print(f"[ok] experiment tables visible: {sorted(tables)}")

    for key in list(models) + ["boosted"]:
        scores = dashboard.model_scores(tables, key)
        if scores is None or scores.empty:
            raise AssertionError(f"no leaderboard scores for {key}")
    print("[ok] per-model leaderboard scores resolve for every tab")

    market = dashboard.data.aggregate_candles(prices, 7)
    if market.empty or not {"Open", "High", "Low", "Close"} <= set(market.columns):
        raise AssertionError("7-day candles were not built")
    print(f"[ok] 7-day candles: {len(market)} rows, last close "
          f"${market['Close'].iloc[-1]:,.2f}")

    figure = dashboard.price_fan_chart(prices, ensemble, 3, "#0f766e")
    if not figure.data:
        raise AssertionError("the price fan chart is empty")
    print(f"[ok] charts render: {len(figure.data)} traces on the price fan")

    print("\nAPP LOGIC OK")


if __name__ == "__main__":
    main()
