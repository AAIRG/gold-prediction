"""Streamlit dashboard for the four gold models plus the boosted ensemble.

Every result -- price fan, return fan, probability table, calibration -- is shown
in its own tab per model, with the boosted mixture in a tab of its own and the
reproduced paper tables in a third. The forecast maths lives in `goldflow` so
that this file only arranges and renders what the notebook trained; nothing about
the target, the covariates or the horizon reconstruction is redefined here.

Reported horizon returns come from `goldflow.forecast.derive_horizon_returns`,
i.e. they are read off the sampled price path spliced onto the observed closes.
Each h-day window ends on its own forecast day and any day inside it that has
already traded enters as a known close. The flow's own free dimensions are shown
separately under "Model calibration" because their disagreement with the
reconstruction measures how well the joint distribution was learned, not what
the return will be.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE / "project"
# GOLDFLOW_ARTIFACTS lets the verification scripts and alternative deployments
# point at an export outside the repository without editing this file.
ARTIFACTS = Path(os.environ.get("GOLDFLOW_ARTIFACTS") or (HERE / "artifacts"))
for path in (HERE, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from goldflow import boosting, data, experiments, features, forecast, variants  # noqa: E402

DEFAULT_TICKER = "GC=F"
DEFAULT_SAMPLES = 500
DOWNLOAD_PERIOD = "1y"
CONTEXT_CANDLES = 45
ENSEMBLE_KEY = "boosted"
ENSEMBLE_NAME = "Boosted ensemble"

TEAL = "#0f766e"
AMBER = "#b45309"
INK = "#111827"
GREY = "#6b7280"
PALETTE = {
    "lstm_realnvp": "#0e7490",
    "lstm_maf": "#7c3aed",
    "transformer_maf": TEAL,
    "veclstm": AMBER,
    ENSEMBLE_KEY: "#be123c",
}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def load_prices(ticker: str) -> pd.DataFrame:
    """One short daily OHLC window -- never the full Yahoo history."""
    try:
        prices = data.download_history(ticker, period=DOWNLOAD_PERIOD)
        if len(prices) < 60:
            raise ValueError(
                f"Yahoo Finance returned only {len(prices)} daily candles, which "
                f"is not enough for the model context plus the 7-day horizon"
            )
        return prices
    except Exception as e:
        raise Exception(f"app - load_prices - Execution error: {e}") from e


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    try:
        path = ARTIFACTS / "gold_yahoo_metadata.json"
        if not path.exists():
            raise FileNotFoundError(
                "artifacts/gold_yahoo_metadata.json is missing. Run "
                "train_gold_yahoo.ipynb in Colab and copy the exported artifacts "
                "into artifacts/."
            )
        metadata = json.loads(path.read_text(encoding="utf-8"))
        features.assert_spec(metadata.get("time_feature_spec"))
        if not metadata.get("models"):
            raise ValueError("the exported metadata lists no models")
        return metadata
    except Exception as e:
        raise Exception(f"app - load_metadata - Execution error: {e}") from e


@st.cache_resource(show_spinner=False)
def load_models(metadata_fingerprint: str) -> dict:
    """Rebuild every exported variant on CPU from its own recorded config."""
    try:
        from flowforecast.training.trainer import load_checkpoint

        del metadata_fingerprint
        metadata = load_metadata()
        models = {}
        for entry in metadata["models"]:
            checkpoint = ARTIFACTS / entry["checkpoint"]
            if not checkpoint.exists():
                raise FileNotFoundError(f"missing checkpoint {checkpoint}")
            model = variants.build_variant(
                entry["key"], entry["config"],
                num_series=int(metadata["num_series"]),
                cond_size=int(metadata["cond_size"]),
                context_length=int(metadata["context_length"]),
                device="cpu",
            )
            load_checkpoint(
                model, str(checkpoint), device="cpu",
                expected_contract=variants.MODEL_CONTRACT,
            )
            model.eval()
            models[entry["key"]] = model
        return models
    except Exception as e:
        raise Exception(f"app - load_models - Execution error: {e}") from e


@st.cache_resource(show_spinner=False)
def load_booster(fingerprint: str):
    """Stage-2 residual booster, or None when it was gated off or is absent."""
    try:
        del fingerprint
        metadata = load_metadata()
        boost_meta = metadata.get("boosting") or {}
        if not boost_meta.get("booster_enabled"):
            return None
        path = ARTIFACTS / "gold_yahoo_booster.joblib"
        if not path.exists():
            return None
        import joblib

        booster = boosting.GradientResidualBooster.__new__(
            boosting.GradientResidualBooster
        )
        booster.model = joblib.load(path)
        booster.fitted = True
        return booster
    except Exception:
        # A missing or unreadable stage 2 must never take the dashboard down;
        # the ensemble tab reports that it is unavailable instead.
        return None


@st.cache_data(ttl=900, show_spinner=False)
def load_experiment_tables() -> dict:
    try:
        directory = ARTIFACTS / "experiments"
        tables = {}
        if not directory.exists():
            return tables
        for csv_path in sorted(directory.glob("*.csv")):
            tables[csv_path.stem] = pd.read_csv(csv_path)
        return tables
    except Exception as e:
        raise Exception(f"app - load_experiment_tables - Execution error: {e}") from e


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def run_forecasts(fingerprint: str, prices: pd.DataFrame, samples: int) -> dict:
    """Forecast every model plus the boosted mixture from one price window."""
    try:
        metadata = load_metadata()
        models = load_models(fingerprint)
        booster = load_booster(fingerprint)

        horizons = tuple(int(value) for value in metadata["return_horizons"])
        lags = [int(value) for value in metadata["lags"]]
        context_length = int(metadata["context_length"])
        prediction_length = int(metadata["prediction_length"])
        returns = data.build_return_targets(prices["Close"], horizons)

        results = {}
        for entry in metadata["models"]:
            results[entry["key"]] = forecast.forecast_prices(
                models[entry["key"]], entry["key"], entry["name"], prices,
                returns, horizons, lags, context_length, prediction_length,
                num_samples=int(samples), device="cpu", seed=int(metadata.get("inference_seed", 20260817)),
            )

        boost_meta = metadata.get("boosting") or {}
        weights = boost_meta.get("weights")
        ensemble = None
        if weights:
            close = prices["Close"].astype(float)
            aligned = close.loc[close.index <= returns.index[-1]]
            ensemble = forecast.ensemble_forecast(
                results, weights, int(boost_meta.get("pool_size", 24)),
                aligned.to_numpy(), model_key=ENSEMBLE_KEY,
                model_name=ENSEMBLE_NAME, booster=booster,
                daily_history=returns[f"return_{horizons[0]}d"].to_numpy(),
            )
        return {
            "results": results,
            "ensemble": ensemble,
            "returns": returns,
            "booster_active": booster is not None,
        }
    except Exception as e:
        raise Exception(f"app - run_forecasts - Execution error: {e}") from e


# --------------------------------------------------------------------------
# charts
# --------------------------------------------------------------------------

def fan_chart(dates, paths: np.ndarray, title: str, yaxis_title: str,
              value_format: str, color: str) -> go.Figure:
    """95% and 80% bands around the sampled median."""
    try:
        low95, low80, median, high80, high95 = np.quantile(
            paths, forecast.QUANTILES, axis=0
        )
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=dates, y=high95, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        figure.add_trace(go.Scatter(
            x=dates, y=low95, mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(14, 116, 144, 0.14)", name="95% interval",
            hovertemplate=f"95% low: {value_format}<extra></extra>",
        ))
        figure.add_trace(go.Scatter(
            x=dates, y=high80, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        figure.add_trace(go.Scatter(
            x=dates, y=low80, mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(14, 116, 144, 0.28)", name="80% interval",
            hovertemplate=f"80% low: {value_format}<extra></extra>",
        ))
        figure.add_trace(go.Scatter(
            x=dates, y=median, mode="lines+markers", name="Median",
            line=dict(color=color, width=3), marker=dict(size=6),
            hovertemplate=f"Median: {value_format}<extra></extra>",
        ))
        figure.update_layout(
            title=title, height=460, template="plotly_white",
            hovermode="x unified", margin=dict(l=16, r=16, t=52, b=16),
            yaxis_title=yaxis_title, xaxis_title=None,
            legend=dict(orientation="h", y=1.12, x=0),
        )
        return figure
    except Exception as e:
        raise Exception(f"app - fan_chart - Execution error: {e}") from e


def price_fan_chart(prices: pd.DataFrame, result: forecast.ForecastResult,
                    candle_window: int, color: str) -> go.Figure:
    """Price fan next to real candles of the selected aggregation window."""
    try:
        figure = fan_chart(
            result.future_dates, result.price_paths,
            f"{result.prediction_length}-day price fan - {result.model_name}",
            "Gold price (USD)", "$%{y:,.2f}", color,
        )
        candles = prices if candle_window == 1 \
            else data.aggregate_candles(prices, candle_window)
        context = candles.tail(CONTEXT_CANDLES)
        label = "Daily candles" if candle_window == 1 \
            else f"{candle_window}-day candles"
        figure.add_trace(go.Candlestick(
            x=context.index, open=context["Open"], high=context["High"],
            low=context["Low"], close=context["Close"], name=label,
            increasing_line_color=TEAL, decreasing_line_color=AMBER,
            increasing_fillcolor="#99f6e4", decreasing_fillcolor="#fed7aa",
        ))
        figure.add_trace(go.Scatter(
            x=[prices.index[-1]], y=[result.last_close], mode="markers",
            marker=dict(color=INK, size=9), name="Latest close",
            hovertemplate="Latest close: $%{y:,.2f}<extra></extra>",
        ))
        figure.update_layout(xaxis_rangeslider_visible=False)
        return figure
    except Exception as e:
        raise Exception(f"app - price_fan_chart - Execution error: {e}") from e


def weights_chart(weights: dict, labels: dict) -> go.Figure:
    try:
        keys = [key for key in weights if float(weights[key]) > 0]
        figure = go.Figure(go.Bar(
            x=[100.0 * float(weights[key]) for key in keys],
            y=[labels.get(key, key) for key in keys],
            orientation="h",
            marker=dict(color=[PALETTE.get(key, TEAL) for key in keys]),
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        ))
        figure.update_layout(
            title="Ensemble weights chosen by forward stagewise CRPS selection",
            height=260, template="plotly_white",
            margin=dict(l=16, r=16, t=52, b=16),
            xaxis_title="Mixture weight (%)",
        )
        return figure
    except Exception as e:
        raise Exception(f"app - weights_chart - Execution error: {e}") from e


# --------------------------------------------------------------------------
# tab bodies
# --------------------------------------------------------------------------

def probability_config(threshold_pct: float) -> dict:
    try:
        percent_columns = ["Median", "80% low", "80% high", "95% low", "95% high"]
        bar_columns = [
            "P(return > 0)", "P(return < 0)",
            f"P(return >= {float(threshold_pct):g}%)",
        ]
        config = {
            column: st.column_config.NumberColumn(format="%.3f%%")
            for column in percent_columns
        }
        config["Observed share"] = st.column_config.NumberColumn(
            "Already observed",
            help="Share of this h-day window that has already traded and enters "
                 "as a known close rather than a sampled one.",
            format="%.0f%%",
        )
        config.update({
            column: st.column_config.ProgressColumn(
                format="%.1f%%", min_value=0.0, max_value=100.0
            )
            for column in bar_columns
        })
        return config
    except Exception as e:
        raise Exception(f"app - probability_config - Execution error: {e}") from e


def render_forecast_tab(result: forecast.ForecastResult, prices: pd.DataFrame,
                        candle_window: int, threshold_pct: float,
                        scores: pd.DataFrame | None, caption: str) -> None:
    try:
        color = PALETTE.get(result.model_key, TEAL)
        quantiles = np.quantile(result.price_paths, forecast.QUANTILES, axis=0)
        columns = st.columns(4)
        columns[0].metric("Latest close", f"${result.last_close:,.2f}")
        columns[1].metric(
            f"Day {result.prediction_length} median", f"${quantiles[2, -1]:,.2f}",
            delta=f"{100.0 * (quantiles[2, -1] / result.last_close - 1.0):+.2f}%",
        )
        columns[2].metric(
            "80% range",
            f"${quantiles[1, -1]:,.0f} - ${quantiles[3, -1]:,.0f}",
        )
        columns[3].metric(
            "95% range",
            f"${quantiles[0, -1]:,.0f} - ${quantiles[4, -1]:,.0f}",
        )
        st.caption(caption)

        st.plotly_chart(
            price_fan_chart(prices, result, candle_window, color),
            width="stretch",
        )

        horizon = st.segmented_control(
            "Return horizon", options=list(result.horizons),
            default=result.horizons[0],
            format_func=lambda value: f"{value}-day return",
            key=f"horizon_{result.model_key}",
        )
        horizon = int(horizon or result.horizons[0])
        st.plotly_chart(
            fan_chart(
                result.future_dates,
                100.0 * result.horizon_returns[horizon],
                f"{horizon}-day return fan - {result.model_name}",
                "Log return (%)", "%{y:.2f}%", color,
            ).add_hline(y=0, line_dash="dot", line_color=GREY),
            width="stretch",
        )
        if horizon > 1:
            observed = result.observed_fraction[horizon]
            settled = int(np.sum(observed > 0))
            st.info(
                f"For the first {settled} forecast day(s) part of the {horizon}-day "
                f"window has already traded, so those returns are the sampled "
                f"path spliced onto known closes. From day {horizon} onward the "
                f"whole window is forecast.",
                icon=":material/info:",
            )

        st.subheader("Return probabilities")
        table = forecast.probability_frame(result, threshold_pct)
        st.dataframe(
            table, hide_index=True, width="stretch",
            column_config=probability_config(threshold_pct),
        )
        st.download_button(
            "Download this table as CSV",
            table.to_csv(index=False).encode("utf-8"),
            file_name=f"{result.model_key}_probabilities.csv",
            mime="text/csv", key=f"download_{result.model_key}",
        )

        if scores is not None and not scores.empty:
            st.subheader("Test-set scores for this model")
            st.dataframe(scores, hide_index=True, width="stretch")

        with st.expander("Model calibration: free dimensions vs reconstruction"):
            st.markdown(
                "The flow samples all three horizons independently, so its raw "
                "`3-day` and `7-day` draws ignore the days of the window that "
                "have already traded. A large gap on the early forecast days is "
                "expected and is exactly why the reported numbers are "
                "reconstructed from the price path instead of read off the "
                "network. A large gap on the last forecast days, where nothing "
                "is observed any more, means the joint distribution is poorly "
                "calibrated."
            )
            st.dataframe(
                forecast.consistency_report(result), hide_index=True,
                width="stretch",
                column_config={
                    "Observed share of window": st.column_config.NumberColumn(
                        format="%.2f"
                    ),
                },
            )
    except Exception as e:
        raise Exception(f"app - render_forecast_tab - Execution error: {e}") from e


def render_market_tab(prices: pd.DataFrame, returns: pd.DataFrame,
                      horizons: tuple) -> None:
    try:
        st.subheader("Real multi-day candles")
        st.markdown(
            "An h-day candle stamped on day `t` covers the daily candles "
            "`t-h+1 .. t`, so it always contains day `t` itself: its open is the "
            "open of the first day in the window, its high and low are the "
            "extremes over the window, and its close is the close of day `t`. "
            "Its close-to-close return is `log(close(t) / close(t-h))`, which is "
            "exactly the `return_{h}d` target the models are trained on."
        )
        window = st.segmented_control(
            "Candle aggregation", options=list(horizons), default=horizons[0],
            format_func=lambda value: "Daily" if value == 1 else f"{value}-day",
            key="market_candles",
        )
        window = int(window or horizons[0])
        candles = prices if window == 1 else data.aggregate_candles(prices, window)
        context = candles.tail(120)
        figure = go.Figure(go.Candlestick(
            x=context.index, open=context["Open"], high=context["High"],
            low=context["Low"], close=context["Close"],
            increasing_line_color=TEAL, decreasing_line_color=AMBER,
            increasing_fillcolor="#99f6e4", decreasing_fillcolor="#fed7aa",
        ))
        figure.update_layout(
            title="Daily candles" if window == 1
            else f"{window}-day candles, each ending on its own day",
            height=460, template="plotly_white",
            xaxis_rangeslider_visible=False,
            margin=dict(l=16, r=16, t=52, b=16),
            yaxis_title="Gold price (USD)",
        )
        st.plotly_chart(figure, width="stretch")

        st.subheader("Observed target matrix")
        tail = returns.tail(15).copy()
        tail.index = [stamp.date() for stamp in tail.index]
        st.dataframe(
            100.0 * tail, width="stretch",
            column_config={
                column: st.column_config.NumberColumn(format="%.3f%%")
                for column in tail.columns
            },
        )
        summary = pd.DataFrame({
            "Horizon": [f"{horizon}-day" for horizon in horizons],
            "Mean (%)": [100.0 * returns[f"return_{h}d"].mean() for h in horizons],
            "Sigma (%)": [100.0 * returns[f"return_{h}d"].std() for h in horizons],
            "Min (%)": [100.0 * returns[f"return_{h}d"].min() for h in horizons],
            "Max (%)": [100.0 * returns[f"return_{h}d"].max() for h in horizons],
        })
        st.dataframe(summary, hide_index=True, width="stretch")
    except Exception as e:
        raise Exception(f"app - render_market_tab - Execution error: {e}") from e


def render_boosting_tab(ensemble, metadata: dict, prices: pd.DataFrame,
                        candle_window: int, threshold_pct: float,
                        labels: dict, booster_active: bool,
                        scores: pd.DataFrame | None) -> None:
    try:
        boost_meta = metadata.get("boosting") or {}
        if ensemble is None or not boost_meta.get("weights"):
            st.warning(
                "No boosting metadata was exported. Re-run the notebook to fit "
                "the ensemble and the residual booster."
            )
            return

        st.markdown(
            "**Stage 1** is forward stagewise selection: each round adds the "
            "model that most reduces the CRPS_sum of the resulting mixture, with "
            "replacement, so the mixture stays a real predictive distribution and "
            "is scored by the same metric as the individual models. **Stage 2** is "
            "a gradient-boosted residual correction that only ships if it beat "
            "the mixture on windows it was not fitted on."
        )
        st.plotly_chart(
            weights_chart(boost_meta["weights"], labels),
            width="stretch",
        )

        columns = st.columns(3)
        rounds = boost_meta.get("rounds") or []
        columns[0].metric("Selection rounds", len(rounds))
        gate = "on" if boost_meta.get("booster_enabled") else "off"
        columns[1].metric(
            "Residual booster", gate,
            delta=f"{float(boost_meta.get('booster_gain_pct', 0.0)):+.2f}% CRPS_sum",
        )
        columns[2].metric(
            "Applied in this forecast", "yes" if booster_active else "no"
        )
        if boost_meta.get("booster_enabled") and not booster_active:
            st.warning(
                "Stage 2 was enabled during training but "
                "artifacts/gold_yahoo_booster.joblib could not be loaded, so only "
                "the stage 1 mixture is shown."
            )

        booster_metrics = boost_meta.get("booster_metrics") or {}
        if booster_metrics:
            with st.expander("Stage 2 gate detail"):
                st.dataframe(
                    pd.DataFrame([
                        {
                            "Metric": name,
                            "Mixture only": booster_metrics.get(f"{name}_before"),
                            "Mixture + booster": booster_metrics.get(f"{name}_after"),
                        }
                        for name in ("crps_sum", "crps", "mse")
                    ]),
                    hide_index=True, width="stretch",
                )

        if rounds:
            with st.expander("Forward stagewise selection trace"):
                trace = pd.DataFrame(rounds)
                trace["picked"] = trace["picked"].map(
                    lambda key: labels.get(key, key)
                )
                st.dataframe(trace, hide_index=True, width="stretch")

        st.divider()
        render_forecast_tab(
            ensemble, prices, candle_window, threshold_pct, scores,
            caption="Weighted mixture of the four variants, reconstructed from "
                    "the pooled joint samples.",
        )
    except Exception as e:
        raise Exception(f"app - render_boosting_tab - Execution error: {e}") from e


def render_experiments_tab(metadata: dict, tables: dict) -> None:
    try:
        st.subheader("Reproduced paper tables for gold")
        st.markdown(
            "Same protocol as §5.2: rolling test windows starting at the end of "
            "the training split, 100 samples for the empirical CDF, and "
            "`mean±stderr` over independent re-runs. CRPS_sum and CRPS are "
            "GluonTS `mean_wQuantileLoss` values aggregated as a ratio of global "
            "sums, not as a mean of per-window ratios."
        )
        protocol = metadata.get("experiment_protocol") or {}
        if protocol:
            columns = st.columns(5)
            columns[0].metric("Test windows", protocol.get("test_windows", "?"))
            columns[1].metric("Horizon", protocol.get("prediction_length", "?"))
            columns[2].metric("Samples", protocol.get("evaluation_samples", "?"))
            columns[3].metric("Seeds", len(protocol.get("seeds", []) or []))
            columns[4].metric("Training rows", protocol.get("training_rows", "?"))

        captions = {
            "table1_crps_sum": "Table 1 - test set CRPS_sum (lower is better)",
            "table4_crps": "Table 4 - test set CRPS (lower is better)",
            "table5_mse": "Table 5 - test set MSE (lower is better)",
        }
        for name, caption in captions.items():
            frame = tables.get(name)
            if frame is None:
                continue
            st.markdown(f"**{caption}**")
            st.dataframe(frame, hide_index=True, width="stretch")

        board = tables.get("leaderboard")
        if board is not None:
            st.subheader("Which model is best on gold")
            display = board.drop(columns=["model_key"], errors="ignore")
            st.dataframe(display, hide_index=True, width="stretch")
            best = board.iloc[0]
            st.success(
                f"**{best['Model']}** has the lowest CRPS_sum on the gold test "
                f"windows at {float(best['crps_sum_mean']):.5f} "
                f"(±{float(best.get('crps_sum_stderr', 0.0)):.5f} over "
                f"{int(best.get('runs', 1))} run(s)).",
                icon=":material/trophy:",
            )

        coverage = tables.get("coverage")
        if coverage is not None:
            st.subheader("Interval calibration")
            st.caption(
                "A model can win CRPS_sum and still be unusable for sizing: if "
                "the 80% band only contains 60% of outcomes the intervals are "
                "too tight. Coverage close to the nominal level with a narrow "
                "width is what you want."
            )
            st.dataframe(coverage, hide_index=True, width="stretch")

        per_window = tables.get("per_window_metrics")
        if per_window is not None:
            with st.expander("Per-window metrics"):
                st.dataframe(per_window, hide_index=True, width="stretch")

        for name, frame in tables.items():
            if name in captions or name in (
                    "leaderboard", "coverage", "per_window_metrics"):
                continue
            with st.expander(name.replace("_", " ")):
                st.dataframe(frame, hide_index=True, width="stretch")
    except Exception as e:
        raise Exception(f"app - render_experiments_tab - Execution error: {e}") from e


def model_scores(tables: dict, model_key: str) -> pd.DataFrame | None:
    try:
        board = tables.get("leaderboard")
        if board is None or "model_key" not in board.columns:
            return None
        row = board[board["model_key"] == model_key]
        if row.empty:
            return None
        record = row.iloc[0]
        rows = []
        for metric in experiments.REPORTED_METRICS:
            mean = record.get(f"{metric}_mean")
            if mean is None or pd.isna(mean):
                continue
            rows.append({
                "Metric": metric,
                "Value": float(mean),
                "Std error": float(record.get(f"{metric}_stderr", 0.0) or 0.0),
                "Rank": int(record.get("Rank", 0)),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        raise Exception(f"app - model_scores - Execution error: {e}") from e


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Gold multi-model forecast", page_icon="G", layout="wide"
    )
    st.markdown(
        """<style>
        .block-container {max-width: 1420px; padding-top: 1.6rem;}
        [data-testid="stMetric"] {border-left: 3px solid #0f766e; padding-left: 0.8rem;}
        </style>""",
        unsafe_allow_html=True,
    )
    st.title("Gold price and return uncertainty")
    st.caption(
        "Four probabilistic variants and a boosted ensemble, each in its own "
        "tab, over a joint 1-day, 3-day and 7-day return forecast"
    )

    try:
        metadata = load_metadata()
    except Exception as error:
        st.error(str(error))
        st.stop()

    fingerprint = json.dumps(
        {
            "trained": metadata.get("trained_at"),
            "models": [entry["key"] for entry in metadata["models"]],
        },
        sort_keys=True,
    )
    horizons = tuple(int(value) for value in metadata["return_horizons"])
    labels = {entry["key"]: entry["name"] for entry in metadata["models"]}
    labels[ENSEMBLE_KEY] = ENSEMBLE_NAME

    with st.sidebar:
        st.header("Forecast settings")
        ticker = st.text_input("Yahoo ticker", metadata.get("ticker", DEFAULT_TICKER))
        samples = st.slider("Monte Carlo samples", 200, 2000, DEFAULT_SAMPLES, step=100)
        threshold_pct = st.number_input(
            "Return probability threshold (%)", value=1.0, step=0.25
        )
        candle_window = st.selectbox(
            "Candles on the price fan", options=list(horizons), index=0,
            format_func=lambda value: "Daily" if value == 1 else f"{value}-day",
        )
        if st.button("Refresh market data", width="stretch"):
            load_prices.clear()
            run_forecasts.clear()
            load_experiment_tables.clear()
        st.divider()
        st.caption(
            f"Trained through {metadata.get('training_end_date', 'unknown')} - "
            f"{len(metadata['models'])} variants - contract "
            f"`{metadata.get('model_contract', 'unknown')}`"
        )

    try:
        prices = load_prices((ticker or DEFAULT_TICKER).strip())
        with st.spinner("Sampling every model on CPU..."):
            bundle = run_forecasts(fingerprint, prices, int(samples))
        tables = load_experiment_tables()
    except Exception as error:
        st.error(str(error))
        st.stop()

    order = [entry["key"] for entry in metadata["models"]]
    tab_labels = [labels[key] for key in order]
    tab_labels += [ENSEMBLE_NAME, "Experiments", "Market data"]
    tabs = st.tabs(tab_labels)

    for index, key in enumerate(order):
        entry = next(e for e in metadata["models"] if e["key"] == key)
        with tabs[index]:
            render_forecast_tab(
                bundle["results"][key], prices, int(candle_window),
                float(threshold_pct), model_scores(tables, key),
                caption=entry.get("description", ""),
            )

    with tabs[len(order)]:
        # When stage 2 shipped, the scores that belong next to the ensemble tab
        # are the ones measured with the correction applied.
        boosted_scores = None
        if bundle["booster_active"]:
            boosted_scores = model_scores(tables, "boosted_gb")
        if boosted_scores is None or boosted_scores.empty:
            boosted_scores = model_scores(tables, ENSEMBLE_KEY)
        render_boosting_tab(
            bundle["ensemble"], metadata, prices, int(candle_window),
            float(threshold_pct), labels, bundle["booster_active"],
            boosted_scores,
        )
    with tabs[len(order) + 1]:
        render_experiments_tab(metadata, tables)
    with tabs[len(order) + 2]:
        render_market_tab(prices, bundle["returns"], horizons)

    st.caption(
        f"{len(metadata['models'])} variants, {int(samples):,} joint trajectories "
        f"each, CPU inference - Yahoo data refreshed "
        f"{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M local')} - horizon returns "
        f"reconstructed from the sampled price path spliced onto observed closes"
    )


if __name__ == "__main__":
    main()
