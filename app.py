"""Streamlit dashboard for the exported multivariate gold return model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE / "project"
ARTIFACTS = HERE / "artifacts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowforecast.data.covariates import time_features  # noqa: E402
from flowforecast.data.scaling import compute_scale, invert_scale  # noqa: E402
from flowforecast.evaluation.sampling import sample_trajectories  # noqa: E402
from flowforecast.training.trainer import load_checkpoint  # noqa: E402
from flowforecast.utils.build import build_model  # noqa: E402


DEFAULT_TICKER = "GC=F"
DEFAULT_SAMPLES = 500
INFERENCE_DOWNLOAD_PERIOD = "3mo"
QUANTILES = (0.025, 0.10, 0.50, 0.90, 0.975)


@st.cache_data(ttl=900, show_spinner=False)
def download_gold(ticker: str) -> pd.DataFrame:
    """Fetch one short daily OHLC window, not the full Yahoo history."""
    try:
        import yfinance as yf

        frame = yf.download(
            ticker, period=INFERENCE_DOWNLOAD_PERIOD, interval="1d", auto_adjust=False,
            progress=False, threads=False,
        )
        if frame.empty:
            raise ValueError(f"Yahoo Finance returned no rows for {ticker}")
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        columns = [column for column in ("Open", "High", "Low", "Close") if column in frame]
        prices = frame[columns].astype(float).dropna(subset=["Close"])
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        prices = prices[~prices.index.duplicated(keep="last")]
        if len(prices) < 40:
            raise ValueError("Yahoo Finance returned too few daily candles for the model context")
        return prices
    except Exception as error:
        raise RuntimeError(
            f"app - download_gold - Yahoo Finance download failed: {error}"
        ) from error


@st.cache_resource(show_spinner=False)
def load_exported_model(model_path: str, metadata_path: str):
    """Recreate the exact trained model using the exported metadata contract."""
    try:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        model = build_model(
            metadata["config"], int(metadata["num_series"]),
            int(metadata["cond_size"]),
            context_length=int(metadata["context_length"]),
        )
        load_checkpoint(
            model, model_path, device="cpu",
            expected_contract="tempflow_v5_official_parity",
        )
        model.eval()
        return model, metadata
    except Exception as error:
        raise RuntimeError(
            f"app - load_exported_model - Model loading failed: {error}"
        ) from error


def make_return_frame(close: pd.Series, horizons: list[int]) -> pd.DataFrame:
    try:
        frame = pd.DataFrame({
            f"return_{horizon}d": np.log(close / close.shift(horizon))
            for horizon in horizons
        }).dropna()
        if frame.empty:
            raise ValueError("not enough prices to calculate all return horizons")
        return frame.astype("float32")
    except Exception as error:
        raise RuntimeError(
            f"app - make_return_frame - Return feature construction failed: {error}"
        ) from error


def forecast_paths(model, metadata: dict, prices: pd.DataFrame, samples: int):
    """Sample joint return paths and derive the internally consistent price paths."""
    try:
        horizons = [int(value) for value in metadata["return_horizons"]]
        context_length = int(metadata["context_length"])
        prediction_length = int(metadata["prediction_length"])
        lags = [int(value) for value in metadata["lags"]]
        max_lag = max(lags)
        close = prices["Close"]
        returns = make_return_frame(close, horizons)
        required = context_length + max_lag
        if len(returns) < required:
            raise ValueError(
                f"need {required} aligned return rows, only {len(returns)} are available"
            )

        history = torch.from_numpy(returns.iloc[-required:].to_numpy()).unsqueeze(0)
        future_dates = pd.bdate_range(
            returns.index[-1] + pd.offsets.BDay(1), periods=prediction_length,
        )
        all_dates = pd.DatetimeIndex(returns.index.tolist() + future_dates.tolist())
        features = time_features(all_dates, "B")[:, -(required + prediction_length):]
        scale = compute_scale(history[0, max_lag:, :], time_dim=0)
        sampled_scaled = sample_trajectories(
            model, history, features.t().unsqueeze(0), lags, prediction_length,
            num_samples=samples, context_length=context_length, scale=scale,
        )
        sampled_returns = invert_scale(sampled_scaled, scale, time_dim=-1)
        returns_paths = sampled_returns.detach().cpu().numpy()
        daily_index = horizons.index(1)
        price_paths = float(close.iloc[-1]) * np.exp(
            np.cumsum(returns_paths[:, daily_index, :], axis=1)
        )
        return future_dates, returns_paths, price_paths, horizons
    except Exception as error:
        raise RuntimeError(
            f"app - forecast_paths - Probabilistic forecast failed: {error}"
        ) from error


@st.cache_data(ttl=900, show_spinner=False)
def cached_forecast(model_path: str, metadata_path: str, prices: pd.DataFrame,
                    samples: int):
    """Cache CPU inference for identical data, model, and sample settings."""
    try:
        model, metadata = load_exported_model(model_path, metadata_path)
        return forecast_paths(model, metadata, prices, samples), metadata
    except Exception as error:
        raise RuntimeError(
            f"app - cached_forecast - Cached inference failed: {error}"
        ) from error


def fan_chart(dates: pd.DatetimeIndex, paths: np.ndarray, title: str,
              yaxis_title: str, value_format: str, color: str) -> go.Figure:
    """Render 95% and 80% fan bands around the sampled median path."""
    q_low_95, q_low_80, q_median, q_high_80, q_high_95 = np.quantile(
        paths, QUANTILES, axis=0
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=q_high_95, mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=q_low_95, mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(14, 116, 144, 0.14)", name="95% interval",
        hovertemplate=f"95% low: {value_format}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=q_high_80, mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=q_low_80, mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(14, 116, 144, 0.28)", name="80% interval",
        hovertemplate=f"80% low: {value_format}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=q_median, mode="lines+markers", name="Median",
        line=dict(color=color, width=3), marker=dict(size=6),
        hovertemplate=f"Median: {value_format}<extra></extra>",
    ))
    fig.update_layout(
        title=title, height=470, template="plotly_white", hovermode="x unified",
        margin=dict(l=16, r=16, t=52, b=16), yaxis_title=yaxis_title,
        xaxis_title=None, legend=dict(orientation="h", y=1.12, x=0),
    )
    return fig


def price_fan_chart(prices: pd.DataFrame, dates: pd.DatetimeIndex,
                    paths: np.ndarray, context_length: int) -> go.Figure:
    close = prices["Close"]
    fig = fan_chart(
        dates, paths, "10-day price fan chart", "Gold price (USD)",
        "$%{y:,.2f}", "#0f766e",
    )
    context = prices.tail(context_length)
    fig.add_trace(go.Candlestick(
        x=context.index, open=context["Open"], high=context["High"],
        low=context["Low"], close=context["Close"], name="Context candles",
        increasing_line_color="#0f766e", decreasing_line_color="#b45309",
        increasing_fillcolor="#99f6e4", decreasing_fillcolor="#fed7aa",
    ))
    fig.add_trace(go.Scatter(
        x=[close.index[-1]], y=[close.iloc[-1]], mode="markers",
        marker=dict(color="#111827", size=9), name="Latest close",
        hovertemplate="Latest close: $%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(xaxis_rangeslider_visible=False)
    return fig


def probability_table(dates: pd.DatetimeIndex, returns_paths: np.ndarray,
                      horizons: list[int], threshold_pct: float) -> pd.DataFrame:
    """Create the realtime probability table from the joint Monte Carlo paths."""
    rows = []
    threshold = threshold_pct / 100.0
    for series_index, horizon in enumerate(horizons):
        paths = returns_paths[:, series_index, :]
        quantiles = np.quantile(paths, QUANTILES, axis=0)
        for day_index, date in enumerate(dates):
            draws = paths[:, day_index]
            rows.append({
                "Date": date.date(),
                "Forecast day": day_index + 1,
                "Return horizon": f"{horizon}-day",
                "Median": 100.0 * quantiles[2, day_index],
                "80% low": 100.0 * quantiles[1, day_index],
                "80% high": 100.0 * quantiles[3, day_index],
                "95% low": 100.0 * quantiles[0, day_index],
                "95% high": 100.0 * quantiles[4, day_index],
                "P(return > 0)": 100.0 * float((draws > 0).mean()),
                "P(return < 0)": 100.0 * float((draws < 0).mean()),
                f"P(return >= {threshold_pct:g}%)": 100.0 * float((draws >= threshold).mean()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Gold Fan Forecast", page_icon="G", layout="wide")
    st.markdown(
        """<style>
        .block-container {max-width: 1280px; padding-top: 1.8rem;}
        [data-testid="stMetric"] {border-left: 3px solid #0f766e; padding-left: 0.8rem;}
        </style>""",
        unsafe_allow_html=True,
    )
    st.title("Gold price and return uncertainty")
    st.caption("Joint 1-day, 3-day, and 7-day return forecast for the next 10 business days")

    with st.sidebar:
        st.header("Forecast settings")
        ticker = st.text_input("Yahoo ticker", DEFAULT_TICKER)
        samples = st.slider("Monte Carlo samples", 200, 1500, DEFAULT_SAMPLES, step=100)
        threshold_pct = st.number_input("Return probability threshold (%)", value=1.0, step=0.25)
        refresh = st.button("Refresh market data", use_container_width=True)
        if refresh:
            download_gold.clear()
            cached_forecast.clear()

    model_path = ARTIFACTS / "gold_yahoo_model.pt"
    metadata_path = ARTIFACTS / "gold_yahoo_metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        st.error("Missing model artifacts. Add gold_yahoo_model.pt and gold_yahoo_metadata.json to artifacts/ after training in Colab.")
        st.stop()

    try:
        prices = download_gold(ticker.strip() or DEFAULT_TICKER)
        forecast, metadata = cached_forecast(
            str(model_path), str(metadata_path), prices, samples
        )
        dates, return_paths, price_paths, horizons = forecast
    except Exception as error:
        st.error(str(error))
        st.stop()

    price_quantiles = np.quantile(price_paths, QUANTILES, axis=0)
    metrics = st.columns(4)
    close = prices["Close"]
    metrics[0].metric("Latest close", f"${close.iloc[-1]:,.2f}")
    metrics[1].metric("Day 10 median", f"${price_quantiles[2, -1]:,.2f}")
    metrics[2].metric("Day 10 80% range", f"${price_quantiles[1, -1]:,.2f} - ${price_quantiles[3, -1]:,.2f}")
    metrics[3].metric("Day 10 95% range", f"${price_quantiles[0, -1]:,.2f} - ${price_quantiles[4, -1]:,.2f}")

    st.plotly_chart(
        price_fan_chart(prices, dates, price_paths, int(metadata["context_length"])),
        use_container_width=True,
    )

    selected_horizon = st.segmented_control(
        "Return fan chart", options=horizons, default=horizons[0],
        format_func=lambda value: f"{value}-day return",
    )
    return_index = horizons.index(selected_horizon)
    return_paths_pct = return_paths[:, return_index, :] * 100.0
    st.plotly_chart(
        fan_chart(
            dates, return_paths_pct,
            f"{selected_horizon}-day return fan chart", "Log return (%)",
            "%{y:.2f}%", "#b45309",
        ).add_hline(y=0, line_dash="dot", line_color="#6b7280"),
        use_container_width=True,
    )

    st.subheader("Realtime return probabilities")
    table = probability_table(dates, return_paths, horizons, threshold_pct)
    probability_columns = [
        "P(return > 0)", "P(return < 0)", f"P(return >= {threshold_pct:g}%)",
    ]
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Median": st.column_config.NumberColumn(format="%.3f%%"),
            "80% low": st.column_config.NumberColumn(format="%.3f%%"),
            "80% high": st.column_config.NumberColumn(format="%.3f%%"),
            "95% low": st.column_config.NumberColumn(format="%.3f%%"),
            "95% high": st.column_config.NumberColumn(format="%.3f%%"),
            **{column: st.column_config.ProgressColumn(format="%.1f%%", min_value=0.0, max_value=100.0) for column in probability_columns},
        },
    )
    st.caption(
        f"Model: {metadata.get('model_name', 'Transformer-MAF')} | "
        f"trained through {metadata.get('training_end_date', 'unknown')} | "
        f"Yahoo data refreshed at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M local')} | "
        f"{samples:,} sampled joint trajectories"
    )


if __name__ == "__main__":
    main()
