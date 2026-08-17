"""Yahoo Finance loading, the multi-horizon target, and real h-day candles.

THE TARGET

The model's cross-section at day `t` is the D = 3 vector of trailing log returns

    return_h(t) = log(close(t) / close(t - h))        h in (1, 3, 7)

Every horizon window ENDS at day `t`, so day `t` is always inside it:
`return_3d(t) = r(t-2) + r(t-1) + r(t)` where `r` is the daily log return. That
is the same quantity as the close-to-close return of a real 3-day candle whose
last daily candle is day `t`, which is what `aggregate_candles` builds below.
The trailing definition is also the only one that is causal: a centred or
forward-looking window would leak days after `t` into the features for `t`.

WHAT WAS ACTUALLY WRONG

The target was never the problem -- the reporting was. The model samples all
three dimensions freely, so its raw `return_3d` for forecast day 1 ignored the
fact that two of those three days have already happened and are known exactly.
The exported table could therefore claim a median 1-day return of +0.23% and a
median 3-day return of -0.41% for the same day, which cannot both be true. Any
number you would size a position with has to come from
`goldflow.forecast.derive_horizon_returns`, which rebuilds every horizon from
the sampled price path spliced onto the observed closes. The free model
dimensions are kept, but only as a training signal and a calibration
diagnostic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RETURN_HORIZONS: tuple[int, ...] = (1, 3, 7)
OHLC = ("Open", "High", "Low", "Close")


def download_history(ticker: str, period: str = "max") -> pd.DataFrame:
    """Daily OHLC for one ticker, de-duplicated and timezone naive."""
    try:
        import yfinance as yf

        frame = yf.download(
            ticker, period=period, interval="1d", auto_adjust=False,
            progress=False, threads=False,
        )
        if frame is None or frame.empty:
            raise ValueError(f"Yahoo Finance returned no rows for {ticker!r}")
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)

        columns = [column for column in OHLC if column in frame.columns]
        if "Close" not in columns:
            raise ValueError(f"Yahoo Finance returned no Close column for {ticker!r}")
        prices = frame[columns].astype(float).dropna(subset=["Close"])
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        prices = prices[~prices.index.duplicated(keep="last")].sort_index()
        if prices.empty:
            raise ValueError(f"every row for {ticker!r} was dropped as invalid")
        return prices
    except Exception as e:
        raise Exception(f"data - download_history - Execution error: {e}") from e


def latest_close(prices: pd.DataFrame) -> float:
    try:
        return float(prices["Close"].iloc[-1])
    except Exception as e:
        raise Exception(f"data - latest_close - Execution error: {e}") from e


def build_return_targets(close: pd.Series,
                         horizons=RETURN_HORIZONS) -> pd.DataFrame:
    """Aligned trailing log-return matrix, one column per horizon."""
    try:
        horizons = [int(horizon) for horizon in horizons]
        if not horizons or any(horizon < 1 for horizon in horizons):
            raise ValueError(f"horizons must be positive integers, got {horizons}")
        if 1 not in horizons:
            raise ValueError(
                "horizon 1 is required: the price path is reconstructed from the "
                "daily return dimension"
            )

        series = pd.Series(close).astype(float).dropna()
        frame = pd.DataFrame(
            {f"return_{horizon}d": np.log(series / series.shift(horizon))
             for horizon in horizons},
            index=series.index,
        ).dropna()
        if frame.empty:
            raise ValueError(
                f"{len(series)} closes are not enough for horizons {horizons}"
            )
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError("non-finite log returns: the price series has zeros or gaps")
        return frame.astype("float32")
    except Exception as e:
        raise Exception(f"data - build_return_targets - Execution error: {e}") from e


def aggregate_candles(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling h-day OHLC candles that end on -- and include -- each day.

    The candle stamped at day `t` covers the daily candles `t - window + 1 .. t`:

        Open  = Open(t - window + 1)
        High  = max High over the window
        Low   = min Low over the window
        Close = Close(t)

    Its close-to-close return is `log(Close(t) / Close(t - window))`, i.e.
    exactly `return_{window}d(t)`, because the close immediately preceding the
    window is `Close(t - window)`. That identity is what keeps the candle chart
    and the forecast tables talking about the same quantity.
    """
    try:
        window = int(window)
        if window < 1:
            raise ValueError(f"window must be at least 1, got {window}")
        missing = [column for column in OHLC if column not in prices.columns]
        if missing:
            raise ValueError(f"prices is missing OHLC columns {missing}")
        if len(prices) < window:
            raise ValueError(
                f"{len(prices)} rows cannot form a {window}-day candle"
            )

        frame = pd.DataFrame(index=prices.index)
        frame["Open"] = prices["Open"].shift(window - 1)
        frame["High"] = prices["High"].rolling(window).max()
        frame["Low"] = prices["Low"].rolling(window).min()
        frame["Close"] = prices["Close"]
        frame["WindowStart"] = pd.Series(prices.index, index=prices.index).shift(window - 1)
        return frame.dropna()
    except Exception as e:
        raise Exception(f"data - aggregate_candles - Execution error: {e}") from e


def split_indices(total_rows: int, prediction_length: int, test_windows: int,
                  validation_windows: int) -> dict:
    """Train / validation / test boundaries for the rolling-window protocol.

    The test range is the last `test_windows * prediction_length` rows, the
    validation range the block immediately before it, and training may only see
    what is left. Returned indices are exclusive ends into the target matrix.
    """
    try:
        total_rows = int(total_rows)
        prediction_length = int(prediction_length)
        test_windows = int(test_windows)
        validation_windows = int(validation_windows)
        if min(prediction_length, test_windows, validation_windows) < 1:
            raise ValueError("prediction_length and window counts must be positive")

        test_end = total_rows
        validation_end = test_end - test_windows * prediction_length
        train_end = validation_end - validation_windows * prediction_length
        if train_end <= 0:
            raise ValueError(
                f"{total_rows} rows cannot host {validation_windows} validation "
                f"and {test_windows} test windows of {prediction_length} days"
            )
        return {
            "train_end": train_end,
            "validation_end": validation_end,
            "test_end": test_end,
        }
    except Exception as e:
        raise Exception(f"data - split_indices - Execution error: {e}") from e
