"""Time covariates and lag features (paper §4.2 and Appendix D.1).

§4.2: covariates are time-dependent features (day of week, hour of day),
time-independent embeddings if applicable, and lag features depending on the
data frequency. "All covariates are thus known for the time periods we wish to
forecast."

Appendix D.1, exact values:

    hourly      -> hour-of-day, day-of-week, day-of-month;  lags [1, 24, 168]
    daily       -> day-of-week;                             lags [1, 7, 14]
    half-hourly -> minute-of-hour, hour-of-day, day-of-week; lags [1, 2, 4, 12, 24, 48]

"The normalized features are concatenated to the RNN or Transformer input at
each time step" -- covariates feed the backbone, NOT the flow. The flow only
ever sees x_t and h_t.

Two things this module is careful about, because both were bugs before:

1. TIME features need REAL timestamps. Passing `range(T)` to pd.DatetimeIndex
   makes pandas read the values as nanoseconds since the epoch, so day-of-week
   is constant and the feature is dead weight. Use `make_timestamps(...)`, which
   wraps pd.date_range with the dataset's own start date and frequency.

2. LAG features are functions of x, so they are NOT known over the forecast
   horizon (lag 1 at horizon step 2 is the value we are trying to predict).
   Every lag consumer here therefore builds them from an explicit history
   buffer: during training from the sampled window's own history
   (`lag_features_from_history`), during sampling from the running buffer of
   already generated values (`lag_vector_from_history`). Both produce the same
   channel ordering: for each lag, the D series, in the order of `lags`.

Feature layout of the covariate vector (kept identical in training and
sampling): concat(lag_features [D * len(lags)], time_features [C_time]).
"""

import pandas as pd
import torch

LAGS_BY_FREQ = {
    "H": [1, 24, 168],              # [paper] Appendix D.1, hourly
    "D": [1, 7, 14],                # [paper] Appendix D.1, daily
    "B": [1, 2],                    # [official] pytorch-ts Fourier lag helper
    "30min": [1, 4, 12, 24, 48],    # [official] pytorch-ts Fourier lag helper
}

FEATURES_BY_FREQ = {
    "H": ["hour", "dayofweek"],
    "D": ["dayofweek"],
    "B": ["dayofweek", "dayofyear"],
    "30min": ["minute", "hour", "dayofweek"],
}

# Canonical frequency keys used across the project, and the pandas alias used to
# build a DatetimeIndex for each of them (lower case: 'H'/'T' are deprecated in
# pandas >= 2.2). 'B' (business day) is deliberately NOT folded into 'D': the
# GluonTS Exchange dataset is business-daily, and indexing it with a plain daily
# range would drift the day-of-week feature by two days every week.
_FREQ_CANON = {
    "H": "H", "1H": "H", "HOURLY": "H", "60MIN": "H", "60T": "H",
    "D": "D", "1D": "D", "DAILY": "D",
    "B": "B", "1B": "B", "BUSINESS": "B",
    "30MIN": "30min", "30T": "30min", "0.5H": "30min",
}

_PANDAS_ALIAS = {"H": "h", "D": "D", "B": "B", "30min": "30min"}


def normalize_freq(freq: str) -> str:
    """Map any spelling of a supported frequency to the canonical key."""
    try:
        key = str(freq).strip().upper()
        if key not in _FREQ_CANON:
            raise ValueError(
                f"Unsupported frequency {freq!r}. Supported: hourly (H), "
                f"daily (D), half-hourly (30min)."
            )
        return _FREQ_CANON[key]
    except Exception as e:
        raise Exception(f"covariates - normalize_freq - Execution error: {e}") from e


def pandas_freq(freq: str) -> str:
    """Canonical key -> pandas offset alias."""
    try:
        return _PANDAS_ALIAS[normalize_freq(freq)]
    except Exception as e:
        raise Exception(f"covariates - pandas_freq - Execution error: {e}") from e


def lags_for_freq(freq: str) -> list:
    try:
        return list(LAGS_BY_FREQ[normalize_freq(freq)])
    except Exception as e:
        raise Exception(f"covariates - lags_for_freq - Execution error: {e}") from e


def num_time_features(freq: str) -> int:
    try:
        return 2 * len(FEATURES_BY_FREQ[normalize_freq(freq)])
    except Exception as e:
        raise Exception(f"covariates - num_time_features - Execution error: {e}") from e


def covariate_dim(num_series: int, freq: str, lags: list) -> int:
    """C = C_time + D * len(lags). This is the backbone's `cond_size`."""
    try:
        return num_time_features(freq) + num_series * len(lags)
    except Exception as e:
        raise Exception(f"covariates - covariate_dim - Execution error: {e}") from e


def make_timestamps(start, periods: int, freq: str) -> pd.DatetimeIndex:
    """Real calendar timestamps for the whole series (Appendix D.1 needs them)."""
    try:
        if isinstance(start, pd.Period):
            start = start.to_timestamp()
        start = pd.Timestamp(start)
        return pd.date_range(start=start, periods=int(periods),
                             freq=pandas_freq(freq))
    except Exception as e:
        raise Exception(f"covariates - make_timestamps - Execution error: {e}") from e


def time_features(timestamps, freq: str) -> torch.Tensor:
    """Official Fourier sine/cosine calendar features, shape [C_time, T]."""
    try:
        freq = normalize_freq(freq)
        dt = timestamps if isinstance(timestamps, pd.DatetimeIndex) \
            else pd.DatetimeIndex(timestamps)
        if len(dt) > 1 and (dt[1] - dt[0]) < pd.Timedelta("1min"):
            # This is the range(T) bug: pandas reads plain integers as
            # nanoseconds, every timestamp lands on 1970-01-01 and every time
            # feature becomes a constant.
            raise ValueError(
                "timestamps are spaced less than a minute apart -- they look "
                "like raw integers. Build them with make_timestamps(start, T, freq)."
            )

        feats = []
        for feature in FEATURES_BY_FREQ[freq]:
            if feature == "minute":
                values = dt.minute.to_numpy()
            elif feature == "hour":
                values = dt.hour.to_numpy()
            elif feature == "dayofweek":
                values = dt.dayofweek.to_numpy()
            elif feature == "dayofyear":
                values = dt.dayofyear.to_numpy()
            else:
                raise ValueError(f"Unknown time feature {feature!r}")
            period = max(int(values.max()) + 1, 1)
            phase = torch.as_tensor(values, dtype=torch.float32) \
                * (2.0 * torch.pi / period)
            feats.extend([torch.cos(phase), torch.sin(phase)])

        return torch.stack(feats)
    except Exception as e:
        raise Exception(f"covariates - time_features - Execution error: {e}") from e


def lag_features(series: torch.Tensor, lags: list) -> torch.Tensor:
    """[D * len(lags), T] with zero padding for the first max(lags) steps.

    Only used for whole-series diagnostics and the unit tests: the training and
    sampling paths use the two history-based helpers below, which never pad.
    """
    try:
        D, T = series.shape
        lag_feats = []
        for lag in lags:
            padded = torch.zeros((D, T), dtype=series.dtype, device=series.device)
            if T > lag:
                padded[:, lag:] = series[:, :T - lag]
            lag_feats.append(padded)
        return torch.cat(lag_feats, dim=0)
    except Exception as e:
        raise Exception(f"covariates - lag_features - Execution error: {e}") from e


def lag_features_from_history(extended: torch.Tensor, lags: list,
                              window_len: int) -> torch.Tensor:
    """Lag channels for a window that carries its own history.

    `extended` is [D, max(lags) + window_len]; `extended[:, max_lag + i]` is the
    window value at index i. The lag-l channel at window index i is
    `extended[:, max_lag + i - l]`, so nothing is ever zero padded.
    """
    try:
        max_lag = max(lags)
        D, n = extended.shape
        if n < max_lag + window_len:
            raise ValueError(
                f"extended history too short: got {n}, need {max_lag + window_len}"
            )
        blocks = []
        for lag in lags:
            offset = max_lag - lag
            blocks.append(extended[:, offset:offset + window_len])
        return torch.cat(blocks, dim=0)
    except Exception as e:
        raise Exception(
            f"covariates - lag_features_from_history - Execution error: {e}"
        ) from e


def lag_vector_from_history(history: torch.Tensor, lags: list) -> torch.Tensor:
    """Lag channels for the single next time step of an autoregressive rollout.

    `history` is [B, n, D] holding x_0 .. x_{n-1}; the returned [B, D * len(lags)]
    vector is the lag part of the covariates at index n, i.e. x_{n-l} for each l.
    """
    try:
        n = history.shape[1]
        max_lag = max(lags)
        if n < max_lag:
            raise ValueError(
                f"history of length {n} cannot serve lag {max_lag}; warm up the "
                f"buffer with at least max(lags) steps"
            )
        return torch.cat([history[:, n - lag, :] for lag in lags], dim=-1)
    except Exception as e:
        raise Exception(
            f"covariates - lag_vector_from_history - Execution error: {e}"
        ) from e


def build_covariates(timestamps, series: torch.Tensor, freq: str,
                     lags: list) -> torch.Tensor:
    """[C, T] = concat(time features, zero-padded lag features).

    Whole-series convenience version. The training / sampling paths do NOT use
    it (they build the lag part per window from real history); it is kept for
    diagnostics, plots and tests.
    """
    try:
        tf = time_features(timestamps, freq)
        lf = lag_features(series, lags)
        return torch.cat([lf, tf], dim=0)
    except Exception as e:
        raise Exception(f"covariates - build_covariates - Execution error: {e}") from e
