"""Sampling, price-path reconstruction, and horizon returns you can trade on.

The model emits a joint sample of `[return_1d, return_3d, return_7d]` for every
forecast day, but those three numbers are free parameters of the flow: nothing in
the architecture forces `return_3d` at forecast day 1 to agree with the daily
returns the same sample drew, nor with the two closes that have already been
observed. Reading them straight out produced tables that contradicted their own
price fan.

`forecast_prices` therefore does what a desk would do:

    1. take the daily dimension of the sample, `r_1 .. r_P`,
    2. compound it onto the last observed close to get a price path,
       `P_k = close(t) * exp(r_1 + ... + r_k)`,
    3. splice that path onto the observed closes and read every horizon off the
       spliced series,
           return_h at forecast day k = log(P(t+k) / P(t+k-h)),
       where any index at or before `t` is a real observed close.

Step 3 is the fix. For `k < h` part of the window is history, so those days enter
as known constants instead of being re-imagined by the flow: at forecast day 1
the 3-day return is `r_1` plus two closes that are already on the tape. Every
horizon window still ends on -- and includes -- its own forecast day, the numbers
are mutually consistent by construction, and `P(return > 0)` refers to a
quantity that actually settles.

The raw model dimensions are kept in `ForecastResult.model_returns` and compared
against the derived ones by `consistency_report`, because their disagreement is
a genuine calibration diagnostic for how well the flow learned the multi-scale
structure.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from .features import stable_time_features

QUANTILES: tuple[float, ...] = (0.025, 0.10, 0.50, 0.90, 0.975)


@dataclass
class ForecastResult:
    """One model's forecast, in both raw and reconstructed form."""

    model_key: str
    model_name: str
    future_dates: pd.DatetimeIndex
    last_close: float
    horizons: tuple[int, ...]
    price_paths: np.ndarray                     # [S, P]
    model_returns: np.ndarray                   # [S, D, P] raw flow output
    horizon_returns: dict = field(repr=False)   # h -> [S, P] reconstructed
    observed_fraction: dict = field(repr=False)  # h -> [P] share already on the tape

    @property
    def num_samples(self) -> int:
        return int(self.price_paths.shape[0])

    @property
    def prediction_length(self) -> int:
        return int(self.price_paths.shape[1])


def sample_return_paths(model, returns: pd.DataFrame, lags: list,
                        context_length: int, prediction_length: int,
                        num_samples: int, device: str = "cpu",
                        seed: int | None = None) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Autoregressive joint samples in original units -> [S, D, P]."""
    try:
        from flowforecast.data.scaling import compute_scale, invert_scale
        from flowforecast.evaluation.sampling import sample_trajectories

        lags = [int(lag) for lag in lags]
        context_length = int(context_length)
        prediction_length = int(prediction_length)
        max_lag = max(lags)
        required = context_length + max_lag
        if len(returns) < required:
            raise ValueError(
                f"need {required} aligned return rows for context "
                f"{context_length} plus max lag {max_lag}, got {len(returns)}"
            )

        history_frame = returns.iloc[-required:]
        history = torch.from_numpy(
            np.ascontiguousarray(history_frame.to_numpy(dtype="float32"))
        ).unsqueeze(0)

        future_dates = pd.bdate_range(
            history_frame.index[-1] + pd.offsets.BDay(1),
            periods=prediction_length,
        )
        all_dates = pd.DatetimeIndex(
            list(history_frame.index) + list(future_dates)
        )
        features = stable_time_features(all_dates).t().unsqueeze(0)

        scale = compute_scale(history[0, max_lag:, :], time_dim=0)
        if seed is not None:
            random.seed(int(seed))
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        sampled_scaled = sample_trajectories(
            model, history.to(device), features.to(device), lags,
            prediction_length, num_samples=int(num_samples),
            context_length=context_length, scale=scale,
        )
        sampled = invert_scale(sampled_scaled.cpu(), scale, time_dim=-1)
        return future_dates, sampled.detach().numpy().astype("float64")
    except Exception as e:
        raise Exception(f"forecast - sample_return_paths - Execution error: {e}") from e


def reconstruct_price_paths(last_close: float, daily_returns: np.ndarray) -> np.ndarray:
    """Compound the daily log-return dimension onto the last observed close."""
    try:
        if daily_returns.ndim != 2:
            raise ValueError(
                f"daily_returns must be [S, P], got {daily_returns.shape}"
            )
        if not np.isfinite(last_close) or last_close <= 0:
            raise ValueError(f"last_close must be positive, got {last_close}")
        return float(last_close) * np.exp(np.cumsum(daily_returns, axis=1))
    except Exception as e:
        raise Exception(
            f"forecast - reconstruct_price_paths - Execution error: {e}"
        ) from e


def derive_horizon_returns(observed_close: np.ndarray, price_paths: np.ndarray,
                           horizons) -> dict:
    """Trailing h-day log returns read off observed closes spliced to the path.

    `observed_close` must end with the close the path starts from. The returned
    array for horizon `h` is [S, P]; column `k - 1` is the h-day return whose
    window ends on forecast day `k` and therefore contains forecast day `k`.
    """
    try:
        horizons = tuple(int(horizon) for horizon in horizons)
        if price_paths.ndim != 2:
            raise ValueError(f"price_paths must be [S, P], got {price_paths.shape}")
        closes = np.asarray(observed_close, dtype="float64").reshape(-1)
        max_horizon = max(horizons)
        if closes.size < max_horizon:
            raise ValueError(
                f"need {max_horizon} observed closes to close the longest "
                f"horizon window, got {closes.size}"
            )
        if (closes[-max_horizon:] <= 0).any():
            raise ValueError("observed closes contain non-positive values")

        num_samples, prediction_length = price_paths.shape
        tail = np.broadcast_to(closes[-max_horizon:], (num_samples, max_horizon))
        extended = np.concatenate([tail, price_paths], axis=1)

        derived = {}
        for horizon in horizons:
            start = max_horizon - horizon
            numerator = extended[:, max_horizon:max_horizon + prediction_length]
            denominator = extended[:, start:start + prediction_length]
            derived[horizon] = np.log(numerator / denominator)
        return derived
    except Exception as e:
        raise Exception(
            f"forecast - derive_horizon_returns - Execution error: {e}"
        ) from e


def observed_fractions(horizons, prediction_length: int) -> dict:
    """Share of each horizon window that is already observed, per forecast day.

    The window ending on forecast day `k` spans the `h` daily returns of days
    `t + k - h + 1 .. t + k`, of which `max(0, h - k)` are already on the tape.
    """
    try:
        steps = np.arange(1, int(prediction_length) + 1)
        return {
            int(horizon): np.maximum(0, int(horizon) - steps) / float(horizon)
            for horizon in horizons
        }
    except Exception as e:
        raise Exception(f"forecast - observed_fractions - Execution error: {e}") from e


def forecast_prices(model, model_key: str, model_name: str,
                    prices: pd.DataFrame, returns: pd.DataFrame,
                    horizons, lags: list, context_length: int,
                    prediction_length: int, num_samples: int,
                    device: str = "cpu",
                    seed: int | None = None) -> ForecastResult:
    """Full forecast for one model: samples, price path, derived horizons."""
    try:
        horizons = tuple(int(horizon) for horizon in horizons)
        if 1 not in horizons:
            raise ValueError("horizon 1 is required to reconstruct the price path")

        future_dates, model_returns = sample_return_paths(
            model, returns, lags, context_length, prediction_length,
            num_samples, device=device, seed=seed,
        )
        if model_returns.shape[1] != len(horizons):
            raise ValueError(
                f"model returned {model_returns.shape[1]} dimensions for "
                f"{len(horizons)} horizons"
            )

        close = prices["Close"].astype(float)
        aligned = close.loc[close.index <= returns.index[-1]]
        if aligned.empty:
            raise ValueError("no observed close is aligned with the return matrix")

        daily_index = horizons.index(1)
        price_paths = reconstruct_price_paths(
            float(aligned.iloc[-1]), model_returns[:, daily_index, :]
        )
        horizon_returns = derive_horizon_returns(
            aligned.to_numpy(), price_paths, horizons
        )
        return ForecastResult(
            model_key=str(model_key),
            model_name=str(model_name),
            future_dates=future_dates,
            last_close=float(aligned.iloc[-1]),
            horizons=horizons,
            price_paths=price_paths,
            model_returns=model_returns,
            horizon_returns=horizon_returns,
            observed_fraction=observed_fractions(horizons, prediction_length),
        )
    except Exception as e:
        raise Exception(f"forecast - forecast_prices - Execution error: {e}") from e


def ensemble_forecast(results: dict, weights: dict, pool_size: int,
                      observed_close: np.ndarray, model_key: str = "boosted",
                      model_name: str = "Boosted ensemble",
                      booster=None,
                      daily_history: np.ndarray | None = None) -> ForecastResult:
    """Pool per-model samples into one mixture forecast, then reconstruct.

    Pooling happens on the raw joint samples, so the mixture is a real
    predictive distribution rather than an average of summary statistics: the
    price path and every horizon are rebuilt from the pooled draws by the same
    code path the individual models use. When `booster` is supplied its residual
    correction is applied to the pooled draws before reconstruction, so stage 2
    shifts the distribution the app actually reports.
    """
    try:
        from .boosting import mix_samples

        if not results:
            raise ValueError("no model forecasts were supplied")
        reference = next(iter(results.values()))
        horizons = reference.horizons
        for key, result in results.items():
            if result.horizons != horizons:
                raise ValueError(
                    f"model {key!r} forecasts horizons {result.horizons}, "
                    f"expected {horizons}"
                )
            if result.prediction_length != reference.prediction_length:
                raise ValueError(f"model {key!r} has a different horizon length")

        blocks = {key: result.model_returns for key, result in results.items()}
        pooled = mix_samples(blocks, weights, pool_size).numpy()

        if booster is not None:
            if daily_history is None:
                raise ValueError(
                    "the residual booster needs the observed daily returns"
                )
            pooled = booster.shift_live(
                pooled, daily_history, reference.future_dates, list(horizons)
            )

        daily_index = horizons.index(1)
        price_paths = reconstruct_price_paths(
            reference.last_close, pooled[:, daily_index, :]
        )
        return ForecastResult(
            model_key=str(model_key),
            model_name=str(model_name),
            future_dates=reference.future_dates,
            last_close=reference.last_close,
            horizons=horizons,
            price_paths=price_paths,
            model_returns=pooled,
            horizon_returns=derive_horizon_returns(
                observed_close, price_paths, horizons
            ),
            observed_fraction=reference.observed_fraction,
        )
    except Exception as e:
        raise Exception(f"forecast - ensemble_forecast - Execution error: {e}") from e


def consistency_report(result: ForecastResult) -> pd.DataFrame:
    """How far the flow's free dimensions drift from the reconstructed truth."""
    try:
        rows = []
        for index, horizon in enumerate(result.horizons):
            raw = result.model_returns[:, index, :]
            derived = result.horizon_returns[horizon]
            for step in range(result.prediction_length):
                rows.append({
                    "Forecast day": step + 1,
                    "Return horizon": f"{horizon}-day",
                    "Observed share of window": float(
                        result.observed_fraction[horizon][step]
                    ),
                    "Model median (%)": 100.0 * float(np.median(raw[:, step])),
                    "Reconstructed median (%)": 100.0 * float(
                        np.median(derived[:, step])
                    ),
                    "Median gap (pp)": 100.0 * float(
                        np.median(derived[:, step]) - np.median(raw[:, step])
                    ),
                    "Model sigma (%)": 100.0 * float(raw[:, step].std(ddof=1)),
                    "Reconstructed sigma (%)": 100.0 * float(
                        derived[:, step].std(ddof=1)
                    ),
                })
        return pd.DataFrame(rows)
    except Exception as e:
        raise Exception(f"forecast - consistency_report - Execution error: {e}") from e


def quantile_frame(dates: pd.DatetimeIndex, paths: np.ndarray,
                   scale: float = 1.0) -> pd.DataFrame:
    """Quantile bands of a [S, P] sample block, one row per forecast day."""
    try:
        low95, low80, median, high80, high95 = np.quantile(paths, QUANTILES, axis=0)
        return pd.DataFrame({
            "Date": [date.date() for date in dates],
            "Forecast day": np.arange(1, paths.shape[1] + 1),
            "95% low": scale * low95,
            "80% low": scale * low80,
            "Median": scale * median,
            "80% high": scale * high80,
            "95% high": scale * high95,
        })
    except Exception as e:
        raise Exception(f"forecast - quantile_frame - Execution error: {e}") from e


def probability_frame(result: ForecastResult, threshold_pct: float,
                      use_reconstructed: bool = True) -> pd.DataFrame:
    """Return probabilities per horizon and forecast day, in percent."""
    try:
        threshold = float(threshold_pct) / 100.0
        rows = []
        for index, horizon in enumerate(result.horizons):
            paths = result.horizon_returns[horizon] if use_reconstructed \
                else result.model_returns[:, index, :]
            quantiles = np.quantile(paths, QUANTILES, axis=0)
            for step, date in enumerate(result.future_dates):
                draws = paths[:, step]
                rows.append({
                    "Date": date.date(),
                    "Forecast day": step + 1,
                    "Return horizon": f"{horizon}-day",
                    "Observed share": 100.0 * float(
                        result.observed_fraction[horizon][step]
                    ),
                    "Median": 100.0 * float(quantiles[2, step]),
                    "80% low": 100.0 * float(quantiles[1, step]),
                    "80% high": 100.0 * float(quantiles[3, step]),
                    "95% low": 100.0 * float(quantiles[0, step]),
                    "95% high": 100.0 * float(quantiles[4, step]),
                    "P(return > 0)": 100.0 * float((draws > 0).mean()),
                    "P(return < 0)": 100.0 * float((draws < 0).mean()),
                    f"P(return >= {float(threshold_pct):g}%)": 100.0 * float(
                        (draws >= threshold).mean()
                    ),
                })
        return pd.DataFrame(rows)
    except Exception as e:
        raise Exception(f"forecast - probability_frame - Execution error: {e}") from e
