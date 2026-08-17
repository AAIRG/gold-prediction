"""Two boosting layers stacked on top of the four trained models.

STAGE 1 -- greedy forward stagewise CRPS ensemble

Classic forward stagewise additive selection (the same greedy scheme Caruana's
ensemble selection uses), with CRPS_sum as the loss. Starting from an empty
ensemble, each round adds the one model that most reduces the CRPS_sum of the
resulting mixture, with replacement, so a strong model can be picked repeatedly
and ends up with a larger weight. The output is a genuine predictive
distribution -- a weighted mixture of the four flows -- so it is scored with the
same `compute_metrics` as the individual models and the comparison is fair.

The mixture loss is evaluated in closed form rather than by materializing pooled
sample arrays. For a mixture with weights `w_m` over empirical sample sets,

    CRPS = sum_m w_m E|X_m - y| - 0.5 * sum_m sum_l w_m w_l E|X_m - X_l|

so the two expectation tables `A` and `B` are computed once and every candidate
weight vector after that costs O(M^2). `E|X - X'|` uses the full n^2 plug-in
mean, which is the identity `crps_elementwise` already relies on, so stage 1
agrees with the project's metric to floating point.

STAGE 2 -- gradient-boosted residual correction

A single `HistGradientBoostingRegressor` predicts the residual between the
mixture median and the realized return from features that are all known at
forecast time: the forecast step, the horizon, the mixture's own median and
spread, recent realized volatility and momentum, and the calendar. Its
prediction is applied as a location shift to every sample of that
(horizon, step) cell.

Stage 2 is gated. It is fitted on a dedicated boosting split of stride-1
windows that never touches the test range, then it must demonstrate a CRPS_sum
improvement on a held-out tail of that split before it is enabled. A residual
model on financial returns usually finds very little, and an honest report says
so instead of quietly shipping a shift that makes the forecast worse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

FEATURE_NAMES: tuple[str, ...] = (
    "forecast_step",
    "horizon",
    "mixture_median",
    "mixture_spread",
    "return_last_1d",
    "return_mean_5d",
    "return_std_20d",
    "dayofweek_cos",
    "dayofweek_sin",
    "dayofyear_cos",
    "dayofyear_sin",
)


@dataclass
class BoostingResult:
    """Everything the app needs to rebuild both boosting stages."""

    weights: dict
    counts: dict
    pool_size: int
    rounds: list = field(repr=False)
    ensemble_metrics: dict = field(default_factory=dict)
    booster_enabled: bool = False
    booster_gain_pct: float = 0.0
    booster_metrics: dict = field(default_factory=dict)
    booster: object = field(default=None, repr=False)

    def to_metadata(self) -> dict:
        try:
            return {
                "weights": {key: float(value) for key, value in self.weights.items()},
                "counts": {key: int(value) for key, value in self.counts.items()},
                "pool_size": int(self.pool_size),
                "rounds": list(self.rounds),
                "ensemble_metrics": {
                    key: float(value) for key, value in self.ensemble_metrics.items()
                },
                "booster_enabled": bool(self.booster_enabled),
                "booster_gain_pct": float(self.booster_gain_pct),
                "booster_metrics": {
                    key: float(value) for key, value in self.booster_metrics.items()
                },
                "feature_names": list(FEATURE_NAMES),
            }
        except Exception as e:
            raise Exception(
                f"BoostingResult - to_metadata - Execution error: {e}"
            ) from e


def _sum_over_dimensions(samples: torch.Tensor) -> torch.Tensor:
    try:
        return samples.to(torch.float64).sum(dim=2)
    except Exception as e:
        raise Exception(
            f"boosting - _sum_over_dimensions - Execution error: {e}"
        ) from e


def _expectation_tables(model_samples: dict, targets: torch.Tensor,
                        order: list) -> tuple:
    """Elementwise `E|X_m - y|` and `E|X_m - X_l|` on the sum-aggregated series."""
    try:
        target_sum = targets.to(torch.float64).sum(dim=1)               # [W, P]
        pooled = [_sum_over_dimensions(model_samples[key]) for key in order]
        for key, block in zip(order, pooled):
            if block.shape[0] != target_sum.shape[0] \
                    or block.shape[2] != target_sum.shape[1]:
                raise ValueError(
                    f"model {key!r} has shape {tuple(block.shape)} which does "
                    f"not line up with targets {tuple(target_sum.shape)}"
                )

        absolute_error = torch.stack([
            (block - target_sum.unsqueeze(1)).abs().mean(dim=1) for block in pooled
        ])                                                             # [M, W, P]

        count = len(order)
        pairwise = torch.zeros(
            (count, count, *target_sum.shape), dtype=torch.float64
        )
        for i in range(count):
            for j in range(i, count):
                value = (
                    pooled[i].unsqueeze(2) - pooled[j].unsqueeze(1)
                ).abs().mean(dim=(1, 2))
                pairwise[i, j] = value
                pairwise[j, i] = value
        return absolute_error, pairwise, target_sum
    except Exception as e:
        raise Exception(
            f"boosting - _expectation_tables - Execution error: {e}"
        ) from e


def _mixture_crps_sum(absolute_error: torch.Tensor, pairwise: torch.Tensor,
                      target_sum: torch.Tensor, weights: torch.Tensor) -> float:
    try:
        term1 = torch.einsum("m,mwp->wp", weights, absolute_error)
        term2 = torch.einsum("m,l,mlwp->wp", weights, weights, pairwise)
        elementwise = term1 - 0.5 * term2
        denominator = float(target_sum.abs().sum().item())
        if denominator <= 0:
            return float(elementwise.mean().item())
        return float(elementwise.sum().item()) / denominator
    except Exception as e:
        raise Exception(
            f"boosting - _mixture_crps_sum - Execution error: {e}"
        ) from e


def fit_greedy_crps_ensemble(model_samples: dict, targets: torch.Tensor,
                             rounds: int = 24) -> BoostingResult:
    """Forward stagewise selection of a CRPS_sum-minimizing mixture."""
    try:
        order = list(model_samples.keys())
        if not order:
            raise ValueError("no model samples were supplied")
        rounds = int(rounds)
        if rounds < 1:
            raise ValueError(f"rounds must be at least 1, got {rounds}")

        absolute_error, pairwise, target_sum = _expectation_tables(
            model_samples, targets, order
        )
        counts = {key: 0 for key in order}
        trace = []
        for step in range(rounds):
            best_key, best_score = None, float("inf")
            for index, key in enumerate(order):
                candidate = dict(counts)
                candidate[key] += 1
                total = float(sum(candidate.values()))
                weights = torch.tensor(
                    [candidate[name] / total for name in order], dtype=torch.float64
                )
                score = _mixture_crps_sum(
                    absolute_error, pairwise, target_sum, weights
                )
                if score < best_score - 1e-15:
                    best_key, best_score = key, score
            if best_key is None:
                break
            counts[best_key] += 1
            trace.append({
                "round": step + 1,
                "picked": best_key,
                "crps_sum": best_score,
            })

        total = float(sum(counts.values()))
        weights = {key: counts[key] / total for key in order}
        return BoostingResult(
            weights=weights, counts=counts, pool_size=rounds, rounds=trace,
            ensemble_metrics={"crps_sum_energy_validation": trace[-1]["crps_sum"]}
            if trace else {},
        )
    except Exception as e:
        raise Exception(
            f"boosting - fit_greedy_crps_ensemble - Execution error: {e}"
        ) from e


def mix_samples(model_samples: dict, weights: dict,
                pool_size: int = 24) -> torch.Tensor:
    """Pool per-model sample blocks into one mixture sample block.

    Accepts either [S, D, P] or [W, S, D, P] and returns the same layout with a
    larger sample axis. Counts are the weights rounded to `pool_size` slots,
    which makes the mixture reproducible and independent of sample ordering.
    """
    try:
        order = [key for key in weights if float(weights[key]) > 0]
        if not order:
            raise ValueError("every ensemble weight is zero")
        missing = [key for key in order if key not in model_samples]
        if missing:
            raise ValueError(f"no samples were supplied for {missing}")

        raw = np.array([float(weights[key]) for key in order])
        counts = np.maximum(1, np.round(raw / raw.sum() * int(pool_size)).astype(int))

        blocks = []
        for key, count in zip(order, counts):
            block = model_samples[key]
            tensor = block if isinstance(block, torch.Tensor) \
                else torch.as_tensor(block)
            if tensor.dim() not in (3, 4):
                raise ValueError(
                    f"model {key!r} samples must be [S, D, P] or [W, S, D, P], "
                    f"got {tuple(tensor.shape)}"
                )
            blocks.extend([tensor.to(torch.float64)] * int(count))
        sample_axis = 1 if blocks[0].dim() == 4 else 0
        return torch.cat(blocks, dim=sample_axis)
    except Exception as e:
        raise Exception(f"boosting - mix_samples - Execution error: {e}") from e


def residual_features(forecast_step: int, horizon: int, median: float,
                      spread: float, daily_history: np.ndarray,
                      target_date: pd.Timestamp) -> list:
    """Feature row for the residual booster. Every entry is known at forecast time."""
    try:
        history = np.asarray(daily_history, dtype="float64").reshape(-1)
        if history.size < 20:
            raise ValueError(
                f"the residual booster needs 20 daily returns of context, "
                f"got {history.size}"
            )
        date = pd.Timestamp(target_date)
        dayofweek = 2.0 * np.pi * date.dayofweek / 7.0
        dayofyear = 2.0 * np.pi * date.dayofyear / 366.0
        return [
            float(forecast_step),
            float(horizon),
            float(median),
            float(spread),
            float(history[-1]),
            float(history[-5:].mean()),
            float(history[-20:].std(ddof=1)),
            float(np.cos(dayofweek)),
            float(np.sin(dayofweek)),
            float(np.cos(dayofyear)),
            float(np.sin(dayofyear)),
        ]
    except Exception as e:
        raise Exception(f"boosting - residual_features - Execution error: {e}") from e


def build_residual_dataset(mixture: torch.Tensor, targets: torch.Tensor,
                           series: torch.Tensor, timestamps: pd.DatetimeIndex,
                           forecast_starts: list, horizons: list,
                           daily_index: int = 0) -> tuple:
    """Feature matrix and residual targets over a set of rolling windows."""
    try:
        samples = mixture.detach().cpu().numpy()
        actual = targets.detach().cpu().numpy()
        daily = series[daily_index].detach().cpu().numpy()
        if samples.ndim != 4:
            raise ValueError(f"mixture must be [W, S, D, P], got {samples.shape}")
        if len(forecast_starts) != samples.shape[0]:
            raise ValueError(
                f"{len(forecast_starts)} window starts for "
                f"{samples.shape[0]} sample windows"
            )

        features, residuals = [], []
        for window, forecast_start in enumerate(forecast_starts):
            history = daily[:forecast_start]
            for dimension, horizon in enumerate(horizons):
                block = samples[window, :, dimension, :]
                median = np.median(block, axis=0)
                spread = np.subtract(*np.quantile(block, [0.75, 0.25], axis=0))
                for step in range(block.shape[1]):
                    index = forecast_start + step
                    if index >= len(timestamps):
                        continue
                    features.append(residual_features(
                        step + 1, int(horizon), float(median[step]),
                        float(spread[step]), history, timestamps[index],
                    ))
                    residuals.append(
                        float(actual[window, dimension, step] - median[step])
                    )
        if not features:
            raise ValueError("the residual dataset came out empty")
        return np.asarray(features, dtype="float64"), np.asarray(residuals, dtype="float64")
    except Exception as e:
        raise Exception(
            f"boosting - build_residual_dataset - Execution error: {e}"
        ) from e


class GradientResidualBooster:
    """Gradient-boosted location correction for the mixture median."""

    def __init__(self, max_iter: int = 200, learning_rate: float = 0.05,
                 max_depth: int = 3, min_samples_leaf: int = 40,
                 l2_regularization: float = 1.0, seed: int = 42):
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor

            self.model = HistGradientBoostingRegressor(
                max_iter=int(max_iter), learning_rate=float(learning_rate),
                max_depth=int(max_depth), min_samples_leaf=int(min_samples_leaf),
                l2_regularization=float(l2_regularization),
                early_stopping=True, validation_fraction=0.2,
                random_state=int(seed),
            )
            self.fitted = False
        except ImportError as e:
            raise Exception(
                "boosting - GradientResidualBooster - __init__ - scikit-learn is "
                f"required for the gradient boosting stage: {e}"
            ) from e
        except Exception as e:
            raise Exception(
                f"GradientResidualBooster - __init__ - Init error: {e}"
            ) from e

    def fit(self, features: np.ndarray, residuals: np.ndarray) -> "GradientResidualBooster":
        try:
            if features.shape[0] != residuals.shape[0]:
                raise ValueError(
                    f"{features.shape[0]} feature rows against "
                    f"{residuals.shape[0]} residuals"
                )
            if features.shape[1] != len(FEATURE_NAMES):
                raise ValueError(
                    f"expected {len(FEATURE_NAMES)} features, got {features.shape[1]}"
                )
            self.model.fit(features, residuals)
            self.fitted = True
            return self
        except Exception as e:
            raise Exception(
                f"GradientResidualBooster - fit - Execution error: {e}"
            ) from e

    def predict(self, features: np.ndarray) -> np.ndarray:
        try:
            if not self.fitted:
                raise ValueError("the booster has not been fitted")
            return np.asarray(self.model.predict(features), dtype="float64")
        except Exception as e:
            raise Exception(
                f"GradientResidualBooster - predict - Execution error: {e}"
            ) from e

    def shift(self, mixture: torch.Tensor, series_or_history,
              timestamps: pd.DatetimeIndex, forecast_starts: list,
              horizons: list, daily_index: int = 0) -> torch.Tensor:
        """Apply the predicted residual as a location shift to every sample."""
        try:
            samples = mixture.detach().cpu().numpy().copy()
            if samples.ndim != 4:
                raise ValueError(f"mixture must be [W, S, D, P], got {samples.shape}")
            daily = series_or_history[daily_index].detach().cpu().numpy() \
                if isinstance(series_or_history, torch.Tensor) \
                else np.asarray(series_or_history, dtype="float64")

            for window, forecast_start in enumerate(forecast_starts):
                history = daily[:forecast_start]
                for dimension, horizon in enumerate(horizons):
                    block = samples[window, :, dimension, :]
                    median = np.median(block, axis=0)
                    spread = np.subtract(*np.quantile(block, [0.75, 0.25], axis=0))
                    rows = []
                    valid = []
                    for step in range(block.shape[1]):
                        index = forecast_start + step
                        if index >= len(timestamps):
                            continue
                        rows.append(residual_features(
                            step + 1, int(horizon), float(median[step]),
                            float(spread[step]), history, timestamps[index],
                        ))
                        valid.append(step)
                    if not rows:
                        continue
                    correction = self.predict(np.asarray(rows, dtype="float64"))
                    for offset, step in enumerate(valid):
                        samples[window, :, dimension, step] += correction[offset]
            return torch.from_numpy(samples)
        except Exception as e:
            raise Exception(
                f"GradientResidualBooster - shift - Execution error: {e}"
            ) from e

    def shift_live(self, samples: np.ndarray, daily_history: np.ndarray,
                   future_dates: pd.DatetimeIndex, horizons: list) -> np.ndarray:
        """Location shift for a single live forecast block [S, D, P].

        The app has no window index and no series array, only the observed daily
        returns and the future business dates, so this is the same correction as
        `shift` with the bookkeeping removed. Keeping both on one class is what
        guarantees the app applies exactly the transform the notebook gated.
        """
        try:
            block = np.asarray(samples, dtype="float64").copy()
            if block.ndim != 3:
                raise ValueError(f"samples must be [S, D, P], got {block.shape}")
            if block.shape[1] != len(horizons):
                raise ValueError(
                    f"{block.shape[1]} dimensions for {len(horizons)} horizons"
                )
            if len(future_dates) < block.shape[2]:
                raise ValueError(
                    f"{len(future_dates)} future dates for "
                    f"{block.shape[2]} forecast days"
                )

            for dimension, horizon in enumerate(horizons):
                cell = block[:, dimension, :]
                median = np.median(cell, axis=0)
                spread = np.subtract(*np.quantile(cell, [0.75, 0.25], axis=0))
                rows = [
                    residual_features(
                        step + 1, int(horizon), float(median[step]),
                        float(spread[step]), daily_history, future_dates[step],
                    )
                    for step in range(cell.shape[1])
                ]
                correction = self.predict(np.asarray(rows, dtype="float64"))
                block[:, dimension, :] += correction[None, :]
            return block
        except Exception as e:
            raise Exception(
                f"GradientResidualBooster - shift_live - Execution error: {e}"
            ) from e


def gate_residual_booster(booster: GradientResidualBooster, mixture: torch.Tensor,
                          targets: torch.Tensor, series: torch.Tensor,
                          timestamps: pd.DatetimeIndex, forecast_starts: list,
                          horizons: list, min_gain_pct: float = 1.0) -> dict:
    """Only keep stage 2 if it clearly lowers CRPS_sum on data it never saw.

    `min_gain_pct` exists because a residual model on financial returns will
    often land a fraction of a percent either side of zero, which is noise. A
    shift is only worth shipping if it earns more than that.
    """
    try:
        from .experiments import evaluate_samples, summarize_windows

        before = summarize_windows(evaluate_samples(mixture, targets))
        shifted = booster.shift(
            mixture, series, timestamps, forecast_starts, horizons
        )
        after = summarize_windows(evaluate_samples(shifted, targets))
        baseline = float(before["crps_sum"]["mean"])
        corrected = float(after["crps_sum"]["mean"])
        gain = 100.0 * (1.0 - corrected / baseline) if baseline > 0 else 0.0
        return {
            "enabled": bool(gain > float(min_gain_pct)),
            "gain_pct": gain,
            "min_gain_pct": float(min_gain_pct),
            "crps_sum_before": baseline,
            "crps_sum_after": corrected,
            "crps_before": float(before["crps"]["mean"]),
            "crps_after": float(after["crps"]["mean"]),
            "mse_before": float(before["mse"]["mean"]),
            "mse_after": float(after["mse"]["mean"]),
        }
    except Exception as e:
        raise Exception(
            f"boosting - gate_residual_booster - Execution error: {e}"
        ) from e
