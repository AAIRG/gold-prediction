"""The paper's §5.2 experiment protocol, run on gold.

Reproduces for `GC=F` the outputs the paper's experiments section produces:

    Table 1   test set CRPS_sum   (§5.2, the headline metric)
    Table 4   test set CRPS       (Appendix B.2, marginal CRPS over D and horizon)
    Table 5   test set MSE        (Appendix B.3, marginal MSE of the sample mean)

Orientation and aggregation follow the paper exactly. Columns are models, rows
are the data set; because gold is a single data set the tables also break the
rows down by return horizon, which is the same "average over the D dimensions"
the appendix defines, just reported per dimension as well as pooled. Entries are
`mean±stderr` over independent re-runs -- the paper re-runs each method three
times for Tables 4 and 5 (`SEEDS` in the notebook defaults to the same three).

Two aggregation details that are easy to get wrong and are deliberately copied
from `scripts/evaluate.py`:

    * CRPS_sum and CRPS are GluonTS `mean_wQuantileLoss` values, i.e. a ratio of
      GLOBAL sums over every forecast point, not the arithmetic mean of
      per-window ratios. Averaging already-normalized ratios lets one
      low-activity window dominate.
    * the mean-scaler fallback is pooled across the evaluation windows before
      any window is scored, so a window whose context is degenerate borrows the
      pooled scale instead of 1e-10.

`num_samples` defaults to 100 because §5.2 says "We take 100 samples to estimate
the empirical CDF in practice".
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

WEIGHTED_METRICS = {
    "crps_sum": ("_crps_sum_numerator", "_crps_sum_denominator"),
    "crps_sum_energy": ("_crps_sum_energy_numerator", "_crps_sum_denominator"),
    "crps": ("_crps_numerator", "_crps_denominator"),
    "crps_energy": ("_crps_energy_numerator", "_crps_denominator"),
    "nd": ("_nd_numerator", "_crps_denominator"),
}

REPORTED_METRICS = ("crps_sum", "crps", "mse", "nd")

# metric -> (caption, print style, break the rows down per return horizon).
# CRPS_sum is defined on the sum across the D dimensions, so it has no
# per-dimension counterpart and stays a single row; CRPS and MSE are marginal
# metrics that the appendix averages over D, so reporting them per horizon as
# well is the same quantity at finer granularity.
PAPER_TABLES = {
    "table1_crps_sum": ("crps_sum", "Test set CRPS_sum (lower is better)",
                        "decimal", False),
    "table4_crps": ("crps", "Test set CRPS (lower is better)",
                    "decimal", True),
    "table5_mse": ("mse", "Test set MSE (lower is better)",
                   "scientific", True),
}


@dataclass
class SeedSummary:
    """One model, one seed, one full pass over the rolling test windows."""

    model_key: str
    model_name: str
    seed: int
    metrics: dict
    per_horizon: dict
    per_window: list


def pooled_default_scale(series: torch.Tensor, forecast_starts: list,
                         context_length: int) -> torch.Tensor:
    """Mean-scaler fallback pooled over every evaluation window's context."""
    try:
        numerator = torch.zeros(series.shape[0], dtype=series.dtype)
        denominator = torch.zeros_like(numerator)
        for forecast_start in forecast_starts:
            context = series[:, forecast_start - context_length:forecast_start]
            if context.shape[1] != context_length:
                raise ValueError(
                    f"window at {forecast_start} has {context.shape[1]} context "
                    f"steps, expected {context_length}"
                )
            numerator += context.abs().sum(dim=-1)
            denominator += torch.full_like(numerator, float(context_length))
        return numerator / denominator.clamp_min(1.0)
    except Exception as e:
        raise Exception(
            f"experiments - pooled_default_scale - Execution error: {e}"
        ) from e


def window_starts(end_index: int, num_windows: int,
                  prediction_length: int) -> list:
    """Forecast starts of consecutive non-overlapping windows ending at `end_index`."""
    try:
        end_index = int(end_index)
        num_windows = int(num_windows)
        prediction_length = int(prediction_length)
        first = end_index - num_windows * prediction_length
        if first <= 0:
            raise ValueError(
                f"{num_windows} windows of {prediction_length} do not fit before "
                f"index {end_index}"
            )
        return [first + i * prediction_length for i in range(num_windows)]
    except Exception as e:
        raise Exception(f"experiments - window_starts - Execution error: {e}") from e


def rolling_window_samples(model, series: torch.Tensor, time_feat: torch.Tensor,
                           lags: list, context_length: int,
                           prediction_length: int, num_windows: int,
                           end_index: int, num_samples: int = 100,
                           device: str = "cpu", seed: int = 1,
                           stride: int | None = None) -> dict:
    """Sample every rolling window once. Returns samples, targets and starts.

    `samples` is [W, S, D, P] in original units and `targets` is [W, D, P], so
    downstream scoring, per-horizon breakdowns and the boosting layers all read
    the same arrays instead of re-sampling.
    """
    try:
        from flowforecast.data.scaling import compute_scale, invert_scale
        from flowforecast.evaluation.sampling import sample_trajectories

        lags = [int(lag) for lag in lags]
        max_lag = max(lags)
        context_length = int(context_length)
        prediction_length = int(prediction_length)
        step = int(stride) if stride else prediction_length

        first = int(end_index) - (int(num_windows) - 1) * step - prediction_length
        starts = [first + i * step for i in range(int(num_windows))]
        if starts[0] - context_length - max_lag < 0:
            raise ValueError(
                f"the first window at {starts[0]} lacks "
                f"{context_length + max_lag} steps of history"
            )
        if starts[-1] + prediction_length > series.shape[1]:
            raise ValueError(
                f"the last window ends at {starts[-1] + prediction_length} "
                f"beyond the series length {series.shape[1]}"
            )

        default_scale = pooled_default_scale(series, starts, context_length)

        random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        all_samples, all_targets = [], []
        for forecast_start in starts:
            hist_start = forecast_start - context_length - max_lag
            window_end = forecast_start + prediction_length
            history = series[:, hist_start:forecast_start]
            target = series[:, forecast_start:window_end]
            scale = compute_scale(
                history[:, max_lag:], time_dim=-1, default_scale=default_scale
            )
            features = time_feat[:, hist_start:window_end].t().unsqueeze(0)
            with torch.no_grad():
                scaled = sample_trajectories(
                    model, history.t().unsqueeze(0).to(device),
                    features.to(device), lags, prediction_length,
                    num_samples=int(num_samples), context_length=context_length,
                    scale=scale,
                )
            all_samples.append(invert_scale(scaled.cpu(), scale, time_dim=-1))
            all_targets.append(target)

        return {
            "samples": torch.stack(all_samples),
            "targets": torch.stack(all_targets),
            "forecast_starts": starts,
        }
    except Exception as e:
        raise Exception(
            f"experiments - rolling_window_samples - Execution error: {e}"
        ) from e


def evaluate_samples(samples: torch.Tensor, targets: torch.Tensor) -> list:
    """Per-window metrics for [W, S, D, P] samples against [W, D, P] targets."""
    try:
        from flowforecast.evaluation.metrics import compute_metrics

        if samples.dim() != 4:
            raise ValueError(f"samples must be [W, S, D, P], got {tuple(samples.shape)}")
        if targets.dim() != 3:
            raise ValueError(f"targets must be [W, D, P], got {tuple(targets.shape)}")
        if samples.shape[0] != targets.shape[0]:
            raise ValueError(
                f"{samples.shape[0]} sample windows against "
                f"{targets.shape[0]} target windows"
            )

        per_window = []
        for index in range(samples.shape[0]):
            metrics = compute_metrics(samples[index], targets[index])
            metrics["window"] = index
            per_window.append(metrics)
        return per_window
    except Exception as e:
        raise Exception(f"experiments - evaluate_samples - Execution error: {e}") from e


def summarize_windows(per_window: list) -> dict:
    """Aggregate per-window metrics with the paper's ratio-of-global-sums rule."""
    try:
        if not per_window:
            raise ValueError("no windows were scored")
        keys = [key for key, value in per_window[0].items()
                if isinstance(value, float)]
        summary = {}
        for key in keys:
            values = [float(window[key]) for window in per_window]
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            summary[key] = {
                "mean": statistics.fmean(values),
                "std": std,
                "stderr": std / math.sqrt(len(values)),
                "values": values,
            }

        for key, (num_key, den_key) in WEIGHTED_METRICS.items():
            if not all(num_key in w and den_key in w for w in per_window):
                continue
            numerator = sum(float(w[num_key]) for w in per_window)
            denominator = sum(float(w[den_key]) for w in per_window)
            if denominator > 0:
                summary[key]["mean"] = numerator / denominator
                summary[key]["aggregation"] = "ratio_of_global_sums"
        return summary
    except Exception as e:
        raise Exception(f"experiments - summarize_windows - Execution error: {e}") from e


def per_dimension_metrics(samples: torch.Tensor, targets: torch.Tensor,
                          horizons) -> dict:
    """Appendix B.2 / B.3 metrics for each return horizon separately.

    CRPS keeps the GluonTS normalization (pinball loss summed over all forecast
    points of that dimension, divided by the summed absolute target), and MSE is
    the squared error of the sample mean, both pooled over every window.
    """
    try:
        from flowforecast.evaluation.metrics import (crps_elementwise,
                                                     quantile_loss_parts)

        horizons = [int(horizon) for horizon in horizons]
        if samples.shape[2] != len(horizons):
            raise ValueError(
                f"samples carry {samples.shape[2]} dimensions for "
                f"{len(horizons)} horizons"
            )

        results = {}
        for index, horizon in enumerate(horizons):
            crps_numerator = 0.0
            crps_denominator = 0.0
            energy_numerator = 0.0
            squared_error = 0.0
            absolute_error = 0.0
            count = 0
            for window in range(samples.shape[0]):
                window_samples = samples[window, :, index:index + 1, :]
                window_target = targets[window, index:index + 1, :]
                numerator, denominator = quantile_loss_parts(
                    window_samples, window_target
                )
                crps_numerator += numerator
                crps_denominator += denominator
                energy_numerator += float(
                    crps_elementwise(window_samples, window_target).sum().item()
                )
                mean_prediction = window_samples.mean(dim=0)
                squared_error += float(
                    (mean_prediction - window_target).square().sum().item()
                )
                absolute_error += float(
                    (window_samples.median(dim=0).values - window_target)
                    .abs().sum().item()
                )
                count += int(window_target.numel())

            results[horizon] = {
                "crps": crps_numerator / crps_denominator
                if crps_denominator > 0 else float("nan"),
                "crps_energy": energy_numerator / crps_denominator
                if crps_denominator > 0 else float("nan"),
                "mse": squared_error / count,
                "nd": absolute_error / crps_denominator
                if crps_denominator > 0 else float("nan"),
            }
        return results
    except Exception as e:
        raise Exception(
            f"experiments - per_dimension_metrics - Execution error: {e}"
        ) from e


def summarize_seeds(summaries: list, metric: str) -> dict:
    """`mean±stderr` of one metric across independent re-runs, per model."""
    try:
        grouped: dict = {}
        for summary in summaries:
            grouped.setdefault(
                (summary.model_key, summary.model_name), []
            ).append(float(summary.metrics[metric]["mean"]))

        result = {}
        for (key, name), values in grouped.items():
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            result[key] = {
                "model_name": name,
                "mean": statistics.fmean(values),
                "std": std,
                "stderr": std / math.sqrt(len(values)),
                "runs": len(values),
                "values": values,
            }
        return result
    except Exception as e:
        raise Exception(f"experiments - summarize_seeds - Execution error: {e}") from e


def format_entry(mean: float, stderr: float, style: str = "decimal") -> str:
    """`0.0064±0.003` for scores, `2.4x10^-4` for MSE, as the paper prints them."""
    try:
        if not math.isfinite(mean):
            return "-"
        if style == "scientific":
            if mean == 0:
                return "0"
            exponent = int(math.floor(math.log10(abs(mean))))
            mantissa = mean / (10.0 ** exponent)
            return f"{mantissa:.1f}x10^{exponent}"
        digits = 4 if abs(mean) < 0.01 else 3
        return f"{mean:.{digits}f}±{stderr:.{digits}f}"
    except Exception as e:
        raise Exception(f"experiments - format_entry - Execution error: {e}") from e


def paper_table(summaries: list, metric: str, horizons, dataset_label: str,
                model_order: list, model_labels: dict,
                style: str = "decimal",
                include_horizons: bool = True) -> pd.DataFrame:
    """One paper-style table: rows are the data set and its horizons, columns models."""
    try:
        pooled = summarize_seeds(summaries, metric)
        rows = [_table_row(dataset_label, pooled, model_order, model_labels, style)]

        for horizon in horizons if include_horizons else []:
            grouped: dict = {}
            for summary in summaries:
                value = summary.per_horizon.get(int(horizon), {}).get(metric)
                if value is None or not math.isfinite(float(value)):
                    continue
                grouped.setdefault(summary.model_key, []).append(float(value))
            per_horizon = {}
            for key, values in grouped.items():
                std = statistics.pstdev(values) if len(values) > 1 else 0.0
                per_horizon[key] = {
                    "mean": statistics.fmean(values),
                    "stderr": std / math.sqrt(len(values)),
                }
            rows.append(_table_row(
                f"{dataset_label} - {horizon}-day return", per_horizon,
                model_order, model_labels, style,
            ))
        return pd.DataFrame(rows).set_index("Data set")
    except Exception as e:
        raise Exception(f"experiments - paper_table - Execution error: {e}") from e


def build_paper_tables(summaries: list, horizons, dataset_label: str,
                       model_order: list, model_labels: dict) -> dict:
    """All three paper tables at once, keyed by their export file name."""
    try:
        tables = {}
        for name, (metric, caption, style, per_horizon) in PAPER_TABLES.items():
            tables[name] = {
                "caption": caption,
                "metric": metric,
                "frame": paper_table(
                    summaries, metric, horizons, dataset_label, model_order,
                    model_labels, style=style, include_horizons=per_horizon,
                ),
            }
        return tables
    except Exception as e:
        raise Exception(f"experiments - build_paper_tables - Execution error: {e}") from e


def _table_row(label: str, values: dict, model_order: list,
               model_labels: dict, style: str) -> dict:
    try:
        row = {"Data set": label}
        for key in model_order:
            entry = values.get(key)
            column = model_labels.get(key, key)
            row[column] = "-" if entry is None else format_entry(
                float(entry["mean"]), float(entry.get("stderr", 0.0)), style
            )
        return row
    except Exception as e:
        raise Exception(f"experiments - _table_row - Execution error: {e}") from e


def leaderboard(summaries: list, primary: str = "crps_sum") -> pd.DataFrame:
    """Every metric for every model, ranked by the paper's headline score."""
    try:
        rows = []
        for metric in REPORTED_METRICS:
            pooled = summarize_seeds(summaries, metric)
            for key, entry in pooled.items():
                rows.append({
                    "model_key": key,
                    "Model": entry["model_name"],
                    "metric": metric,
                    "mean": entry["mean"],
                    "stderr": entry["stderr"],
                    "runs": entry["runs"],
                })
        long_frame = pd.DataFrame(rows)
        runs = long_frame.groupby(["model_key", "Model"])["runs"].max()
        wide = long_frame.pivot_table(
            index=["model_key", "Model"], columns="metric",
            values=["mean", "stderr"],
        )
        wide.columns = [f"{metric}_{stat}" for stat, metric in wide.columns]
        wide = wide.join(runs).reset_index().sort_values(f"{primary}_mean")
        wide.insert(0, "Rank", range(1, len(wide) + 1))
        best = wide.iloc[0]
        wide["Gap vs best (%)"] = 100.0 * (
            wide[f"{primary}_mean"] / float(best[f"{primary}_mean"]) - 1.0
        )
        return wide.reset_index(drop=True)
    except Exception as e:
        raise Exception(f"experiments - leaderboard - Execution error: {e}") from e


def coverage_report(samples: torch.Tensor, targets: torch.Tensor,
                    horizons, levels=(0.80, 0.95)) -> pd.DataFrame:
    """Empirical coverage of the predictive intervals -- a calibration check.

    A model can win CRPS_sum and still be useless for sizing a position if its
    80% band only contains 60% of outcomes, so this is reported next to the
    paper metrics rather than instead of them.
    """
    try:
        horizons = [int(horizon) for horizon in horizons]
        numpy_samples = samples.detach().cpu().numpy()
        numpy_targets = targets.detach().cpu().numpy()
        rows = []
        for index, horizon in enumerate(horizons):
            block = numpy_samples[:, :, index, :]
            actual = numpy_targets[:, index, :]
            row = {"Return horizon": f"{horizon}-day"}
            for level in levels:
                tail = (1.0 - level) / 2.0
                low = np.quantile(block, tail, axis=1)
                high = np.quantile(block, 1.0 - tail, axis=1)
                inside = (actual >= low) & (actual <= high)
                row[f"{int(level * 100)}% coverage"] = 100.0 * float(inside.mean())
                row[f"{int(level * 100)}% width"] = float((high - low).mean())
            median = np.median(block, axis=1)
            row["Median bias"] = float((median - actual).mean())
            row["Direction hit rate"] = 100.0 * float(
                (np.sign(median) == np.sign(actual)).mean()
            )
            rows.append(row)
        return pd.DataFrame(rows)
    except Exception as e:
        raise Exception(f"experiments - coverage_report - Execution error: {e}") from e
