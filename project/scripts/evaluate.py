"""Evaluate a trained model: CRPS_sum, CRPS, MSE, fan charts (roadmap M5).

    python scripts/evaluate.py --config configs/exchange.yaml \
        --checkpoint checkpoints/exchange_lstm_realnvp_best.pt
    python scripts/evaluate.py --config configs/exchange.yaml --baseline random

Protocol (§5.2): "for testing we use a rolling windows prediction starting from
the last point seen in the training data set and compare it to the test set".
The cut is the one scripts/train.py used (`train_test_split`), so window i
forecasts [train_end + i*P, train_end + (i+1)*P) and only ever conditions on
data before its own forecast start.

Details that decide whether the numbers mean anything:

  * eval mode. BatchNormFlow must use its running statistics (§2.1); sampling in
    train mode silently rewrites them with statistics of the samples themselves.
  * the scale comes from the CONTEXT of each window only, and the samples are
    inverted with the same scale, so metrics are in original units.
  * covariates over the horizon are rebuilt from the sampled values
    (evaluation/sampling.py), never from the ground truth.
  * CRPS_sum is reported normalized (divided by sum_t |sum_i x_t^i|), which is
    the convention behind Table 1 -- a raw value in dollars is not comparable
    with 0.0064.

`--baseline random` scores N(0, 1) samples instead of the model. That is the M5
acceptance test for the metric itself: if the trained model does not score much
better, the metric is wrong, not the model.
"""

import argparse
import os
import statistics
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from flowforecast.data.scaling import compute_scale, invert_scale
from flowforecast.data.toy_pipes import empirical_cross_covariance
from flowforecast.evaluation.metrics import compute_metrics
from flowforecast.evaluation.plots import plot_covariance_comparison, plot_fan_chart
from flowforecast.evaluation.sampling import (random_baseline_samples,
                                              sample_trajectories)
from flowforecast.training.trainer import load_checkpoint
from flowforecast.utils.build import (append_csv_row, build_model, model_tag,
                                      prepare_data, run_name, save_json)
from flowforecast.utils.common import get_device, load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--baseline", type=str, default=None,
                        choices=["random", "veclstm"])
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-test-windows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--outputs", type=str, default="outputs")
    parser.add_argument("--fan-chart-series", type=int, nargs="*", default=[0, 1])
    parser.add_argument("--cross-covariance", action="store_true",
                        help="also produce the Figure 5 cross-covariance panels")
    args = parser.parse_args()
    if args.baseline != "random" and not args.checkpoint:
        parser.error("--checkpoint is required unless --baseline random is used")
    return args


def evaluate(args) -> dict:
    config = load_config(args.config)
    set_seed(args.seed)
    device = torch.device(args.device or get_device().type)

    bundle = prepare_data(config)
    eval_cfg = dict(config.get("evaluation") or {})
    num_samples = int(args.num_samples or eval_cfg.get("num_samples", 100))
    num_windows = int(args.num_test_windows or bundle.num_test_windows)
    pred_len = bundle.prediction_length
    ctx_len = bundle.context_length
    max_lag = bundle.max_lag
    eval_train_end = bundle.dataset.total_timesteps - num_windows * pred_len
    if eval_train_end <= 0:
        raise ValueError(
            f"{num_windows} windows of {pred_len} do not fit in "
            f"{bundle.dataset.total_timesteps} test-array steps"
        )

    model = None
    checkpoint = None
    if args.baseline != "random":
        model = build_model(config, bundle.num_series, bundle.cond_size,
                            baseline=args.baseline,
                            context_length=ctx_len).to(device)
        checkpoint = load_checkpoint(
            model, args.checkpoint, device=device,
            expected_contract="tempflow_v5_official_parity",
        )
        _warn_on_drift(checkpoint, bundle, args)
        model.eval()
        print(f"[evaluate] loaded {args.checkpoint} "
              f"(epoch {checkpoint.get('epoch', '?')}, "
              f"best_score {checkpoint.get('best_score', '?')})")

    tag = args.baseline if args.baseline else model_tag(config)
    name = args.run_name or run_name(config, baseline=args.baseline)
    os.makedirs(args.outputs, exist_ok=True)

    per_window = []
    data_cov, sample_cov = [], []
    last = None

    scale_numerator = torch.zeros(bundle.num_series, dtype=bundle.values.dtype)
    scale_denominator = torch.zeros_like(scale_numerator)
    for i in range(num_windows):
        forecast_start = eval_train_end + i * pred_len
        hist_start = forecast_start - ctx_len - max_lag
        context = bundle.values[:, forecast_start - ctx_len:forecast_start]
        context_observed = bundle.observed[
            :, forecast_start - ctx_len:forecast_start
        ]
        scale_numerator += (context.abs() * context_observed).sum(dim=-1)
        scale_denominator += context_observed.sum(dim=-1)
    eval_default_scale = scale_numerator / scale_denominator.clamp_min(1.0)

    for i in range(num_windows):
        forecast_start = eval_train_end + i * pred_len
        hist_start = forecast_start - ctx_len - max_lag
        if hist_start < 0:
            raise ValueError(
                f"window {i} needs {ctx_len + max_lag} steps of history before "
                f"index {forecast_start}"
            )
        window_end = forecast_start + pred_len
        if window_end > bundle.dataset.total_timesteps:
            raise ValueError(
                f"window {i} ends at {window_end} beyond the series "
                f"({bundle.dataset.total_timesteps})"
            )

        history_raw = bundle.values[:, hist_start:forecast_start]      # [D, n_hist]
        target = bundle.values[:, forecast_start:window_end]           # [D, P]
        target_observed = bundle.observed[:, forecast_start:window_end]

        observed_history = bundle.observed[:, hist_start:forecast_start]
        scale = compute_scale(
            history_raw[:, max_lag:], time_dim=-1,
            observed=observed_history[:, max_lag:],
            default_scale=eval_default_scale,
        )
        history_input = history_raw.t().unsqueeze(0)
        time_feat = bundle.time_feat[:, hist_start:window_end].t().unsqueeze(0)

        if args.baseline == "random":
            samples_scaled = random_baseline_samples(
                num_samples, bundle.num_series, pred_len, device=device)
        else:
            with torch.no_grad():
                samples_scaled = sample_trajectories(
                    model, history_input.to(device), time_feat.to(device),
                    bundle.lags, pred_len, num_samples=num_samples,
                    context_length=ctx_len, scale=scale)

        samples = invert_scale(samples_scaled.cpu(), scale, time_dim=-1)

        metrics = compute_metrics(samples, target, observed=target_observed)
        metrics["window"] = i
        metrics["forecast_start"] = forecast_start
        per_window.append(metrics)
        print(f"[evaluate] window {i}: CRPS_sum={metrics['crps_sum']:.4f} "
              f"(quantile check {metrics['crps_sum_quantile']:.4f}), "
              f"CRPS={metrics['crps']:.4f}, MSE={metrics['mse']:.4g}")

        if args.cross_covariance:
            data_cov.append(empirical_cross_covariance(target))
            sample_cov.append(
                torch.stack([empirical_cross_covariance(samples[s])
                             for s in range(samples.shape[0])]).mean(dim=0))

        last = (history_raw, target, samples)

    summary = _summarize(per_window)
    print("\n[evaluate] "
          f"{num_windows} rolling windows, {num_samples} samples, model={tag}")
    for key in ("crps_sum", "crps_sum_quantile", "crps", "mse", "nd"):
        print(f"  {key:18s} {summary[key]['mean']:.6f} "
              f"+/- {summary[key]['stderr']:.6f} (std {summary[key]['std']:.6f})")

    if last is not None:
        history_raw, target, samples = last
        for series_idx in args.fan_chart_series or []:
            if series_idx >= bundle.num_series:
                continue
            plot_fan_chart(
                history_raw[:, max_lag:], target, samples, series_idx=series_idx,
                save_path=os.path.join(
                    args.outputs, f"fan_chart_{name}_series{series_idx}.png"),
                title=f"{name} -- series {series_idx}, last test window")

    if args.cross_covariance and data_cov:
        gt = torch.stack(data_cov).mean(dim=0)
        fit = torch.stack(sample_cov).mean(dim=0)
        plot_covariance_comparison(
            gt, fit, show_diff=True, titles=("Data", "Samples"),
            suptitle=f"Cross-covariance x_t vs x_(t+1) -- {name} (Figure 5 style)",
            save_path=os.path.join(args.outputs, f"cross_covariance_{name}.png"))

    trainer_contract = (checkpoint or {}).get("trainer_contract") or {}
    checkpoint_config = (checkpoint or {}).get("config") or config
    checkpoint_training = dict(checkpoint_config.get("training") or {})
    training_budget = {
        "epochs": trainer_contract.get("num_epochs"),
        "batches_per_epoch": trainer_contract.get("batches_per_epoch"),
        "batch_size": checkpoint_training.get("batch_size"),
    }

    payload = {
        "run": name,
        "dataset": bundle.dataset.name,
        "model": tag,
        "checkpoint": args.checkpoint,
        "num_samples": num_samples,
        "num_test_windows": num_windows,
        "seed": args.seed,
        "context_length": ctx_len,
        "prediction_length": pred_len,
        "model_contract": "tempflow_v5_official_parity",
        "metric_contract": "gluonts_multivariate_19_quantiles",
        "training_budget": training_budget,
        "summary": summary,
        "per_window": per_window,
    }
    save_json(payload, os.path.join(args.outputs, f"metrics_{name}.json"))
    append_csv_row(os.path.join(args.outputs, "results.csv"), {
        "run": name,
        "dataset": bundle.dataset.name,
        "model": tag,
        "seed": args.seed,
        "num_samples": num_samples,
        "num_test_windows": num_windows,
        "model_contract": "tempflow_v5_official_parity",
        "metric_contract": "gluonts_multivariate_19_quantiles",
        "training_epochs": training_budget["epochs"],
        "training_batches_per_epoch": training_budget["batches_per_epoch"],
        "training_batch_size": training_budget["batch_size"],
        "crps_sum": summary["crps_sum"]["mean"],
        "crps_sum_stderr": summary["crps_sum"]["stderr"],
        "crps_sum_quantile": summary["crps_sum_quantile"]["mean"],
        "crps": summary["crps"]["mean"],
        "mse": summary["mse"]["mean"],
        "nd": summary["nd"]["mean"],
    })
    return payload


def _summarize(per_window: list) -> dict:
    keys = [k for k, v in per_window[0].items() if isinstance(v, float)]
    summary = {}
    for key in keys:
        values = [float(w[key]) for w in per_window]
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        summary[key] = {
            "mean": statistics.fmean(values),
            "std": std,
            "stderr": std / (len(values) ** 0.5) if values else 0.0,
            "values": values,
        }

    # Table 1 / GluonTS mean_wQuantileLoss is a ratio of GLOBAL sums, not the
    # arithmetic mean of per-window ratios.  Keep per-window values/std/stderr
    # for diagnostics, but replace each reported mean with its correctly
    # denominator-weighted counterpart.  This prevents low-activity windows
    # (Taxi windows 39/40) from receiving enormous, disproportionate weight.
    weighted = {
        "crps_sum": ("_crps_sum_numerator", "_crps_sum_denominator"),
        "crps_sum_energy": ("_crps_sum_energy_numerator",
                            "_crps_sum_denominator"),
        "crps": ("_crps_numerator", "_crps_denominator"),
        "crps_energy": ("_crps_energy_numerator", "_crps_denominator"),
        "nd": ("_nd_numerator", "_crps_denominator"),
    }
    for key, (num_key, den_key) in weighted.items():
        if all(num_key in w and den_key in w for w in per_window):
            numerator = sum(float(w[num_key]) for w in per_window)
            denominator = sum(float(w[den_key]) for w in per_window)
            if denominator > 0:
                summary[key]["mean"] = numerator / denominator
                summary[key]["aggregation"] = "ratio_of_global_sums"

    # The quantile estimator has the same denominator as CRPS_sum.  Older
    # per-window records do not expose its raw numerator, so new runs derive it
    # from ratio*denominator and aggregate globally.
    if all("crps_sum_quantile" in w and "_crps_sum_denominator" in w
           for w in per_window):
        denominator = sum(float(w["_crps_sum_denominator"]) for w in per_window)
        numerator = sum(float(w["crps_sum_quantile"])
                        * float(w["_crps_sum_denominator"])
                        for w in per_window)
        if denominator > 0:
            summary["crps_sum_quantile"]["mean"] = numerator / denominator
            summary["crps_sum_quantile"]["aggregation"] = "ratio_of_global_sums"
    return summary


def _warn_on_drift(checkpoint: dict, bundle, args) -> None:
    """Loud failure instead of a silent shape error when config and run differ."""
    if checkpoint.get("model_contract") != "tempflow_v5_official_parity":
        raise ValueError(
            "checkpoint predates the official-parity TempFlow v5 contract; "
            "old weights use incompatible flow or conditioning semantics "
            "and must be retrained"
        )
    saved_cond = checkpoint.get("cond_size")
    if saved_cond is not None and int(saved_cond) != bundle.cond_size:
        raise ValueError(
            f"checkpoint was trained with cond_size={saved_cond} but this config "
            f"produces {bundle.cond_size}: the dataset, lags or frequency changed"
        )
    saved_series = checkpoint.get("num_series")
    if saved_series is not None and int(saved_series) != bundle.num_series:
        raise ValueError(
            f"checkpoint was trained with D={saved_series} but the loaded data "
            f"has D={bundle.num_series}"
        )
    saved_baseline = checkpoint.get("baseline")
    if saved_baseline != args.baseline and (saved_baseline or args.baseline):
        print(f"[evaluate] WARNING checkpoint baseline={saved_baseline!r} but "
              f"--baseline={args.baseline!r}")


def main():
    args = parse_args()
    try:
        evaluate(args)
    except Exception as e:
        raise Exception(f"evaluate - main - Evaluation pipeline failed: {e}") from e


if __name__ == "__main__":
    main()
