"""Toy pipes experiment (paper §5.1, Figures 3 and 4) -- roadmap M6.

    python scripts/pipes_experiment.py                    # default: 10k steps, LSTM-MAF
    python scripts/pipes_experiment.py --flow realnvp --epochs 20

The paper's own sanity check. A 6-dimensional system of pipes is simulated
(data/toy_pipes.py), the model is trained on it exactly like a real dataset, then
100 trajectories are sampled and their covariance and cross-covariance matrices
are compared with the ones estimated from the data:

    Figure 4 left  -> Cov(x_t, x_t)      : dependency ACROSS series
    Figure 4 right -> Cov(x_t, x_{t+1})  : dependency ACROSS time,
                                           S_0 -> {S_1, S_2} -> S_3

The acceptance criterion is quantitative, not "the heatmaps look similar": the
correlation between the entries of the data matrix and the model matrix, plus the
propagation entries of the model, are printed and written to
outputs/pipes_report.json.

Sample covariance matrices are averaged over trajectories (each trajectory gets
its own estimate) instead of being computed from the sample mean -- averaging the
trajectories first would destroy exactly the variance we are trying to measure.
"""

import argparse
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from flowforecast.data.covariates import time_features
from flowforecast.data.scaling import compute_scale
from flowforecast.data.toy_pipes import (PIPE_LABELS, check_pipe_structure,
                                         empirical_covariance,
                                         empirical_cross_covariance,
                                         simulate_pipes)
from flowforecast.data.windowing import WindowedDataset
from flowforecast.evaluation.plots import (plot_covariance_comparison,
                                           plot_loss_curve, plot_series_grid)
from flowforecast.evaluation.sampling import sample_trajectories
from flowforecast.training.trainer import Trainer
from flowforecast.utils.build import build_model, count_parameters, save_json
from flowforecast.utils.common import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=24)
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--flow", type=str, default="maf", choices=["maf", "realnvp"])
    parser.add_argument("--backbone", type=str, default="lstm",
                        choices=["lstm", "transformer"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batches-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--outputs", type=str, default="outputs")
    parser.add_argument("--checkpoints", type=str, default="checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        set_seed(args.seed)
        device = args.device or get_device().type
        os.makedirs(args.outputs, exist_ok=True)

        series = simulate_pipes(args.timesteps, seed=args.seed)     # [6, T]
        num_series, total = series.shape
        print(f"[pipes] simulated {num_series} series x {total} steps")
        plot_series_grid(series[:, :400], num_series=num_series,
                         save_path=os.path.join(args.outputs, "pipes_series.png"),
                         title="Simulated system of pipes (§5.1, Figure 3 style)")

        # The pipe system has no calendar of its own; an hourly index keeps the
        # covariate layout identical to the real datasets (the lags carry the
        # signal, the calendar features are just along for the ride).
        lags = [1, 2]
        freq = "H"
        timestamps = pd.date_range("2020-01-01", periods=total, freq="h")
        time_feat = time_features(timestamps, freq)

        config = {
            "dataset": {"lags": lags, "dequantize": False},
            "backbone": {"type": args.backbone, "hidden_size": 40, "num_layers": 2,
                         "n_heads": 8, "num_encoder_layers": 2,
                         "num_decoder_layers": 2, "dropout": 0.1},
            "flow": {"type": args.flow, "num_blocks": 5, "hidden": 100,
                     "batch_norm": True},
        }

        test_windows = 5
        train_end = total - test_windows * args.prediction_length
        dataset = WindowedDataset(
            series=series, time_feat=time_feat,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            lags=lags,
            num_windows_per_epoch=args.batch_size * args.batches_per_epoch,
            end_index=train_end)
        loader = DataLoader(dataset, batch_size=args.batch_size, drop_last=True)

        cond_size = time_feat.shape[0] + num_series * len(lags)
        model = build_model(config, num_series, cond_size,
                            context_length=args.context_length)
        print(f"[pipes] model {type(model.flow).__name__} with "
              f"{count_parameters(model):,} parameters on {device}")

        trainer = Trainer(model, loader, lr=1e-3, grad_clip=0.0,
                          batches_per_epoch=args.batches_per_epoch, device=device)
        run = f"pipes_{args.backbone}_{args.flow}"
        history = trainer.fit(args.epochs, resume_from=None,
                             checkpoint_dir=args.checkpoints, run_name=run)
        plot_loss_curve(history,
                        save_path=os.path.join(args.outputs, f"loss_{run}.png"),
                        title=f"Learning curve -- {run}")

        # ---- sample trajectories from the end of the training range -----------
        max_lag = max(lags)
        forecast_start = train_end
        hist_start = forecast_start - args.context_length - max_lag
        history_raw = series[:, hist_start:forecast_start]
        scale = compute_scale(history_raw[:, max_lag:], time_dim=-1)
        history_input = history_raw.t().unsqueeze(0)
        tf_window = time_feat[:, hist_start:forecast_start + args.prediction_length]
        tf_window = tf_window.t().unsqueeze(0)

        model.eval()
        with torch.no_grad():
            samples_scaled = sample_trajectories(
                model, history_input.to(device), tf_window.to(device), lags,
                args.prediction_length, num_samples=args.num_samples,
                context_length=args.context_length, scale=scale)
        samples = (samples_scaled.cpu() * scale.unsqueeze(-1))  # [S, D, P]
        print(f"[pipes] sampled {tuple(samples.shape)} trajectories")

        # ---- Figure 4: data vs model, covariance and cross-covariance ---------
        data_cov = empirical_covariance(series[:, :train_end])
        data_cross = empirical_cross_covariance(series[:, :train_end])
        model_cov = torch.stack([empirical_covariance(samples[s])
                                 for s in range(samples.shape[0])]).mean(dim=0)
        model_cross = torch.stack([empirical_cross_covariance(samples[s])
                                   for s in range(samples.shape[0])]).mean(dim=0)

        plot_covariance_comparison(
            data_cov, model_cov, labels=PIPE_LABELS, show_diff=True,
            titles=("Data Cov(x_t, x_t)", "Model Cov(x_t, x_t)"),
            suptitle="Pipes: covariance across series (§5.1, Figure 4 left)",
            save_path=os.path.join(args.outputs, f"{run}_covariance.png"))
        plot_covariance_comparison(
            data_cross, model_cross, labels=PIPE_LABELS, show_diff=True,
            titles=("Data Cov(x_t, x_t+1)", "Model Cov(x_t, x_t+1)"),
            suptitle="Pipes: cross-covariance across time (§5.1, Figure 4 right)",
            save_path=os.path.join(args.outputs, f"{run}_cross_covariance.png"))

        report = {
            "run": run,
            "timesteps": int(total),
            "epochs": int(args.epochs),
            "num_samples": int(args.num_samples),
            "final_train_nll": history["train_nll"][-1] if history.get("train_nll") else None,
            "data_structure": check_pipe_structure(series[:, :train_end]),
            "model_structure": check_pipe_structure_from_samples(samples),
            "cov_entry_correlation": _entry_correlation(data_cov, model_cov),
            "cross_entry_correlation": _entry_correlation(data_cross, model_cross),
            "cov_max_abs_diff": float((data_cov - model_cov).abs().max().item()),
            "cross_max_abs_diff": float((data_cross - model_cross).abs().max().item()),
        }
        save_json(report, os.path.join(args.outputs, "pipes_report.json"))

        print("\n[pipes] M6 acceptance report")
        print(f"  data  propagation visible : "
              f"{report['data_structure']['propagation_visible']}")
        print(f"  model propagation visible : "
              f"{report['model_structure']['propagation_visible']}")
        print(f"  Cov entry correlation     : {report['cov_entry_correlation']:.3f}")
        print(f"  Cross entry correlation   : {report['cross_entry_correlation']:.3f}")
        print(f"  data  Cov(s1,s2)          : "
              f"{report['data_structure']['cov_s1_s2']:.4f}")
        print(f"  model Cov(s1,s2)          : "
              f"{report['model_structure']['cov_s1_s2']:.4f}")
    except Exception as e:
        raise Exception(f"pipes_experiment - main - Experiment failed: {e}") from e


def check_pipe_structure_from_samples(samples: torch.Tensor) -> dict:
    """Average the per-trajectory structure numbers over the sample set."""
    try:
        reports = [check_pipe_structure(samples[s]) for s in range(samples.shape[0])]
        keys = [k for k, v in reports[0].items() if isinstance(v, float)]
        averaged = {k: sum(r[k] for r in reports) / len(reports) for k in keys}
        averaged["propagation_visible"] = bool(
            averaged["cross_s0_s1"] > 0 and averaged["cross_s0_s2"] > 0
            and averaged["cross_s1_s3"] > 0 and averaged["cross_s2_s3"] > 0)
        return averaged
    except Exception as e:
        raise Exception(
            f"pipes_experiment - check_pipe_structure_from_samples - "
            f"Execution error: {e}"
        ) from e


def _entry_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between the entries of two matrices."""
    try:
        x = a.reshape(-1).to(torch.float64)
        y = b.reshape(-1).to(torch.float64)
        x = x - x.mean()
        y = y - y.mean()
        denom = float((x.norm() * y.norm()).item())
        if denom == 0:
            return float("nan")
        return float((x @ y).item() / denom)
    except Exception as e:
        raise Exception(
            f"pipes_experiment - _entry_correlation - Execution error: {e}"
        ) from e


if __name__ == "__main__":
    main()
