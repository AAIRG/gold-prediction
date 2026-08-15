"""Train a ConditionedFlowModel or the Vec-LSTM baseline (roadmap M4).

    python scripts/train.py --config configs/exchange.yaml
    python scripts/train.py --config configs/exchange.yaml --baseline veclstm
    python scripts/train.py --config configs/exchange.yaml --overfit-one-window

What this script is careful about:

  * TRAINING NEVER TOUCHES THE TEST RANGE. §5.2 evaluates on rolling windows
    starting at the last point seen in training, so `train_test_split` decides a
    cut index and the window sampler is bounded by it. Sampling windows from the
    whole series -- which is what this script used to do -- trains on the test
    range and makes every reported CRPS meaningless.
  * Reproduction configs set `val_windows=0`: the fixed-budget official trainer
    uses the complete supplied training split and hands final weights to prediction.
  * Covariates come from real calendar timestamps and per-window lag history
    (flowforecast/data/covariates.py); batch-aware scaling is applied by the model.
  * `cond_size` is the covariate dimension C = C_time + D * len(lags), computed
    once in flowforecast/utils/build.py and shared with scripts/evaluate.py.

Checkpoints: both `checkpoints/<run>.pt` and `<run>_best.pt` hold the final epoch
in reproduction mode; the latter name is retained for CLI compatibility.
"""

import argparse
import os
import sys

from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from flowforecast.data.windowing import RollingWindowDataset, WindowedDataset
from flowforecast.evaluation.plots import plot_loss_curve
from flowforecast.training.trainer import Trainer
from flowforecast.utils.build import (build_model, count_parameters, prepare_data,
                                      run_name, save_json)
from flowforecast.utils.common import get_device, load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--baseline", type=str, choices=["veclstm"],
                        help="train the Vec-LSTM baseline instead of the flow")
    parser.add_argument("--overfit-one-window", action="store_true",
                        help="M4 acceptance test: one fixed window, NLL must collapse")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batches-per-epoch", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore an existing checkpoint and start over")
    parser.add_argument("--outputs", type=str, default="outputs")
    parser.add_argument("--checkpoints", type=str, default="checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        config = load_config(args.config)
        train_cfg = dict(config.get("training") or {})
        flow_type = str((config.get("flow") or {}).get("type", "realnvp")).lower()
        if args.baseline:
            flow_type = "baseline"

        seed = args.seed if args.seed is not None else int(train_cfg.get("seed", 42))
        set_seed(seed)
        device = args.device or get_device().type

        bundle = prepare_data(config)
        batch_size = int(train_cfg.get("batch_size", 64))
        batches_per_epoch = int(args.batches_per_epoch
                                or train_cfg.get("batches_per_epoch", 100))
        num_epochs = int(args.epochs or train_cfg.get("epochs", 40))
        select_metric = str(train_cfg.get("select_metric", "last")).lower()
        val_windows = int(train_cfg.get("val_windows", 0))
        if select_metric in ("crps", "both", "val_crps") and val_windows <= 0:
            raise ValueError(
                f"select_metric={select_metric!r} requires training.val_windows > 0"
            )

        # Training draws from the grouped TRAIN array with its own calendar.
        # `bundle.train_series*` falls back to the contiguous array when gluonts
        # did not give a separate train split (raw-file datasets).
        common = dict(
            series=bundle.train_series,
            time_feat=bundle.train_series_time_feat,
            context_length=bundle.context_length,
            prediction_length=bundle.prediction_length,
            lags=bundle.lags,
            dequantize_counts=bundle.dequantize,
            observed=bundle.train_observed,
        )
        train_range_end = bundle.train_series_end

        if args.overfit_one_window:
            # One fixed window, repeated: the NLL must go very low. If it does
            # not, the bug is in the h_t -> flow wiring, not in the schedule.
            train_dataset = WindowedDataset(
                num_windows_per_epoch=max(batch_size, 8),
                fixed_start=bundle.max_lag,
                end_index=train_range_end, **common)
            val_dataset = None
            batch_size = min(batch_size, len(train_dataset))
            print(f"[train] overfit mode: single window at index {bundle.max_lag}")
        else:
            val_start = train_range_end - val_windows * bundle.prediction_length
            train_dataset = WindowedDataset(
                num_windows_per_epoch=batch_size * batches_per_epoch,
                end_index=val_start if val_windows > 0 else train_range_end,
                **common)
            val_dataset = None
            if val_windows > 0:
                val_dataset = RollingWindowDataset(
                    num_windows=val_windows, end_index=train_range_end, **common)
                print(f"[train] validation windows start at "
                      f"{val_dataset.starts} (train range ends at {val_start})")

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=False, drop_last=True)
        val_loader = None if val_dataset is None else DataLoader(
            val_dataset, batch_size=min(batch_size, len(val_dataset)),
            shuffle=False)

        model = build_model(config, bundle.num_series, bundle.cond_size,
                            baseline=args.baseline,
                            context_length=bundle.context_length)
        print(f"[train] model: {type(model).__name__} / "
              f"{type(model.backbone).__name__} / {type(model.flow).__name__}, "
              f"{count_parameters(model):,} trainable parameters, device={device}")

        name = args.run_name or run_name(
            config, baseline=args.baseline,
            seed=seed if args.seed is not None else None,
            suffix="overfit" if args.overfit_one_window else None)

        trainer_lr = float(train_cfg.get("lr", 1e-3))
        trainer_grad_clip = float(
            train_cfg.get("maf_grad_clip", 10.0)
            if flow_type == "maf"
            else train_cfg.get("grad_clip", 0.0)
        )
        trainer_weight_decay = float(train_cfg.get("weight_decay", 1e-6))
        trainer_max_lr = float(train_cfg.get("max_lr", 1e-2))
        if flow_type == "maf":
            flow_cfg = dict(config.get("flow") or {})
            print(
                "[train] official MAF parity: "
                f"log_scale_bound={float(flow_cfg.get('maf_log_scale_bound', 0.0)):g}, "
                f"lr={trainer_lr:g}, schedule=onecycle, "
                f"max_lr={trainer_max_lr:g}, grad_clip={trainer_grad_clip:g}"
            )

        trainer = Trainer(
            model, train_loader, val_loader=val_loader,
            lr=trainer_lr,
            grad_clip=trainer_grad_clip,
            batches_per_epoch=batches_per_epoch,
            device=device,
            early_stopping_patience=train_cfg.get("early_stopping_patience"),
            select_metric=str(train_cfg.get("select_metric", "last")),
            weight_decay=trainer_weight_decay,
            lr_schedule=str(train_cfg.get("lr_schedule", "onecycle")),
            max_lr=trainer_max_lr,
            validation_num_samples=int(
                train_cfg.get("validation_num_samples", 20)
            ),
            validation_seed=int(train_cfg.get("validation_seed", 1729)),
            validation_interval=int(train_cfg.get("validation_interval", 1)),
        )

        checkpoint_path = os.path.join(args.checkpoints, f"{name}.pt")
        history = trainer.fit(
            num_epochs,
            resume_from=None if args.no_resume else checkpoint_path,
            checkpoint_dir=args.checkpoints,
            run_name=name,
            extra_state={
                "config": config,
                "seed": seed,
                "baseline": args.baseline,
                "num_series": bundle.num_series,
                "cond_size": bundle.cond_size,
                "context_length": bundle.context_length,
                "prediction_length": bundle.prediction_length,
                "lags": list(bundle.lags),
                "freq": bundle.freq,
                "train_end": bundle.train_end,
                "model_contract": "tempflow_v5_official_parity",
            },
        )

        os.makedirs(args.outputs, exist_ok=True)
        save_json({"run": name, "seed": seed, "history": history},
                  os.path.join(args.outputs, f"history_{name}.json"))
        if history.get("train_nll"):
            plot_loss_curve(history,
                            save_path=os.path.join(args.outputs, f"loss_{name}.png"),
                            title=f"Learning curve -- {name}")

        print(f"[train] done. Evaluate with:\n"
              f"  python scripts/evaluate.py --config {args.config} "
              f"--checkpoint {os.path.join(args.checkpoints, name + '_best.pt')}"
              + (f" --baseline {args.baseline}" if args.baseline else ""))
    except Exception as e:
        raise Exception(f"train - main - Training pipeline failed: {e}") from e


if __name__ == "__main__":
    main()
