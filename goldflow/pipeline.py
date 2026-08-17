"""One orchestration path shared by the Colab notebook and the local test.

The notebook is a thin caller: it fills in a `PipelineConfig` and calls `run`.
Everything that decides how models are trained, how windows are scored, how the
boosting stages are fitted and gated, and what ends up in the exported artifacts
lives here, so a local run with tiny settings exercises exactly the same code as
the T4 run with the real budget.

Stages, in order:

    prepare        download gold, build the D = 3 target, cut train / validation
                   / test so that nothing after the training split is ever seen
                   during fitting
    train          every variant, every seed, through the project's Trainer
    evaluate       rolling test windows per variant and seed, §5.2 protocol
    boost stage 1  forward stagewise CRPS mixture, weights fitted on the
                   VALIDATION windows only
    boost stage 2  gradient-boosted residual correction, fitted on stride-1
                   windows inside the validation range and gated on a held-out
                   tail of that range
    export         checkpoints, metadata, paper tables 1/4/5, leaderboard,
                   calibration, per-window metrics, learning curves

Split discipline is the part worth stating plainly: ensemble weights and the
residual booster never touch the test range, so the test numbers the leaderboard
reports are honest for the boosted rows too.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import boosting, data, experiments, features, variants


@dataclass
class PipelineConfig:
    ticker: str = "GC=F"
    horizons: tuple = (1, 3, 7)
    context_length: int = 30
    prediction_length: int = 10
    lags: tuple = (1, 2)

    seeds: tuple = (1, 2, 3)
    epochs: int = 40
    batch_size: int = 64
    batches_per_epoch: int = 100

    test_windows: int = 20
    validation_windows: int = 20
    evaluation_samples: int = 100
    validation_samples: int = 100

    ensemble_rounds: int = 24
    boost_fit_windows: int = 120
    boost_gate_windows: int = 40
    boost_samples: int = 30
    enable_residual_booster: bool = True
    booster_min_gain_pct: float = 1.0

    download_period: str = "max"
    dataset_label: str = "Gold (GC=F)"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    work_dir: Path = Path("/content/gold_yahoo_work")
    export_dir: Path = Path("/content/gold_yahoo_artifacts")
    prices: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def export_seed(self) -> int:
        return int(self.seeds[0])

    @property
    def max_lag(self) -> int:
        return max(int(lag) for lag in self.lags)


BOOSTED_KEY = "boosted"
BOOSTED_NAME = "Boosted ensemble"
BOOSTED_GB_KEY = "boosted_gb"
BOOSTED_GB_NAME = "Boosted + residual GB"


# --------------------------------------------------------------------------
# stage 1: data
# --------------------------------------------------------------------------

def prepare(config: PipelineConfig) -> dict:
    """Download, build the target matrix, and cut the three splits."""
    try:
        prices = config.prices if config.prices is not None else \
            data.download_history(config.ticker, period=config.download_period)
        returns = data.build_return_targets(prices["Close"], config.horizons)
        timestamps = pd.DatetimeIndex(returns.index)
        time_feat = features.stable_time_features(timestamps)
        series = torch.from_numpy(
            np.ascontiguousarray(returns.to_numpy(dtype="float32"))
        ).t().contiguous()

        splits = data.split_indices(
            len(returns), config.prediction_length, config.test_windows,
            config.validation_windows,
        )
        cond_size = int(time_feat.shape[0] + series.shape[0] * len(config.lags))
        needed = config.context_length + config.max_lag
        if splits["train_end"] <= needed:
            raise ValueError(
                f"the training split holds {splits['train_end']} rows, which "
                f"cannot serve context {config.context_length} plus max lag "
                f"{config.max_lag}"
            )
        return {
            "prices": prices,
            "returns": returns,
            "timestamps": timestamps,
            "time_feat": time_feat,
            "series": series,
            "splits": splits,
            "cond_size": cond_size,
            "num_series": int(series.shape[0]),
        }
    except Exception as e:
        raise Exception(f"pipeline - prepare - Execution error: {e}") from e


def base_config_for(config: PipelineConfig, bundle: dict, seed: int) -> dict:
    try:
        return variants.base_config(
            num_series=bundle["num_series"],
            context_length=config.context_length,
            prediction_length=config.prediction_length,
            lags=list(config.lags), epochs=config.epochs,
            batch_size=config.batch_size,
            batches_per_epoch=config.batches_per_epoch,
            validation_windows=config.validation_windows,
            test_windows=config.test_windows,
            evaluation_samples=config.evaluation_samples,
            validation_samples=config.validation_samples,
            seed=seed,
        )
    except Exception as e:
        raise Exception(f"pipeline - base_config_for - Execution error: {e}") from e


# --------------------------------------------------------------------------
# stage 2: training
# --------------------------------------------------------------------------

def build_loaders(config: PipelineConfig, bundle: dict) -> tuple:
    try:
        from flowforecast.data.windowing import (RollingWindowDataset,
                                                 WindowedDataset)

        splits = bundle["splits"]
        train_dataset = WindowedDataset(
            series=bundle["series"], time_feat=bundle["time_feat"],
            context_length=config.context_length,
            prediction_length=config.prediction_length, lags=list(config.lags),
            num_windows_per_epoch=config.batch_size * config.batches_per_epoch,
            end_index=splits["train_end"],
        )
        val_dataset = RollingWindowDataset(
            series=bundle["series"], time_feat=bundle["time_feat"],
            context_length=config.context_length,
            prediction_length=config.prediction_length, lags=list(config.lags),
            num_windows=config.validation_windows,
            end_index=splits["validation_end"],
        )
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=False,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=len(val_dataset), shuffle=False
        )
        return train_loader, val_loader
    except Exception as e:
        raise Exception(f"pipeline - build_loaders - Execution error: {e}") from e


def train_one(config: PipelineConfig, bundle: dict, variant_key: str, seed: int,
              loaders: tuple, checkpoint_dir: Path) -> tuple:
    """Train one variant with one seed and return the model and its history."""
    try:
        import random

        from flowforecast.training.trainer import Trainer
        from flowforecast.utils.build import count_parameters

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        variant = variants.get_variant(variant_key)
        stamped = variants.variant_config(
            variant_key, base_config_for(config, bundle, seed), seed=seed
        )
        model = variants.build_variant(
            variant_key, stamped, num_series=bundle["num_series"],
            cond_size=bundle["cond_size"], context_length=config.context_length,
            device=config.device,
        )
        train_loader, val_loader = loaders
        trainer = Trainer(
            model, train_loader, val_loader=val_loader, device=config.device,
            lr=1e-3, grad_clip=5.0, batches_per_epoch=config.batches_per_epoch,
            select_metric="last", validation_num_samples=config.validation_samples,
            validation_seed=1729, validation_interval=max(1, config.epochs // 4),
            lr_schedule="onecycle", max_lr=3e-3,
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        run_name = f"gold_{variant_key}_seed{seed}"
        history = trainer.fit(
            num_epochs=config.epochs, checkpoint_dir=str(checkpoint_dir),
            run_name=run_name,
            extra_state={
                "config": stamped, "seed": int(seed),
                "baseline": variant.baseline,
                "num_series": bundle["num_series"],
                "cond_size": bundle["cond_size"],
                "context_length": config.context_length,
                "prediction_length": config.prediction_length,
                "lags": list(config.lags), "freq": "B",
                "model_contract": variants.MODEL_CONTRACT,
            },
        )
        model.eval()
        return model, {
            "history": history,
            "checkpoint": checkpoint_dir / f"{run_name}.pt",
            "parameters": int(count_parameters(model)),
            "config": stamped,
        }
    except Exception as e:
        raise Exception(f"pipeline - train_one - Execution error: {e}") from e


# --------------------------------------------------------------------------
# stage 3: rolling-window evaluation
# --------------------------------------------------------------------------

def sample_split(config: PipelineConfig, bundle: dict, model, end_index: int,
                 num_windows: int, num_samples: int, seed: int,
                 stride: int | None = None) -> dict:
    try:
        return experiments.rolling_window_samples(
            model, bundle["series"], bundle["time_feat"], list(config.lags),
            config.context_length, config.prediction_length,
            num_windows=num_windows, end_index=end_index,
            num_samples=num_samples, device=config.device, seed=seed,
            stride=stride,
        )
    except Exception as e:
        raise Exception(f"pipeline - sample_split - Execution error: {e}") from e


def score(config: PipelineConfig, model_key: str, model_name: str, seed: int,
          samples: torch.Tensor, targets: torch.Tensor) -> experiments.SeedSummary:
    try:
        per_window = experiments.evaluate_samples(samples, targets)
        return experiments.SeedSummary(
            model_key=model_key, model_name=model_name, seed=int(seed),
            metrics=experiments.summarize_windows(per_window),
            per_horizon=experiments.per_dimension_metrics(
                samples, targets, config.horizons
            ),
            per_window=per_window,
        )
    except Exception as e:
        raise Exception(f"pipeline - score - Execution error: {e}") from e


# --------------------------------------------------------------------------
# stage 4: boosting
# --------------------------------------------------------------------------

def fit_boosting(config: PipelineConfig, bundle: dict, models: dict,
                 validation: dict, test: dict, seed: int) -> dict:
    """Both boosting stages for one seed. Weights come from validation only."""
    try:
        validation_samples = {key: value["samples"] for key, value in validation.items()}
        reference = next(iter(validation.values()))
        ensemble = boosting.fit_greedy_crps_ensemble(
            validation_samples, reference["targets"], rounds=config.ensemble_rounds
        )

        test_samples = {key: value["samples"] for key, value in test.items()}
        test_targets = next(iter(test.values()))["targets"]
        mixture = boosting.mix_samples(
            test_samples, ensemble.weights, ensemble.pool_size
        )
        summaries = [score(
            config, BOOSTED_KEY, BOOSTED_NAME, seed, mixture, test_targets
        )]
        ensemble.ensemble_metrics = {
            "crps_sum_test": float(summaries[0].metrics["crps_sum"]["mean"]),
            "crps_test": float(summaries[0].metrics["crps"]["mean"]),
            "mse_test": float(summaries[0].metrics["mse"]["mean"]),
        }

        booster = None
        if config.enable_residual_booster:
            booster, gate = _fit_residual_stage(
                config, bundle, models, ensemble, seed
            )
            ensemble.booster_enabled = bool(gate["enabled"])
            ensemble.booster_gain_pct = float(gate["gain_pct"])
            ensemble.booster_metrics = {
                key: float(value) for key, value in gate.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            shifted = booster.shift(
                mixture, bundle["series"], bundle["timestamps"],
                next(iter(test.values()))["forecast_starts"], list(config.horizons),
            )
            summaries.append(score(
                config, BOOSTED_GB_KEY, BOOSTED_GB_NAME, seed, shifted, test_targets
            ))
        return {"ensemble": ensemble, "booster": booster, "summaries": summaries,
                "mixture": mixture}
    except Exception as e:
        raise Exception(f"pipeline - fit_boosting - Execution error: {e}") from e


def _fit_residual_stage(config: PipelineConfig, bundle: dict, models: dict,
                        ensemble: boosting.BoostingResult, seed: int) -> tuple:
    """Fit stage 2 on stride-1 validation windows, gate on a disjoint tail.

    With stride 1 the windows overlap heavily, so the gate block is pushed back
    far enough that its prediction range shares no day with the fit block:
    stride-1 windows ending at `E` cover `[E - N + 1 - P, E)`, therefore the fit
    block must end at or before `E_gate - N_gate + 1 - P`. One extra horizon of
    margin is taken on top of that.
    """
    try:
        splits = bundle["splits"]
        gate_end = splits["validation_end"]
        fit_end = gate_end - config.boost_gate_windows - config.prediction_length
        earliest_start = (gate_end - config.boost_fit_windows
                          - config.boost_gate_windows
                          - 2 * config.prediction_length + 1)
        if earliest_start < splits["train_end"]:
            raise ValueError(
                f"stage 2 would reach back to index {earliest_start}, inside the "
                f"training range that ends at {splits['train_end']}: its residuals "
                f"would be in-sample for the base models. Raise "
                f"validation_windows to at least "
                f"{math.ceil((config.boost_fit_windows + config.boost_gate_windows + 2 * config.prediction_length - 1) / config.prediction_length)} "
                f"or lower boost_fit_windows / boost_gate_windows."
            )
        if earliest_start < config.context_length + config.max_lag:
            raise ValueError(
                f"stage 2 window at {earliest_start} lacks "
                f"{config.context_length + config.max_lag} steps of history"
            )

        blocks = {"fit": (fit_end, config.boost_fit_windows),
                  "gate": (gate_end, config.boost_gate_windows)}
        pooled = {}
        for name, (end_index, count) in blocks.items():
            per_model = {}
            for key, model in models.items():
                per_model[key] = sample_split(
                    config, bundle, model, end_index, count,
                    config.boost_samples, seed + 101, stride=1,
                )
            reference = next(iter(per_model.values()))
            pooled[name] = {
                "mixture": boosting.mix_samples(
                    {k: v["samples"] for k, v in per_model.items()},
                    ensemble.weights, ensemble.pool_size,
                ),
                "targets": reference["targets"],
                "forecast_starts": reference["forecast_starts"],
            }

        overlap = set(range(
            min(pooled["fit"]["forecast_starts"]),
            max(pooled["fit"]["forecast_starts"]) + config.prediction_length,
        )) & set(range(
            min(pooled["gate"]["forecast_starts"]),
            max(pooled["gate"]["forecast_starts"]) + config.prediction_length,
        ))
        if overlap:
            raise ValueError(
                f"the stage 2 fit and gate blocks share {len(overlap)} days, so "
                f"the gate would not be held out"
            )

        matrix, residuals = boosting.build_residual_dataset(
            pooled["fit"]["mixture"], pooled["fit"]["targets"], bundle["series"],
            bundle["timestamps"], pooled["fit"]["forecast_starts"],
            list(config.horizons),
        )
        booster = boosting.GradientResidualBooster(seed=seed).fit(matrix, residuals)
        gate = boosting.gate_residual_booster(
            booster, pooled["gate"]["mixture"], pooled["gate"]["targets"],
            bundle["series"], bundle["timestamps"],
            pooled["gate"]["forecast_starts"], list(config.horizons),
            min_gain_pct=config.booster_min_gain_pct,
        )
        gate["fit_rows"] = float(matrix.shape[0])
        return booster, gate
    except Exception as e:
        raise Exception(f"pipeline - _fit_residual_stage - Execution error: {e}") from e


# --------------------------------------------------------------------------
# stage 5: orchestration and export
# --------------------------------------------------------------------------

def run(config: PipelineConfig, verbose: bool = True) -> dict:
    """Train, evaluate, boost and export everything. Returns the full report."""
    try:
        bundle = prepare(config)
        loaders = build_loaders(config, bundle)
        splits = bundle["splits"]
        if verbose:
            print(f"[pipeline] {len(bundle['returns']):,} aligned target rows "
                  f"{bundle['returns'].index.min().date()} -> "
                  f"{bundle['returns'].index.max().date()}")
            print(f"[pipeline] train [0, {splits['train_end']}) - validation "
                  f"[{splits['train_end']}, {splits['validation_end']}) - test "
                  f"[{splits['validation_end']}, {splits['test_end']})")
            print(f"[pipeline] cond_size={bundle['cond_size']}, "
                  f"device={config.device}, seeds={list(config.seeds)}")

        summaries: list = []
        training_meta: dict = {}
        checkpoints: dict = {}
        export_payload: dict = {}
        variant_keys = variants.variant_names()

        for seed in config.seeds:
            models: dict = {}
            test: dict = {}
            validation: dict = {}
            for key in variant_keys:
                variant = variants.get_variant(key)
                if verbose:
                    print(f"\n[pipeline] === {variant.name}, seed {seed} ===")
                model, meta = train_one(
                    config, bundle, key, seed, loaders,
                    Path(config.work_dir) / "checkpoints",
                )
                models[key] = model
                test[key] = sample_split(
                    config, bundle, model, splits["test_end"],
                    config.test_windows, config.evaluation_samples, seed,
                )
                validation[key] = sample_split(
                    config, bundle, model, splits["validation_end"],
                    config.validation_windows, config.evaluation_samples, seed + 7,
                )
                summary = score(
                    config, key, variant.name, seed, test[key]["samples"],
                    test[key]["targets"],
                )
                summaries.append(summary)
                if verbose:
                    print(f"[pipeline] {variant.name} seed {seed}: "
                          f"CRPS_sum={summary.metrics['crps_sum']['mean']:.5f} "
                          f"CRPS={summary.metrics['crps']['mean']:.5f} "
                          f"MSE={summary.metrics['mse']['mean']:.4g}")

                if seed == config.export_seed:
                    checkpoints[key] = meta["checkpoint"]
                    training_meta[key] = {
                        "name": variant.name,
                        "baseline": variant.baseline,
                        "description": variant.description,
                        "parameters": meta["parameters"],
                        "config": meta["config"],
                        "history": meta["history"],
                    }

            boost = fit_boosting(config, bundle, models, validation, test, seed)
            summaries.extend(boost["summaries"])
            if verbose:
                print(f"\n[pipeline] seed {seed} ensemble weights: "
                      f"{ {k: round(v, 3) for k, v in boost['ensemble'].weights.items()} }")
                for summary in boost["summaries"]:
                    print(f"[pipeline] {summary.model_name} seed {seed}: "
                          f"CRPS_sum={summary.metrics['crps_sum']['mean']:.5f}")
            if seed == config.export_seed:
                export_payload = {
                    "ensemble": boost["ensemble"],
                    "booster": boost["booster"],
                    "test": test,
                    "mixture": boost["mixture"],
                    "models": models,
                }

        report = build_report(config, bundle, summaries, training_meta,
                              export_payload)
        export(config, bundle, report, checkpoints, training_meta, export_payload)
        report["bundle"] = bundle
        report["export"] = export_payload
        report["models"] = export_payload.get("models", {})
        return report
    except Exception as e:
        raise Exception(f"pipeline - run - Execution error: {e}") from e


def build_report(config: PipelineConfig, bundle: dict, summaries: list,
                 training_meta: dict, export_payload: dict) -> dict:
    """Paper tables, leaderboard, calibration and per-window records."""
    try:
        labels = {v.key: v.name for v in variants.MODEL_VARIANTS}
        labels[BOOSTED_KEY] = BOOSTED_NAME
        labels[BOOSTED_GB_KEY] = BOOSTED_GB_NAME
        present = []
        for key in list(labels):
            if any(summary.model_key == key for summary in summaries):
                present.append(key)

        tables = experiments.build_paper_tables(
            summaries, config.horizons, config.dataset_label, present, labels
        )
        board = experiments.leaderboard(summaries)

        coverage = None
        if export_payload.get("test"):
            targets = next(iter(export_payload["test"].values()))["targets"]
            frames = []
            for key, value in export_payload["test"].items():
                frame = experiments.coverage_report(
                    value["samples"], value["targets"], config.horizons
                )
                frame.insert(0, "Model", labels[key])
                frames.append(frame)
            if export_payload.get("mixture") is not None:
                frame = experiments.coverage_report(
                    export_payload["mixture"], targets, config.horizons
                )
                frame.insert(0, "Model", BOOSTED_NAME)
                frames.append(frame)
            coverage = pd.concat(frames, ignore_index=True)

        per_window_rows = []
        for summary in summaries:
            for window in summary.per_window:
                per_window_rows.append({
                    "Model": summary.model_name,
                    "model_key": summary.model_key,
                    "seed": summary.seed,
                    "window": window["window"],
                    "crps_sum": window["crps_sum"],
                    "crps": window["crps"],
                    "mse": window["mse"],
                    "nd": window["nd"],
                })

        per_seed_rows = []
        for summary in summaries:
            row = {
                "Model": summary.model_name, "model_key": summary.model_key,
                "seed": summary.seed,
            }
            for metric in experiments.REPORTED_METRICS:
                row[metric] = float(summary.metrics[metric]["mean"])
            per_seed_rows.append(row)

        horizon_rows = []
        for summary in summaries:
            for horizon, values in summary.per_horizon.items():
                horizon_rows.append({
                    "Model": summary.model_name, "model_key": summary.model_key,
                    "seed": summary.seed, "Return horizon": f"{horizon}-day",
                    **{name: float(value) for name, value in values.items()},
                })

        curves = {
            key: meta["history"] for key, meta in training_meta.items()
        }
        best = board.iloc[0]
        return {
            "labels": labels,
            "model_order": present,
            "tables": tables,
            "leaderboard": board,
            "coverage": coverage,
            "per_window": pd.DataFrame(per_window_rows),
            "per_seed": pd.DataFrame(per_seed_rows),
            "per_horizon": pd.DataFrame(horizon_rows),
            "learning_curves": curves,
            "summaries": summaries,
            "winner": {
                "model_key": str(best["model_key"]),
                "model_name": str(best["Model"]),
                "crps_sum": float(best["crps_sum_mean"]),
                "crps_sum_stderr": float(best.get("crps_sum_stderr", 0.0) or 0.0),
            },
        }
    except Exception as e:
        raise Exception(f"pipeline - build_report - Execution error: {e}") from e


def export(config: PipelineConfig, bundle: dict, report: dict,
           checkpoints: dict, training_meta: dict,
           export_payload: dict) -> Path:
    """Write checkpoints, metadata and every table the app reads."""
    try:
        export_dir = Path(config.export_dir)
        models_dir = export_dir / "models"
        tables_dir = export_dir / "experiments"
        for directory in (export_dir, models_dir, tables_dir):
            directory.mkdir(parents=True, exist_ok=True)

        model_entries = []
        for key, source in checkpoints.items():
            target = models_dir / f"{key}.pt"
            shutil.copy2(source, target)
            meta = training_meta[key]
            model_entries.append({
                "key": key,
                "name": meta["name"],
                "baseline": meta["baseline"],
                "description": meta["description"],
                "parameters": meta["parameters"],
                "checkpoint": f"models/{key}.pt",
                "config": meta["config"],
            })

        ensemble = export_payload.get("ensemble")
        booster = export_payload.get("booster")
        if booster is not None and ensemble is not None and ensemble.booster_enabled:
            import joblib

            joblib.dump(booster.model, export_dir / "gold_yahoo_booster.joblib")

        splits = bundle["splits"]
        returns = bundle["returns"]
        metadata = {
            "model_name": report["winner"]["model_name"],
            "ticker": config.ticker,
            "target": "multi_horizon_log_return",
            "target_definition": "return_h(t) = log(close(t) / close(t - h))",
            "reporting": (
                "horizon returns are reconstructed from the sampled price path "
                "spliced onto observed closes, so every h-day window ends on its "
                "own forecast day and already-traded days enter as known closes"
            ),
            "return_horizons": [int(h) for h in config.horizons],
            "series_labels": [f"{h}-day return" for h in config.horizons],
            "freq": "B",
            "num_series": bundle["num_series"],
            "cond_size": bundle["cond_size"],
            "context_length": config.context_length,
            "prediction_length": config.prediction_length,
            "lags": [int(lag) for lag in config.lags],
            "time_feature_spec": features.TIME_FEATURE_SPEC,
            "checkpoint_selection": "last_epoch",
            "training_epochs": config.epochs,
            "training_end_date": str(returns.index[splits["train_end"] - 1].date()),
            "validation_end_date": str(
                returns.index[splits["validation_end"] - 1].date()
            ),
            "test_end_date": str(returns.index[splits["test_end"] - 1].date()),
            "trained_at": pd.Timestamp.utcnow().isoformat(),
            "inference_seed": 20260817,
            "models": model_entries,
            "boosting": ensemble.to_metadata() if ensemble is not None else {},
            "experiment_protocol": {
                "seeds": [int(seed) for seed in config.seeds],
                "test_windows": config.test_windows,
                "validation_windows": config.validation_windows,
                "prediction_length": config.prediction_length,
                "evaluation_samples": config.evaluation_samples,
                "training_rows": int(splits["train_end"]),
                "total_rows": int(len(returns)),
                "ensemble_rounds": config.ensemble_rounds,
                "boost_fit_windows": config.boost_fit_windows,
                "boost_gate_windows": config.boost_gate_windows,
            },
            "winner": report["winner"],
            "model_contract": variants.MODEL_CONTRACT,
        }
        (export_dir / "gold_yahoo_metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )

        for name, payload in report["tables"].items():
            payload["frame"].to_csv(tables_dir / f"{name}.csv", encoding="utf-8")
        report["leaderboard"].to_csv(
            tables_dir / "leaderboard.csv", index=False, encoding="utf-8"
        )
        for name in ("per_window", "per_seed", "per_horizon"):
            frame = report.get(name)
            if frame is not None and not frame.empty:
                suffix = "per_window_metrics" if name == "per_window" else name
                frame.to_csv(
                    tables_dir / f"{suffix}.csv", index=False, encoding="utf-8"
                )
        if report.get("coverage") is not None:
            report["coverage"].to_csv(
                tables_dir / "coverage.csv", index=False, encoding="utf-8"
            )
        (tables_dir / "learning_curves.json").write_text(
            json.dumps(report["learning_curves"], indent=2, default=float),
            encoding="utf-8",
        )
        print(f"[pipeline] exported {len(model_entries)} checkpoints and "
              f"{len(list(tables_dir.glob('*')))} result files to {export_dir}")
        return export_dir
    except Exception as e:
        raise Exception(f"pipeline - export - Execution error: {e}") from e


def copy_to_drive(config: PipelineConfig,
                  destination: str = "/content/drive/MyDrive/gold_yahoo_artifacts") -> Path:
    try:
        target = Path(destination)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(Path(config.export_dir), target)
        print(f"[pipeline] copied artifacts to {target}")
        return target
    except Exception as e:
        raise Exception(f"pipeline - copy_to_drive - Execution error: {e}") from e


def format_winner(report: dict) -> str:
    try:
        winner = report["winner"]
        stderr = winner["crps_sum_stderr"]
        margin = f" ± {stderr:.5f}" if math.isfinite(stderr) and stderr > 0 else ""
        return (
            f"Best on gold by CRPS_sum: {winner['model_name']} at "
            f"{winner['crps_sum']:.5f}{margin}"
        )
    except Exception as e:
        raise Exception(f"pipeline - format_winner - Execution error: {e}") from e
