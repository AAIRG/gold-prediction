"""One place that turns a config file into data and a model.

Both scripts/train.py and scripts/evaluate.py go through here. That is the whole
point: the previous version built the LSTM with `cond_size = C - D` in one script
and fed it C + D features in the other, which cannot work, and it scaled the
targets only at evaluation time. A single builder makes those mismatches
impossible -- if training and evaluation disagree, they disagree everywhere at
once and the unit tests catch it.

Config keys, all optional unless stated:

    dataset:      name (required), cache_dir, fallback_path, context_length,
                  prediction_length, lags, dequantize, num_series, strict
    backbone:     type (lstm|transformer), hidden_size, num_layers, n_heads,
                  num_encoder_layers, num_decoder_layers, dropout
    flow:         type (realnvp|maf), num_blocks, hidden, batch_norm
    training:     lr, epochs, batch_size, batches_per_epoch, grad_clip, seed,
                  windows_per_epoch, val_windows, early_stopping_patience
    evaluation:   num_samples, num_test_windows
"""

import csv
import copy
import json
import os
from dataclasses import dataclass, field

import pandas as pd
import torch

from ..data.covariates import (covariate_dim, lags_for_freq, make_timestamps,
                               num_time_features, time_features)
from ..data.datasets import DATASET_SPECS, MultivariateDataset, load_dataset


@dataclass
class DataBundle:
    """Everything the scripts need after the data layer has done its job."""
    dataset: MultivariateDataset
    time_feat: torch.Tensor          # [C_time, T_total]
    lags: list
    context_length: int
    prediction_length: int
    num_test_windows: int
    train_end: int                   # forecast start of the first test window
    dequantize: bool
    timestamps: pd.DatetimeIndex = field(repr=False, default=None)

    # The TRAINING array and its own calendar features. GluonTS groups the train
    # and test splits with two MultivariateGrouper instances, each computing its
    # own `first_timestamp`, so the two arrays sit on DIFFERENT calendar origins
    # and neither is a prefix of the other. The reference implementation trains
    # on the train array and evaluates the test entries separately; we do the
    # same. `train_end` below still means "first forecast start inside the TEST
    # array", which is what scripts/evaluate.py needs.
    train_series_values: torch.Tensor = field(repr=False, default=None)
    train_time_feat: torch.Tensor = field(repr=False, default=None)
    train_timestamps: pd.DatetimeIndex = field(repr=False, default=None)
    observed_values: torch.Tensor = field(repr=False, default=None)
    train_observed_values: torch.Tensor = field(repr=False, default=None)

    @property
    def values(self) -> torch.Tensor:
        return self.dataset.values

    @property
    def train_series(self) -> torch.Tensor:
        """Array training windows are drawn from."""
        return self.dataset.values if self.train_series_values is None \
            else self.train_series_values

    @property
    def train_series_time_feat(self) -> torch.Tensor:
        """Calendar features aligned with `train_series`."""
        return self.time_feat if self.train_time_feat is None \
            else self.train_time_feat

    @property
    def observed(self) -> torch.Tensor:
        return torch.ones_like(self.values) if self.observed_values is None \
            else self.observed_values

    @property
    def train_observed(self) -> torch.Tensor:
        if self.train_observed_values is not None:
            return self.train_observed_values
        if self.train_series_values is None:
            return self.observed
        return torch.ones_like(self.train_series_values)

    @property
    def train_series_end(self) -> int:
        """Usable end of the training range, inside `train_series`.

        With a separate train array the whole array is training data: GluonTS
        already cut the test range out of it. Without one we fall back to the
        contiguous-array split.
        """
        if self.train_series_values is None:
            return self.train_end
        return int(self.train_series_values.shape[1])

    @property
    def num_series(self) -> int:
        return self.dataset.num_series

    @property
    def freq(self) -> str:
        return self.dataset.freq

    @property
    def cond_size(self) -> int:
        return self.time_feat.shape[0] + self.num_series * len(self.lags)

    @property
    def max_lag(self) -> int:
        return max(self.lags)


def prepare_data(config: dict, strict: bool | None = None) -> DataBundle:
    try:
        data_cfg = dict(config.get("dataset") or {})
        eval_cfg = dict(config.get("evaluation") or {})
        name = data_cfg.get("name")
        if not name:
            raise ValueError("config['dataset']['name'] is required")

        dataset = load_dataset(
            name,
            cache_dir=data_cfg.get("cache_dir"),
            fallback_path=data_cfg.get("fallback_path") or data_cfg.get("path"),
            strict=bool(data_cfg.get("strict", False)) if strict is None else strict,
            require_gluonts=bool(data_cfg.get("require_gluonts", False)),
        )
        print(f"[build] {dataset.summary()}")

        spec = DATASET_SPECS[str(name).strip().lower()]
        prediction_length = int(data_cfg.get("prediction_length")
                                or dataset.prediction_length
                                or spec.prediction_length)
        context_length = int(data_cfg.get("context_length") or prediction_length)
        lags = list(data_cfg.get("lags") or lags_for_freq(dataset.freq))
        num_test_windows = int(eval_cfg.get("num_test_windows", 5))

        configured_series = data_cfg.get("num_series")
        if configured_series and int(configured_series) != dataset.num_series:
            print(f"[build] WARNING config says num_series="
                  f"{configured_series} but the loaded data has "
                  f"{dataset.num_series}; using the data.")

        timestamps = make_timestamps(dataset.start, dataset.total_timesteps,
                                     dataset.freq)
        tf = time_features(timestamps, dataset.freq)

        dequantize = bool(data_cfg.get("dequantize", dataset.is_count))
        train_end = dataset.total_timesteps - prediction_length * num_test_windows

        # The grouped TRAIN array, when gluonts gave us one, with its own
        # calendar. It is not a prefix of the test array: the two groupers use
        # different time origins (Electricity: train 5833 steps from 2014-01-01,
        # test 4144 steps from 2014-03-19). Training reads this array, evaluation
        # reads `dataset.values`; that separation is what makes Taxi's published
        # 56-window protocol expressible at all.
        train_series_values = dataset.train_values
        train_tf = None
        train_timestamps = None
        if train_series_values is not None:
            train_timestamps = make_timestamps(
                dataset.train_start or dataset.start,
                int(train_series_values.shape[1]), dataset.freq)
            train_tf = time_features(train_timestamps, dataset.freq)

        train_range_end = int(train_series_values.shape[1]) \
            if train_series_values is not None else train_end
        if train_range_end <= context_length + max(lags):
            raise ValueError(
                f"training range of {train_range_end} steps is too short for "
                f"context {context_length} plus max lag {max(lags)}"
            )
        if train_end <= 0:
            raise ValueError(
                f"the test array holds {dataset.total_timesteps} steps but "
                f"{num_test_windows} windows of {prediction_length} need "
                f"{prediction_length * num_test_windows}: lower "
                f"evaluation.num_test_windows"
            )

        bundle = DataBundle(
            dataset=dataset, time_feat=tf, lags=lags,
            context_length=context_length, prediction_length=prediction_length,
            num_test_windows=num_test_windows, train_end=train_end,
            dequantize=dequantize, timestamps=timestamps,
            train_series_values=train_series_values, train_time_feat=train_tf,
            train_timestamps=train_timestamps,
            observed_values=dataset.observed_values,
            train_observed_values=dataset.train_observed_values,
        )
        print(f"[build] windows: context={context_length}, "
              f"prediction={prediction_length}, lags={lags}, "
              f"C_time={tf.shape[0]}, cond_size={bundle.cond_size}, "
              f"train array [0, {bundle.train_series_end}), "
              f"test array {dataset.total_timesteps} steps with "
              f"{num_test_windows} windows from {train_end}, "
              f"dequantize={dequantize}")
        return bundle
    except Exception as e:
        raise Exception(f"build - prepare_data - Execution error: {e}") from e


def build_backbone(config: dict, num_series: int, cond_size: int,
                   context_length: int | None = None) -> torch.nn.Module:
    try:
        backbone_cfg = dict(config.get("backbone") or {})
        data_cfg = dict(config.get("dataset") or {})
        configured_lags = list(data_cfg.get("lags") or [])
        time_feature_size = cond_size - num_series * len(configured_lags) \
            if configured_lags else cond_size % num_series
        kind = str(backbone_cfg.get("type", "lstm")).lower()

        if kind == "lstm":
            from ..models.backbone_lstm import LSTMBackbone
            return LSTMBackbone(
                input_size=num_series,
                cond_size=cond_size,
                hidden_size=int(backbone_cfg.get("hidden_size", 40)),
                num_layers=int(backbone_cfg.get("num_layers", 2)),
                dropout=float(backbone_cfg.get("dropout", 0.0)),
                time_feature_size=time_feature_size,
            )
        if kind == "transformer":
            from ..models.backbone_transformer import TransformerBackbone
            return TransformerBackbone(
                input_size=num_series,
                cond_size=cond_size,
                d_model=int(backbone_cfg.get("d_model", 32)),
                n_heads=int(backbone_cfg.get("n_heads", 8)),
                num_encoder_layers=int(backbone_cfg.get("num_encoder_layers", 3)),
                num_decoder_layers=int(backbone_cfg.get("num_decoder_layers", 3)),
                dropout=float(backbone_cfg.get("dropout", 0.1)),
                context_length=context_length,
                time_feature_size=time_feature_size,
            )
        raise ValueError(
            f"Unknown backbone type {kind!r}; expected 'lstm' or 'transformer'"
        )
    except Exception as e:
        raise Exception(f"build - build_backbone - Execution error: {e}") from e


def build_model(config: dict, num_series: int, cond_size: int,
                baseline: str | None = None,
                context_length: int | None = None) -> torch.nn.Module:
    """Backbone + head. `baseline='veclstm'` swaps the flow for a diagonal Gaussian."""
    try:
        model_config = config
        if baseline == "veclstm":
            model_config = copy.deepcopy(config)
            model_config["backbone"] = {
                "type": "lstm",
                "hidden_size": 40,
                "num_layers": 2,
                "dropout": 0.1,
            }
        backbone = build_backbone(
            model_config, num_series, cond_size, context_length
        )
        hidden_size = backbone.hidden_size
        conditioning_length = int(
            (config.get("model") or {}).get("conditioning_length", 200)
        )
        data_cfg = dict(model_config.get("dataset") or {})
        configured_lags = list(data_cfg.get("lags") or [])
        time_feature_size = cond_size - num_series * len(configured_lags) \
            if configured_lags else cond_size % num_series
        dequantize_target = bool(data_cfg.get("dequantize", False))

        if baseline == "veclstm":
            from ..models.vec_lstm_model import VecLSTMModel
            return VecLSTMModel(
                backbone, feature_size=num_series, hidden_size=hidden_size,
                context_length=int(context_length),
                time_feature_size=time_feature_size,
                dequantize_target=dequantize_target,
            )

        flow_cfg = dict(config.get("flow") or {})
        from ..flows.flow import ConditionalFlow
        from ..models.conditioned_flow_model import ConditionedFlowModel
        block_type = str(flow_cfg.get("type", "realnvp")).lower()
        flow = ConditionalFlow(
            features=num_series,
            cond_size=conditioning_length,
            num_blocks=int(flow_cfg.get("num_blocks", 5)),
            hidden=int(flow_cfg.get("hidden", 100)),
            block_type=block_type,
            batch_norm=bool(flow_cfg.get("batch_norm", True)),
            bn_momentum=float(flow_cfg.get("bn_momentum", 0.1)),
            bn_min_var=float(flow_cfg.get("bn_min_var", 1e-4)),
            maf_log_scale_bound=float(
                flow_cfg.get("maf_log_scale_bound", 3.0)
            ),
            maf_mu_bound=float(flow_cfg.get("maf_mu_bound", 10.0)),
        )
        return ConditionedFlowModel(
            backbone, flow, context_length=context_length,
            conditioning_length=conditioning_length,
            time_feature_size=time_feature_size,
            dequantize_target=dequantize_target,
            condition_bound=float((config.get("model") or {}).get("condition_bound", 10.0)),
        )
    except Exception as e:
        raise Exception(f"build - build_model - Execution error: {e}") from e


def model_tag(config: dict, baseline: str | None = None) -> str:
    """Short identifier of the variant, e.g. lstm_realnvp / transformer_maf."""
    try:
        if baseline:
            return str(baseline)
        backbone = str((config.get("backbone") or {}).get("type", "lstm")).lower()
        flow = str((config.get("flow") or {}).get("type", "realnvp")).lower()
        return f"{backbone}_{flow}"
    except Exception as e:
        raise Exception(f"build - model_tag - Execution error: {e}") from e


def run_name(config: dict, baseline: str | None = None,
             seed: int | None = None, suffix: str | None = None) -> str:
    try:
        parts = [str((config.get("dataset") or {}).get("name", "dataset")).lower(),
                 model_tag(config, baseline)]
        if seed is not None:
            parts.append(f"seed{seed}")
        if suffix:
            parts.append(suffix)
        return "_".join(parts)
    except Exception as e:
        raise Exception(f"build - run_name - Execution error: {e}") from e


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(payload: dict, path: str) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        print(f"[build] wrote {path}")
    except Exception as e:
        raise Exception(f"build - save_json - Execution error: {e}") from e


def append_csv_row(path: str, row: dict) -> None:
    """Upsert one result row by ``run``, preserving a union CSV schema."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        exists = os.path.exists(path)
        rows = []
        fields = []
        if exists:
            with open(path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fields = list(reader.fieldnames or [])

        key = row.get("run")
        if key is not None:
            rows = [existing for existing in rows
                    if existing.get("run") != str(key)]
        rows.append({k: str(v) for k, v in row.items()})
        fields.extend(k for k in row if k not in fields)

        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[build] upserted run {key!r} in {path}")
    except Exception as e:
        raise Exception(f"build - append_csv_row - Execution error: {e}") from e


def expected_covariate_dim(num_series: int, freq: str, lags: list) -> int:
    """C_time + D * len(lags) -- the backbone cond_size, spelled out."""
    try:
        return covariate_dim(num_series, freq, lags)
    except Exception as e:
        raise Exception(f"build - expected_covariate_dim - Execution error: {e}") from e


def time_feature_count(freq: str) -> int:
    try:
        return num_time_features(freq)
    except Exception as e:
        raise Exception(f"build - time_feature_count - Execution error: {e}") from e
