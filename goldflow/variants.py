"""The four model variants that are trained, evaluated and served.

    LSTM-Real-NVP           paper Table 1 variant, RNN backbone + coupling flow
    LSTM-MAF                paper Table 1 variant, RNN backbone + MAF
    Transformer-MAF         paper Table 1 variant, attention backbone + MAF
    Vec-LSTM-ind-scaling    Salinas et al. (2019a) baseline, diagonal Gaussian

The first three are the paper's own models (roadmap.md, "Variants reported by
the paper"). The fourth is the baseline the paper compares against, and it is
the control that makes the comparison mean anything: it shares the backbone and
the objective but its head factorizes over the three horizons, so whatever the
flows win on CRPS_sum is attributable to modelling the joint cross-section
rather than to capacity.

Hyperparameters follow the project's own configs. The LSTM sizes are the
Salinas et al. values the paper defers to (Appendix D.2 only says "the LSTM
hyperparameters were the ones from Salinas et al. (2019a)"), so they are our
choice and are recorded in every checkpoint.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Variant:
    key: str
    name: str
    backbone: str
    flow: str | None
    baseline: str | None
    description: str


MODEL_VARIANTS: tuple[Variant, ...] = (
    Variant(
        key="lstm_realnvp",
        name="LSTM-Real-NVP",
        backbone="lstm",
        flow="realnvp",
        baseline=None,
        description="LSTM backbone conditioning a Real-NVP coupling flow over the three horizons.",
    ),
    Variant(
        key="lstm_maf",
        name="LSTM-MAF",
        backbone="lstm",
        flow="maf",
        baseline=None,
        description="LSTM backbone conditioning a masked autoregressive flow over the three horizons.",
    ),
    Variant(
        key="transformer_maf",
        name="Transformer-MAF",
        backbone="transformer",
        flow="maf",
        baseline=None,
        description="Attention backbone conditioning a masked autoregressive flow; the paper's strongest variant.",
    ),
    Variant(
        key="veclstm",
        name="Vec-LSTM-ind-scaling",
        backbone="lstm",
        flow=None,
        baseline="veclstm",
        description="Baseline with a diagonal Gaussian head: no cross-horizon dependency at a fixed day.",
    ),
)

VARIANTS_BY_KEY = {variant.key: variant for variant in MODEL_VARIANTS}

MODEL_CONTRACT = "tempflow_v5_official_parity"


def variant_names() -> list[str]:
    try:
        return [variant.key for variant in MODEL_VARIANTS]
    except Exception as e:
        raise Exception(f"variants - variant_names - Execution error: {e}") from e


def get_variant(key: str) -> Variant:
    try:
        normalized = str(key).strip().lower()
        if normalized not in VARIANTS_BY_KEY:
            raise ValueError(
                f"unknown variant {key!r}; expected one of {variant_names()}"
            )
        return VARIANTS_BY_KEY[normalized]
    except Exception as e:
        raise Exception(f"variants - get_variant - Execution error: {e}") from e


def base_config(num_series: int, context_length: int, prediction_length: int,
                lags: list, epochs: int, batch_size: int,
                batches_per_epoch: int, validation_windows: int,
                test_windows: int, evaluation_samples: int,
                validation_samples: int, seed: int,
                dataset_name: str = "gold_yahoo_multi_horizon") -> dict:
    """Shared config skeleton; `variant_config` stamps the backbone and flow."""
    try:
        return {
            "dataset": {
                "name": dataset_name,
                "freq": "B",
                "num_series": int(num_series),
                "prediction_length": int(prediction_length),
                "context_length": int(context_length),
                "lags": [int(lag) for lag in lags],
                "dequantize": False,
            },
            "flow": {
                "type": "maf",
                "num_blocks": 5,
                "hidden": 100,
                "batch_norm": True,
                "bn_min_var": 1e-4,
                "maf_log_scale_bound": 3.0,
                "maf_mu_bound": 10.0,
            },
            "backbone": {
                "type": "transformer",
                "d_model": 32,
                "n_heads": 8,
                "num_encoder_layers": 3,
                "num_decoder_layers": 3,
                "dropout": 0.1,
                "hidden_size": 40,
                "num_layers": 2,
            },
            "model": {"conditioning_length": 200, "condition_bound": 10.0},
            "training": {
                "lr": 1e-3,
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "batches_per_epoch": int(batches_per_epoch),
                "maf_grad_clip": 5.0,
                "grad_clip": 5.0,
                "seed": int(seed),
                "val_windows": int(validation_windows),
                "select_metric": "last",
                "validation_num_samples": int(validation_samples),
                "validation_seed": 1729,
                "validation_interval": 5,
                "lr_schedule": "onecycle",
                "max_lr": 3e-3,
            },
            "evaluation": {
                "num_samples": int(evaluation_samples),
                "num_test_windows": int(test_windows),
            },
        }
    except Exception as e:
        raise Exception(f"variants - base_config - Execution error: {e}") from e


def variant_config(key: str, config: dict, seed: int | None = None) -> dict:
    """Deep copy of `config` with the variant's backbone and flow applied."""
    try:
        variant = get_variant(key)
        stamped = copy.deepcopy(config)
        stamped["backbone"] = dict(stamped.get("backbone") or {})
        stamped["backbone"]["type"] = variant.backbone
        if variant.flow is not None:
            stamped["flow"] = dict(stamped.get("flow") or {})
            stamped["flow"]["type"] = variant.flow
        if seed is not None:
            stamped["training"] = dict(stamped.get("training") or {})
            stamped["training"]["seed"] = int(seed)
        return stamped
    except Exception as e:
        raise Exception(f"variants - variant_config - Execution error: {e}") from e


def build_variant(key: str, config: dict, num_series: int, cond_size: int,
                  context_length: int, device: str = "cpu") -> torch.nn.Module:
    """Instantiate one variant through the project's single model builder."""
    try:
        from flowforecast.utils.build import build_model

        variant = get_variant(key)
        model = build_model(
            variant_config(key, config), num_series=int(num_series),
            cond_size=int(cond_size), baseline=variant.baseline,
            context_length=int(context_length),
        )
        return model.to(device)
    except Exception as e:
        raise Exception(f"variants - build_variant - Execution error: {e}") from e
