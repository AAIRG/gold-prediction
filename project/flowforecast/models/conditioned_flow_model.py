"""Backbone (LSTM / Transformer) + conditional flow head (paper eq. 4).

    p_X(x_{t0:T} | x_{1:t0-1}, c_{1:T}; theta) = prod_{t=t0}^{T} p_X(x_t | h_t; theta)

theta covers the parameters of BOTH the flow and the backbone -- the model is
trained end to end.

Training objective (§4.1):

    L = 1 / (|D| * T) * sum_{x_{1:T} in D} sum_{t=1}^{T} log p_X(x_t | h_t; theta)

The sum starts at t = 1, not t = t0: the likelihood is scored over the whole
sampled window, context included. p_X(x_t | h_t) is already the JOINT density of
the D-dimensional cross-section, so nothing is summed over D here -- the flow
returns one log-density per (batch, time) row.

Sanity check before anything else (roadmap M4): `--overfit-one-window`. The NLL
must go very low. If it does not, the bug is in the h_t -> flow wiring (most
likely the off-by-one shift), not in the hyperparameters.
"""

import torch
import torch.nn as nn

from .inputs import prepare_model_inputs


class ConditionedFlowModel(nn.Module):
    def __init__(self, backbone: nn.Module, flow: nn.Module,
                 context_length: int | None = None,
                 conditioning_length: int = 200,
                 time_feature_size: int = 0,
                 dequantize_target: bool = False,
                 condition_bound: float = 10.0):
        super().__init__()
        try:
            self.backbone = backbone
            self.flow = flow
            self.context_length = context_length
            self.conditioning_length = int(conditioning_length)
            self.time_feature_size = int(time_feature_size)
            self.dequantize_target = bool(dequantize_target)
            self.condition_bound = float(condition_bound)
            self.condition_proj = nn.Linear(
                int(backbone.hidden_size), self.conditioning_length
            )
            nn.init.normal_(self.condition_proj.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.condition_proj.bias)
            nn.init.normal_(self.condition_proj.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.condition_proj.bias)
        except Exception as e:
            raise Exception(f"ConditionedFlowModel - __init__ - Init error: {e}") from e

    def log_prob(self, x: torch.Tensor, covariates: torch.Tensor,
                 observed: torch.Tensor | None = None) -> torch.Tensor:
        """[B, T, D] + [B, T, C] -> log p_X(x_t | h_t) as [B, T]."""
        try:
            x, covariates, _, _ = prepare_model_inputs(
                x, covariates, observed, self.context_length,
                self.flow.features, self.time_feature_size,
                self.dequantize_target,
                future_only_target=getattr(
                    self.backbone, "loss_on_future_only", False
                ),
            )
            h = self.backbone(x, covariates, context_length=self.context_length)
            if getattr(self.backbone, "loss_on_future_only", False):
                x = x[:, self.context_length:, :]
            B, T, D = x.shape
            condition = self.project_condition(h)
            log_prob = self.flow.log_prob(
                x.reshape(B * T, D),
                condition.reshape(B * T, condition.shape[-1]),
            )
            return log_prob.reshape(B, T)
        except Exception as e:
            raise Exception(f"ConditionedFlowModel - log_prob - Execution error: {e}") from e

    def loss(self, x: torch.Tensor, covariates: torch.Tensor,
             observed: torch.Tensor | None = None) -> torch.Tensor:
        """Reference-weighted negative log-likelihood."""
        try:
            values = self.log_prob(x, covariates, observed=observed)
            if getattr(self.backbone, "loss_on_future_only", False) and observed is not None:
                observed = observed[:, self.context_length:, :]
            elif observed is not None:
                observed = observed
            if observed is None or not getattr(
                    self.backbone, "loss_uses_observed_mask", True):
                return -values.mean()
            weights = observed.to(values.dtype).amin(dim=-1)
            weighted = torch.where(
                weights > 0, values * weights, torch.zeros_like(values)
            )
            per_item = weighted.sum(dim=1) / weights.sum(
                dim=1
            ).clamp_min(1.0)
            return -per_item.mean()
        except Exception as e:
            raise Exception(f"ConditionedFlowModel - loss - Execution error: {e}") from e

    def project_condition(self, h: torch.Tensor) -> torch.Tensor:
        try:
            projected = self.condition_proj(h)
            projected = torch.nan_to_num(projected, nan=0.0, posinf=1e3, neginf=-1e3)
            return self.condition_bound * torch.tanh(projected / self.condition_bound)
        except Exception as e:
            raise Exception(
                f"ConditionedFlowModel - project_condition - Execution error: {e}"
            ) from e

    def forecast(self, x_history: torch.Tensor, time_feat_full: torch.Tensor,
                 lags: list, prediction_length: int, num_samples: int = 100,
                 context_length: int | None = None) -> torch.Tensor:
        """Autoregressive sampling (§4.3) -> [num_samples, D, prediction_length].

        Inputs and outputs are in the mean-scaled space; the caller inverts the
        scaling (data/scaling.py) to get original units.
        """
        try:
            from ..evaluation.sampling import sample_trajectories
            return sample_trajectories(
                self, x_history, time_feat_full, lags, prediction_length,
                num_samples=num_samples, context_length=context_length,
            )
        except Exception as e:
            raise Exception(f"ConditionedFlowModel - forecast - Execution error: {e}") from e
