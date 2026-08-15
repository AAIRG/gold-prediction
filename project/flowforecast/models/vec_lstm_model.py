"""Vec-LSTM-ind-scaling baseline (Salinas et al., 2019a; paper Table 1).

Same backbone and same §4.1 objective as the flow model, but the head is a
diagonal Gaussian instead of a normalizing flow:

    mu_t, log_sigma_t = Linear(h_t)
    p(x_t | h_t) = prod_i N(x_t^i | mu_t^i, sigma_t^i)

That product is the point of the baseline: it has NO cross-series dependency at
a fixed time step, so any improvement the flow shows on CRPS_sum comes from
modelling the joint distribution -- which is exactly the paper's claim.

The head is exposed as `self.flow` and implements `sample(h)` / `log_prob(x, h)`,
so the §4.3 sampling loop in evaluation/sampling.py works unchanged for both
models.
"""

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from .inputs import prepare_model_inputs


class DiagonalGaussianHead(nn.Module):
    """Independent Gaussian per series -- the 'ind' in Vec-LSTM-ind-scaling."""

    def __init__(self, hidden_size: int, feature_size: int,
                 min_log_sigma: float = -7.0, max_log_sigma: float = 7.0):
        super().__init__()
        try:
            self.mu_proj = nn.Linear(hidden_size, feature_size)
            self.log_sigma_proj = nn.Linear(hidden_size, feature_size)
            self.min_log_sigma = min_log_sigma
            self.max_log_sigma = max_log_sigma
        except Exception as e:
            raise Exception(f"DiagonalGaussianHead - __init__ - Init error: {e}") from e

    def distribution(self, h: torch.Tensor) -> Normal:
        try:
            mu = self.mu_proj(h)
            log_sigma = self.log_sigma_proj(h).clamp(self.min_log_sigma,
                                                     self.max_log_sigma)
            return Normal(mu, torch.exp(log_sigma))
        except Exception as e:
            raise Exception(f"DiagonalGaussianHead - distribution - Execution error: {e}") from e

    def log_prob(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """[N, D] + [N, H] -> [N]: joint log-density over the D series."""
        try:
            return self.distribution(h).log_prob(x).sum(dim=-1)
        except Exception as e:
            raise Exception(f"DiagonalGaussianHead - log_prob - Execution error: {e}") from e

    def sample(self, h: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """One draw per row of h -> [N, D] (same contract as ConditionalFlow)."""
        try:
            if h.dim() == 1:
                h = h.unsqueeze(0)
            if num_samples > 1 and h.shape[0] == 1:
                h = h.expand(num_samples, -1)
            return self.distribution(h).sample()
        except Exception as e:
            raise Exception(f"DiagonalGaussianHead - sample - Execution error: {e}") from e


class VecLSTMModel(nn.Module):
    def __init__(self, backbone: nn.Module, feature_size: int, hidden_size: int,
                 context_length: int, time_feature_size: int,
                 dequantize_target: bool = False):
        super().__init__()
        try:
            self.backbone = backbone
            self.feature_size = feature_size
            self.flow = DiagonalGaussianHead(hidden_size, feature_size)
            self.context_length = int(context_length)
            self.time_feature_size = int(time_feature_size)
            self.dequantize_target = bool(dequantize_target)
        except Exception as e:
            raise Exception(f"VecLSTMModel - __init__ - Init error: {e}") from e

    def log_prob(self, x: torch.Tensor, covariates: torch.Tensor,
                 observed: torch.Tensor | None = None) -> torch.Tensor:
        try:
            x, covariates, _, _ = prepare_model_inputs(
                x, covariates, observed, self.context_length,
                self.feature_size, self.time_feature_size,
                self.dequantize_target,
                future_only_target=False,
            )
            h = self.backbone(x, covariates)
            B, T, D = x.shape
            log_prob = self.flow.log_prob(x.reshape(B * T, D),
                                          h.reshape(B * T, h.shape[-1]))
            return log_prob.reshape(B, T)
        except Exception as e:
            raise Exception(f"VecLSTMModel - log_prob - Execution error: {e}") from e

    def loss(self, x: torch.Tensor, covariates: torch.Tensor,
             observed: torch.Tensor | None = None) -> torch.Tensor:
        try:
            values = self.log_prob(x, covariates, observed=observed)
            if observed is None:
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
            raise Exception(f"VecLSTMModel - loss - Execution error: {e}") from e

    def forecast(self, x_history: torch.Tensor, time_feat_full: torch.Tensor,
                 lags: list, prediction_length: int, num_samples: int = 100,
                 context_length: int | None = None) -> torch.Tensor:
        try:
            from ..evaluation.sampling import sample_trajectories
            return sample_trajectories(
                self, x_history, time_feat_full, lags, prediction_length,
                num_samples=num_samples, context_length=context_length,
            )
        except Exception as e:
            raise Exception(f"VecLSTMModel - forecast - Execution error: {e}") from e
