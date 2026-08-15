"""Shared official-style scaling and dequantization for model inputs."""

import torch

from ..data.scaling import compute_scale, dequantize


def prepare_model_inputs(x: torch.Tensor, covariates: torch.Tensor,
                         observed: torch.Tensor | None,
                         context_length: int, num_series: int,
                         time_feature_size: int, dequantize_target: bool,
                         future_only_target: bool = False):
    """Scale raw target/lags from the context and dequantize density targets."""
    try:
        if observed is None:
            observed = torch.ones_like(x)
        context = x[:, :context_length, :]
        context_observed = observed[:, :context_length, :]
        scale = compute_scale(
            context, time_dim=1, observed=context_observed
        )

        density_target = x
        if dequantize_target:
            density_target = x.clone()
            start = context_length if future_only_target else 0
            density_target[:, start:] = dequantize(density_target[:, start:])
        density_target = density_target / scale.unsqueeze(1)

        # Every producer uses [lag blocks, time features]. Keep the same layout
        # here so training and autoregressive sampling receive identical inputs.
        if time_feature_size > 0:
            lag_part = covariates[:, :, :-time_feature_size]
            time_part = covariates[:, :, -time_feature_size:]
        else:
            lag_part = covariates
            time_part = covariates[:, :, :0]
        if lag_part.shape[-1] % num_series != 0:
            raise ValueError(
                f"lag width {lag_part.shape[-1]} is not divisible by "
                f"num_series={num_series}"
            )
        num_lags = lag_part.shape[-1] // num_series
        lag_scale = scale.repeat(1, num_lags).unsqueeze(1)
        scaled_covariates = torch.cat(
            [lag_part / lag_scale, time_part], dim=-1
        )
        return density_target, scaled_covariates, observed, scale
    except Exception as e:
        raise Exception(
            f"inputs - prepare_model_inputs - Execution error: {e}"
        ) from e
