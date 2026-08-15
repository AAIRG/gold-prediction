"""Autoregressive trajectory sampling aligned with TempFlow prediction."""

import torch

from ..data.covariates import lag_vector_from_history
from ..data.scaling import compute_scale


def _project(model, hidden: torch.Tensor) -> torch.Tensor:
    return model.project_condition(hidden) \
        if hasattr(model, "project_condition") else hidden


def sample_trajectories(model, x_history: torch.Tensor,
                        time_feat_full: torch.Tensor, lags: list,
                        prediction_length: int, num_samples: int = 100,
                        context_length: int | None = None,
                        scale: torch.Tensor | None = None) -> torch.Tensor:
    """Return scaled samples from raw history.

    ``x_history`` is raw [1, max_lag + context, D] data. Scaling is performed
    once here, using the supplied context-only scale when evaluation already
    computed the official batch fallback.
    """
    try:
        if x_history.dim() != 3 or x_history.shape[0] != 1:
            raise ValueError(
                f"x_history must be [1, n_hist, D], got {tuple(x_history.shape)}"
            )
        if time_feat_full.dim() != 3 or time_feat_full.shape[0] != 1:
            raise ValueError(
                f"time_feat_full must be [1, T, C_time], got "
                f"{tuple(time_feat_full.shape)}"
            )

        n_hist = x_history.shape[1]
        max_lag = max(lags)
        ctx_len = int(context_length or (n_hist - max_lag))
        if n_hist != max_lag + ctx_len:
            raise ValueError(
                f"history must contain max_lag + context_length = "
                f"{max_lag + ctx_len} rows, got {n_hist}"
            )
        if time_feat_full.shape[1] < n_hist + prediction_length:
            raise ValueError("time_feat_full does not cover history and horizon")

        was_training = model.training
        model.eval()
        with torch.no_grad():
            device = next(model.parameters()).device
            raw_history = x_history.to(device)
            if scale is None:
                scale = compute_scale(
                    raw_history[:, max_lag:, :], time_dim=1
                )[0]
            else:
                scale = scale.to(device=device, dtype=raw_history.dtype)
            history = (raw_history / scale[None, None, :]).repeat(
                num_samples, 1, 1
            )
            time_feat = time_feat_full.to(device).repeat(num_samples, 1, 1)

            def covariate_at(index: int) -> torch.Tensor:
                lag_part = lag_vector_from_history(history[:, :index, :], lags)
                return torch.cat([lag_part, time_feat[:, index, :]], dim=-1)

            context_cov = torch.stack(
                [covariate_at(t) for t in range(max_lag, n_hist)], dim=1
            )
            samples = []

            if hasattr(model.backbone, "step"):
                state = None
                for row in range(context_cov.shape[1]):
                    _, state = model.backbone.step(
                        history[:, max_lag + row:max_lag + row + 1, :],
                        context_cov[:, row:row + 1, :], state,
                    )

                for k in range(prediction_length):
                    index = n_hist + k
                    c_next = covariate_at(index).unsqueeze(1)
                    hidden, state = model.backbone.step(
                        history[:, -1:, :], c_next, state
                    )
                    x_next = model.flow.sample(_project(model, hidden[:, -1, :]))
                    samples.append(x_next)
                    history = torch.cat([history, x_next.unsqueeze(1)], dim=1)
            else:
                memory = model.backbone.encode(context_cov, context_cov.shape[1])
                for k in range(prediction_length):
                    index = n_hist + k
                    c_next = covariate_at(index).unsqueeze(1)
                    hidden = model.backbone.decode(c_next, memory)
                    x_next = model.flow.sample(_project(model, hidden[:, -1, :]))
                    samples.append(x_next)
                    history = torch.cat([history, x_next.unsqueeze(1)], dim=1)

            out = torch.stack(samples, dim=1).transpose(1, 2).contiguous()

        if was_training:
            model.train()
        return out
    except Exception as e:
        raise Exception(
            f"sampling - sample_trajectories - Execution error: {e}"
        ) from e


def random_baseline_samples(num_samples: int, num_series: int,
                            prediction_length: int, device=None) -> torch.Tensor:
    try:
        return torch.randn(
            num_samples, num_series, prediction_length, device=device
        )
    except Exception as e:
        raise Exception(
            f"sampling - random_baseline_samples - Execution error: {e}"
        ) from e
