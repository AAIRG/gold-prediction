"""Mean-scaling and dequantization.

Mean-scaling (paper §4, paragraph before 4.1): "we divide each individual time
series by their training window means before feeding it into the model. At
inference the distributions are then correspondingly transformed with the same
mean values to match the original scale."

Dequantization (§2.1 end, §5.2): discrete data has differential entropy of
negative infinity, which lets a flow reach arbitrarily high likelihood even on
test data. Fix: add Uniform[0, 1) noise to integer-valued data. The resulting
continuous log-likelihood lower-bounds the discrete one (Theis et al., 2016).
Order matters -- §5.2 says Taxi and Wikipedia "are thus dequantized before
being fed to the flow (and mean-scaled)".

LEAKAGE WARNING (our decision, the paper does not spell this out): the scale is
always computed from the CONTEXT window only, never from a window that includes
the prediction horizon. See data/windowing.py and scripts/evaluate.py -- both
compute it from the context part exclusively.

Shape convention (this is what used to break): every function takes an explicit
`time_dim`, the axis along which time runs in `x`. `compute_scale` reduces that
axis, and `apply_scale` / `invert_scale` re-insert it, so the same three
functions work for

    raw series      [D, T]              time_dim=-1  -> scale [D]
    model batches   [B, T, D]           time_dim=1   -> scale [B, D]
    sample sets     [num_samples, D, P]  time_dim=-1  -> scale [D] (broadcast)

Acceptance criterion (M3): invert_scale(apply_scale(x, s), s) == x.
"""

import torch


def compute_scale(x_context: torch.Tensor, time_dim: int = -1,
                  observed: torch.Tensor | None = None,
                  default_scale: torch.Tensor | None = None) -> torch.Tensor:
    """Mean absolute value per series over the context window.

    Series that are identically zero over the context use the official
    per-dimension batch fallback when a batch is available, then the official
    minimum scale of 1e-10.
    """
    try:
        if observed is None:
            weights = torch.ones_like(x_context)
        else:
            if observed.shape != x_context.shape:
                raise ValueError(
                    f"observed {tuple(observed.shape)} must match "
                    f"x_context {tuple(x_context.shape)}"
                )
            weights = observed.to(dtype=x_context.dtype)

        denominator = weights.sum(dim=time_dim)
        numerator = (x_context.abs() * weights).sum(dim=time_dim)
        means = numerator / denominator.clamp_min(1.0)

        if default_scale is None and x_context.dim() >= 3:
            batch_dim = 0
            global_denominator = denominator.sum(dim=batch_dim)
            global_numerator = numerator.sum(dim=batch_dim)
            default_scale = global_numerator / global_denominator.clamp_min(1.0)

        if default_scale is None:
            fallback = torch.full_like(means, 1e-10)
        else:
            fallback = default_scale.to(device=means.device, dtype=means.dtype)
            while fallback.dim() < means.dim():
                fallback = fallback.unsqueeze(0)
            fallback = torch.broadcast_to(fallback, means.shape)

        # Official GluonTS MeanScaler falls back not only when observations are
        # missing, but also when an observed context is identically zero. This
        # is essential for sparse series such as Solar at night: using 1e-10
        # there makes the next non-zero lag roughly 1e10 in model space.
        scale = torch.where(numerator > 0, means, fallback)
        return scale.clamp_min(1e-10).detach()
    except Exception as e:
        raise Exception(f"scaling - compute_scale - Execution error: {e}") from e


def apply_scale(x: torch.Tensor, scale: torch.Tensor,
                time_dim: int = -1) -> torch.Tensor:
    """x / scale, where `scale` is the output of compute_scale for the same layout."""
    try:
        return x / _align(scale, x, time_dim)
    except Exception as e:
        raise Exception(f"scaling - apply_scale - Execution error: {e}") from e


def invert_scale(x_scaled: torch.Tensor, scale: torch.Tensor,
                 time_dim: int = -1) -> torch.Tensor:
    """x_scaled * scale -- brings samples back to the original units."""
    try:
        return x_scaled * _align(scale, x_scaled, time_dim)
    except Exception as e:
        raise Exception(f"scaling - invert_scale - Execution error: {e}") from e


def dequantize(x_counts: torch.Tensor,
               generator: torch.Generator | None = None) -> torch.Tensor:
    """x + Uniform[0, 1) for integer valued (count) data, §2.1 / §5.2."""
    try:
        noise = torch.rand(x_counts.shape, dtype=x_counts.dtype,
                           device=x_counts.device, generator=generator)
        return x_counts + noise
    except Exception as e:
        raise Exception(f"scaling - dequantize - Execution error: {e}") from e


def _align(scale: torch.Tensor, x: torch.Tensor, time_dim: int) -> torch.Tensor:
    """Re-insert the reduced time axis so `scale` broadcasts against `x`."""
    try:
        if scale.dim() == x.dim():
            return scale
        if scale.dim() != x.dim() - 1:
            # e.g. per-series scale [D] against samples [S, D, P]: unsqueeze the
            # time axis and let broadcasting handle the leading dims.
            axis = time_dim if time_dim >= 0 else x.dim() + time_dim
            axis = min(axis, scale.dim())
            return scale.unsqueeze(axis)
        axis = time_dim if time_dim >= 0 else x.dim() + time_dim
        return scale.unsqueeze(axis)
    except Exception as e:
        raise Exception(f"scaling - _align - Execution error: {e}") from e
